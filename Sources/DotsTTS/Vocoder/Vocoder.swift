import Foundation
import MLX
import MLXNN

/// dots.tts-soar AudioVAE decoder (latent -> 48 kHz waveform).
///
/// BigVGAN-style causal decoder with snakebeta activations and anti-aliased
/// up/down resampling, preceded by a mutual-information LSTM block. Swift port
/// of the architecture in `spec_vocoder.md`. Parameter paths mirror the converted
/// `vocoder_decoder_mlx.safetensors` keys exactly (snake_case via @ModuleInfo).
///
/// All math runs in float32. mlx conv ops are channels-last (B, L, C); the public
/// entry point takes/returns PyTorch-style (B, C, L) and permutes internally.
public final class Vocoder: Module {
    public struct Config: Sendable {
        public var latentDim = 128
        public var upsampleInitialChannel = 1536
        public var upsampleRates = [10, 6, 4, 2, 2, 2]
        public var upsampleKernelSizes = [20, 12, 8, 4, 4, 4]
        public var resblockKernelSizes = [3, 7, 11]
        public var resblockDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        public var miNumLayers = 4
        public init() {}
    }

    @ModuleInfo(key: "post_proj") var postProj: Conv1d
    // nn.Sequential: dec_mi_layer.0 (Linear), .1 (SLSTM), .2 (Linear).
    @ModuleInfo(key: "dec_mi_layer") var decMiLayer: [UnaryLayer]
    @ModuleInfo(key: "decoder") var decoder: BigVGANDecoder

    public init(_ cfg: Config = Config()) {
        // post_proj: Conv1d 128->128, k=1, NON-causal (pad=0).
        self._postProj.wrappedValue = Conv1d(
            inputChannels: cfg.latentDim, outputChannels: cfg.latentDim,
            kernelSize: 1, stride: 1, padding: 0, bias: true)
        self._decMiLayer.wrappedValue = [
            Linear(cfg.latentDim, 512, bias: true),
            SLSTM(size: 512, numLayers: cfg.miNumLayers),
            Linear(512, cfg.latentDim, bias: true),
        ]
        self._decoder.wrappedValue = BigVGANDecoder(cfg)
        super.init()
    }

    /// x: (B, 128, T) denormalized latent. Returns (B, 1, T*1920) waveform.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // post_proj on (B, T, C) channels-last.
        var h = postProj(x.transposed(0, 2, 1))  // (B, T, 128)
        for layer in decMiLayer { h = layer(h) }   // (B, T, 128)
        // decoder works channels-last too; keep (B, T, C).
        let out = decoder(h)                        // (B, T*1920, 1)
        return out.transposed(0, 2, 1)              // (B, 1, T*1920)
    }
}

/// Unidirectional multi-layer LSTM with a residual skip (y = lstm(x) + x), no
/// proj_out. Hand-rolled since MLXNN ships no LSTM here. PyTorch gate order is
/// [input, forget, cell, output] in the 4*hidden rows; both bias_ih and bias_hh
/// are kept and summed at runtime.
public final class SLSTM: Module, UnaryLayer {
    /// The safetensors flattens all layers under a single `lstm` prefix with
    /// `_l{i}` suffixes (e.g. `lstm.weight_ih_l0`). MLX's array/dotted-index key
    /// scheme cannot produce that name, so we expose a nested `lstm` module whose
    /// parameters are keyed verbatim per layer.
    @ModuleInfo(key: "lstm") var lstm: LSTMParams
    let hidden: Int
    let numLayers: Int

    public init(size: Int, numLayers: Int) {
        self.hidden = size
        self.numLayers = numLayers
        self._lstm.wrappedValue = LSTMParams(hidden: size, numLayers: numLayers)
        super.init()
    }

    /// x: (B, T, H) -> (B, T, H), with residual skip over the stacked LSTM.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var layerIn = x
        for l in 0 ..< numLayers {
            layerIn = runLayer(layerIn, l: l)
        }
        return layerIn + x
    }

    private func runLayer(_ x: MLXArray, l: Int) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        let wih = lstm.weightIH(l)                 // (4H, H_in)
        let whh = lstm.weightHH(l)                 // (4H, H)
        let bias = lstm.biasIH(l) + lstm.biasHH(l) // (4H,)

        // Precompute input projection for all timesteps: (B, T, 4H).
        let xProj = x.matmul(wih.transposed(1, 0)) + bias

        var h = MLXArray.zeros([B, hidden])
        var c = MLXArray.zeros([B, hidden])
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(T)

        for t in 0 ..< T {
            let gates = xProj[0..., t, 0...] + h.matmul(whh.transposed(1, 0))  // (B, 4H)
            let parts = split(gates, parts: 4, axis: -1)  // i, f, g, o
            let i = sigmoid(parts[0])
            let f = sigmoid(parts[1])
            let g = tanh(parts[2])
            let o = sigmoid(parts[3])
            c = f * c + i * g
            h = o * tanh(c)
            outputs.append(h.expandedDimensions(axis: 1))
        }
        return concatenated(outputs, axis: 1)  // (B, T, H)
    }
}

/// Holds the flat LSTM parameters under the `lstm` prefix, keyed verbatim as in
/// PyTorch (`weight_ih_l0`, `weight_hh_l0`, `bias_ih_l0`, `bias_hh_l0`, ...).
/// Fixed at 4 layers (the SLSTM config in this model).
public final class LSTMParams: Module {
    @ParameterInfo(key: "weight_ih_l0") var weightIHL0: MLXArray
    @ParameterInfo(key: "weight_ih_l1") var weightIHL1: MLXArray
    @ParameterInfo(key: "weight_ih_l2") var weightIHL2: MLXArray
    @ParameterInfo(key: "weight_ih_l3") var weightIHL3: MLXArray
    @ParameterInfo(key: "weight_hh_l0") var weightHHL0: MLXArray
    @ParameterInfo(key: "weight_hh_l1") var weightHHL1: MLXArray
    @ParameterInfo(key: "weight_hh_l2") var weightHHL2: MLXArray
    @ParameterInfo(key: "weight_hh_l3") var weightHHL3: MLXArray
    @ParameterInfo(key: "bias_ih_l0") var biasIHL0: MLXArray
    @ParameterInfo(key: "bias_ih_l1") var biasIHL1: MLXArray
    @ParameterInfo(key: "bias_ih_l2") var biasIHL2: MLXArray
    @ParameterInfo(key: "bias_ih_l3") var biasIHL3: MLXArray
    @ParameterInfo(key: "bias_hh_l0") var biasHHL0: MLXArray
    @ParameterInfo(key: "bias_hh_l1") var biasHHL1: MLXArray
    @ParameterInfo(key: "bias_hh_l2") var biasHHL2: MLXArray
    @ParameterInfo(key: "bias_hh_l3") var biasHHL3: MLXArray

    public init(hidden: Int, numLayers: Int) {
        precondition(numLayers == 4, "LSTMParams is fixed at 4 layers")
        // Distinct array per slot - sharing one MLXArray identity across slots
        // collapses them to a single node in MLX's parameter graph, so update()
        // only fills one and the rest alias it.
        func w() -> MLXArray { MLXArray.zeros([4 * hidden, hidden]) }
        func b() -> MLXArray { MLXArray.zeros([4 * hidden]) }
        self._weightIHL0.wrappedValue = w(); self._weightIHL1.wrappedValue = w()
        self._weightIHL2.wrappedValue = w(); self._weightIHL3.wrappedValue = w()
        self._weightHHL0.wrappedValue = w(); self._weightHHL1.wrappedValue = w()
        self._weightHHL2.wrappedValue = w(); self._weightHHL3.wrappedValue = w()
        self._biasIHL0.wrappedValue = b(); self._biasIHL1.wrappedValue = b()
        self._biasIHL2.wrappedValue = b(); self._biasIHL3.wrappedValue = b()
        self._biasHHL0.wrappedValue = b(); self._biasHHL1.wrappedValue = b()
        self._biasHHL2.wrappedValue = b(); self._biasHHL3.wrappedValue = b()
        super.init()
    }

    func weightIH(_ l: Int) -> MLXArray { [weightIHL0, weightIHL1, weightIHL2, weightIHL3][l] }
    func weightHH(_ l: Int) -> MLXArray { [weightHHL0, weightHHL1, weightHHL2, weightHHL3][l] }
    func biasIH(_ l: Int) -> MLXArray { [biasIHL0, biasIHL1, biasIHL2, biasIHL3][l] }
    func biasHH(_ l: Int) -> MLXArray { [biasHHL0, biasHHL1, biasHHL2, biasHHL3][l] }
}
