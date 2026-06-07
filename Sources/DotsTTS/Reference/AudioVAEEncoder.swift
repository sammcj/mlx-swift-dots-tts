import Foundation
import MLX
import MLXNN

/// AudioVAE encoder: waveform -> latent mean||logstd.
///
/// Swift port of `bigvgan.py::AudioVAE.extract_latents` (do_sample=False).
/// Pipeline:
///   audio_encoder (conv downsample stack, causal) -> (B, 128, L)
///   permute -> (B, L, 128); enc_mi_layer (Linear 128->512, 4-layer skip-LSTM,
///     Linear 512->128); permute back -> (B, 128, L)
///   pre_proj (Conv1d 128->256, k1) -> (B, 256, L) = mean||logstd
///
/// Conv weights are stored pre-folded (weight_norm) and transposed to MLX
/// channels-last layout [out, k, in] by the converter. Everything runs fp32.
/// Parameter paths mirror `audiovae_encoder_mlx.safetensors`.
public final class AudioVAEEncoder: Module {
    @ModuleInfo(key: "audio_encoder") var audioEncoder: BigVGANEncoder
    // enc_mi_layer is a 3-slot nn.Sequential [Linear, SkipLSTM, Linear]; keys
    // `enc_mi_layer.{0,1,2}.*` unflatten into an array, so it must be an array
    // property keyed directly `enc_mi_layer` (no extra path segment).
    @ModuleInfo(key: "enc_mi_layer") var encMiLayer: [Module]
    @ModuleInfo(key: "pre_proj") var preProj: Conv1d

    public override init() {
        self._audioEncoder.wrappedValue = BigVGANEncoder()
        self._encMiLayer.wrappedValue = [
            Linear(128, 512, bias: true),
            SkipLSTM(dimension: 512, numLayers: 4),
            Linear(512, 128, bias: true),
        ]
        // pre_proj: plain Conv1d 128->256 k1 s1 (channels-last).
        self._preProj.wrappedValue = Conv1d(
            inputChannels: 128, outputChannels: 256, kernelSize: 1, stride: 1, bias: true)
        super.init()
    }

    private func runEncMi(_ x: MLXArray) -> MLXArray {
        var h = (encMiLayer[0] as! Linear)(x)
        h = (encMiLayer[1] as! SkipLSTM)(h)
        h = (encMiLayer[2] as! Linear)(h)
        return h
    }

    /// waveform: (B, 1, T) or (B, T) or (T,). Returns (B, 256, L) mean||logstd.
    public func callAsFunction(_ waveform: MLXArray) -> MLXArray {
        // Internally everything is channels-last (B, L, C). Caller supplies the
        // raw waveform; we present it to the conv stack as (B, T, 1).
        var w = waveform
        if w.ndim == 1 { w = w.reshaped(1, w.dim(0), 1) }            // (1, T, 1)
        else if w.ndim == 2 { w = w.expandedDimensions(axis: -1) }   // (B, T, 1)
        else if w.ndim == 3 {
            // accept (B, 1, T) PyTorch layout -> (B, T, 1)
            w = w.transposed(0, 2, 1)
        }

        var x = audioEncoder(w)        // (B, L, 128)   channels-last
        x = runEncMi(x)                // (B, L, 128)
        x = preProj(x)                 // (B, L, 256)
        // Return PyTorch-style (B, 256, L) to match the reference fixture layout.
        return x.transposed(0, 2, 1)
    }
}

/// A causal/non-causal 1D conv with manual left-pad, channels-last activations.
public final class Conv1dS: Module {
    let kernelSize: Int
    let stride: Int
    let dilation: Int
    let causal: Bool

    @ModuleInfo(key: "layer") var layer: Conv1d

    public init(inChannels: Int, outChannels: Int, kernelSize: Int,
                stride: Int = 1, dilation: Int = 1, causal: Bool) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.dilation = dilation
        self.causal = causal
        // mlx Conv1d does the symmetric padding for the non-causal post conv.
        let symPad = causal ? 0 : dilation * (kernelSize - 1) / 2
        self._layer.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: kernelSize, stride: stride, padding: symPad,
            dilation: dilation, bias: true)
        super.init()
    }

    /// x: (B, L, C). Causal => left-pad dilation*(k-1) then conv pad0.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if causal {
            let lp = dilation * (kernelSize - 1)
            if lp > 0 {
                let p = padded(x, widths: [.init((0, 0)), .init((lp, 0)), .init((0, 0))])
                return layer(p)
            }
        }
        return layer(x)
    }
}

/// Param-free placeholder for the LeakyReLU / ConstantPad slots in a ResStack
/// block so the inner array indices line up with the PyTorch nn.Sequential.
public final class ResStackNoop: Module {}

/// ResStack: `nums` residual blocks, dilation = base**i. causal=True.
/// Each block is the PyTorch nn.Sequential
///   [LeakyReLU, ConstantPad, Conv(.2), LeakyReLU, ConstantPad, Conv(.5)]
/// stored as a 6-slot array so flat safetensors keys `layers.<i>.{2,5}.*`
/// unflatten into the right positions. forward: x = x + block(x), where
/// block = LeakyReLU -> leftpad(dil*(k-1)) -> conv(k,dil) -> LeakyReLU
///   -> leftpad(k-1) -> conv(k,1).
public final class ResStack: Module {
    let kernelSize: Int
    let dilations: [Int]

    @ModuleInfo(key: "layers") var layers: [[Module]]

    public init(channels: Int, kernelSize: Int = 3, base: Int = 2, nums: Int = 6) {
        self.kernelSize = kernelSize
        self.dilations = (0..<nums).map { Int(Foundation.pow(Double(base), Double($0))) }
        self._layers.wrappedValue = dilations.map { dil in
            let conv0 = Conv1d(
                inputChannels: channels, outputChannels: channels,
                kernelSize: kernelSize, stride: 1, padding: 0, dilation: dil, bias: true)
            let conv1 = Conv1d(
                inputChannels: channels, outputChannels: channels,
                kernelSize: kernelSize, stride: 1, padding: 0, dilation: 1, bias: true)
            return [ResStackNoop(), ResStackNoop(), conv0,
                    ResStackNoop(), ResStackNoop(), conv1]
        }
        super.init()
    }

    private func leakyRelu(_ x: MLXArray) -> MLXArray { maximum(x, 0.01 * x) }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for (i, block) in layers.enumerated() {
            let conv0 = block[2] as! Conv1d
            let conv1 = block[5] as! Conv1d
            let dil = dilations[i]
            var y = leakyRelu(h)
            let pad1 = dil * (kernelSize - 1)
            y = padded(y, widths: [.init((0, 0)), .init((pad1, 0)), .init((0, 0))])
            y = conv0(y)
            y = leakyRelu(y)
            let pad2 = kernelSize - 1
            y = padded(y, widths: [.init((0, 0)), .init((pad2, 0)), .init((0, 0))])
            y = conv1(y)
            h = h + y
        }
        return h
    }
}

/// Param-free LeakyReLU(0.2) placeholder occupying a generator slot so the
/// `generator` array indices stay aligned with the PyTorch nn.Sequential.
public final class LeakyReLU02: Module, UnaryLayer {
    public func callAsFunction(_ x: MLXArray) -> MLXArray { maximum(x, 0.2 * x) }
}

/// The BigVGAN encoder conv stack (`Encoder.generator`), channels-last.
/// The submodules live in an ordered array whose indices mirror the PyTorch
/// nn.Sequential exactly (`generator.0`, `generator.1` (LeakyReLU), ...),
/// because flat safetensors keys with numeric segments unflatten into an array.
/// LeakyReLU slots are param-free placeholders. Down kernels = factor*2; pre
/// k3; post k5 (non-causal, lookahead 2). LeakyReLU slope 0.2.
public final class BigVGANEncoder: Module {
    @ModuleInfo(key: "generator") var generator: [Module]

    public override init() {
        // channels [12,24,48,96,192,384,768]; down factors [2,2,2,4,6,10].
        let channels = [12, 24, 48, 96, 192, 384, 768]
        let factors = [2, 2, 2, 4, 6, 10]

        var gen: [Module] = []
        // idx 0: pre conv 1->12 k3 causal; idx 1: LeakyReLU
        gen.append(Conv1dS(inChannels: 1, outChannels: channels[0],
                           kernelSize: 3, stride: 1, causal: true))
        gen.append(LeakyReLU02())

        // For each factor: down conv, resstack, leaky (idx 2..19).
        for (j, f) in factors.enumerated() {
            let inC = channels[j], outC = channels[j + 1]
            gen.append(Conv1dS(inChannels: inC, outChannels: outC,
                               kernelSize: f * 2, stride: f, causal: true))
            gen.append(ResStack(channels: outC, kernelSize: 3, base: 2, nums: 6))
            gen.append(LeakyReLU02())
        }

        // idx 20: post conv 768->128 k5 NON-causal (lookahead 2, symmetric pad 2)
        gen.append(Conv1dS(inChannels: channels.last!, outChannels: 128,
                           kernelSize: 5, stride: 1, causal: false))

        self._generator.wrappedValue = gen
        super.init()
    }

    /// x: (B, T, 1) channels-last. Returns (B, L, 128).
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in generator {
            switch layer {
            case let m as Conv1dS: h = m(h)
            case let m as ResStack: h = m(h)
            case let m as LeakyReLU02: h = m(h)
            default: break
            }
        }
        return h
    }
}

/// 4-layer unidirectional LSTM with skip residual `y = lstm(x) + x`, fp32.
/// PyTorch packed weights (weight_ih_l{n}/weight_hh_l{n}/bias_ih_l{n}/bias_hh_l{n}),
/// gate order i,f,g,o. The packed parameters live in a `[String: MLXArray]`
/// dictionary keyed `lstm` so paths render as `lstm.weight_ih_l0` etc., matching
/// the safetensors exactly. fp32 throughout.
public final class SkipLSTM: Module {
    let numLayers: Int
    let hidden: Int

    @ParameterInfo(key: "lstm") var lstm: [String: MLXArray]

    public init(dimension: Int, numLayers: Int) {
        self.numLayers = numLayers
        self.hidden = dimension
        let H = dimension
        var p: [String: MLXArray] = [:]
        for n in 0..<numLayers {
            p["weight_ih_l\(n)"] = MLXArray.zeros([4 * H, H])
            p["weight_hh_l\(n)"] = MLXArray.zeros([4 * H, H])
            p["bias_ih_l\(n)"] = MLXArray.zeros([4 * H])
            p["bias_hh_l\(n)"] = MLXArray.zeros([4 * H])
        }
        self._lstm.wrappedValue = p
        super.init()
    }

    /// x: (B, L, H). Returns (B, L, H) with skip residual.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        var input = x
        for n in 0..<numLayers {
            let wih = lstm["weight_ih_l\(n)"]!
            let whh = lstm["weight_hh_l\(n)"]!
            let bih = lstm["bias_ih_l\(n)"]!
            let bhh = lstm["bias_hh_l\(n)"]!
            // Precompute input contribution for all timesteps: x @ W_ih^T + b_ih.
            let xProj = addMM(bih, input.reshaped(B * L, -1), wih.T).reshaped(B, L, 4 * hidden)
            var h = MLXArray.zeros([B, hidden])
            var c = MLXArray.zeros([B, hidden])
            var outs: [MLXArray] = []
            outs.reserveCapacity(L)
            for t in 0..<L {
                let gatesX = xProj[0..., t, 0...]                 // (B, 4H)
                let gates = addMM(bhh, h, whh.T) + gatesX         // (B, 4H)
                let parts = split(gates, parts: 4, axis: -1)
                let i = sigmoid(parts[0])
                let f = sigmoid(parts[1])
                let g = tanh(parts[2])
                let o = sigmoid(parts[3])
                c = f * c + i * g
                h = o * tanh(c)
                outs.append(h)
            }
            input = stacked(outs, axis: 1)                        // (B, L, H)
        }
        return input + x   // skip residual
    }
}
