import Foundation
import MLX
import MLXFast
import MLXNN

/// Flow-matching velocity-field predictor (the dots DiT).
///
/// Swift port of the validated reference `dit_mlx.py`. 18 modulated transformer
/// blocks: adaLN (6 params/block), NeoX rotary (theta 10000), affine-free qk
/// RMSNorm, affine-free LayerNorm, GELU-tanh FFN. Conditioning `c =
/// timeEmbed(t) + gCond`. Norms with no learnable params carry no weight keys.
/// Parameter paths mirror `dit/model.safetensors` exactly (snake_case via
/// @ModuleInfo keys).
public final class DiT: Module {
    public struct Config: Sendable {
        public var hidden = 1024
        public var numLayers = 18
        public var numHeads = 16
        public var ffn = 4096
        public var inDim = 1024
        public var outDim = 128
        public var theta: Float = 10000
        public var rmsEps: Float = 1.1920929e-7  // torch finfo(f32).eps
        public var lnEps: Float = 1e-5
        public init() {}
    }

    @ModuleInfo(key: "input_layer") var inputLayer: Linear
    @ModuleInfo(key: "time_embedder") var timeEmbedder: TimestepEmbedder
    @ModuleInfo(key: "blocks") var blocks: [DiTBlock]
    @ModuleInfo(key: "output_layer") var outputLayer: FinalLayer

    public init(_ cfg: Config = Config(), quant: QuantizationSettings = .none) {
        self._inputLayer.wrappedValue = QuantizedLayerFactory.linear(cfg.inDim, cfg.hidden, bias: true, settings: quant)
        self._timeEmbedder.wrappedValue = TimestepEmbedder(hidden: cfg.hidden, quant: quant)
        self._blocks.wrappedValue = (0..<cfg.numLayers).map { _ in
            DiTBlock(hidden: cfg.hidden, numHeads: cfg.numHeads, ffn: cfg.ffn,
                     theta: cfg.theta, rmsEps: cfg.rmsEps, lnEps: cfg.lnEps, quant: quant)
        }
        self._outputLayer.wrappedValue = FinalLayer(hidden: cfg.hidden, outDim: cfg.outDim, lnEps: cfg.lnEps, quant: quant)
        super.init()
    }

    /// x: (B, L, inDim); timesteps: (B,); gCond: (B, hidden); mask: optional additive
    /// attention bias broadcastable to (B, numHeads, L, L) (0 keep / -inf drop).
    /// Returns (B, L, outDim).
    public func callAsFunction(_ x: MLXArray, timesteps: MLXArray, gCond: MLXArray?, mask: MLXArray? = nil) -> MLXArray {
        var c = timeEmbedder(timesteps)
        if let gCond { c = c + gCond }
        var h = inputLayer(x)
        for blk in blocks { h = blk(h, c, mask) }
        return outputLayer(h, c)
    }
}

/// x * (1 + scale) + shift, with scale/shift of shape (B, hidden) broadcast over L.
@inline(__always)
func modulate(_ x: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
    x * (1 + scale.expandedDimensions(axis: 1)) + shift.expandedDimensions(axis: 1)
}

@inline(__always)
func geluTanh(_ x: MLXArray) -> MLXArray {
    0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
}

public final class TimestepEmbedder: Module {
    @ModuleInfo(key: "mlp") var mlp: [UnaryLayer]  // [Linear, SiLU, Linear]
    let freqSize: Int

    public init(hidden: Int, freqSize: Int = 256, quant: QuantizationSettings = .none) {
        self.freqSize = freqSize
        self._mlp.wrappedValue = [
            QuantizedLayerFactory.linear(freqSize, hidden, settings: quant),
            SiLU(),
            QuantizedLayerFactory.linear(hidden, hidden, settings: quant),
        ]
        super.init()
    }

    /// Sinusoidal timestep embedding, matching dit.py TimestepEmbedder.
    private func embedding(_ t: MLXArray) -> MLXArray {
        let half = freqSize / 2
        let scale = -Foundation.log(10000.0) / Double(half)
        let freqs = MLX.exp(MLXArray(0..<half).asType(.float32) * Float(scale))
        let args = t.expandedDimensions(axis: 1).asType(.float32) * freqs.expandedDimensions(axis: 0)
        return concatenated([cos(args), sin(args)], axis: -1)
    }

    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        var h = embedding(t)
        for layer in mlp { h = layer(h) }
        return h
    }
}

public final class DiTAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "o_proj") var oProj: Linear
    let rope: RoPE

    public init(hidden: Int, numHeads: Int, theta: Float, rmsEps: Float, quant: QuantizationSettings = .none) {
        self.numHeads = numHeads
        self.headDim = hidden / numHeads
        self.scale = Foundation.pow(Double(headDim), -0.5).floatValue
        self._qProj.wrappedValue = QuantizedLayerFactory.linear(hidden, hidden, bias: false, settings: quant)
        self._kProj.wrappedValue = QuantizedLayerFactory.linear(hidden, hidden, bias: false, settings: quant)
        self._vProj.wrappedValue = QuantizedLayerFactory.linear(hidden, hidden, bias: false, settings: quant)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: rmsEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: rmsEps)
        self._oProj.wrappedValue = QuantizedLayerFactory.linear(hidden, hidden, bias: true, settings: quant)
        self.rope = RoPE(dimensions: headDim, traditional: false, base: theta)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ mask: MLXArray? = nil) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        var q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        q = qNorm(q)
        k = kNorm(k)
        q = rope(q)
        k = rope(k)
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, numHeads * headDim))
    }
}

public final class DiTMlp: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    public init(hidden: Int, ffn: Int, quant: QuantizationSettings = .none) {
        self._fc1.wrappedValue = QuantizedLayerFactory.linear(hidden, ffn, bias: true, settings: quant)
        self._fc2.wrappedValue = QuantizedLayerFactory.linear(ffn, hidden, bias: true, settings: quant)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluTanh(fc1(x)))
    }
}

public final class DiTBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attn: DiTAttention
    @ModuleInfo(key: "ffn") var ffn: DiTMlp
    @ModuleInfo(key: "adaLN_modulation") var adaLN: [UnaryLayer]  // [SiLU, Linear]

    public init(hidden: Int, numHeads: Int, ffn: Int, theta: Float, rmsEps: Float, lnEps: Float, quant: QuantizationSettings = .none) {
        self._norm1.wrappedValue = LayerNorm(dimensions: hidden, eps: lnEps, affine: false)
        self._norm2.wrappedValue = LayerNorm(dimensions: hidden, eps: lnEps, affine: false)
        self._attn.wrappedValue = DiTAttention(hidden: hidden, numHeads: numHeads, theta: theta, rmsEps: rmsEps, quant: quant)
        self._ffn.wrappedValue = DiTMlp(hidden: hidden, ffn: ffn, quant: quant)
        self._adaLN.wrappedValue = [SiLU(), QuantizedLayerFactory.linear(hidden, 6 * hidden, settings: quant)]
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ c: MLXArray, _ mask: MLXArray? = nil) -> MLXArray {
        let mod = adaLN[1](adaLN[0](c))
        let p = split(mod, parts: 6, axis: -1)
        var h = x + p[2].expandedDimensions(axis: 1) * attn(modulate(norm1(x), shift: p[0], scale: p[1]), mask)
        h = h + p[5].expandedDimensions(axis: 1) * ffn(modulate(norm2(h), shift: p[3], scale: p[4]))
        return h
    }
}

public final class FinalLayer: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "adaLN_modulation") var adaLN: [UnaryLayer]  // [SiLU, Linear]

    public init(hidden: Int, outDim: Int, lnEps: Float, quant: QuantizationSettings = .none) {
        self._norm.wrappedValue = LayerNorm(dimensions: hidden, eps: lnEps, affine: false)
        self._linear.wrappedValue = QuantizedLayerFactory.linear(hidden, outDim, bias: true, settings: quant)
        self._adaLN.wrappedValue = [SiLU(), QuantizedLayerFactory.linear(hidden, 2 * hidden, settings: quant)]
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ c: MLXArray) -> MLXArray {
        let mod = adaLN[1](adaLN[0](c))
        let p = split(mod, parts: 2, axis: -1)
        return linear(modulate(norm(x), shift: p[0], scale: p[1]))
    }
}

private extension Double {
    var floatValue: Float { Float(self) }
}
