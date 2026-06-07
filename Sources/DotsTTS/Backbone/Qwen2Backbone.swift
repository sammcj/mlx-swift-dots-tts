import Foundation
import MLX
import MLXFast
import MLXNN

/// Qwen2 AR backbone (the dots `llm.*` namespace, here under the `model.` prefix).
///
/// Produces the last hidden state per audio patch (the dots pipeline conditions
/// the flow-matching DiT on these) and tied-embedding logits (for the EOS/codec
/// head). GQA (12 q / 2 kv heads, head_dim 128), RoPE theta 1e6, RMSNorm,
/// SwiGLU MLP, q/k/v bias. Loads fp32 or mlx_lm int4: build plain, optionally
/// `quantize(...)`, then load weights.
public final class Qwen2Backbone: Module {
    public struct Config: Sendable {
        public var vocabSize = 151672
        public var hidden = 1536
        public var layers = 28
        public var heads = 12
        public var kvHeads = 2
        public var headDim = 128
        public var intermediate = 8960
        public var rmsEps: Float = 1e-6
        public var ropeTheta: Float = 1_000_000
        public init() {}
    }

    @ModuleInfo(key: "model") var model: Qwen2Model
    let cfg: Config

    public init(_ cfg: Config = Config()) {
        self.cfg = cfg
        self._model.wrappedValue = Qwen2Model(cfg)
        super.init()
    }

    /// Last hidden state (B, L, hidden).
    public func hidden(_ ids: MLXArray) -> MLXArray { model(ids) }

    /// Tied-embedding logits (B, L, vocab).
    public func logits(_ ids: MLXArray) -> MLXArray {
        model.embedTokens.asLinear(model(ids))
    }
}

public final class Qwen2Model: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen2Layer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public init(_ cfg: Qwen2Backbone.Config) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: cfg.vocabSize, dimensions: cfg.hidden)
        self._layers.wrappedValue = (0..<cfg.layers).map { _ in Qwen2Layer(cfg) }
        self._norm.wrappedValue = RMSNorm(dimensions: cfg.hidden, eps: cfg.rmsEps)
        super.init()
    }

    public func callAsFunction(_ ids: MLXArray) -> MLXArray {
        var h = embedTokens(ids)
        let L = h.dim(1)
        let mask: MLXArray? = L > 1 ? causalMask(L, dtype: h.dtype) : nil
        for layer in layers { h = layer(h, mask: mask) }
        return norm(h)
    }
}

/// Additive (L, L) causal mask: 0 on/below diagonal, -inf above.
func causalMask(_ L: Int, dtype: DType) -> MLXArray {
    let r = MLXArray(0..<Int32(L)).reshaped(L, 1)
    let c = MLXArray(0..<Int32(L)).reshaped(1, L)
    let keep = c .<= r
    return MLX.where(keep, MLXArray(0, dtype: dtype), MLXArray(-Float.infinity, dtype: dtype))
}

public final class Qwen2Layer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen2Attention
    @ModuleInfo(key: "mlp") var mlp: Qwen2MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    public init(_ cfg: Qwen2Backbone.Config) {
        self._selfAttn.wrappedValue = Qwen2Attention(cfg)
        self._mlp.wrappedValue = Qwen2MLP(cfg)
        self._inputLayernorm.wrappedValue = RMSNorm(dimensions: cfg.hidden, eps: cfg.rmsEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: cfg.hidden, eps: cfg.rmsEps)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let h = x + selfAttn(inputLayernorm(x), mask: mask)
        return h + mlp(postAttentionLayernorm(h))
    }
}

public final class Qwen2Attention: Module {
    let heads: Int, kvHeads: Int, headDim: Int, scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    let rope: RoPE

    public init(_ cfg: Qwen2Backbone.Config) {
        heads = cfg.heads
        kvHeads = cfg.kvHeads
        headDim = cfg.headDim
        scale = Float(1.0 / Foundation.sqrt(Double(cfg.headDim)))
        self._qProj.wrappedValue = Linear(cfg.hidden, cfg.heads * cfg.headDim, bias: true)
        self._kProj.wrappedValue = Linear(cfg.hidden, cfg.kvHeads * cfg.headDim, bias: true)
        self._vProj.wrappedValue = Linear(cfg.hidden, cfg.kvHeads * cfg.headDim, bias: true)
        self._oProj.wrappedValue = Linear(cfg.heads * cfg.headDim, cfg.hidden, bias: false)
        self.rope = RoPE(dimensions: cfg.headDim, traditional: false, base: cfg.ropeTheta)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        var q = qProj(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = rope(q)
        k = rope(k)
        let nRep = heads / kvHeads
        k = repeatKV(k, nRep)
        v = repeatKV(v, nRep)
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, heads * headDim))
    }
}

/// Expand kv heads for GQA: (B, nKV, L, D) -> (B, nKV*nRep, L, D).
func repeatKV(_ x: MLXArray, _ nRep: Int) -> MLXArray {
    if nRep == 1 { return x }
    let b = x.dim(0), nKV = x.dim(1), l = x.dim(2), d = x.dim(3)
    return repeated(x.expandedDimensions(axis: 2), count: nRep, axis: 2)
        .reshaped(b, nKV * nRep, l, d)
}

public final class Qwen2MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(_ cfg: Qwen2Backbone.Config) {
        self._gateProj.wrappedValue = Linear(cfg.hidden, cfg.intermediate, bias: false)
        self._upProj.wrappedValue = Linear(cfg.hidden, cfg.intermediate, bias: false)
        self._downProj.wrappedValue = Linear(cfg.intermediate, cfg.hidden, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}
