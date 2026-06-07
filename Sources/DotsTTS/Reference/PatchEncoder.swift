import Foundation
import MLX
import MLXFast
import MLXNN

/// PatchEncoder (`VAESemanticEncoder`): latents -> backbone patch embeddings.
///
/// Swift port of the reference-audio conditioning patch encoder. Pipeline:
///   ds_proj (causal Conv1d 128->128, k2 s2)  -> halve time
///   in_proj (Linear 128->1024)
///   24x causal pre-norm transformer (16 heads, head_dim 64, RMSNorm,
///     NeoX rotary theta 10000, affine-free qk-norm, SiLU FFN 1024->4096->1024)
///   group every 2 tokens (-> 2048)
///   out_proj (Linear 2048->1536)
///
/// Parameter paths mirror `patchencoder_mlx.safetensors` exactly via
/// @ModuleInfo keys. Conv weight is stored pre-transposed to MLX layout
/// [out, k, in] by the converter.
public final class PatchEncoder: Module {
    public struct Config: Sendable {
        public var inDim = 128
        public var outDim = 1536
        public var hidden = 1024
        public var numLayers = 24
        public var numHeads = 16
        public var ffn = 4096
        // RMSNorm eps; 1e-5 validated against the reference (rel 4e-4).
        public var rmsEps: Float = 1e-5
        public var outDsRate = 2  // group 2 transformer tokens per output patch
        public init() {}
    }

    let cfg: Config

    // ds_proj is a causal Conv1d (k=2, s=2). We hold the raw conv weight/bias
    // as parameters and run a manual left-pad + grouped conv.
    @ModuleInfo(key: "ds_proj") var dsProj: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "encoder") var encoder: TransformerStack
    @ModuleInfo(key: "out_proj") var outProj: Linear

    public init(_ cfg: Config = Config()) {
        self.cfg = cfg
        // Conv1d(channels-last). padding=0; we left-pad manually for causality.
        self._dsProj.wrappedValue = Conv1d(
            inputChannels: cfg.inDim, outputChannels: cfg.inDim,
            kernelSize: 2, stride: 2, padding: 0, bias: true)
        self._inProj.wrappedValue = Linear(cfg.inDim, cfg.hidden, bias: true)
        self._encoder.wrappedValue = TransformerStack(cfg)
        self._outProj.wrappedValue = Linear(cfg.hidden * cfg.outDsRate, cfg.outDim, bias: true)
        super.init()
    }

    /// x: (B, L, 128) un-normalised, trimmed latents. Returns (B, L/4, 1536).
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // ds_proj: causal Conv1d over time. mlx Conv1d is channels-last (B,L,C),
        // matching x. Causal k2 s2 = left-pad by dilation*(k-1)=1, then conv pad0.
        let padded = padded(x, widths: [.init((0, 0)), .init((1, 0)), .init((0, 0))])
        var h = dsProj(padded)              // (B, L/2, 128)
        h = inProj(h)                       // (B, L/2, 1024)
        h = encoder(h)                      // (B, L/2, 1024)
        return projectEmbeddings(h)         // (B, L/4, 1536)
    }

    /// Test hook: per-stage outputs for localising parity bugs.
    public func debugStages(_ x: MLXArray) -> [String: MLXArray] {
        let pd = padded(x, widths: [.init((0, 0)), .init((1, 0)), .init((0, 0))])
        let ds = dsProj(pd)
        let inp = inProj(ds)
        let enc = encoder(inp)
        let final = projectEmbeddings(enc)
        return ["after_downsample": ds, "after_in_proj": inp, "after_encoder": enc, "final": final]
    }

    /// group every `outDsRate` consecutive tokens along the feature dim then project.
    /// rearrange "b (s d) h -> b s (d h)" with d=2: token[2s] feats then token[2s+1].
    private func projectEmbeddings(_ z: MLXArray) -> MLXArray {
        let B = z.dim(0), L = z.dim(1), H = z.dim(2)
        let d = cfg.outDsRate
        let grouped = z.reshaped(B, L / d, d * H)
        return outProj(grouped)
    }
}

/// 24-layer causal pre-norm transformer.
public final class TransformerStack: Module {
    @ModuleInfo(key: "layers") var layers: [TransformerEncoderLayer]

    public init(_ cfg: PatchEncoder.Config) {
        self._layers.wrappedValue = (0..<cfg.numLayers).map { _ in
            TransformerEncoderLayer(cfg)
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let L = x.dim(1)
        // Additive causal mask, float, broadcastable to (B, heads, L, L).
        let mask = MultiHeadSelfAttention.causalMask(L, dtype: x.dtype)
        var h = x
        for layer in layers { h = layer(h, mask: mask) }
        return h
    }
}

/// Pre-norm causal transformer layer:
///   x = x + attn(attn_norm(x)); x = x + ffn(ffn_norm(x))
public final class TransformerEncoderLayer: Module {
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "attn") var attn: MultiHeadSelfAttention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "ffn") var ffn: SemanticMlp

    public init(_ cfg: PatchEncoder.Config) {
        self._attnNorm.wrappedValue = RMSNorm(dimensions: cfg.hidden, eps: cfg.rmsEps)
        self._attn.wrappedValue = MultiHeadSelfAttention(cfg)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: cfg.hidden, eps: cfg.rmsEps)
        self._ffn.wrappedValue = SemanticMlp(hidden: cfg.hidden, ffn: cfg.ffn)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var h = x + attn(attnNorm(x), mask: mask)
        h = h + ffn(ffnNorm(h))
        return h
    }
}

/// MHA, causal additive mask. The PatchEncoder config has qk_norm=False and
/// rotary_bias=False, so this is plain multi-head attention (no qk RMSNorm, no
/// rotary). Confirmed against the reference: enabling either pushes rel from
/// 4e-4 to 0.13.
public final class MultiHeadSelfAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    public init(_ cfg: PatchEncoder.Config) {
        self.numHeads = cfg.numHeads
        self.headDim = cfg.hidden / cfg.numHeads
        self.scale = Foundation.pow(Double(headDim), -0.5).floatValue
        self._qProj.wrappedValue = Linear(cfg.hidden, cfg.hidden, bias: false)
        self._kProj.wrappedValue = Linear(cfg.hidden, cfg.hidden, bias: false)
        self._vProj.wrappedValue = Linear(cfg.hidden, cfg.hidden, bias: false)
        self._oProj.wrappedValue = Linear(cfg.hidden, cfg.hidden, bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        let q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, numHeads * headDim))
    }

    /// additive float causal mask of shape (L, L): 0 on/below diagonal, -inf above.
    static func causalMask(_ L: Int, dtype: DType) -> MLXArray {
        let r = MLXArray(0..<L).reshaped(L, 1)
        let c = MLXArray(0..<L).reshaped(1, L)
        let allow = c .<= r  // bool (L, L)
        let neg = MLXArray(Float(-1e9))
        let zero = MLXArray(Float(0))
        return MLX.where(allow, zero, neg).asType(dtype)
    }
}

/// FFN: fc1 (hidden->ffn) -> SiLU -> fc2 (ffn->hidden). Both have bias.
public final class SemanticMlp: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    public init(hidden: Int, ffn: Int) {
        self._fc1.wrappedValue = Linear(hidden, ffn, bias: true)
        self._fc2.wrappedValue = Linear(ffn, hidden, bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(silu(fc1(x)))
    }
}

private extension Double {
    var floatValue: Float { Float(self) }
}
