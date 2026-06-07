import Foundation

/// Mirrors the dots.tts-soar `config.json` (model_type "dots_tts").
/// Decoded from the HF model repo so the Swift modules stay data-driven.
public struct DotsConfig: Codable, Sendable {
    public var latentDim: Int
    public var patchSize: Int
    public var cfgDroprate: Double
    public var fmSigma: Double
    public var xvecDropRate: Double
    public var campplusEmbeddingSize: Int
    public var xvecMaxAudioSeconds: Double
    public var patchEncoder: TransformerConfig
    public var dit: TransformerConfig
    public var vocoder: VocoderConfig
    public var backbone: BackboneConfig

    enum CodingKeys: String, CodingKey {
        case latentDim = "latent_dim"
        case patchSize = "patch_size"
        case cfgDroprate = "cfg_droprate"
        case fmSigma = "fm_sigma"
        case xvecDropRate = "xvec_drop_rate"
        case campplusEmbeddingSize = "campplus_embedding_size"
        case xvecMaxAudioSeconds = "xvec_max_audio_seconds"
        case patchEncoder = "PatchEncoder"
        case dit = "DiT"
        case vocoder
        case backbone
    }
}

/// Shared transformer block config for the PatchEncoder and the DiT.
public struct TransformerConfig: Codable, Sendable {
    public var numLayers: Int
    public var numHeads: Int
    public var hiddenSize: Int
    public var ffnHiddenSize: Int
    public var modulation: Bool
    public var qkvBias: Bool
    public var qkNorm: Bool
    public var normLayer: String
    public var rotaryBias: Bool
    public var rotaryTheta: Double
    public var inputDim: Int?
    public var causal: Bool?

    enum CodingKeys: String, CodingKey {
        case numLayers = "num_layers"
        case numHeads = "num_heads"
        case hiddenSize = "hidden_size"
        case ffnHiddenSize = "ffn_hidden_size"
        case modulation
        case qkvBias = "qkv_bias"
        case qkNorm = "qk_norm"
        case normLayer = "norm_layer"
        case rotaryBias = "rotary_bias"
        case rotaryTheta = "rotary_theta"
        case inputDim = "input_dim"
        case causal
    }
}

/// Qwen2 AR backbone config (from the upstream `llm_config.json`).
public struct BackboneConfig: Codable, Sendable {
    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var rmsNormEps: Double
    public var ropeTheta: Double
    public var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
    }
}

public struct VocoderConfig: Codable, Sendable {
    public var sampleRate: Int
    public var latentDim: Int
    public var upsampleRates: [Int]
    public var upsampleKernelSizes: [Int]
    public var upsampleInitialChannel: Int
    public var resblockKernelSizes: [Int]
    public var resblockDilationSizes: [[Int]]
    public var activation: String
    public var snakeLogscale: Bool
    public var causal: Bool

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case latentDim = "latent_dim"
        case upsampleRates = "upsample_rates"
        case upsampleKernelSizes = "upsample_kernel_sizes"
        case upsampleInitialChannel = "upsample_initial_channel"
        case resblockKernelSizes = "resblock_kernel_sizes"
        case resblockDilationSizes = "resblock_dilation_sizes"
        case activation
        case snakeLogscale = "snake_logscale"
        case causal
    }
}
