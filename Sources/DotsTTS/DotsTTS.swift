/// Native MLX (mlx-swift) port of the dots.tts-soar TTS pipeline.
///
/// Components (each ported under its own directory):
///   - Backbone  - Qwen2 AR backbone (hidden states per audio patch)
///   - Reference - AudioVAE encoder + PatchEncoder (reference -> conditioning)
///   - Speaker   - CAM++ x-vector encoder (reference -> g_cond)
///   - DiT       - flow-matching velocity field predictor
///   - FlowMatching - Euler ODE solver + classifier-free guidance
///   - Vocoder   - BigVGAN/AudioVAE decoder (latents -> 48kHz waveform)
///   - Pipeline  - end-to-end orchestration
///
/// Weights load from the converted HF model repo `smcleod/dots.tts-soar-mlx`.
public enum DotsTTS {
    public static let version = "0.0.1"
}
