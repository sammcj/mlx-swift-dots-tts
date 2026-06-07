# mlx-swift-dots-tts

Native [MLX](https://github.com/ml-explore/mlx-swift) port of [dots.tts-soar](https://huggingface.co/rednote-hilab/dots.tts-soar), a continuous autoregressive TTS system with flow-matching synthesis and voice cloning.

Runs the full pipeline on Apple silicon with no Python daemon: Qwen2 AR backbone, flow-matching DiT, BigVGAN/AudioVAE vocoder, and CAM++ x-vector speaker conditioning. Converted and (optionally) quantised weights are published at [smcleod/dots.tts-soar-mlx](https://huggingface.co/smcleod/dots.tts-soar-mlx).

## Status

Early port. See the task tracker for component progress.

## Why MLX

On an M5 Max, the AR backbone decodes ~2x faster in MLX than PyTorch-MPS at fp32 and ~3.8x at int4 (it's memory-bandwidth-bound). The flow-matching DiT - the dominant cost - runs ~2-2.8x faster in MLX fp32 (compute-bound, so quantisation there saves memory, not time). Quantisation shrinks the 8.2GB fp32 core to ~2-3GB.

## Build

```
swift build
swift test
```

## License

Apache-2.0. Model weights and architecture derive from rednote-hilab's dots.tts-soar (Apache-2.0).
