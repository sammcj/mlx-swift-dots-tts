# mlx-swift-dots-tts

Native [MLX](https://github.com/ml-explore/mlx-swift) port of [dots.tts-soar](https://huggingface.co/rednote-hilab/dots.tts-soar), a continuous autoregressive TTS system with flow-matching synthesis and voice cloning.

Runs the full pipeline on Apple silicon with no Python daemon: Qwen2 AR backbone, flow-matching DiT, BigVGAN/AudioVAE vocoder, and CAM++ x-vector speaker conditioning. Converted and (optionally) quantised weights are published at [smcleod/dots.tts-soar-mlx](https://huggingface.co/smcleod/dots.tts-soar-mlx).

## Status

All components are ported and numerically parity-checked against the PyTorch reference, and the end-to-end pipeline renders voice-cloned speech. The backbone, DiT and patch encoder support per-component MLX quantisation (int4/int8); the vocoder and AudioVAE encoder can run at full or reduced (bf16/fp16) precision.

## Usage

```swift
import DotsTTS
import Tokenizers

let tokenizer = try await AutoTokenizer.from(modelFolder: modelRepo.appendingPathComponent("backbone"))
let pipeline = try DotsTTSPipeline(modelRepo: modelRepo, tokenizer: tokenizer)

var params = DotsTTSPipeline.Params()
params.numSteps = 10
params.seed = 1
let audio48k = pipeline.generate(
    targetText: "Hello world.",
    refAudio48k: referenceSamples,   // MLXArray, mono 48 kHz
    refTranscript: "the reference transcript",
    params: params)
```

`modelRepo` is a directory with one subdirectory per component (`backbone/`, `dit/`, `patch_encoder/`, `vocoder/`, `speaker/`, `audiovae_encoder/`, `heads/`) plus shared config, as published at [smcleod/dots.tts-soar-mlx](https://huggingface.co/smcleod/dots.tts-soar-mlx).

## Why MLX

On an M5 Max, the AR backbone decodes ~2x faster in MLX than PyTorch-MPS at fp32 and ~3.8x at int4 (it's memory-bandwidth-bound). The flow-matching DiT - the dominant cost - runs ~2-2.8x faster in MLX fp32 (compute-bound, so quantisation there saves memory, not time). Quantisation shrinks the 8.2GB fp32 core to ~2-3GB.

## Build

```
swift build
swift test
```

`swift build` compiles, but the SwiftPM debug/test binary crashes on the first MLX op because the Metal kernels aren't compiled into it. Run MLX tests with `mlx.metallib` colocated next to the test runner, or consume the package from an app built with xcodebuild. Heavy parity and end-to-end tests are gated behind `DOTS_RUN_E2E=1` / `DOTS_RUN_DECODE_PARITY=1` and read fixtures from paths set by `DOTS_MODEL_REPO` / `DOTS_FIXTURES`.

## License

Apache-2.0. Model weights and architecture derive from rednote-hilab's dots.tts-soar (Apache-2.0).
