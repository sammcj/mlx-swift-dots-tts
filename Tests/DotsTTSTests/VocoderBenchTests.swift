import Foundation
import MLX
import MLXNN
import XCTest
@testable import DotsTTS

/// Spike bench (#50): BigVGAN/AudioVAE vocoder decode at f32 vs bf16 vs fp16, plus
/// an int8 feasibility breakdown. Measures decode latency, peak GPU memory, and
/// output drift vs the f32 path and the torch reference, and writes each variant's
/// waveform to a safetensors for ear-check (convert with dots-mlx-spike/st_to_wav.py).
///
/// Opt-in (needs Metal + the spike fixtures, so it's skipped otherwise):
///   DOTS_RUN_VOCODER_BENCH=1
///   DOTS_FIXTURES        -> dir holding ref_vocoder.safetensors + vocoder_decoder_mlx.safetensors
///   DOTS_BENCH_OUT       -> dir for the per-variant output safetensors (default $TMPDIR)
///   DOTS_BENCH_ITERS     -> timed iterations per variant (default 5)
///   DOTS_BENCH_TILES     -> repeat the fixture latent N times along T to bench a
///                           longer decode and show RAM scaling (default 1)
final class VocoderBenchTests: XCTestCase {
    private struct Variant {
        let name: String
        let dtype: DType
    }

    func testVocoderPrecisionBench() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["DOTS_RUN_VOCODER_BENCH"] == "1", "set DOTS_RUN_VOCODER_BENCH=1 to run")

        let spike = env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike"
        let weightsPath = "\(spike)/vocoder_decoder_mlx.safetensors"
        let refPath = "\(spike)/ref_vocoder.safetensors"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: weightsPath)
                && FileManager.default.fileExists(atPath: refPath),
            "vocoder weights or reference fixture not present"
        )
        let outDir = env["DOTS_BENCH_OUT"] ?? NSTemporaryDirectory()
        try FileManager.default.createDirectory(
            atPath: outDir, withIntermediateDirectories: true)
        let iters = Int(env["DOTS_BENCH_ITERS"] ?? "5") ?? 5
        let tiles = max(1, Int(env["DOTS_BENCH_TILES"] ?? "1") ?? 1)

        // Hard guard: cap MLX at 16 GB so a runaway decode can't lock the machine.
        // The fixture decode needs well under 2 GB even at modest tiling.
        MLX.Memory.memoryLimit = 16 * 1024 * 1024 * 1024
        MLX.Memory.cacheLimit = 2 * 1024 * 1024 * 1024

        let f32Weights = try MLX.loadArrays(url: URL(fileURLWithPath: weightsPath))
        let ref = try MLX.loadArrays(url: URL(fileURLWithPath: refPath))
        var latent = ref["latent"]!.asType(.float32)        // (1, 128, T)
        if tiles > 1 {
            latent = concatenated(Array(repeating: latent, count: tiles), axis: 2)
        }
        let target = ref["waveform"]!.asType(.float32)      // (1, 1, T*1920)
        let outSamples = latent.dim(2) * 1920
        print(String(format: "\n=== Vocoder bench: latent T=%d -> %d samples (%.2fs @48k), iters=%d ===",
                     latent.dim(2), outSamples, Double(outSamples) / 48000.0, iters))

        printParamBreakdown(f32Weights)

        let variants = [
            Variant(name: "f32", dtype: .float32),
            Variant(name: "bf16", dtype: .bfloat16),
            Variant(name: "fp16", dtype: .float16),
        ]

        var f32Output: MLXArray?
        print("\nvariant   load(MB)  decode(ms)  peakGPU(MB)  relVsTorch  relVsF32   note")
        print(String(repeating: "-", count: 78))

        for v in variants {
            // Cast every float weight to the target dtype (mirrors cast_component.py).
            let castWeights = f32Weights.mapValues { arr -> MLXArray in
                arr.dtype == .float32 || arr.dtype == .float16 || arr.dtype == .bfloat16
                    ? arr.asType(v.dtype) : arr
            }
            let loadBytes = castWeights.values.reduce(0) { $0 + $1.nbytes }

            let vocoder = Vocoder()
            try vocoder.update(parameters: ModuleParameters.unflattened(castWeights), verify: .all)
            eval(vocoder)
            let x = latent.asType(v.dtype)

            // Warmup (kernel compile + graph build excluded from timing).
            let warm = vocoder(x); eval(warm)

            MLX.GPU.resetPeakMemory()
            var times: [Double] = []
            var last: MLXArray = warm
            for _ in 0 ..< iters {
                let t0 = DispatchTime.now().uptimeNanoseconds
                let out = vocoder(x)
                eval(out)
                let t1 = DispatchTime.now().uptimeNanoseconds
                times.append(Double(t1 - t0) / 1_000_000.0)
                last = out
            }
            let peak = MLX.Memory.peakMemory
            let median = times.sorted()[times.count / 2]

            let outF32 = last.asType(.float32)
            if v.name == "f32" { f32Output = outF32 }
            // The torch target only matches the un-tiled fixture length.
            let relTorch = tiles == 1 ? relDiff(outF32, target) : Float.nan
            let relF32 = f32Output.map { relDiff(outF32, $0) } ?? 0
            let finite = isFinite(outF32).all().item(Bool.self)
            let note = finite ? "" : "NON-FINITE (NaN/Inf)"

            print(String(format: "%-8s  %7.1f  %9.2f  %10.1f  %9.5f  %8.5f   %@",
                         (v.name as NSString).utf8String!,
                         Double(loadBytes) / 1e6, median, Double(peak) / 1e6,
                         relTorch, relF32, note as NSString))

            // Save waveform for ear-check.
            let outPath = "\(outDir)/vocoder_bench_\(v.name).safetensors"
            try MLX.save(
                arrays: ["audio": outF32.reshaped([-1]), "sample_rate": MLXArray(Int32(48000))],
                url: URL(fileURLWithPath: outPath))
            MLX.GPU.clearCache()
        }

        print("\nint8: see param breakdown above. The decoder is Conv1d/LSTM-dominated;")
        print("the only 2 Linear layers are array-nested (dec_mi_layer.0/.2), which MLX")
        print("quantize(model:) cannot reach, and MLX has no int8 Conv kernel. So int8")
        print("over the vocoder quantises ~0% of compute -> not pursued.\n")
        print("WAVs: for f in f32 bf16 fp16; do python \(spike)/st_to_wav.py \(outDir)/vocoder_bench_$f.safetensors \(outDir)/vocoder_bench_$f.wav; done\n")
    }

    private func relDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        let maxAbs = abs(a - b).max().item(Float.self)
        let scale = abs(b).max().item(Float.self)
        return scale > 0 ? maxAbs / scale : maxAbs
    }

    /// Categorise vocoder parameters by tensor rank to size the int8 opportunity.
    private func printParamBreakdown(_ weights: [String: MLXArray]) {
        var convBytes = 0, linearLstmBytes = 0, otherBytes = 0
        for (k, v) in weights {
            let b = v.size * 4  // f32-equivalent element bytes
            if v.ndim == 3 { convBytes += b }            // Conv1d weight (O,K,I)
            else if v.ndim == 2 {                         // Linear / LSTM matmul
                _ = k
                linearLstmBytes += b
            } else { otherBytes += b }                    // bias / norm / alpha / beta
        }
        let total = convBytes + linearLstmBytes + otherBytes
        func pct(_ x: Int) -> Double { total > 0 ? 100.0 * Double(x) / Double(total) : 0 }
        print(String(format: "param breakdown (f32-equiv): conv(3D) %.1f%%  linear/lstm(2D) %.1f%%  other(1D) %.1f%%  total %.0f MB",
                     pct(convBytes), pct(linearLstmBytes), pct(otherBytes), Double(total) / 1e6))
    }
}
