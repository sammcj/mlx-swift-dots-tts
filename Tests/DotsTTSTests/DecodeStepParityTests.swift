import Foundation
import MLX
import Tokenizers
import XCTest
@testable import DotsTTS

/// Injected-noise decode-step parity. Loads Python's REAL first FM decode-step
/// buffers (input_sequence / cfg_sequence / g_cond / attn_mask / pos_ids) and a
/// FIXED noise, runs the loaded Swift solver+DiT, and compares the solved
/// normalised latent to torch's `z_out`. Removes RNG divergence so any gap is a
/// solver / DiT / coordinate_proj bug on REAL conditioning (the synthetic DiT
/// fixture used random conditioning and passed).
final class DecodeStepParityTests: XCTestCase {
    func testDecodeStepMatchesPython() async throws {
        let env = ProcessInfo.processInfo.environment
        let repo = env["DOTS_MODEL_REPO"] ?? "/Users/samm/git/sammcj/dots.tts-soar-mlx"
        let fixtures = env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike"
        let fixURL = URL(fileURLWithPath: fixtures).appendingPathComponent("decode_step0.safetensors")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: fixURL.path), "decode_step0 fixture missing")

        let f = try MLX.loadArrays(url: fixURL)
        let inputSeq = f["input_sequence"]!.asType(.float32)
        let cfgSeq = f["cfg_sequence"]!.asType(.float32)
        let gCond = f["g_cond"]!.asType(.float32)
        let noise = f["noise"]!.asType(.float32)
        let zOut = f["z_out"]!.asType(.float32)
        let attnBool = f["attn_mask"]!.asType(.float32)         // (1, L, L) 1 keep / 0 drop
        let meta = f["meta"]!.asType(.float32)
        let numSteps = Int(meta[0].item(Float.self))
        let guidance = meta[1].item(Float.self)

        // additive mask (1,1,L,L): 0 keep / -inf drop
        let L = attnBool.dim(1)
        let additive = MLX.where(attnBool .> 0.5,
                                 MLXArray(Float(0)), MLXArray(-Float.infinity)).reshaped(1, 1, L, L)

        let tokenizer = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: repo).appendingPathComponent("backbone"))
        let pipeline = try DotsTTSPipeline(modelRepo: URL(fileURLWithPath: repo), tokenizer: tokenizer)

        let z = pipeline.debugSolveStep(inputSeq: inputSeq, cfgSeq: cfgSeq, gCond: gCond,
                                        noise: noise, mask: additive, numSteps: numSteps, guidance: guidance)
        eval(z)

        let diff = abs(z - zOut)
        let rel = (diff.max() / abs(zOut).max()).item(Float.self)
        let cos = (sum(z * zOut) / (sqrt(sum(z * z)) * sqrt(sum(zOut * zOut)))).item(Float.self)
        let zRMS = sqrt((z * z).mean()).item(Float.self)
        let refRMS = sqrt((zOut * zOut).mean()).item(Float.self)
        print("[decode-parity] rel=\(rel) cos=\(cos) swiftRMS=\(zRMS) torchRMS=\(refRMS) steps=\(numSteps) g=\(guidance)")
        XCTAssertGreaterThan(cos, 0.99, "Swift solver diverges from torch decode latent")
    }
}
