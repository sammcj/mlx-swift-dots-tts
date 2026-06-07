import Foundation
import MLX
import Tokenizers
import XCTest
@testable import DotsTTS

/// Stage-by-stage parity for the orchestration glue (the components are verified
/// elsewhere). Compares deterministic intermediates against the Python model
/// dumped in pipeline_intermediates.safetensors. Gated on the fixtures.
final class PipelineStageTests: XCTestCase {
    private func loadPipeline() async throws -> (DotsTTSPipeline, [String: MLXArray], MLXArray) {
        let env = ProcessInfo.processInfo.environment
        let repo = env["DOTS_MODEL_REPO"] ?? "/Users/samm/git/sammcj/dots.tts-soar-mlx"
        let fixtures = env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike"
        let interURL = URL(fileURLWithPath: fixtures).appendingPathComponent("pipeline_intermediates.safetensors")
        let fixURL = URL(fileURLWithPath: fixtures).appendingPathComponent("e2e_fixture.safetensors")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: interURL.path)
                && FileManager.default.fileExists(atPath: fixURL.path),
            "pipeline intermediates / e2e fixture not present")
        let inter = try MLX.loadArrays(url: interURL)
        let refAudio = try MLX.loadArrays(url: fixURL)["ref_audio_48k"]!.asType(.float32)
        let tokenizer = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: repo).appendingPathComponent("backbone"))
        let pipeline = try DotsTTSPipeline(modelRepo: URL(fileURLWithPath: repo), tokenizer: tokenizer)
        return (pipeline, inter, refAudio)
    }

    private func rel(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = abs(a - b)
        return (sqrt((d * d).sum()) / sqrt((b * b).sum())).item(Float.self)
    }
    private func cos(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.reshaped(-1).asType(.float32), y = b.reshaped(-1).asType(.float32)
        return ((x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()))).item(Float.self)
    }

    func testSpeakerCondMatchesPython() async throws {
        let (pipeline, _, refAudio) = try await loadPipeline()
        let env = ProcessInfo.processInfo.environment
        let fixtures = env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike"
        let refURL = URL(fileURLWithPath: fixtures).appendingPathComponent("speaker_ref2.safetensors")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: refURL.path), "speaker_ref2 not present")
        let spkRef = try MLX.loadArrays(url: refURL)
        let stages = pipeline.debugSpeakerStages(refAudio48k: refAudio)
        for key in ["resampled_16k", "fbank", "xvector_scaled", "g_cond"] {
            let got = stages[key]!
            let exp = spkRef[key]!.asType(.float32).reshaped(got.shape)
            eval(got)
            print("[stage \(key)] rel=\(rel(got, exp)) cos=\(cos(got, exp)) shape=\(got.shape)")
        }
        let g = stages["g_cond"]!
        let expected = spkRef["g_cond"]!.asType(.float32)
        XCTAssertLessThan(rel(g, expected), 0.03, "g_cond rel too high")
        XCTAssertGreaterThan(cos(g, expected), 0.999, "g_cond cos too low")
    }

    /// Prefill parity using Python's exact sampled latents (bypasses VAE RNG).
    func testPrefillMatchesPython() async throws {
        let (pipeline, inter, _) = try await loadPipeline()
        let transcript = "I'm very sorry I can't be with you all today and such an important gathering. Some have speculated that I've seen more of the natural world than anyone else."
        let target = "The natural world is the greatest source of wonder we will ever know."
        let refLatents = inter["prompt_latents_sampled"]!.asType(.float32)
        let got = pipeline.debugPrefill(refLatentsTrim: refLatents, targetText: target, refTranscript: transcript)
        // cross-check: pipeline's own instance via callAsFunction vs debugStages.
        let viaCall = got["prompt_patch_embeddings"]!
        let viaStages = got["ppe_via_stages"]!
        eval(viaCall, viaStages)
        print("[prefill call-vs-stages] cos=\(cos(viaCall, viaStages))")
        for key in ["prompt_patch_embeddings", "ppe_via_stages", "fm_sequence", "fm_cfg_sequence", "llm_hidden_last"] {
            let g = got[key]!
            let refKey = key == "ppe_via_stages" ? "prompt_patch_embeddings" : key
            let exp = inter[refKey]!.asType(.float32).reshaped(g.shape)
            eval(g)
            print("[prefill \(key)] rel=\(rel(g, exp)) cos=\(cos(g, exp)) shape=\(g.shape)")
        }
        // ppe and the cfg (uncond) buffer are quant-independent -> near exact.
        let ppe = got["prompt_patch_embeddings"]!
        XCTAssertGreaterThan(cos(ppe, inter["prompt_patch_embeddings"]!.asType(.float32).reshaped(ppe.shape)), 0.999,
                             "patch embeddings cos too low")
        let cfg = got["fm_cfg_sequence"]!
        XCTAssertGreaterThan(cos(cfg, inter["fm_cfg_sequence"]!.asType(.float32).reshaped(cfg.shape)), 0.999,
                             "fm cfg buffer cos too low")
        // cond buffer + llm hidden embed the int4-quantised backbone hiddens, so
        // they carry quantisation drift vs the fp32 Python reference (~0.98).
        let llm = got["llm_hidden_last"]!
        let expLlm = inter["llm_hidden_last"]!.asType(.float32).reshaped(llm.shape)
        XCTAssertGreaterThan(cos(llm, expLlm), 0.97, "prefill llm hidden cos too low even for int4")
    }

    /// Localise the patch-encoder divergence at the real 276-len input.
    func testPatchEncoderStages() async throws {
        let env = ProcessInfo.processInfo.environment
        let repo = env["DOTS_MODEL_REPO"] ?? "/Users/samm/git/sammcj/dots.tts-soar-mlx"
        let fixtures = env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike"
        let stagesURL = URL(fileURLWithPath: fixtures).appendingPathComponent("pe_stages.safetensors")
        let interURL = URL(fileURLWithPath: fixtures).appendingPathComponent("pipeline_intermediates.safetensors")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: stagesURL.path)
            && FileManager.default.fileExists(atPath: interURL.path), "pe stages not present")
        let stagesRef = try MLX.loadArrays(url: stagesURL)
        let x = try MLX.loadArrays(url: interURL)["prompt_latents_sampled"]!.asType(.float32)
        let pe = PatchEncoder()
        try WeightLoading.load(pe, from: URL(fileURLWithPath: repo).appendingPathComponent("patch_encoder"))
        eval(pe)
        let got = pe.debugStages(x)
        for key in ["after_downsample", "after_in_proj", "after_encoder", "final"] {
            let g = got[key]!
            let exp = stagesRef[key]!.asType(.float32).reshaped(g.shape)
            eval(g)
            print("[pe \(key)] rel=\(rel(g, exp)) cos=\(cos(g, exp)) shape=\(g.shape)")
        }
    }
}
