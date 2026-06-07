import Foundation
import MLX
import Tokenizers
import XCTest
@testable import DotsTTS

/// Isolates the LLM decode-step (cached, L==1) path. Feeds Python's EXACT
/// per-step patch-encoder embedding into the LLM step (plus injected prompt
/// latents / gCond / noise), so the only thing under test is backbone.step's
/// cached attention. If the hidden still diverges from Python with identical
/// input, the bug is in the cached decode path, not the patch encoder.
final class LLMStepParityTests: XCTestCase {
    func testLLMDecodeStepMatchesPython() async throws {
        let env = ProcessInfo.processInfo.environment
        let repo = env["DOTS_MODEL_REPO"] ?? "/Users/samm/git/sammcj/dots.tts-soar-mlx"
        let fixtures = env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike"
        let trajURL = URL(fileURLWithPath: fixtures).appendingPathComponent("decode_traj.safetensors")
        let fixURL = URL(fileURLWithPath: fixtures).appendingPathComponent("e2e_fixture.safetensors")
        let metaURL = URL(fileURLWithPath: fixtures).appendingPathComponent("e2e_fixture.json")
        // Investigation scaffolding (fp16-clean; int4 drifts) - opt-in.
        try XCTSkipUnless(env["DOTS_RUN_DECODE_PARITY"] == "1", "set DOTS_RUN_DECODE_PARITY=1")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: trajURL.path), "decode_traj missing")

        struct Meta: Codable { let transcript: String; let target_text: String; let sample_rate: Int }
        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        let refAudio = try MLX.loadArrays(url: fixURL)["ref_audio_48k"]!.asType(.float32)
        let traj = try MLX.loadArrays(url: trajURL)

        let noisesAll = traj["noises"]!.asType(.float32)
        let npatch = noisesAll.dim(0)
        var noises: [MLXArray] = []
        var embeds: [MLXArray] = []
        for i in 0 ..< npatch {
            noises.append(noisesAll[i])
            if let e = traj[String(format: "embed_%02d", i)] { embeds.append(e.asType(.float32)) }
        }

        let tokenizer = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: repo).appendingPathComponent("backbone"))
        let pipeline = try DotsTTSPipeline(modelRepo: URL(fileURLWithPath: repo), tokenizer: tokenizer)
        pipeline.debugInjectNoise = noises
        pipeline.debugInjectEmbed = embeds
        if let pl = traj["prompt_latents"] { pipeline.debugInjectPromptLatents = pl.asType(.float32) }
        if let g = traj["prompt_g_cond"] { pipeline.debugInjectGCond = g.asType(.float32) }

        var params = DotsTTSPipeline.Params()
        params.seed = 1
        params.maxOutputPatches = npatch + 6
        params.eosThreshold = 2.0
        _ = pipeline.generate(targetText: meta.target_text, refAudio48k: refAudio,
                              refTranscript: meta.transcript, params: params)

        func cosOf(_ a: MLXArray, _ b: MLXArray) -> Float {
            (sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)))).item(Float.self)
        }
        let n = min(pipeline.debugCapturedHidden.count, embeds.count)
        var worst: Float = 1
        for i in 0 ..< n {
            guard let ph = traj[String(format: "hidden_%02d", i)] else { continue }
            // embedCos here should be ~1.0 since we inject Python's embed
            let ec = cosOf(pipeline.debugCapturedEmbed[i], embeds[i])
            let hc = cosOf(pipeline.debugCapturedHidden[i], ph.asType(.float32))
            worst = min(worst, hc)
            print(String(format: "  step %02d injEmbedCos=%.5f hiddenCos=%.5f", i, ec, hc))
        }
        print("[llmstep] worst hiddenCos with injected embed = \(worst)")
        XCTAssertGreaterThan(worst, 0.99, "cached LLM decode step diverges even with identical input")
    }
}
