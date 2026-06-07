import Foundation
import MLX
import Tokenizers
import XCTest
@testable import DotsTTS

/// Replays Python's decode trajectory with identical injected per-patch noise and
/// compares each solved latent. Step 0 already matches (DecodeStepParityTests);
/// the first diverging patch here localises the decode-loop FEEDBACK bug (LLM
/// step + patch-encoder recompute), which is what makes the Swift render run long
/// with leading silence.
final class DecodeTrajParityTests: XCTestCase {
    func testDecodeTrajectoryMatchesPython() async throws {
        let env = ProcessInfo.processInfo.environment
        let repo = env["DOTS_MODEL_REPO"] ?? "/Users/samm/git/sammcj/dots.tts-soar-mlx"
        let fixtures = env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike"
        let trajURL = URL(fileURLWithPath: fixtures).appendingPathComponent("decode_traj.safetensors")
        let fixURL = URL(fileURLWithPath: fixtures).appendingPathComponent("e2e_fixture.safetensors")
        let metaURL = URL(fileURLWithPath: fixtures).appendingPathComponent("e2e_fixture.json")
        // Investigation scaffolding: full-trajectory parity holds under fp16 but
        // int4 quantisation drifts in the autoregressive loop (~patch 9), so this
        // is opt-in rather than a default regression guard.
        try XCTSkipUnless(env["DOTS_RUN_DECODE_PARITY"] == "1", "set DOTS_RUN_DECODE_PARITY=1")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: trajURL.path), "decode_traj fixture missing")

        struct Meta: Codable { let transcript: String; let target_text: String; let sample_rate: Int }
        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        let refAudio = try MLX.loadArrays(url: fixURL)["ref_audio_48k"]!.asType(.float32)

        let traj = try MLX.loadArrays(url: trajURL)
        let noisesAll = traj["noises"]!.asType(.float32)         // (NPATCH, 1, 4, 128)
        let npatch = noisesAll.dim(0)
        var noises: [MLXArray] = []
        for i in 0 ..< npatch { noises.append(noisesAll[i]) }     // (1, 4, 128)

        let tokenizer = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: repo).appendingPathComponent("backbone"))
        let pipeline = try DotsTTSPipeline(modelRepo: URL(fileURLWithPath: repo), tokenizer: tokenizer)
        pipeline.debugInjectNoise = noises
        // inject Python's exact prompt latents + speaker g_cond to remove RNG.
        if let pl = traj["prompt_latents"] { pipeline.debugInjectPromptLatents = pl.asType(.float32) }
        if let g = traj["prompt_g_cond"] { pipeline.debugInjectGCond = g.asType(.float32) }

        var params = DotsTTSPipeline.Params()
        params.seed = 1
        params.maxOutputPatches = npatch + 6
        params.eosThreshold = 2.0   // never EOS-break, so we capture the full window
        _ = pipeline.generate(targetText: meta.target_text, refAudio48k: refAudio,
                              refTranscript: meta.transcript, params: params)

        let captured = pipeline.debugCapturedZ
        print("[traj] swift captured \(captured.count) patches, python has \(npatch)")
        let n = min(captured.count, npatch)
        var firstDiverge = -1
        for i in 0 ..< n {
            let zi = captured[i]
            let pz = traj[String(format: "z_%02d", i)]!.asType(.float32)
            let cos = (sum(zi * pz) / (sqrt(sum(zi * zi)) * sqrt(sum(pz * pz)))).item(Float.self)
            let sRMS = sqrt((zi * zi).mean()).item(Float.self)
            let pRMS = sqrt((pz * pz).mean()).item(Float.self)
            let flag = cos < 0.99 ? "  <-- DIVERGE" : ""
            print(String(format: "  patch %02d cos=%.5f swiftRMS=%.4f pyRMS=%.4f%@", i, cos, sRMS, pRMS, flag))
            if cos < 0.99 && firstDiverge < 0 { firstDiverge = i }
        }
        // split the feedback path: patch-encoder embedding vs LLM hidden, per step
        func cosOf(_ a: MLXArray, _ b: MLXArray) -> Float {
            (sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)))).item(Float.self)
        }
        let ne = min(pipeline.debugCapturedEmbed.count, npatch)
        for i in 0 ..< ne {
            guard let pe = traj[String(format: "embed_%02d", i)] else { continue }
            guard let ph = traj[String(format: "hidden_%02d", i)] else { continue }
            let ec = cosOf(pipeline.debugCapturedEmbed[i], pe.asType(.float32))
            let hc = cosOf(pipeline.debugCapturedHidden[i], ph.asType(.float32))
            print(String(format: "  step %02d embedCos=%.5f hiddenCos=%.5f", i, ec, hc))
        }
        XCTAssertEqual(firstDiverge, -1, "decode trajectory diverges at patch \(firstDiverge)")
    }
}
