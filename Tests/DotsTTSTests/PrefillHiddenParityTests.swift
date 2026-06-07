import Foundation
import MLX
import Tokenizers
import XCTest
@testable import DotsTTS

/// Per-position parity of the full prefill hidden states. The prefill last-token
/// hidden matched (~1.0) but early decode steps gave hidden cos ~0.94 even with
/// identical input, implicating the prefill-populated KV cache. If non-final
/// prefill positions diverge here, the backbone forward has a position-dependent
/// bug; if all positions match, the bug is in the cached decode path itself.
final class PrefillHiddenParityTests: XCTestCase {
    func testFullPrefillHiddenMatchesPython() async throws {
        let env = ProcessInfo.processInfo.environment
        let repo = env["DOTS_MODEL_REPO"] ?? "/Users/samm/git/sammcj/dots.tts-soar-mlx"
        let fixtures = env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike"
        let trajURL = URL(fileURLWithPath: fixtures).appendingPathComponent("decode_traj.safetensors")
        let fixURL = URL(fileURLWithPath: fixtures).appendingPathComponent("e2e_fixture.safetensors")
        let metaURL = URL(fileURLWithPath: fixtures).appendingPathComponent("e2e_fixture.json")
        // Investigation scaffolding (fp16-clean; int4 per-position drifts) - opt-in.
        try XCTSkipUnless(env["DOTS_RUN_DECODE_PARITY"] == "1", "set DOTS_RUN_DECODE_PARITY=1")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: trajURL.path), "decode_traj missing")

        struct Meta: Codable { let transcript: String; let target_text: String; let sample_rate: Int }
        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        let refAudio = try MLX.loadArrays(url: fixURL)["ref_audio_48k"]!.asType(.float32)
        let traj = try MLX.loadArrays(url: trajURL)
        let pyPrefill = traj["prefill_hidden"]!.asType(.float32)   // (1, 125, 1536)

        let noisesAll = traj["noises"]!.asType(.float32)
        var noises: [MLXArray] = []
        for i in 0 ..< noisesAll.dim(0) { noises.append(noisesAll[i]) }

        let tokenizer = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: repo).appendingPathComponent("backbone"))
        let pipeline = try DotsTTSPipeline(modelRepo: URL(fileURLWithPath: repo), tokenizer: tokenizer)
        pipeline.debugInjectNoise = noises
        if let pl = traj["prompt_latents"] { pipeline.debugInjectPromptLatents = pl.asType(.float32) }
        if let g = traj["prompt_g_cond"] { pipeline.debugInjectGCond = g.asType(.float32) }
        var params = DotsTTSPipeline.Params()
        params.seed = 1
        params.maxOutputPatches = 2
        params.eosThreshold = 2.0
        _ = pipeline.generate(targetText: meta.target_text, refAudio48k: refAudio,
                              refTranscript: meta.transcript, params: params)

        let sw = pipeline.debugFullPrefillHidden!   // (1, L, 1536)
        let L = min(sw.dim(1), pyPrefill.dim(1))
        print("[prefill-hidden] swift L=\(sw.dim(1)) python L=\(pyPrefill.dim(1))")
        func cosAt(_ p: Int) -> Float {
            let a = sw[0..., p], b = pyPrefill[0..., p]
            return (sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)))).item(Float.self)
        }
        var worst: Float = 1; var worstPos = -1
        for p in 0 ..< L {
            let c = cosAt(p)
            if c < worst { worst = c; worstPos = p }
            if p < 8 || p >= L - 8 || c < 0.99 {
                print(String(format: "  pos %03d cos=%.5f", p, c))
            }
        }
        print("[prefill-hidden] worst cos=\(worst) at pos \(worstPos)")
        XCTAssertGreaterThan(worst, 0.99, "prefill hidden diverges at pos \(worstPos)")
    }
}
