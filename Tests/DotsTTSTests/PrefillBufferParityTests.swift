import Foundation
import MLX
import Tokenizers
import XCTest
@testable import DotsTTS

/// Compares Swift's prefill-built first-decode FM conditioning (input_sequence /
/// cfg_sequence / g_cond) against Python's REAL decode-step buffers. Step 0 of
/// the solver matched perfectly when fed Python's buffer, but diverged with
/// Swift's own prefill buffer - so the gap is in how prefill builds the decode FM
/// sequence. This pins down which buffer and where.
final class PrefillBufferParityTests: XCTestCase {
    func testPrefillDecodeBufferMatchesPython() async throws {
        let env = ProcessInfo.processInfo.environment
        let repo = env["DOTS_MODEL_REPO"] ?? "/Users/samm/git/sammcj/dots.tts-soar-mlx"
        let fixtures = env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike"
        let stepURL = URL(fileURLWithPath: fixtures).appendingPathComponent("decode_step0.safetensors")
        let trajURL = URL(fileURLWithPath: fixtures).appendingPathComponent("decode_traj.safetensors")
        let fixURL = URL(fileURLWithPath: fixtures).appendingPathComponent("e2e_fixture.safetensors")
        let metaURL = URL(fileURLWithPath: fixtures).appendingPathComponent("e2e_fixture.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: stepURL.path), "decode_step0 missing")

        struct Meta: Codable { let transcript: String; let target_text: String; let sample_rate: Int }
        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        let refAudio = try MLX.loadArrays(url: fixURL)["ref_audio_48k"]!.asType(.float32)
        let step = try MLX.loadArrays(url: stepURL)
        let pyInput = step["input_sequence"]!.asType(.float32)
        let pyCfg = step["cfg_sequence"]!.asType(.float32)
        let pyG = step["g_cond"]!.asType(.float32)

        let traj = try MLX.loadArrays(url: trajURL)
        let noisesAll = traj["noises"]!.asType(.float32)
        var noises: [MLXArray] = []
        for i in 0 ..< noisesAll.dim(0) { noises.append(noisesAll[i]) }

        let tokenizer = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: repo).appendingPathComponent("backbone"))
        let pipeline = try DotsTTSPipeline(modelRepo: URL(fileURLWithPath: repo), tokenizer: tokenizer)
        pipeline.debugInjectNoise = noises
        var params = DotsTTSPipeline.Params()
        params.seed = 1
        params.maxOutputPatches = 2
        params.eosThreshold = 2.0
        _ = pipeline.generate(targetText: meta.target_text, refAudio48k: refAudio,
                              refTranscript: meta.transcript, params: params)

        let sw = pipeline.debugFirstInputSeq!
        let swCfg = pipeline.debugFirstCfgSeq!
        let swG = pipeline.debugFirstGCond!
        print("[buf] swift inputSeq \(sw.shape) python \(pyInput.shape)")
        print("[buf] swift gCond \(swG.shape) python \(pyG.shape)")

        func compare(_ a: MLXArray, _ b: MLXArray, _ name: String) {
            let L = min(a.dim(1), b.dim(1))
            let aa = a[0..., 0 ..< L], bb = b[0..., 0 ..< L]
            let cos = (sum(aa * bb) / (sqrt(sum(aa * aa)) * sqrt(sum(bb * bb)))).item(Float.self)
            print("[buf] \(name) overall cos=\(cos) (L=\(L))")
            // per-position cos to find where they diverge
            var firstBad = -1
            for p in 0 ..< L {
                let ap = aa[0..., p], bp = bb[0..., p]
                let c = (sum(ap * bp) / (sqrt(sum(ap * ap)) * sqrt(sum(bp * bp)) + 1e-9)).item(Float.self)
                if c < 0.99 && firstBad < 0 { firstBad = p }
            }
            print("[buf] \(name) first diverging position=\(firstBad)")
        }
        // gCond
        let gcos = (sum(swG * pyG) / (sqrt(sum(swG * swG)) * sqrt(sum(pyG * pyG)))).item(Float.self)
        print("[buf] gCond cos=\(gcos)")
        compare(sw, pyInput, "input_sequence")
        compare(swCfg, pyCfg, "cfg_sequence")
    }
}
