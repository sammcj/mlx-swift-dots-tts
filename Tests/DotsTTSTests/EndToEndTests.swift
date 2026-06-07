import Foundation
import MLX
import Tokenizers
import XCTest
@testable import DotsTTS

/// End-to-end smoke + ear-check render. Gated behind DOTS_RUN_E2E=1 (slow,
/// stochastic). Loads a 48 kHz reference clip + transcript, synthesises the
/// target text, and writes the output samples to DOTS_E2E_OUT for WAV
/// conversion + listening. Asserts a plausible non-empty waveform.
final class EndToEndTests: XCTestCase {
    struct Meta: Codable { let transcript: String; let target_text: String; let sample_rate: Int }

    func testRenderProducesAudio() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["DOTS_RUN_E2E"] == "1", "set DOTS_RUN_E2E=1 to run the slow e2e render")
        let repo = env["DOTS_MODEL_REPO"] ?? "/Users/samm/git/sammcj/dots.tts-soar-mlx"
        let fixtures = env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike"
        let outPath = env["DOTS_E2E_OUT"] ?? "\(fixtures)/dots_mlx_out.safetensors"

        let fixURL = URL(fileURLWithPath: fixtures).appendingPathComponent("e2e_fixture.safetensors")
        let metaURL = URL(fileURLWithPath: fixtures).appendingPathComponent("e2e_fixture.json")
        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        let refAudio = try MLX.loadArrays(url: fixURL)["ref_audio_48k"]!.asType(.float32)

        let tokenizer = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: repo).appendingPathComponent("backbone"))
        let pipeline = try DotsTTSPipeline(modelRepo: URL(fileURLWithPath: repo), tokenizer: tokenizer)

        var params = DotsTTSPipeline.Params()
        params.seed = UInt64(env["DOTS_SEED"].flatMap { UInt64($0) } ?? 1)
        params.maxOutputPatches = 200
        let wav = pipeline.generate(
            targetText: meta.target_text, refAudio48k: refAudio,
            refTranscript: meta.transcript, params: params)
        eval(wav)
        let samples = wav.reshaped(-1)
        print("[e2e] output samples = \(samples.dim(0)) (\(Double(samples.dim(0)) / Double(meta.sample_rate))s)")
        XCTAssertGreaterThan(samples.dim(0), meta.sample_rate / 2, "render shorter than 0.5s")
        let peak = abs(samples).max().item(Float.self)
        print("[e2e] peak amplitude = \(peak)")
        XCTAssertGreaterThan(peak, 0.01, "render is near-silent")
        try MLX.save(arrays: ["audio": samples, "sample_rate": MLXArray(Int32(meta.sample_rate))], url: URL(fileURLWithPath: outPath))
        print("[e2e] wrote \(outPath)")
    }
}
