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

    /// Split target text into sentence-ish chunks so each render (and its
    /// vocoder decode) stays small. Splits after . ! ? keeping the punctuation.
    static func splitIntoChunks(_ text: String) -> [String] {
        var chunks: [String] = []
        var current = ""
        for ch in text {
            current.append(ch)
            if ch == "." || ch == "!" || ch == "?" {
                let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { chunks.append(trimmed) }
                current = ""
            }
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { chunks.append(tail) }
        return chunks.isEmpty ? [text] : chunks
    }

    func testRenderProducesAudio() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["DOTS_RUN_E2E"] == "1", "set DOTS_RUN_E2E=1 to run the slow e2e render")
        let repo = env["DOTS_MODEL_REPO"] ?? "/Users/samm/git/sammcj/dots.tts-soar-mlx"
        let fixtures = env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike"
        let outPath = env["DOTS_E2E_OUT"] ?? "\(fixtures)/dots_mlx_out.safetensors"
        let fixtureName = env["DOTS_FIXTURE"] ?? "e2e_fixture"

        let fixURL = URL(fileURLWithPath: fixtures).appendingPathComponent("\(fixtureName).safetensors")
        let metaURL = URL(fileURLWithPath: fixtures).appendingPathComponent("\(fixtureName).json")
        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        let refAudio = try MLX.loadArrays(url: fixURL)["ref_audio_48k"]!.asType(.float32)

        let tokenizer = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: repo).appendingPathComponent("backbone"))
        let pipeline = try DotsTTSPipeline(modelRepo: URL(fileURLWithPath: repo), tokenizer: tokenizer)

        var params = DotsTTSPipeline.Params()
        params.seed = UInt64(env["DOTS_SEED"].flatMap { UInt64($0) } ?? 1)
        params.maxOutputPatches = env["DOTS_MAX_PATCHES"].flatMap { Int($0) } ?? 200
        // For MeanFlow models this is the NFE (published default 4); for
        // flow-matching it's the Euler step count.
        if let n = env["DOTS_NUM_STEPS"].flatMap({ Int($0) }) { params.numSteps = n }
        if let g = env["DOTS_GUIDANCE"].flatMap({ Float($0) }) { params.guidance = g }
        if let m = env["DOTS_ODE_METHOD"] { params.odeMethod = ODEMethod(m) }

        // Bound MLX's buffer-recycle cache so this multi-chunk test can't hoard
        // freed buffers up to physical RAM (the host app sets this via its own
        // budget; a bare `swift test` has no such policy). Active allocations are
        // never limited by this - only the idle free pool.
        MLX.Memory.cacheLimit = 4 * 1024 * 1024 * 1024

        // Render sentence-by-sentence and join. A single-shot render of long
        // text makes one huge vocoder call (the latent->48kHz decode is the
        // memory hot spot), which can spike RAM into the tens of GB. Chunking
        // bounds each vocoder call and mirrors how the app renders long text.
        let chunks = Self.splitIntoChunks(meta.target_text)
        let gap = MLXArray.zeros([Int(Double(meta.sample_rate) * 0.12)], dtype: .float32)
        var pieces: [MLXArray] = []
        for (i, chunk) in chunks.enumerated() {
            let piece = pipeline.generate(
                targetText: chunk, refAudio48k: refAudio,
                refTranscript: meta.transcript, params: params).reshaped(-1)
            eval(piece)   // force this chunk's vocoder graph to run + free before the next
            // Return this chunk's idle Metal buffers to the OS before the next
            // chunk allocates, so peak memory tracks one chunk, not all of them.
            MLX.GPU.clearCache()
            print("[e2e] chunk \(i + 1)/\(chunks.count) samples=\(piece.dim(0)): \(chunk)")
            if i > 0 { pieces.append(gap) }
            pieces.append(piece)
        }
        let samples = concatenated(pieces, axis: 0)
        eval(samples)
        print("[e2e] output samples = \(samples.dim(0)) (\(Double(samples.dim(0)) / Double(meta.sample_rate))s)")
        XCTAssertGreaterThan(samples.dim(0), meta.sample_rate / 2, "render shorter than 0.5s")
        let peak = abs(samples).max().item(Float.self)
        print("[e2e] peak amplitude = \(peak)")
        XCTAssertGreaterThan(peak, 0.01, "render is near-silent")
        try MLX.save(arrays: ["audio": samples, "sample_rate": MLXArray(Int32(meta.sample_rate))], url: URL(fileURLWithPath: outPath))
        print("[e2e] wrote \(outPath)")
    }
}
