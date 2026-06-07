import Foundation
import Tokenizers
import XCTest
@testable import DotsTTS

/// Parity: the Swift schedule builder + special-token resolution must match the
/// Python build_generation_schedule output (schedule_reference.json) token-for-token.
final class TextScheduleParityTests: XCTestCase {
    struct Reference: Codable {
        let special_ids: [String: Int]
        let prompt_text: String
        let target_text: String
        let max_audio_tokens: Int
        let schedule_ids: [Int]
    }

    func testScheduleMatchesPythonReference() async throws {
        let env = ProcessInfo.processInfo.environment
        let repo = env["DOTS_MODEL_REPO"] ?? "/Users/samm/git/sammcj/dots.tts-soar-mlx"
        let fixtures = env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike"
        let backbone = URL(fileURLWithPath: repo).appendingPathComponent("backbone")
        let refURL = URL(fileURLWithPath: fixtures).appendingPathComponent("schedule_reference.json")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: backbone.appendingPathComponent("tokenizer.json").path)
                && FileManager.default.fileExists(atPath: refURL.path),
            "tokenizer or schedule reference not present")

        let ref = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: refURL))
        let tokenizer = try await AutoTokenizer.from(modelFolder: backbone)
        let special = try DotsSpecialTokens(tokenizer: tokenizer)

        XCTAssertEqual(special.audioGenStart, ref.special_ids["audio_gen_start"])
        XCTAssertEqual(special.audioGenSpan, ref.special_ids["audio_gen_span"])
        XCTAssertEqual(special.audioCompSpan, ref.special_ids["audio_comp_span"])
        XCTAssertEqual(special.textCondEnd, ref.special_ids["text_cond_end"])

        let ids = DotsTemplate.generationSchedule(
            promptText: ref.prompt_text,
            targetText: ref.target_text,
            maxAudioTokens: ref.max_audio_tokens,
            tokenizer: tokenizer,
            special: special)
        XCTAssertEqual(ids, ref.schedule_ids, "schedule token IDs diverge from Python reference")
    }
}
