import Foundation
import MLX
import Tokenizers
import XCTest
@testable import DotsTTS

/// Prints Swift's schedule prefix (non-span) token ids so they can be diffed
/// against Python's build_generation_schedule. A tokenization mismatch shifts the
/// whole prefill and corrupts the KV cache.
final class ScheduleDumpTests: XCTestCase {
    func testDumpSchedulePrefix() async throws {
        let env = ProcessInfo.processInfo.environment
        let repo = env["DOTS_MODEL_REPO"] ?? "/Users/samm/git/sammcj/dots.tts-soar-mlx"
        let transcript = "I'm very sorry I can't be with you all today and such an important gathering. Some have speculated that I've seen more of the natural world than anyone else."
        let target = "The natural world is the greatest source of wonder we will ever know."

        let tokenizer = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: repo).appendingPathComponent("backbone"))
        let special = try DotsSpecialTokens(tokenizer: tokenizer)
        let schedule = DotsTemplate.generationSchedule(
            promptText: transcript, targetText: target, maxAudioTokens: 512,
            tokenizer: tokenizer, special: special)
        let prefix = schedule.filter { $0 != special.audioGenSpan }
        print("[sched] prefix_len_nonspan \(prefix.count)")
        print("[sched] prefix \(prefix)")
        // sub-piece tokenizations
        print("[sched] textPrefix \(tokenizer.encode(text: "[文本]", addSpecialTokens: false))")
        print("[sched] audioPrefix \(tokenizer.encode(text: "[文本对应语音]", addSpecialTokens: false))")
        let combined = transcript + target
        print("[sched] textOnly_len \(tokenizer.encode(text: combined, addSpecialTokens: false).count)")
        print("[sched] textOnly \(tokenizer.encode(text: combined, addSpecialTokens: false))")
    }
}
