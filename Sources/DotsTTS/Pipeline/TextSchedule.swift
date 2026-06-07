import Foundation
import Tokenizers

/// Special-token IDs the dots schedule + decode loop key on. Resolved from the
/// loaded tokenizer's added vocab (Qwen2 base + dots audio tokens).
public struct DotsSpecialTokens: Sendable {
    public let audioGenStart: Int
    public let audioGenSpan: Int
    public let audioCompSpan: Int
    public let textCondEnd: Int

    static let audioGenStartToken = "<|audio_gen_start|>"
    static let audioGenSpanToken = "<|audio_gen_span|>"
    static let audioCompSpanToken = "<|audio_comp_span|>"
    static let textCondEndToken = "<|text_cond_end|>"

    public init(tokenizer: Tokenizer) throws {
        func id(_ token: String) throws -> Int {
            guard let v = tokenizer.convertTokenToId(token) else {
                throw DotsTextError.missingSpecialToken(token)
            }
            return v
        }
        self.audioGenStart = try id(Self.audioGenStartToken)
        self.audioGenSpan = try id(Self.audioGenSpanToken)
        self.audioCompSpan = try id(Self.audioCompSpanToken)
        self.textCondEnd = try id(Self.textCondEndToken)
    }
}

public enum DotsTextError: Error {
    case missingSpecialToken(String)
}

/// Builds the generation schedule for the default "tts" template
/// `[文本]{text}[文本对应语音]{audio}`. Mirrors build_generation_schedule:
/// each literal segment and the text are tokenized independently
/// (add_special_tokens=False), then the audio block expands to one
/// audio_gen_start followed by `maxAudioTokens` audio_gen_span tokens.
///
/// For voice cloning the reference transcript is prepended to the target text
/// (runtime concatenates `prompt_text + text`); the first `prompt_patch_count`
/// of the audio_gen_span positions are consumed by prefill as the reference
/// audio, the rest are generated.
public enum DotsTemplate {
    public static let textPrefix = "[文本]"
    public static let audioPrefix = "[文本对应语音]"

    public static func generationSchedule(
        promptText: String?,
        targetText: String,
        maxAudioTokens: Int,
        tokenizer: Tokenizer,
        special: DotsSpecialTokens
    ) -> [Int] {
        precondition(maxAudioTokens > 0, "maxAudioTokens must be positive")
        // Mirror runtime `_process_prompt_text` / `_process_text`: strip both, and
        // for non-CJK prompts append a trailing space before concatenating with the
        // target. Without the space the boundary (".The") tokenises differently from
        // Python's ". The", shifting the whole prefill by a token and corrupting the
        // KV cache, which degrades the cloned voice and runs generation long.
        let prompt = (promptText ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let target = targetText.trimmingCharacters(in: .whitespacesAndNewlines)
        let text: String
        if prompt.isEmpty {
            text = target
        } else {
            let separator = containsCJK(prompt) ? "" : " "
            text = prompt + separator + target
        }
        var ids: [Int] = []
        ids += tokenizer.encode(text: textPrefix, addSpecialTokens: false)
        ids += tokenizer.encode(text: text, addSpecialTokens: false)
        ids += tokenizer.encode(text: audioPrefix, addSpecialTokens: false)
        ids.append(special.audioGenStart)
        ids += Array(repeating: special.audioGenSpan, count: maxAudioTokens)
        return ids
    }

    /// True if the text contains CJK ideographs or Japanese kana. Used to match
    /// the runtime's language rule for whether to insert a prompt/target space.
    static func containsCJK(_ s: String) -> Bool {
        for scalar in s.unicodeScalars {
            let v = scalar.value
            if (0x4E00...0x9FFF).contains(v)      // CJK Unified Ideographs
                || (0x3400...0x4DBF).contains(v)  // CJK Ext A
                || (0x3040...0x309F).contains(v)  // Hiragana
                || (0x30A0...0x30FF).contains(v)  // Katakana
                || (0xF900...0xFAFF).contains(v)  // CJK Compatibility Ideographs
            { return true }
        }
        return false
    }
}
