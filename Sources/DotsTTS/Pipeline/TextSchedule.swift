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
        let text = (promptText ?? "") + targetText
        var ids: [Int] = []
        ids += tokenizer.encode(text: textPrefix, addSpecialTokens: false)
        ids += tokenizer.encode(text: text, addSpecialTokens: false)
        ids += tokenizer.encode(text: audioPrefix, addSpecialTokens: false)
        ids.append(special.audioGenStart)
        ids += Array(repeating: special.audioGenSpan, count: maxAudioTokens)
        return ids
    }
}
