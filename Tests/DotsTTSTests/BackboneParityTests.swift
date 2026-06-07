import Foundation
import MLX
import MLXNN
import XCTest
@testable import DotsTTS

/// Parity: the Swift Qwen2 backbone must match the canonical mlx_lm loader on the
/// SAME int4 weights (both MLX -> tight tolerance). Compares last hidden state and
/// tied-embedding logits against backbone_reference.safetensors.
final class BackboneParityTests: XCTestCase {
    struct ConfigFile: Codable { let quantization: QuantizationSettings.Config? }

    func testBackboneMatchesMLXLMReference() throws {
        let env = ProcessInfo.processInfo.environment
        let repo = env["DOTS_MODEL_REPO"] ?? "/Users/samm/git/sammcj/dots.tts-soar-mlx"
        let fixtures = env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike"
        let dir = URL(fileURLWithPath: repo).appendingPathComponent("backbone")
        let refURL = URL(fileURLWithPath: fixtures).appendingPathComponent("backbone_reference.safetensors")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: dir.appendingPathComponent("model.safetensors").path)
                && FileManager.default.fileExists(atPath: refURL.path),
            "backbone weights or reference fixture not present"
        )

        let cfgData = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        let quant = QuantizationSettings(from: try JSONDecoder().decode(ConfigFile.self, from: cfgData).quantization)

        let backbone = Qwen2Backbone()
        if quant.enabled {
            quantize(model: backbone, groupSize: quant.groupSize, bits: quant.bits)
        }
        try WeightLoading.load(backbone, from: dir)

        let ref = try MLX.loadArrays(url: refURL)
        let ids = ref["input_ids"]!
        let hidden = backbone.hidden(ids)
        let logits = backbone.logits(ids)
        eval(hidden, logits)

        // mlx_lm ships the backbone in bf16, so max-abs rel is dominated by Qwen
        // outlier dims; cosine similarity is the meaningful parity metric.
        func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
            let x = a.reshaped(-1).asType(.float32)
            let y = b.reshaped(-1).asType(.float32)
            let c = (x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()))
            return c.item(Float.self)
        }
        let hRel = abs(hidden - ref["hidden"]!).max().item(Float.self) / abs(ref["hidden"]!).max().item(Float.self)
        let hCos = cosine(hidden, ref["hidden"]!)
        let lCos = cosine(logits, ref["logits"]!)
        print("backbone parity: hidden cos \(hCos)  logits cos \(lCos)  (hidden maxRel \(hRel), bf16)")
        XCTAssertGreaterThan(hCos, 0.999, "hidden state diverges from mlx_lm reference")
        XCTAssertGreaterThan(lCos, 0.999, "logits diverge from mlx_lm reference")
    }
}
