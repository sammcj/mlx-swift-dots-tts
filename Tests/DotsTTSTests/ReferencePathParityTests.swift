import Foundation
import MLX
import MLXNN
import XCTest
@testable import DotsTTS

/// Parity checks for the reference-audio path (AudioVAE encoder + PatchEncoder).
///
/// Two independent gates, each fed a stored intermediate so the module under
/// test is isolated:
///   1. PatchEncoder: trimmed_latent (1,72,128) -> patch_embeddings (1,18,1536)
///   2. AudioVAE encoder: waveform (145920,) -> latents_mean_logstd (1,256,76)
///
/// Fixtures live in DOTS_FIXTURES (default /Users/samm/git/dots-mlx-spike):
///   patchencoder_mlx.safetensors, audiovae_encoder_mlx.safetensors,
///   ref_refpath.safetensors. Tests are skipped (not failed) if absent.
final class ReferencePathParityTests: XCTestCase {
    private var fixturesDir: URL {
        let env = ProcessInfo.processInfo.environment
        return URL(fileURLWithPath: env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike")
    }

    private func relError(_ out: MLXArray, _ ref: MLXArray) -> Float {
        let maxAbs = abs(out - ref).max().item(Float.self)
        let scale = abs(ref).max().item(Float.self)
        return maxAbs / scale
    }

    /// Load a flat safetensors into a module and assert keys match (verify:.all).
    private func loadFlat(_ module: Module, _ url: URL) throws {
        let weights = try MLX.loadArrays(url: url)
        try module.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(module)
    }

    // MARK: Gate 1 - PatchEncoder (primary)

    func testPatchEncoderMatchesReference() throws {
        let weightsURL = fixturesDir.appendingPathComponent("patchencoder_mlx.safetensors")
        let refURL = fixturesDir.appendingPathComponent("ref_refpath.safetensors")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: weightsURL.path)
                && FileManager.default.fileExists(atPath: refURL.path),
            "patch encoder weights or reference fixture not present")

        let ref = try MLX.loadArrays(url: refURL)
        let input = ref["trimmed_latent"]!.asType(.float32)       // (1, 72, 128)
        let expected = ref["patch_embeddings"]!.asType(.float32)  // (1, 18, 1536)

        let pe = PatchEncoder()
        try loadFlat(pe, weightsURL)

        let out = pe(input)
        eval(out)

        XCTAssertEqual(out.shape, expected.shape, "patch embedding shape mismatch")
        let rel = relError(out, expected)
        print("PatchEncoder parity: rel \(rel)  shape \(out.shape)")
        XCTAssertLessThan(rel, 0.03, "patch embeddings diverge from torch reference")
    }

    // MARK: Gate 2 - AudioVAE encoder

    func testAudioVAEEncoderMatchesReference() throws {
        let weightsURL = fixturesDir.appendingPathComponent("audiovae_encoder_mlx.safetensors")
        let refURL = fixturesDir.appendingPathComponent("ref_refpath.safetensors")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: weightsURL.path)
                && FileManager.default.fileExists(atPath: refURL.path),
            "audiovae encoder weights or reference fixture not present")

        let ref = try MLX.loadArrays(url: refURL)
        let waveform = ref["waveform"]!.asType(.float32)               // (145920,)
        let expected = ref["latents_mean_logstd"]!.asType(.float32)    // (1, 256, 76)

        let enc = AudioVAEEncoder()
        try loadFlat(enc, weightsURL)

        let out = enc(waveform)
        eval(out)

        XCTAssertEqual(out.shape, expected.shape, "latent mean/logstd shape mismatch")
        let rel = relError(out, expected)
        print("AudioVAEEncoder parity: rel \(rel)  shape \(out.shape)")
        XCTAssertLessThan(rel, 0.03, "latents diverge from torch reference")
    }

    /// Verifies the sampling + normalize formulae against stored intermediates
    /// (documents the math; not on the gated module path).
    func testSamplingFormulaMatchesReference() throws {
        let refURL = fixturesDir.appendingPathComponent("ref_refpath.safetensors")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: refURL.path),
                          "reference fixture not present")
        let ref = try MLX.loadArrays(url: refURL)

        let meanLogstd = ref["latents_mean_logstd"]!.asType(.float32)  // (1, 256, 76)
        let noise = ref["sample_noise"]!.asType(.float32)             // (1, 128, 76)
        let sampled = ref["sampled_latent"]!.asType(.float32)         // (1, 76, 128)
        let trimmed = ref["trimmed_latent"]!.asType(.float32)         // (1, 72, 128)
        let normalized = ref["normalized_latent"]!.asType(.float32)   // (1, 72, 128)
        let mean = ref["latent_mean"]!.asType(.float32)               // (128,)
        let variance = ref["latent_var"]!.asType(.float32)            // (128,)
        let patch = Int(ref["patch_size"]!.item(Int32.self))

        // sample_from_latent: split channels -> m, logs; z = m + noise*exp(logs); transpose to (B,L,128)
        let parts = split(meanLogstd, parts: 2, axis: 1)
        let m = parts[0], logs = parts[1]
        let z = (m + noise * exp(logs)).transposed(0, 2, 1)
        eval(z)
        XCTAssertLessThan(relError(z, sampled), 1e-4, "sampling formula mismatch")

        // trimmed = sampled[:, :-patch_size]
        let L = sampled.dim(1)
        let tr = sampled[0..., 0..<(L - patch), 0...]
        eval(tr)
        XCTAssertLessThan(relError(tr, trimmed), 1e-5, "trim mismatch")

        // normalize: (x - mean) / sqrt(var)
        let norm = (trimmed - mean) / sqrt(variance)
        eval(norm)
        XCTAssertLessThan(relError(norm, normalized), 1e-4, "normalize formula mismatch")
    }
}
