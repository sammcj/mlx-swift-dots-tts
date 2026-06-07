import Foundation
import MLX
import MLXNN
import XCTest
@testable import DotsTTS

/// Parity check for the CAM++ speaker encoder.
///
/// PRIMARY gate: feed the stored `fbank` (298,80) through the network and compare
/// the 512-d output to the stored `xvector`. This isolates the CAM++ network from
/// the fbank front-end. Target relative error < 0.02.
///
/// SECONDARY (looser): waveform -> fbank vs the stored fbank. Windowing/edge
/// handling can differ from torchaudio, so this is a sanity check, not a gate.
///
/// Fixtures live in `dots-mlx-spike` by default (env-overridable):
///   DOTS_FIXTURES       -> dir holding ref_speaker.safetensors
///   DOTS_SPEAKER_WEIGHTS -> speaker_encoder_mlx.safetensors (defaults to fixtures dir)
final class SpeakerParityTests: XCTestCase {
    private func fixturesDir() -> URL {
        let env = ProcessInfo.processInfo.environment
        return URL(fileURLWithPath: env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike")
    }

    private func weightsURL() -> URL {
        let env = ProcessInfo.processInfo.environment
        if let p = env["DOTS_SPEAKER_WEIGHTS"] { return URL(fileURLWithPath: p) }
        return fixturesDir().appendingPathComponent("speaker_encoder_mlx.safetensors")
    }

    private func refURL() -> URL {
        fixturesDir().appendingPathComponent("ref_speaker.safetensors")
    }

    private func loadEncoder() throws -> CAMPPlus {
        let weights = try MLX.loadArrays(url: weightsURL())
        let model = CAMPPlus()
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)
        return model
    }

    /// PRIMARY: stored fbank -> xvector parity.
    func testNetworkParityFromStoredFbank() throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: weightsURL().path)
                && FileManager.default.fileExists(atPath: refURL().path),
            "speaker weights or reference fixture not present")

        let ref = try MLX.loadArrays(url: refURL())
        let fbank = ref["fbank"]!.asType(.float32)        // (298, 80)
        let expected = ref["xvector"]!.asType(.float32)   // (512,)

        let model = try loadEncoder()
        let input = fbank.expandedDimensions(axis: 0)     // (1, 298, 80)
        let out = model(input).reshaped(512)              // (512,)
        eval(out)

        let diff = abs(out - expected)
        let maxAbs = diff.max().item(Float.self)
        let relError = (sqrt((diff * diff).sum()) / sqrt((expected * expected).sum())).item(Float.self)
        let cosine = ((out * expected).sum()
            / (sqrt((out * out).sum()) * sqrt((expected * expected).sum()))).item(Float.self)
        print("[speaker network parity] rel=\(relError) maxAbs=\(maxAbs) cos=\(cosine)")

        XCTAssertLessThan(relError, 0.02, "network rel error \(relError) exceeds 0.02")
        XCTAssertGreaterThan(cosine, 0.999, "cosine \(cosine) below 0.999")
    }

    /// SECONDARY (looser, non-gating): waveform -> fbank.
    func testFbankFrontEndApproximate() throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: refURL().path),
            "reference fixture not present")

        let ref = try MLX.loadArrays(url: refURL())
        let waveform = ref["waveform"]!.asType(.float32)   // (48000,)
        let expected = ref["fbank"]!.asType(.float32)      // (298, 80)

        let fbank = KaldiFbank()
        let got = fbank(waveform)
        eval(got)

        XCTAssertEqual(got.dim(0), expected.dim(0), "frame count mismatch")
        XCTAssertEqual(got.dim(1), expected.dim(1), "mel bin count mismatch")

        let diff = abs(got - expected)
        let maxAbs = diff.max().item(Float.self)
        let relError = (sqrt((diff * diff).sum()) / sqrt((expected * expected).sum())).item(Float.self)
        print("[speaker fbank parity] rel=\(relError) maxAbs=\(maxAbs) shape=\(got.shape)")
        // Looser tolerance: fbank windowing/edge handling differs from torchaudio.
        XCTAssertLessThan(relError, 0.25, "fbank rel error \(relError) unexpectedly large")
    }
}
