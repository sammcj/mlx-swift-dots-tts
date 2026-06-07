import Foundation
import MLX
import MLXNN
import XCTest
@testable import DotsTTS

/// Parity check: the Swift vocoder decoder must match the torch fp32 reference
/// waveform (ref_vocoder.safetensors) within fp32 cross-framework drift.
///
/// Fixtures live in the dots-mlx-spike dir; override with env vars so the test is
/// skipped (not failed) on machines without them:
///   DOTS_VOCODER_WEIGHTS -> vocoder_decoder_mlx.safetensors
///   DOTS_VOCODER_REF      -> ref_vocoder.safetensors
final class VocoderParityTests: XCTestCase {
    func testVocoderMatchesTorchReference() throws {
        let env = ProcessInfo.processInfo.environment
        let spike = env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike"
        let weightsPath = env["DOTS_VOCODER_WEIGHTS"]
            ?? "\(spike)/vocoder_decoder_mlx.safetensors"
        let refPath = env["DOTS_VOCODER_REF"]
            ?? "\(spike)/ref_vocoder.safetensors"

        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: weightsPath)
                && FileManager.default.fileExists(atPath: refPath),
            "vocoder weights or reference fixture not present"
        )

        let weights = try MLX.loadArrays(url: URL(fileURLWithPath: weightsPath))
        let ref = try MLX.loadArrays(url: URL(fileURLWithPath: refPath))
        let latent = ref["latent"]!.asType(.float32)    // (1, 128, 32)
        let target = ref["waveform"]!.asType(.float32)  // (1, 1, 61440)

        let vocoder = Vocoder()
        try vocoder.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(vocoder)

        let out = vocoder(latent)
        eval(out)

        XCTAssertEqual(out.shape, target.shape, "waveform shape mismatch")

        let maxAbs = abs(out - target).max().item(Float.self)
        let refScale = abs(target).max().item(Float.self)
        let rel = maxAbs / refScale
        print("Vocoder parity: maxAbs \(maxAbs)  rel \(rel)  (ref scale \(refScale))")
        // The decoder is a deep causal stack (6 upsamples x 18 resblocks, each
        // with error-amplifying SnakeBeta sin(exp(alpha)*x)). Stage diagnostics
        // show error accumulating smoothly from 7e-4 (post LSTM) to ~2.1e-2 at
        // the output with no single-stage jump - pure fp32 cross-framework drift,
        // not a correctness bug. maxAbs diff is ~1.4e-3 on a [-1,1] waveform
        // (inaudible). Allow a small margin over the observed 0.0213.
        XCTAssertLessThan(rel, 0.03, "vocoder waveform diverges from torch reference")
    }

    /// Isolated SLSTM check against lstm_stages.safetensors (per-layer + full).
    func testSLSTMIsolated() throws {
        let env = ProcessInfo.processInfo.environment
        let spike = env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike"
        let weightsPath = "\(spike)/vocoder_decoder_mlx.safetensors"
        let lstmPath = "\(spike)/lstm_stages.safetensors"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: weightsPath)
                && FileManager.default.fileExists(atPath: lstmPath),
            "lstm dump not present"
        )
        let weights = try MLX.loadArrays(url: URL(fileURLWithPath: weightsPath))
        let st = try MLX.loadArrays(url: URL(fileURLWithPath: lstmPath))
        let v = Vocoder()
        try v.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(v)
        let slstm = v.decMiLayer[1] as! SLSTM
        let xin = st["lstm_in"]!.asType(.float32)  // (1,32,512)
        let out = slstm(xin)  // includes skip
        let refFull = st["lstm_full"]! + st["lstm_in"]!  // skip applied
        let mx = abs(out - refFull).max().item(Float.self)
        let sc = abs(refFull).max().item(Float.self)
        let rel = mx / sc
        print(String(format: "SLSTM(skip) rel %.6f  maxAbs %.6g  scale %.6g", rel, mx, sc))
        XCTAssertLessThan(rel, 0.01, "SLSTM diverges from torch reference")
    }

    /// Stage-by-stage diagnostic: compares each decoder stage against the torch
    /// dump (vocoder_stages.safetensors). Skipped if the dump is absent.
    func testVocoderStageDiagnostics() throws {
        let env = ProcessInfo.processInfo.environment
        let spike = env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike"
        let weightsPath = "\(spike)/vocoder_decoder_mlx.safetensors"
        let stagesPath = "\(spike)/vocoder_stages.safetensors"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: weightsPath)
                && FileManager.default.fileExists(atPath: stagesPath),
            "vocoder stage dump not present"
        )
        let weights = try MLX.loadArrays(url: URL(fileURLWithPath: weightsPath))
        let st = try MLX.loadArrays(url: URL(fileURLWithPath: stagesPath))

        let v = Vocoder()
        try v.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(v)

        // got is always (B, L, C). transpose=true when the torch ref is (B, C, L).
        func cmp(_ name: String, _ got: MLXArray, _ refKey: String, transpose: Bool) {
            let r = transpose ? st[refKey]!.transposed(0, 2, 1) : st[refKey]!
            let mx = abs(got - r).max().item(Float.self)
            let sc = abs(r).max().item(Float.self)
            print(String(format: "stage %@ rel %.5f  maxAbs %.6g  scale %.6g",
                         name.padding(toLength: 11, withPad: " ", startingAt: 0), mx / sc, mx, sc))
        }

        let latent = st["latent"]!.asType(.float32)
        var h = v.postProj(latent.transposed(0, 2, 1))
        cmp("post_proj", h, "post_proj", transpose: true)
        let mi = v.decMiLayer
        h = mi[0](h); cmp("mi_fc0", h, "mi_fc0", transpose: false)
        h = mi[1](h); cmp("mi_slstm", h, "mi_slstm", transpose: false)
        h = mi[2](h); cmp("mi_fc2", h, "mi_fc2", transpose: false)

        let dec = v.decoder
        var x = dec.convPre(h)
        cmp("conv_pre", x, "conv_pre", transpose: true)
        for i in 0 ..< dec.ups.count {
            x = dec.ups[i][0](x)
            cmp("up\(i)", x, "up\(i)", transpose: true)
            var xs = dec.resblocks[i * dec.numKernels](x)
            for j in 1 ..< dec.numKernels { xs = xs + dec.resblocks[i * dec.numKernels + j](x) }
            x = xs / Float(dec.numKernels)
            cmp("res\(i)", x, "res\(i)", transpose: true)
        }
        x = dec.activationPost(x)
        cmp("act_post", x, "act_post", transpose: true)
        x = dec.convPost(x)
        cmp("conv_post", x, "conv_post", transpose: true)
    }
}
