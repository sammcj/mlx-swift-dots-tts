import Foundation
import MLX
import MLXNN
import XCTest
@testable import DotsTTS

/// Parity: the Swift FM Euler+CFG solver must match the torch reference
/// (`solver_reference.safetensors`) integrating the real fork DiT + coordinate_proj.
/// Full attention / arange positions, fixed seed, 10 steps, guidance 3.0.
final class SolverParityTests: XCTestCase {
    func testSolverMatchesTorchReference() throws {
        let env = ProcessInfo.processInfo.environment
        let repo = env["DOTS_MODEL_REPO"] ?? "/Users/samm/git/sammcj/dots.tts-soar-mlx"
        let fixtures = env["DOTS_FIXTURES"] ?? "/Users/samm/git/dots-mlx-spike"
        let ditDir = URL(fileURLWithPath: repo).appendingPathComponent("dit")
        let headsURL = URL(fileURLWithPath: repo).appendingPathComponent("heads/model.safetensors")
        let refURL = URL(fileURLWithPath: fixtures).appendingPathComponent("solver_reference.safetensors")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: ditDir.appendingPathComponent("model.safetensors").path)
                && FileManager.default.fileExists(atPath: headsURL.path)
                && FileManager.default.fileExists(atPath: refURL.path),
            "dit weights, heads, or solver reference not present")

        let dit = DiT()
        try WeightLoading.load(dit, from: ditDir)

        let solver = EulerSolver(dit: dit)
        let heads = try MLX.loadArrays(url: headsURL)
        var coord: [String: MLXArray] = [:]
        for (k, v) in heads where k.hasPrefix("coordinate_proj.") {
            coord[String(k.dropFirst("coordinate_proj.".count))] = v
        }
        try solver.coordinateProj.update(parameters: ModuleParameters.unflattened(coord), verify: .all)
        eval(solver)

        let ref = try MLX.loadArrays(url: refURL)
        let steps = ref["num_steps"]!.item(Int.self)
        let guidance = ref["guidance"]!.item(Float.self)
        let out = solver.solve(
            noise: ref["noise"]!.asType(.float32),
            inputSeq: ref["input_seq"]!.asType(.float32),
            cfgSeq: ref["cfg_seq"]!.asType(.float32),
            gCond: ref["g_cond"]!.asType(.float32),
            numSteps: steps, guidance: guidance)
        eval(out)

        let expected = ref["out_latent"]!.asType(.float32)
        let diff = abs(out - expected)
        let maxAbs = diff.max().item(Float.self)
        let rel = (sqrt((diff * diff).sum()) / sqrt((expected * expected).sum())).item(Float.self)
        let cos = ((out * expected).sum()
            / (sqrt((out * out).sum()) * sqrt((expected * expected).sum()))).item(Float.self)
        print("[solver parity] rel=\(rel) maxAbs=\(maxAbs) cos=\(cos)")
        XCTAssertLessThan(rel, 0.05, "solver rel error \(rel) exceeds 0.05")
        XCTAssertGreaterThan(cos, 0.999, "solver cosine \(cos) below 0.999")
    }
}
