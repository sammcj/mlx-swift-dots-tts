import Foundation
import MLX
import MLXNN

/// Flow-matching Euler solver with classifier-free guidance.
///
/// Swift port of core.py `_flow_matching_step_fm` / `fm_solver_step`. The DiT
/// denoises the last `patchSize` latent slots of the FM sequence while attending
/// over the whole sequence. Each Euler step projects the current latent `z`
/// through `coordinateProj` (128 -> 1024), writes it into the trailing slots of
/// both the conditional and unconditional prompt sequences, runs the DiT on the
/// stacked CFG batch, slices the trailing velocity, and combines:
///   v = v_cond + guidance * (v_cond - v_uncond)
/// then integrates `z += dt * v` with dt = 1 / numSteps over t = n*dt.
public final class EulerSolver: Module {
    @ModuleInfo(key: "coordinate_proj") var coordinateProj: Linear

    let dit: DiT
    let patchSize: Int
    let latentDim: Int

    public init(dit: DiT, hidden: Int = 1024, latentDim: Int = 128, patchSize: Int = 4) {
        self.dit = dit
        self.patchSize = patchSize
        self.latentDim = latentDim
        self._coordinateProj.wrappedValue = Linear(latentDim, hidden, bias: true)
        super.init()
    }

    /// One CFG-guided velocity evaluation at scalar time `t` for latent `z`.
    /// - inputSeq: (1, L, hidden) conditional prompt sequence
    /// - cfgSeq:   (1, L, hidden) unconditional prompt sequence
    /// - gCond:    (1, hidden) global conditioning (uncond branch zeroed)
    /// - z:        (1, patchSize, latentDim)
    /// Returns guided velocity (1, patchSize, latentDim).
    private func solverStep(
        t: MLXArray, z: MLXArray,
        inputSeq: MLXArray, cfgSeq: MLXArray, gCond: MLXArray, mask: MLXArray?
    ) -> MLXArray {
        let L = inputSeq.dim(1)
        let latentStart = L - patchSize
        let zc = coordinateProj(z)  // (1, patchSize, hidden)

        let zCond = concatenated([inputSeq[0..., 0 ..< latentStart], zc], axis: 1)
        let zUnc = concatenated([cfgSeq[0..., 0 ..< latentStart], zc], axis: 1)
        let zz = concatenated([zCond, zUnc], axis: 0)  // (2, L, hidden)

        let gg = concatenated([gCond, MLXArray.zeros(like: gCond)], axis: 0)  // (2, hidden)
        let tt = broadcast(t.reshaped(1), to: [2])

        var vt = dit(zz, timesteps: tt, gCond: gg, mask: mask)  // (2, L, latentDim)
        vt = vt[0..., latentStart...]               // (2, patchSize, latentDim)
        let vCond = vt[0 ..< 1]
        let vUnc = vt[1 ..< 2]
        return vCond + guidanceScale * (vCond - vUnc)
    }

    private var guidanceScale: Float = 3.0

    /// Integrate the FM ODE from noise to a clean latent.
    /// - noise:    (1, patchSize, latentDim) initial sample
    /// Returns the integrated latent (1, patchSize, latentDim).
    /// - mask: optional additive attention bias (0 keep / -inf drop) broadcastable
    ///   to (2, numHeads, L, L). When nil the DiT uses full attention. The mask is
    ///   constant across Euler steps, so the caller builds it once.
    public func solve(
        noise: MLXArray, inputSeq: MLXArray, cfgSeq: MLXArray, gCond: MLXArray,
        numSteps: Int = 10, guidance: Float = 3.0, mask: MLXArray? = nil
    ) -> MLXArray {
        guidanceScale = guidance
        var z = noise
        let dt = 1.0 / Float(numSteps)
        for n in 0 ..< numSteps {
            let t = MLXArray(Float(n) * dt)
            z = z + dt * solverStep(t: t, z: z, inputSeq: inputSeq, cfgSeq: cfgSeq, gCond: gCond, mask: mask)
            eval(z)
        }
        return z
    }

    /// MeanFlow (few-step distilled) integrator. The student predicts the average
    /// velocity over the interval, so there is no CFG (no cond/uncond branch) and
    /// the DiT takes the step interval `dt` as a second time input. Schedule is
    /// linspace(0, 1, nfe+1); for uniform steps t = k/nfe and dt = 1/nfe, with
    /// z += v * dt. Single batch (the prompt's conditional sequence only).
    /// Mirrors core.py `_meanflow_step_fm` / `meanflow_solver_step`.
    public func solveMeanFlow(
        noise: MLXArray, inputSeq: MLXArray, gCond: MLXArray, nfe: Int = 4, mask: MLXArray? = nil
    ) -> MLXArray {
        let L = inputSeq.dim(1)
        let latentStart = L - patchSize
        var z = noise
        let inv = 1.0 / Float(nfe)
        for k in 0 ..< nfe {
            let t = MLXArray(Float(k) * inv).reshaped(1)
            let dt = MLXArray(inv).reshaped(1)
            let zc = coordinateProj(z)  // (1, patchSize, hidden)
            let seq = concatenated([inputSeq[0..., 0 ..< latentStart], zc], axis: 1)  // (1, L, hidden)
            var vt = dit(seq, timesteps: t, gCond: gCond, mask: mask, duration: dt)
            vt = vt[0..., latentStart...]  // (1, patchSize, latentDim)
            z = z + vt * inv
            eval(z)
        }
        return z
    }
}
