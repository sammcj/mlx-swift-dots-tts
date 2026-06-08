import Foundation
import MLX
import MLXNN
import MLXRandom
import Tokenizers

/// End-to-end dots.tts inference in MLX, mirroring DotsTtsModel/_generate_latents_stream.
///
/// Voice cloning (ICL): the reference clip drives BOTH the CAM++ speaker
/// embedding (g_cond into the DiT) AND, with a transcript, the prefill that
/// seeds the AR/FM history with reference patches. The decode loop interleaves a
/// KV-cached Qwen2 step with a masked flow-matching solve per latent patch, the
/// patch encoder re-runs over the causal latent history to produce each LLM
/// input embedding (prefix-stable, so equivalent to the streaming decode_patch),
/// and the EOS head stops generation. Generated latents are denormalised and the
/// BigVGAN/AudioVAE decoder renders 48 kHz audio.
///
/// See the verified algorithm notes for the FM buffer layout and CFG asymmetry.
///
/// `@unchecked Sendable`: the pipeline holds MLX model state and mutable debug
/// hooks, but it is meant to be owned by a single serialising actor (Cloney's
/// `DotsSwiftSynthesiser`), which never calls it concurrently. The debug-inject
/// properties are test-only and unused on the actor path.
/// Opt-in per-stage wall-clock profiler (env DOTS_PROFILE_STAGES=1). Accumulates
/// time per labelled stage across the decode loop so we can see where wall-clock
/// actually goes (DiT solve vs patch encoder vs backbone vs vocoder). Off by
/// default; when on it forces an `eval` at each tagged stage to attribute time.
final class StageProfiler {
    private var totals: [String: Double] = [:]
    private var counts: [String: Int] = [:]
    private var order: [String] = []

    func add(_ label: String, _ seconds: Double) {
        if totals[label] == nil { order.append(label) }
        totals[label, default: 0] += seconds
        counts[label, default: 0] += 1
    }

    func report() -> String {
        let grand = totals.values.reduce(0, +)
        var lines = ["[dots-profile] per-stage wall-clock (tagged total \(String(format: "%.2f", grand))s):"]
        for label in order.sorted(by: { (totals[$0] ?? 0) > (totals[$1] ?? 0) }) {
            let t = totals[label] ?? 0
            let n = counts[label] ?? 0
            let pct = grand > 0 ? 100 * t / grand : 0
            lines.append(String(format: "  %-16@ %7.2fs  %5.1f%%  (%d calls, %.1fms/call)",
                                label as NSString, t, pct, n, n > 0 ? 1000 * t / Double(n) : 0))
        }
        return lines.joined(separator: "\n")
    }
}

public final class DotsTTSPipeline: @unchecked Sendable {
    public struct Params: Sendable {
        public var numSteps = 10
        public var guidance: Float = 1.2          // runtime default (NOT core.py's 3.0)
        public var speakerScale: Float = 1.5
        /// Fixed-step ODE solver for the flow-matching path. Ignored by MeanFlow.
        public var odeMethod: ODEMethod = .euler
        public var eosThreshold: Float = 0.8
        public var maxOutputPatches = 600
        public var seed: UInt64 = 0
        public init() {}
    }

    let backbone: Qwen2Backbone
    let dit: DiT
    let solver: EulerSolver
    let vocoder: Vocoder
    let speaker: CAMPPlus
    let fbank: KaldiFbank
    let audioVAE: AudioVAEEncoder
    let patchEncoder: PatchEncoder
    let resampler: Resampler
    let tokenizer: Tokenizer
    let special: DotsSpecialTokens
    /// Opt-in per-stage profiler (env DOTS_PROFILE_STAGES=1); nil = no extra syncs.
    let profiler: StageProfiler?
    /// True when the DiT is a MeanFlow-distilled checkpoint (dit/config.json
    /// `mode == "meanflow"`): few-step, no CFG, duration-conditioned solve.
    let meanflow: Bool

    /// Debug-only: when set, decodeNextPatch uses these injected noises (indexed
    /// by decode step) instead of MLXRandom.normal, and each solved patch latent
    /// is recorded in `debugCapturedZ`. Lets the decode loop be replayed against
    /// Python's trajectory with identical noise to localise feedback drift.
    public var debugInjectNoise: [MLXArray]? = nil
    public var debugCapturedZ: [MLXArray] = []
    public var debugFirstInputSeq: MLXArray? = nil
    public var debugFirstCfgSeq: MLXArray? = nil
    public var debugFirstGCond: MLXArray? = nil
    /// Debug-only: inject Python's exact stochastic conditioning to remove RNG
    /// divergence (prompt-latent VAE sample, speaker random-crop) from decode
    /// parity. debugInjectPromptLatents is the unnorm, last-patch-trimmed
    /// reference latents (Python `prompt_latents_sampled`).
    public var debugInjectPromptLatents: MLXArray? = nil
    public var debugInjectGCond: MLXArray? = nil
    public var debugCapturedEmbed: [MLXArray] = []
    public var debugCapturedHidden: [MLXArray] = []
    /// When set, replaces the patch-encoder embedding fed to the LLM decode step
    /// with Python's exact embedding, isolating the LLM/cache path from the patch
    /// encoder. The FM latent feedback still uses Swift's own z.
    public var debugInjectEmbed: [MLXArray]? = nil
    public var debugFullPrefillHidden: MLXArray? = nil

    // projection heads (raw weights; small, kept as arrays)
    let hiddenProjW, hiddenProjB: MLXArray
    let latentProjW, latentProjB: MLXArray
    let xvecLinW, xvecLinB, xvecLnW, xvecLnB: MLXArray
    let eos0W, eos0B, eos2W, eos2B: MLXArray
    let latentMean, latentStd: MLXArray

    let latentDim = 128
    let patchSize = 4
    let hiddenPatchSize = 1
    let hopSize = 1920
    let maxSpeakerSeconds = 10.0   // SpeakerXVectorFeatures.max_audio_seconds
    let speakerSourceRate = 48000

    public init(modelRepo: URL, tokenizer: Tokenizer) throws {
        struct CfgFile: Codable { let quantization: QuantizationSettings.Config? }
        // Per-component quantisation: read the component's config.json `quantization`
        // block (mlx_lm format). Absent block -> fp32. Lets backbone / DiT /
        // patch_encoder each ship at their own precision in a given model repo.
        func quantOf(_ dir: URL) -> QuantizationSettings {
            guard let data = try? Data(contentsOf: dir.appendingPathComponent("config.json")),
                  let cfg = try? JSONDecoder().decode(CfgFile.self, from: data) else { return .none }
            return QuantizationSettings(from: cfg.quantization)
        }

        let backboneDir = modelRepo.appendingPathComponent("backbone")
        let q = quantOf(backboneDir)
        let bb = Qwen2Backbone()
        if q.enabled { quantize(model: bb, groupSize: q.groupSize, bits: q.bits) }
        try WeightLoading.load(bb, from: backboneDir)
        self.backbone = bb

        // Read dit/config.json `mode`: "meanflow" selects the distilled few-step
        // solver and a DiT carrying the extra duration_embedder weights.
        func modeOf(_ dir: URL) -> String {
            guard let data = try? Data(contentsOf: dir.appendingPathComponent("config.json")),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let mode = obj["mode"] as? String else { return "flow_matching" }
            return mode
        }

        let ditDir = modelRepo.appendingPathComponent("dit")
        let isMeanflow = modeOf(ditDir) == "meanflow"
        self.meanflow = isMeanflow
        // DiT nests Linears inside [UnaryLayer] arrays (adaLN_modulation,
        // time_embedder.mlp) that quantize(model:) cannot reach, so build them
        // quantised at construction via the factory instead.
        var ditCfg = DiT.Config()
        ditCfg.meanflow = isMeanflow
        let dit = DiT(ditCfg, quant: quantOf(ditDir))
        try WeightLoading.load(dit, from: ditDir)
        self.dit = dit
        self.solver = EulerSolver(dit: dit)

        let voc = Vocoder()
        try WeightLoading.load(voc, from: modelRepo.appendingPathComponent("vocoder"))
        self.vocoder = voc

        let spk = CAMPPlus()
        try WeightLoading.load(spk, from: modelRepo.appendingPathComponent("speaker"))
        self.speaker = spk
        self.fbank = KaldiFbank()

        let vae = AudioVAEEncoder()
        try WeightLoading.load(vae, from: modelRepo.appendingPathComponent("audiovae_encoder"))
        self.audioVAE = vae

        let peDir = modelRepo.appendingPathComponent("patch_encoder")
        let pe = PatchEncoder(quant: quantOf(peDir))
        try WeightLoading.load(pe, from: peDir)
        self.patchEncoder = pe

        let heads = try MLX.loadArrays(url: modelRepo.appendingPathComponent("heads/model.safetensors"))
        func h(_ k: String) throws -> MLXArray {
            guard let v = heads[k] else { throw DotsTextError.missingSpecialToken(k) }
            return v.asType(.float32)
        }
        self.hiddenProjW = try h("hidden_proj.weight"); self.hiddenProjB = try h("hidden_proj.bias")
        self.latentProjW = try h("latent_proj.weight"); self.latentProjB = try h("latent_proj.bias")
        self.xvecLinW = try h("xvec_proj.0.weight"); self.xvecLinB = try h("xvec_proj.0.bias")
        self.xvecLnW = try h("xvec_proj.1.weight"); self.xvecLnB = try h("xvec_proj.1.bias")
        self.eos0W = try h("eos_proj.0.weight"); self.eos0B = try h("eos_proj.0.bias")
        self.eos2W = try h("eos_proj.2.weight"); self.eos2B = try h("eos_proj.2.bias")
        // coordinate_proj into the solver.
        var coord: [String: MLXArray] = [:]
        for (k, v) in heads where k.hasPrefix("coordinate_proj.") {
            coord[String(k.dropFirst("coordinate_proj.".count))] = v.asType(.float32)
        }
        try solver.coordinateProj.update(parameters: ModuleParameters.unflattened(coord), verify: .all)

        let stats = try JSONDecoder().decode(
            LatentStats.self, from: Data(contentsOf: modelRepo.appendingPathComponent("latent_stats.json")))
        self.latentMean = MLXArray(stats.mean)
        self.latentStd = sqrt(MLXArray(stats.var))

        let rs = try MLX.loadArrays(url: modelRepo.appendingPathComponent("resampler_48k_16k.safetensors"))
        self.resampler = Resampler(
            origRate: rs["orig"]!.item(Int.self), newRate: rs["new"]!.item(Int.self),
            gcd: rs["gcd"]!.item(Int.self), width: rs["width"]!.item(Int.self),
            torchKernel: rs["kernel"]!.asType(.float32))

        self.tokenizer = tokenizer
        self.special = try DotsSpecialTokens(tokenizer: tokenizer)
        self.profiler = ProcessInfo.processInfo.environment["DOTS_PROFILE_STAGES"] == "1" ? StageProfiler() : nil
        eval(bb, dit, voc, spk, vae, pe, solver)
    }

    struct LatentStats: Codable { let mean: [Float]; let `var`: [Float] }

    /// Tag a stage for profiling. When the profiler is on, force-eval `a` and
    /// record the elapsed time under `label`. When off (the normal path), return
    /// `a` untouched - no eval, so the decode loop only syncs where it must (the
    /// per-patch EOS `.item()` read), instead of eagerly evaluating every stage.
    @discardableResult
    private func timedEval(_ label: String, _ a: MLXArray) -> MLXArray {
        guard let profiler else { return a }
        let t0 = DispatchTime.now().uptimeNanoseconds
        eval(a)
        profiler.add(label, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
        return a
    }

    /// Time a stage that already forces its own eval internally (e.g. the FM
    /// solver evals each Euler step), so the elapsed call time is the real cost.
    private func timed<T>(_ label: String, _ body: () -> T) -> T {
        guard let profiler else { return body() }
        let t0 = DispatchTime.now().uptimeNanoseconds
        let r = body()
        profiler.add(label, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
        return r
    }

    // MARK: projection helpers (channels-last x @ W^T + b)
    private func linear(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray { matmul(x, w.T) + b }
    private func hiddenProj(_ x: MLXArray) -> MLXArray { linear(x, hiddenProjW, hiddenProjB) }
    private func latentProj(_ x: MLXArray) -> MLXArray { linear(x, latentProjW, latentProjB) }
    private func xvecProj(_ x: MLXArray) -> MLXArray {
        let h = linear(x, xvecLinW, xvecLinB)
        let mu = h.mean(axis: -1, keepDims: true)
        let centered = h - mu
        let v = (centered * centered).mean(axis: -1, keepDims: true)
        return centered * rsqrt(v + 1e-5) * xvecLnW + xvecLnB
    }
    private func normalize(_ x: MLXArray) -> MLXArray { (x - latentMean) / latentStd }
    private func denormalize(_ x: MLXArray) -> MLXArray { x * latentStd + latentMean }
    private func sampleFromLatent(_ meanLogstd: MLXArray) -> MLXArray {
        // meanLogstd: (1, 256, L). chunk on channel dim, sample, -> (1, L, 128).
        let parts = split(meanLogstd, parts: 2, axis: 1)
        let mean = parts[0], logStd = parts[1]
        let z = mean + MLXRandom.normal(mean.shape) * exp(logStd)
        return z.transposed(0, 2, 1)
    }

    /// Test hook: g_cond from a raw 48k clip, padding exactly as generate() does.
    public func debugSpeakerCond(refAudio48k: MLXArray, scale: Float = 1.5) -> MLXArray {
        speakerCond(refAudio48k: padTo(refAudio48k, multiple: patchSize * hopSize), scale: scale)
    }

    /// Test hook: speaker sub-stages for localising glue bugs.
    public func debugSpeakerStages(refAudio48k: MLXArray, scale: Float = 1.5) -> [String: MLXArray] {
        var w = padTo(refAudio48k, multiple: patchSize * hopSize)
        let maxSamples = Int((maxSpeakerSeconds * Double(speakerSourceRate)).rounded())
        if w.dim(0) > maxSamples { w = w[0 ..< maxSamples] }
        let mono16k = resampler(w)
        let fb = fbank(mono16k)
        let xvec = speaker(fb.expandedDimensions(axis: 0)).reshaped(1, 512) * scale
        let g = xvecProj(xvec)
        return ["resampled_16k": mono16k, "fbank": fb, "xvector_scaled": xvec, "g_cond": g]
    }

    /// Speaker conditioning g_cond (1, 1024) from a 48 kHz mono reference clip.
    /// The reference is cropped to `maxSpeakerSeconds` (the encoder's training
    /// window; Python uses a random start, we use start 0 for determinism), then
    /// resampled to 16 kHz, fbank'd, and run through CAM++ + xvec_proj.
    public func speakerCond(refAudio48k: MLXArray, scale: Float) -> MLXArray {
        var w = refAudio48k
        if w.ndim == 2 { w = w.reshaped(w.dim(w.ndim - 1)) }
        let maxSamples = Int((maxSpeakerSeconds * Double(speakerSourceRate)).rounded())
        if w.dim(0) > maxSamples { w = w[0 ..< maxSamples] }
        let mono16k = resampler(w)
        let fb = fbank(mono16k).expandedDimensions(axis: 0)   // (1, T, 80)
        let xvec = speaker(fb).reshaped(1, 512) * scale       // (1, 512)
        return xvecProj(xvec)                                  // (1, 1024)
    }

    // MARK: FM history buffers (interleaved hidden(1)/latent(patchSize))
    private final class State {
        var cond: MLXArray? = nil       // (1, T, 1024)
        var uncond: MLXArray? = nil     // (1, T, 1024)
        var len = 0
        var unnormLatents: MLXArray? = nil  // (1, T*4, 128) for patch-encoder recompute
        var llmCache: [KVCache] = []
        var llmHidden: MLXArray? = nil  // (1, 1, 1536)
    }

    private func append(_ s: State, cond: MLXArray, uncond: MLXArray) {
        s.cond = s.cond.map { concatenated([$0, cond], axis: 1) } ?? cond
        s.uncond = s.uncond.map { concatenated([$0, uncond], axis: 1) } ?? uncond
        s.len += cond.dim(1)
    }
    private func appendHidden(_ s: State, _ hidden: MLXArray) {
        let proj = hiddenProj(hidden)
        append(s, cond: proj, uncond: hiddenProj(MLXArray.zeros(like: hidden)))
    }
    private func appendLatent(_ s: State, _ latent: MLXArray) {
        let proj = latentProj(latent)  // same in both branches
        append(s, cond: proj, uncond: proj)
    }
    private func appendUnnormLatent(_ s: State, _ unnorm: MLXArray) {
        s.unnormLatents = s.unnormLatents.map { concatenated([$0, unnorm], axis: 1) } ?? unnorm
    }

    /// Structured FM-decode attention mask -> additive (1,1,total,total).
    private func fmMask(len: Int, total: Int, dtype: DType) -> MLXArray {
        let latentStart = Int32(total - patchSize)
        let blockStart = Int32(len - hiddenPatchSize)
        let lenA = Int32(len)
        let idx = MLXArray(0 ..< Int32(total))
        let rows = idx.reshaped(total, 1), cols = idx.reshaped(1, total)
        // context = keys in [0..len) (history prefix) OR [latentStart..) (latent block)
        let context = (cols .< lenA) .|| (cols .>= latentStart)
        // last hidden block rows [blockStart..len) and latent rows [latentStart..)
        // both attend to the full context.
        let hidRows = (rows .>= blockStart) .&& (rows .< lenA)
        let latRows = rows .>= latentStart
        var keep = (hidRows .|| latRows) .&& context
        // history-prefix rows [0..blockStart) are causal among themselves.
        let causal = (cols .<= rows) .&& (rows .< blockStart)
        keep = keep .|| causal
        let additive = MLX.where(keep, MLXArray(Float(0), dtype: dtype), MLXArray(-Float.infinity, dtype: dtype))
        return additive.reshaped(1, 1, total, total)
    }

    /// Test hook: run the loaded FM solver on EXTERNALLY supplied conditioning
    /// (Python's real decode-step buffers) with an INJECTED fixed noise, so the
    /// solver+DiT can be compared bit-for-bit against the torch reference with no
    /// RNG divergence. `mask` is the additive (1,1,L,L) bias (0 keep / -inf drop).
    public func debugSolveStep(inputSeq: MLXArray, cfgSeq: MLXArray, gCond: MLXArray,
                               noise: MLXArray, mask: MLXArray, numSteps: Int, guidance: Float) -> MLXArray {
        solver.solve(noise: noise, inputSeq: inputSeq, cfgSeq: cfgSeq, gCond: gCond,
                     numSteps: numSteps, guidance: guidance, mask: mask)
    }

    /// Solve one latent patch from the current FM history.
    private func decodeNextPatch(_ s: State, gCond: MLXArray, p: Params) -> MLXArray {
        let total = s.len + patchSize
        let pad = MLXArray.zeros([1, patchSize, 1024])
        let inputSeq = concatenated([s.cond!, pad], axis: 1)
        let cfgSeq = concatenated([s.uncond!, pad], axis: 1)
        if debugInjectNoise != nil && debugCapturedZ.isEmpty && debugFirstInputSeq == nil {
            debugFirstInputSeq = inputSeq
            debugFirstCfgSeq = cfgSeq
            debugFirstGCond = gCond
        }
        let mask = fmMask(len: s.len, total: total, dtype: inputSeq.dtype)
        let noise: MLXArray
        if let inj = debugInjectNoise {
            noise = inj[min(debugCapturedZ.count, inj.count - 1)]
        } else {
            noise = MLXRandom.normal([1, patchSize, latentDim])
        }
        if meanflow {
            // Few-step, no CFG; p.numSteps is the NFE (published default 4).
            return solver.solveMeanFlow(noise: noise, inputSeq: inputSeq, gCond: gCond,
                                        nfe: p.numSteps, mask: mask)
        }
        return solver.solve(noise: noise, inputSeq: inputSeq, cfgSeq: cfgSeq, gCond: gCond,
                            numSteps: p.numSteps, guidance: p.guidance, method: p.odeMethod,
                            mask: mask)
    }

    /// EOS stop probability: softmax(eos_proj(hidden))[...,1].
    private func eosProb(_ hidden: MLXArray) -> Float {
        let h = silu(linear(hidden, eos0W, eos0B))
        let logits = linear(h, eos2W, eos2B)               // (1,1,2)
        return softmax(logits, axis: -1)[0, 0, 1].item(Float.self)
    }

    /// New LLM input embedding for the just-generated patch: re-run the patch
    /// encoder over the whole (causal) unnorm latent history, take the last token.
    private func patchEmbedding(_ s: State) -> MLXArray {
        let emb = patchEncoder(s.unnormLatents!)            // (1, K, 1536)
        return emb[0..., (emb.dim(1) - 1) ..< emb.dim(1)]   // (1, 1, 1536)
    }

    /// Test hook: run the prefill with EXTERNALLY supplied (unnorm, trimmed)
    /// prompt latents so the glue is comparable to Python bit-for-bit (bypasses
    /// the stochastic VAE sample). Returns promptEmbed (ppe), the FM cond/uncond
    /// buffers after prefill, and the last prefill llm hidden.
    public func debugPrefill(refLatentsTrim: MLXArray, targetText: String, refTranscript: String,
                             maxOutputPatches: Int = 200) -> [String: MLXArray] {
        let pc = refLatentsTrim.dim(1) / patchSize
        let promptPatches = normalize(refLatentsTrim).reshaped(1, pc, patchSize, latentDim)
        let maxAudio = pc + maxOutputPatches
        let schedule = DotsTemplate.generationSchedule(
            promptText: refTranscript, targetText: targetText, maxAudioTokens: maxAudio,
            tokenizer: tokenizer, special: special)
        let promptEmbed = patchEncoder(refLatentsTrim)
        let spans = schedule.enumerated().filter { $0.element == special.audioGenSpan }.map { $0.offset }
        let prefillEnd = spans[pc]
        let s = State()
        s.llmCache = backbone.makeCache()
        s.unnormLatents = refLatentsTrim
        let ids = MLXArray(schedule[0 ..< prefillEnd].map { Int32($0) }).reshaped(1, prefillEnd)
        let tokEmbeds = backbone.embed(ids)
        let p0 = spans[0]
        let embeds = concatenated([tokEmbeds[0..., 0 ..< p0, 0...], promptEmbed], axis: 1)
        let prefillHidden = backbone.step(embeds: embeds, cache: s.llmCache)
        eval(prefillHidden)
        var cursor = 0
        for i in 0 ..< pc {
            let sp = spans[i]
            if sp > cursor { appendHidden(s, prefillHidden[0..., (sp - 1) ..< sp, 0...]) }
            appendLatent(s, promptPatches[0..., i])
            appendHidden(s, prefillHidden[0..., sp ..< (sp + 1), 0...])
            cursor = sp + 1
        }
        return [
            "prompt_patch_embeddings": promptEmbed,
            "ppe_via_stages": patchEncoder.debugStages(refLatentsTrim)["final"]!,
            "fm_sequence": s.cond!,
            "fm_cfg_sequence": s.uncond!,
            "llm_hidden_last": prefillHidden[0..., (prefillEnd - 1) ..< prefillEnd, 0...],
        ]
    }

    /// Generate a 48 kHz waveform (1, 1, samples). Voice cloning needs a
    /// reference clip + its transcript; targetText is what to speak.
    public func generate(targetText: String, refAudio48k: MLXArray, refTranscript: String, params: Params = Params()) -> MLXArray {
        MLXRandom.seed(params.seed)
        // Python pads the prompt audio before BOTH the speaker encoder and the VAE.
        let pad = padTo(refAudio48k, multiple: patchSize * hopSize)
        let gCond = timedEval("speaker", debugInjectGCond ?? speakerCond(refAudio48k: pad, scale: params.speakerScale))

        // reference -> sampled latents (unnorm), trim last patch, normalised patches.
        let refLatentsTrim: MLXArray
        if let inj = debugInjectPromptLatents {
            refLatentsTrim = inj
        } else {
            var refLatents = sampleFromLatent(audioVAE(pad))       // (1, L, 128) unnorm
            refLatents = refLatents[0..., 0 ..< (refLatents.dim(1) - patchSize)]
            let pcLocal = refLatents.dim(1) / patchSize
            refLatentsTrim = refLatents[0..., 0 ..< (pcLocal * patchSize)]
        }
        timedEval("audio_vae", refLatentsTrim)
        let pc = refLatentsTrim.dim(1) / patchSize
        let promptPatches = normalize(refLatentsTrim).reshaped(1, pc, patchSize, latentDim)

        // schedule + prefill embeddings (patch encoder over the reference latents).
        let maxAudio = pc + params.maxOutputPatches
        let schedule = DotsTemplate.generationSchedule(
            promptText: refTranscript, targetText: targetText, maxAudioTokens: maxAudio,
            tokenizer: tokenizer, special: special)
        let promptEmbed = timedEval("patch_encoder_prefill", patchEncoder(refLatentsTrim))  // (1, pc, 1536)

        // span positions (audio_gen_span); prefill consumes the first pc.
        let spans = schedule.enumerated().filter { $0.element == special.audioGenSpan }.map { $0.offset }
        let prefillEnd = spans[pc]   // first decode span

        let s = State()
        s.llmCache = backbone.makeCache()
        s.unnormLatents = refLatentsTrim

        // prefill: embeds for schedule[:prefillEnd] with prompt patches injected at spans[0..<pc].
        let ids = MLXArray(schedule[0 ..< prefillEnd].map { Int32($0) }).reshaped(1, prefillEnd)
        let tokEmbeds = backbone.embed(ids)
        // prompt span positions are contiguous (spans[0..<pc]); replace that trailing
        // block with the patch embeddings (prefillEnd == spans[0] + pc).
        let p0 = spans[0]
        let embeds = concatenated([tokEmbeds[0..., 0 ..< p0, 0...], promptEmbed], axis: 1)
        let prefillHidden = backbone.step(embeds: embeds, cache: s.llmCache)  // (1, prefillEnd, 1536)
        timedEval("backbone_prefill", prefillHidden)
        if debugInjectNoise != nil { debugFullPrefillHidden = prefillHidden }
        s.llmHidden = prefillHidden[0..., (prefillEnd - 1) ..< prefillEnd, 0...]

        // build FM history from the reference (mirrors _prefill).
        var cursor = 0
        for i in 0 ..< pc {
            let sp = spans[i]
            if sp > cursor { appendHidden(s, prefillHidden[0..., (sp - 1) ..< sp, 0...]) }
            appendLatent(s, promptPatches[0..., i])  // (1, patchSize, 128)
            appendHidden(s, prefillHidden[0..., sp ..< (sp + 1), 0...])  // next is always a span
            cursor = sp + 1
        }

        // decode loop.
        var outPatches: [MLXArray] = []
        var dropFirst = true   // prompt prefill regenerates the prompt tail
        let totalSpans = spans.count
        let debugEos = ProcessInfo.processInfo.environment["DOTS_DEBUG_EOS"] == "1"
        for step in 0 ..< (totalSpans - pc) {
            let prob = eosProb(s.llmHidden!)
            let stop = prob > params.eosThreshold
            // The FM solver evals each Euler step internally, so z is already
            // realised on return; `timed` just measures the call. No separate
            // eval(z) (it would be a redundant no-op on the normal path).
            let z = timed("dit_solve") { decodeNextPatch(s, gCond: gCond, p: params) }  // (1, patchSize, 128) normalised
            if debugInjectNoise != nil { debugCapturedZ.append(z) }
            // consume: append latent history (normalised), patch-encode, step LLM.
            appendLatent(s, z)
            let unnorm = denormalize(z)
            if debugEos && (step % 5 == 0 || prob > 0.3) {
                let rms = sqrt((unnorm * unnorm).mean()).item(Float.self)
                print("[dec] patch \(step) eos=\(prob) latentRMS=\(rms)")
            }
            appendUnnormLatent(s, unnorm)
            var llmEmbed = timedEval("patch_encoder", patchEmbedding(s))
            if debugInjectNoise != nil { debugCapturedEmbed.append(llmEmbed) }
            if let inj = debugInjectEmbed, debugCapturedHidden.count < inj.count {
                llmEmbed = inj[debugCapturedHidden.count]
            }
            s.llmHidden = timedEval("backbone_step", backbone.step(embeds: llmEmbed, cache: s.llmCache))
            if debugInjectNoise != nil { debugCapturedHidden.append(s.llmHidden!) }
            let isLast = (step == totalSpans - pc - 1)
            if !isLast { appendHidden(s, s.llmHidden!) }
            if dropFirst { dropFirst = false } else { outPatches.append(unnorm) }
            // No eval(s.llmHidden!) here: the next iteration's eosProb(...).item()
            // forces it, so one sync/patch instead of three. On the final patch it
            // isn't needed (we break and consume outPatches).
            if stop { break }
        }

        guard !outPatches.isEmpty else { return MLXArray.zeros([1, 1, 0]) }
        let latents = concatenated(outPatches, axis: 1).transposed(0, 2, 1)  // (1, 128, T)
        // Debug-only: dump the real denormalised vocoder input for the precision
        // bench (a genuine speech latent, unlike the normalised parity fixture).
        if let dumpPath = ProcessInfo.processInfo.environment["DOTS_DUMP_VOCODER_LATENT"] {
            eval(latents)
            try? MLX.save(arrays: ["latent": latents], url: URL(fileURLWithPath: dumpPath))
        }
        // timedEval evals + times wav when profiling; otherwise realise it here.
        let wav = vocoder(latents)
        if let profiler {
            timedEval("vocoder", wav)
            print(profiler.report())
        } else {
            eval(wav)
        }
        return wav
    }

    private func padTo(_ audio: MLXArray, multiple: Int) -> MLXArray {
        var w = audio
        if w.ndim == 2 { w = w.reshaped(w.dim(w.ndim - 1)) }
        let n = w.dim(0)
        let target = Int((Double(n) / Double(multiple)).rounded(.up)) * multiple
        if target > n { w = padded(w, widths: [.init((0, target - n))]) }
        return w
    }
}
