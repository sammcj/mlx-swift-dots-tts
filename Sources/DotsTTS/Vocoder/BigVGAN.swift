import Foundation
import MLX
import MLXNN

// MARK: - Causal conv helpers (channels-last, B, L, C)

/// Causal Conv1d: left-pad input by dilation*(k-1) zeros then conv with pad=0.
/// Weight stored as (out, k, in/groups). Bias optional. Operates on (B, L, C).
final class CausalConv1d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray  // (out, k, in)
    @ParameterInfo(key: "bias") var bias: MLXArray?     // (out,) optional
    let leftPad: Int
    let dilation: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, dilation: Int, bias: Bool) {
        self.leftPad = dilation * (kernelSize - 1)
        self.dilation = dilation
        self._weight.wrappedValue = MLXArray.zeros([outChannels, kernelSize, inChannels])
        self._bias.wrappedValue = bias ? MLXArray.zeros([outChannels]) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, L, C). Pad along L (axis 1) on the left, then conv with pad=0.
        let padded = MLX.padded(x, widths: [.init((0, 0)), .init((leftPad, 0)), .init((0, 0))])
        var y = MLX.conv1d(padded, weight, stride: 1, padding: 0, dilation: dilation)
        if let bias { y = y + bias }
        return y
    }
}

// MARK: - SnakeBeta activation (snake_logscale=true)

/// y = x + (1/(exp(beta)+1e-9)) * sin(exp(alpha)*x)^2, per-channel alpha/beta.
/// Operates on channels-last (B, L, C).
final class SnakeBeta: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray  // (C,)
    @ParameterInfo(key: "beta") var beta: MLXArray    // (C,)

    init(channels: Int) {
        self._alpha.wrappedValue = MLXArray.zeros([channels])
        self._beta.wrappedValue = MLXArray.zeros([channels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = MLX.exp(alpha)  // (C,) broadcasts over (B, L, C)
        let b = MLX.exp(beta)
        let s = MLX.sin(a * x)
        return x + (1.0 / (b + 1e-9)) * (s * s)
    }
}

// MARK: - Anti-alias up/down resample (ratio=2, k=12, causal)

/// Activation1d: downsample( SnakeBeta( upsample(x) ) ), all channels-last.
/// fixed_filter=true  -> shared (1,1,12) filter, broadcast across channels.
/// fixed_filter=false -> per-channel (C,1,12) filters.
final class Activation1d: Module {
    @ModuleInfo(key: "act") var act: SnakeBeta
    @ModuleInfo(key: "upsample") var upsample: UpSample1d
    @ModuleInfo(key: "downsample") var downsample: DownSample1d

    init(channels: Int, fixedFilter: Bool) {
        self._act.wrappedValue = SnakeBeta(channels: channels)
        self._upsample.wrappedValue = UpSample1d(channels: channels, fixedFilter: fixedFilter)
        self._downsample.wrappedValue = DownSample1d(channels: channels, fixedFilter: fixedFilter)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downsample(act(upsample(x)))
    }
}

/// UpSample1d (ratio=2, k=12, causal): ratio * convT(filter, stride=2, groups=C),
/// then right-trim by (k-stride)=10. filter buffer is (C,1,12) or (1,1,12).
final class UpSample1d: Module {
    @ParameterInfo(key: "filter") var filter: MLXArray  // (C,1,12) or (1,1,12)
    let channels: Int
    let fixedFilter: Bool
    let ratio = 2
    let kernel = 12

    init(channels: Int, fixedFilter: Bool) {
        self.channels = channels
        self.fixedFilter = fixedFilter
        let outC = fixedFilter ? 1 : channels
        self._filter.wrappedValue = MLXArray.zeros([outC, 1, kernel])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, L, C). convTransposed1d weight expects (C_out, k, C_in/groups).
        // Depthwise: groups=C, weight (C, k, 1). Saved filter is (C,1,k) -> (C,k,1).
        var w = filter
        if fixedFilter {
            w = MLX.broadcast(w, to: [channels, 1, kernel])  // (C,1,k)
        }
        w = w.transposed(0, 2, 1)  // (C, k, 1)
        var y = ratio * MLX.convTransposed1d(x, w, stride: ratio, padding: 0, groups: channels)
        // right-trim (k - stride) = 10 along L (axis 1).
        let L = y.dim(1)
        y = y[0..., 0 ..< (L - (kernel - ratio)), 0...]
        return y
    }
}

/// DownSample1d (ratio=2, k=12, causal) -> LowPassFilter1d: replicate-pad
/// left by k-1=11, then stride-2 depthwise conv with filter, groups=C.
final class DownSample1d: Module {
    @ModuleInfo(key: "lowpass") var lowpass: LowPassFilter1d

    init(channels: Int, fixedFilter: Bool) {
        self._lowpass.wrappedValue = LowPassFilter1d(channels: channels, fixedFilter: fixedFilter)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { lowpass(x) }
}

final class LowPassFilter1d: Module {
    @ParameterInfo(key: "filter") var filter: MLXArray  // (C,1,12) or (1,1,12)
    let channels: Int
    let fixedFilter: Bool
    let ratio = 2
    let kernel = 12

    init(channels: Int, fixedFilter: Bool) {
        self.channels = channels
        self.fixedFilter = fixedFilter
        let outC = fixedFilter ? 1 : channels
        self._filter.wrappedValue = MLXArray.zeros([outC, 1, kernel])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, L, C). Replicate-pad left by k-1=11 along L.
        let padLeft = kernel - 1
        let edge = x[0..., 0 ..< 1, 0...]  // (B, 1, C)
        let pad = MLX.broadcast(edge, to: [x.dim(0), padLeft, x.dim(2)])
        let padded = MLX.concatenated([pad, x], axis: 1)
        // conv1d weight (C_out, k, C_in/groups); depthwise groups=C, weight (C,k,1).
        var w = filter
        if fixedFilter {
            w = MLX.broadcast(w, to: [channels, 1, kernel])  // (C,1,k)
        }
        w = w.transposed(0, 2, 1)  // (C, k, 1)
        return MLX.conv1d(padded, w, stride: ratio, padding: 0, groups: channels)
    }
}

// MARK: - AMPBlock1 (resblock)

/// AMPBlock1: 3 dilated causal convs (convs1) + 3 dilation-1 causal convs (convs2),
/// interleaved with 6 fixed-filter Activation1d (snakebeta). Residual per pair.
final class AMPBlock1: Module {
    @ModuleInfo(key: "convs1") var convs1: [CausalConv1d]
    @ModuleInfo(key: "convs2") var convs2: [CausalConv1d]
    @ModuleInfo(key: "activations") var activations: [Activation1d]

    init(channels: Int, kernel: Int, dilations: [Int]) {
        self._convs1.wrappedValue = dilations.map {
            CausalConv1d(inChannels: channels, outChannels: channels,
                         kernelSize: kernel, dilation: $0, bias: true)
        }
        self._convs2.wrappedValue = dilations.map { _ in
            CausalConv1d(inChannels: channels, outChannels: channels,
                         kernelSize: kernel, dilation: 1, bias: true)
        }
        self._activations.wrappedValue = (0 ..< 6).map { _ in
            Activation1d(channels: channels, fixedFilter: true)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for i in 0 ..< convs1.count {
            let a1 = activations[2 * i]
            let a2 = activations[2 * i + 1]
            var xt = a1(h)
            xt = convs1[i](xt)
            xt = a2(xt)
            xt = convs2[i](xt)
            h = xt + h
        }
        return h
    }
}

// MARK: - Decoder (BigVGAN)

/// BigVGAN decoder. conv_pre (non-causal), 6 upsample stages each followed by 3
/// resblocks summed/3, activation_post (per-channel filter), conv_post (causal,
/// no bias), clamp to [-1, 1]. Operates channels-last (B, L, C).
public final class BigVGANDecoder: Module {
    @ModuleInfo(key: "conv_pre") var convPre: Conv1d
    @ModuleInfo(key: "ups") var ups: [[CausalConvTranspose1d]]
    @ModuleInfo(key: "resblocks") var resblocks: [AMPBlock1]
    @ModuleInfo(key: "activation_post") var activationPost: Activation1d
    @ModuleInfo(key: "conv_post") var convPost: CausalConv1d

    let numKernels: Int

    public init(_ cfg: Vocoder.Config) {
        let ic = cfg.upsampleInitialChannel
        // conv_pre: Conv1d 128->1536, k=5, NON-causal SAME pad=(5-1)/2=2.
        self._convPre.wrappedValue = Conv1d(
            inputChannels: cfg.latentDim, outputChannels: ic,
            kernelSize: 5, stride: 1, padding: 2, bias: true)

        self._ups.wrappedValue = (0 ..< cfg.upsampleRates.count).map { i in
            let inC = ic / (1 << i)
            let outC = ic / (1 << (i + 1))
            return [CausalConvTranspose1d(
                inChannels: inC, outChannels: outC,
                kernelSize: cfg.upsampleKernelSizes[i], stride: cfg.upsampleRates[i])]
        }

        var blocks: [AMPBlock1] = []
        for i in 0 ..< cfg.upsampleRates.count {
            let ch = ic / (1 << (i + 1))
            for j in 0 ..< cfg.resblockKernelSizes.count {
                blocks.append(AMPBlock1(
                    channels: ch, kernel: cfg.resblockKernelSizes[j],
                    dilations: cfg.resblockDilations[j]))
            }
        }
        self._resblocks.wrappedValue = blocks
        self.numKernels = cfg.resblockKernelSizes.count

        let finalC = ic / (1 << cfg.upsampleRates.count)  // 24
        self._activationPost.wrappedValue = Activation1d(channels: finalC, fixedFilter: false)
        self._convPost.wrappedValue = CausalConv1d(
            inChannels: finalC, outChannels: 1, kernelSize: 7, dilation: 1, bias: false)
        super.init()
    }

    /// z: (B, T, 128) channels-last. Returns (B, T*1920, 1).
    public func callAsFunction(_ z: MLXArray) -> MLXArray {
        var x = convPre(z)
        for i in 0 ..< ups.count {
            x = ups[i][0](x)
            var xs = resblocks[i * numKernels](x)
            for j in 1 ..< numKernels {
                xs = xs + resblocks[i * numKernels + j](x)
            }
            x = xs / Float(numKernels)
        }
        x = activationPost(x)
        x = convPost(x)
        return MLX.clip(x, min: -1.0, max: 1.0)
    }
}

/// Causal ConvTranspose1d: nn.ConvTranspose1d then right-trim by `stride`.
/// k == 2*stride, padding=0. Weight stored as (out, k, in/groups). (B, L, C).
final class CausalConvTranspose1d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray  // (out, k, in)
    @ParameterInfo(key: "bias") var bias: MLXArray      // (out,)
    let stride: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int) {
        self.stride = stride
        // mlx convTransposed1d weight: (C_out, k, C_in). Converter produced exactly this.
        self._weight.wrappedValue = MLXArray.zeros([outChannels, kernelSize, inChannels])
        self._bias.wrappedValue = MLXArray.zeros([outChannels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = MLX.convTransposed1d(x, weight, stride: stride, padding: 0)
        y = y + bias  // broadcast over (B, L, C_out)
        // right-trim by `stride` along L (axis 1).
        let L = y.dim(1)
        y = y[0..., 0 ..< (L - stride), 0...]
        return y
    }
}
