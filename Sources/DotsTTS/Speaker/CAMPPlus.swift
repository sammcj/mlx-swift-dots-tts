import Foundation
import MLX
import MLXNN

/// CAM++ x-vector speaker encoder (3D-Speaker / dots.tts-soar).
///
/// Swift port of `dots_tts/modules/speaker/{campplus,campplus_layers}.py`. Maps an
/// 80-d Kaldi fbank `(B, T, 80)` to a 512-d speaker embedding `(B, 512)`.
///
/// Parameter paths mirror `speaker_encoder_mlx.safetensors` exactly:
/// `head.*` (FCM 2D conv stem) and `xvector.*` (1D TDNN trunk + stats + dense).
/// All convolution weights in the safetensors are already in MLX channels-last
/// layout (the converter transposed them). The whole path runs in float32.
public final class CAMPPlus: Module {
    @ModuleInfo(key: "head") var head: FCM
    @ModuleInfo(key: "xvector") var xvector: XVectorTrunk

    public init(featDim: Int = 80, embeddingSize: Int = 512,
                growthRate: Int = 32, bnSize: Int = 4, initChannels: Int = 128) {
        let headOutChannels = 32 * (featDim / 8)   // 320
        self._head.wrappedValue = FCM(featDim: featDim)
        self._xvector.wrappedValue = XVectorTrunk(
            inChannels: headOutChannels, featDim: featDim, embeddingSize: embeddingSize,
            growthRate: growthRate, bnSize: bnSize, initChannels: initChannels)
        super.init()
    }

    /// x: `(B, T, 80)` fbank. Returns `(B, 512)` x-vector.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // torch: x.permute(0,2,1) -> (B, 80, T). The FCM works on (B, freq, time);
        // in MLX it converts to/from channels-last internally.
        let xf = x.transposed(0, 2, 1)  // (B, 80, T)
        var h = head(xf)                // (B, 320, T)
        h = xvector(h)                  // (B, 512)
        return h
    }
}

// MARK: - Inference BatchNorm (frozen running stats)

/// BatchNorm1d/2d at inference: affine transform with stored running stats.
/// `y = (x - running_mean) / sqrt(running_var + eps) * weight + bias`.
/// For `affine == false` (config_str "batchnorm_") weight=1, bias=0 and only the
/// running buffers exist. Operates on the channel axis of a channels-LAST tensor
/// (last dim = C), which is how the conv outputs are laid out in MLX.
public final class InferenceBatchNorm: Module {
    let eps: Float
    let affine: Bool
    let weight: MLXArray?
    let bias: MLXArray?
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray

    public init(_ channels: Int, eps: Float = 1e-5, affine: Bool = true) {
        self.eps = eps
        self.affine = affine
        if affine {
            self.weight = MLXArray.ones([channels])
            self.bias = MLXArray.zeros([channels])
        } else {
            self.weight = nil
            self.bias = nil
        }
        self._runningMean.wrappedValue = MLXArray.zeros([channels])
        self._runningVar.wrappedValue = MLXArray.ones([channels])
        super.init()
    }

    /// x: channels-last, last dim = C.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = (x - runningMean) * rsqrt(runningVar + eps)
        if let weight { y = y * weight }
        if let bias { y = y + bias }
        return y
    }
}

@inline(__always)
private func bnRelu(_ bn: InferenceBatchNorm, _ x: MLXArray) -> MLXArray {
    relu(bn(x))
}

// MARK: - FCM (2D ResNet stem)

/// FCM head: `(B, 80, T)` -> `(B, 320, T)`. Inside it treats the input as a 2D
/// image with H = freq (80), W = time (T), 1 input channel, and downsamples the
/// freq axis by 8 (80 -> 10), keeping time. The 32 conv channels * 10 freq bins =
/// 320 output channels fed to the 1D trunk.
public final class FCM: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: InferenceBatchNorm
    @ModuleInfo(key: "layer1") var layer1: [BasicResBlock]
    @ModuleInfo(key: "layer2") var layer2: [BasicResBlock]
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "bn2") var bn2: InferenceBatchNorm

    let mChannels: Int
    let outChannels: Int

    public init(block numBlocks: (Int, Int) = (2, 2), mChannels: Int = 32, featDim: Int = 80) {
        self.mChannels = mChannels
        self.outChannels = mChannels * (featDim / 8)
        self._conv1.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: mChannels, kernelSize: 3,
            stride: 1, padding: 1, bias: false)
        self._bn1.wrappedValue = InferenceBatchNorm(mChannels)
        // layer1/layer2: first block stride (2,1) with shortcut conv, rest identity.
        self._layer1.wrappedValue = FCM.makeLayer(numBlocks.0, inPlanes: mChannels, planes: mChannels, stride: 2)
        self._layer2.wrappedValue = FCM.makeLayer(numBlocks.1, inPlanes: mChannels, planes: mChannels, stride: 2)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: mChannels, outputChannels: mChannels, kernelSize: 3,
            stride: IntOrPair((2, 1)), padding: 1, bias: false)
        self._bn2.wrappedValue = InferenceBatchNorm(mChannels)
        super.init()
    }

    private static func makeLayer(_ n: Int, inPlanes: Int, planes: Int, stride: Int) -> [BasicResBlock] {
        var blocks: [BasicResBlock] = []
        var inP = inPlanes
        for i in 0..<n {
            let s = i == 0 ? stride : 1
            blocks.append(BasicResBlock(inPlanes: inP, planes: planes, stride: s))
            inP = planes
        }
        return blocks
    }

    /// x: `(B, 80, T)`. Returns `(B, 320, T)`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), F = x.dim(1), T = x.dim(2)
        // torch: x.unsqueeze(1) -> (B,1,80,T) = (B,C,H,W). MLX Conv2d is NHWC:
        // (B, H=freq, W=time, C=1).
        var h = x.reshaped(B, F, T, 1)            // (B, 80, T, 1)
        h = relu(bn1(conv1(h)))
        for b in layer1 { h = b(h) }
        for b in layer2 { h = b(h) }
        h = relu(bn2(conv2(h)))                    // (B, 10, T, 32) NHWC, H=freq=10
        // torch flattens (B, C, F, T) row-major -> (B, C*F, T), C outer / F inner.
        // h here is (B, F=10, T, C=32); permute to (B, C, F, T) then reshape.
        let Fp = h.dim(1), Tp = h.dim(2), C = h.dim(3)
        h = h.transposed(0, 3, 1, 2)              // (B, C=32, F=10, T)
        h = h.reshaped(B, C * Fp, Tp)             // (B, 320, T)
        return h
    }
}

/// BasicResBlock (expansion 1). conv1 carries the stride (downsamples freq);
/// shortcut is a 1x1 stride conv + BN when stride != 1, else identity.
public final class BasicResBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: InferenceBatchNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "bn2") var bn2: InferenceBatchNorm
    // shortcut.0 = Conv2d, shortcut.1 = BN. Empty when identity.
    @ModuleInfo(key: "shortcut") var shortcut: [Module]

    let hasShortcut: Bool

    public init(inPlanes: Int, planes: Int, stride: Int) {
        self._conv1.wrappedValue = Conv2d(
            inputChannels: inPlanes, outputChannels: planes, kernelSize: 3,
            stride: IntOrPair((stride, 1)), padding: 1, bias: false)
        self._bn1.wrappedValue = InferenceBatchNorm(planes)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: planes, outputChannels: planes, kernelSize: 3,
            stride: 1, padding: 1, bias: false)
        self._bn2.wrappedValue = InferenceBatchNorm(planes)
        if stride != 1 || inPlanes != planes {
            self.hasShortcut = true
            let sc = Conv2d(
                inputChannels: inPlanes, outputChannels: planes, kernelSize: 1,
                stride: IntOrPair((stride, 1)), bias: false)
            self._shortcut.wrappedValue = [sc, InferenceBatchNorm(planes)]
        } else {
            self.hasShortcut = false
            self._shortcut.wrappedValue = []
        }
        super.init()
    }

    /// x: NHWC `(B, H, W, C)`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = relu(bn1(conv1(x)))
        out = bn2(conv2(out))
        let sc: MLXArray
        if hasShortcut {
            let conv = shortcut[0] as! Conv2d
            let bn = shortcut[1] as! InferenceBatchNorm
            sc = bn(conv(x))
        } else {
            sc = x
        }
        return relu(out + sc)
    }
}

// MARK: - 1D conv helper (channels-last)

/// Run a `Conv1d` (channels-last in MLX) over a channel-major `(B, C, T)` tensor,
/// returning `(B, Cout, T')`. The trunk keeps tensors channel-major (matching the
/// torch layout) and converts at each conv boundary.
@inline(__always)
private func conv1dCM(_ conv: Conv1d, _ x: MLXArray) -> MLXArray {
    let y = conv(x.transposed(0, 2, 1))   // (B, T, Cin) -> (B, T', Cout)
    return y.transposed(0, 2, 1)          // (B, Cout, T')
}

// MARK: - xvector trunk

/// The 1D TDNN trunk: tdnn -> block1 -> transit1 -> block2 -> transit2 ->
/// block3 -> transit3 -> out_nonlinear(BN+ReLU) -> stats -> dense. Tensors are
/// kept channel-major `(B, C, T)` to mirror the torch reference; convs convert to
/// channels-last internally.
public final class XVectorTrunk: Module {
    @ModuleInfo(key: "tdnn") var tdnn: TDNNLayer
    @ModuleInfo(key: "block1") var block1: CAMDenseTDNNBlock
    @ModuleInfo(key: "transit1") var transit1: TransitLayer
    @ModuleInfo(key: "block2") var block2: CAMDenseTDNNBlock
    @ModuleInfo(key: "transit2") var transit2: TransitLayer
    @ModuleInfo(key: "block3") var block3: CAMDenseTDNNBlock
    @ModuleInfo(key: "transit3") var transit3: TransitLayer
    @ModuleInfo(key: "out_nonlinear") var outNonlinear: NonlinearBNReLU
    @ModuleInfo(key: "dense") var dense: DenseLayer

    public init(inChannels: Int, featDim: Int, embeddingSize: Int,
                growthRate: Int, bnSize: Int, initChannels: Int) {
        let bnChannels = bnSize * growthRate            // 128
        // tdnn: Conv1d(320->128, k5, s2, dil1, pad2) + BN(128)+ReLU
        self._tdnn.wrappedValue = TDNNLayer(
            inChannels: inChannels, outChannels: initChannels,
            kernelSize: 5, stride: 2, padding: 2, dilation: 1)
        var channels = initChannels                      // 128
        // block1: 12 layers, dil 1
        self._block1.wrappedValue = CAMDenseTDNNBlock(
            numLayers: 12, inChannels: channels, outChannels: growthRate,
            bnChannels: bnChannels, kernelSize: 3, dilation: 1)
        channels = channels + 12 * growthRate            // 512
        self._transit1.wrappedValue = TransitLayer(inChannels: channels, outChannels: channels / 2)
        channels = channels / 2                           // 256
        // block2: 24 layers, dil 2
        self._block2.wrappedValue = CAMDenseTDNNBlock(
            numLayers: 24, inChannels: channels, outChannels: growthRate,
            bnChannels: bnChannels, kernelSize: 3, dilation: 2)
        channels = channels + 24 * growthRate            // 1024
        self._transit2.wrappedValue = TransitLayer(inChannels: channels, outChannels: channels / 2)
        channels = channels / 2                           // 512
        // block3: 16 layers, dil 2
        self._block3.wrappedValue = CAMDenseTDNNBlock(
            numLayers: 16, inChannels: channels, outChannels: growthRate,
            bnChannels: bnChannels, kernelSize: 3, dilation: 2)
        channels = channels + 16 * growthRate            // 1024
        self._transit3.wrappedValue = TransitLayer(inChannels: channels, outChannels: channels / 2)
        channels = channels / 2                           // 512
        self._outNonlinear.wrappedValue = NonlinearBNReLU(channels)
        // dense: stats doubles channels (mean,std) -> 1024 -> 512
        self._dense.wrappedValue = DenseLayer(inChannels: channels * 2, outChannels: embeddingSize)
        super.init()
    }

    /// x: `(B, 320, T)`. Returns `(B, 512)`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = tdnn(x)
        h = block1(h)
        h = transit1(h)
        h = block2(h)
        h = transit2(h)
        h = block3(h)
        h = transit3(h)
        h = outNonlinear(h)
        let stats = statisticsPooling(h)   // (B, 1024)
        return dense(stats)                // (B, 512)
    }
}

/// BN(channels)+ReLU on a channel-major `(B, C, T)` tensor (the `nonlinear` of a
/// "batchnorm-relu" Sequential). The stored key is `<prefix>.batchnorm.*`.
public final class NonlinearBNReLU: Module {
    @ModuleInfo(key: "batchnorm") var batchnorm: InferenceBatchNorm
    public init(_ channels: Int) {
        self._batchnorm.wrappedValue = InferenceBatchNorm(channels)
        super.init()
    }
    /// x: `(B, C, T)`. BN over channel axis (axis 1).
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // InferenceBatchNorm expects channels-last; transpose, apply, transpose back.
        let y = batchnorm(x.transposed(0, 2, 1))
        return relu(y.transposed(0, 2, 1))
    }
}

/// BN(channels, affine=false) only (config_str "batchnorm_"), channel-major.
public final class NonlinearBNOnly: Module {
    @ModuleInfo(key: "batchnorm") var batchnorm: InferenceBatchNorm
    public init(_ channels: Int) {
        self._batchnorm.wrappedValue = InferenceBatchNorm(channels, affine: false)
        super.init()
    }
    /// x: `(B, C, T)`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = batchnorm(x.transposed(0, 2, 1))
        return y.transposed(0, 2, 1)
    }
}

/// TDNNLayer: Conv1d then BN+ReLU. forward = nonlinear(conv(x)) but here the
/// reference applies conv FIRST then nonlinear (post-activation for the entry
/// TDNN, matching `linear` -> `nonlinear` in source).
public final class TDNNLayer: Module {
    @ModuleInfo(key: "linear") var linear: Conv1d
    @ModuleInfo(key: "nonlinear") var nonlinear: NonlinearBNReLU

    public init(inChannels: Int, outChannels: Int, kernelSize: Int,
                stride: Int, padding: Int, dilation: Int) {
        self._linear.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: kernelSize,
            stride: stride, padding: padding, dilation: dilation, bias: false)
        self._nonlinear.wrappedValue = NonlinearBNReLU(outChannels)
        super.init()
    }

    /// x: `(B, Cin, T)`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        nonlinear(conv1dCM(linear, x))
    }
}

/// TransitLayer: pre-activation BN+ReLU, then a 1x1 Conv1d. forward =
/// linear(nonlinear(x)).
public final class TransitLayer: Module {
    @ModuleInfo(key: "nonlinear") var nonlinear: NonlinearBNReLU
    @ModuleInfo(key: "linear") var linear: Conv1d

    public init(inChannels: Int, outChannels: Int) {
        self._nonlinear.wrappedValue = NonlinearBNReLU(inChannels)
        self._linear.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, bias: false)
        super.init()
    }

    /// x: `(B, Cin, T)`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv1dCM(linear, nonlinear(x))
    }
}

/// DenseLayer (`xvector.dense`): input is `(B, 1024)` after stats. Unsqueeze to
/// `(B, 1024, 1)`, Conv1d(1024->512, k1, bias=false), squeeze, then BN(512,
/// affine=false). No ReLU.
public final class DenseLayer: Module {
    @ModuleInfo(key: "linear") var linear: Conv1d
    @ModuleInfo(key: "nonlinear") var nonlinear: NonlinearBNOnly

    public init(inChannels: Int, outChannels: Int) {
        self._linear.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, bias: false)
        self._nonlinear.wrappedValue = NonlinearBNOnly(outChannels)
        super.init()
    }

    /// x: `(B, 1024)`. Returns `(B, 512)`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xt = x.expandedDimensions(axis: -1)   // (B, 1024, 1)
        var h = conv1dCM(linear, xt)              // (B, 512, 1)
        h = nonlinear(h)                          // (B, 512, 1)
        return h.squeezed(axis: -1)              // (B, 512)
    }
}

/// CAMDenseTDNNBlock: dense block. forward concatenates each layer's output to
/// the running tensor along the channel dim. Layers are named `tdnnd1..tdnndN`
/// (1-indexed) to match the safetensors keys.
///
/// The layers are held in a plain dictionary (not `@ModuleInfo`) so MLX does not
/// auto-discover them under a `layers.` path segment. `items()` is overridden to
/// hoist each `tdnndK` directly as a child of the block, matching the torch key
/// layout `xvector.blockN.tdnndK.*`.
public final class CAMDenseTDNNBlock: Module {
    private let layerDict: [String: CAMDenseTDNNLayer]
    let orderedKeys: [String]

    public init(numLayers: Int, inChannels: Int, outChannels: Int,
                bnChannels: Int, kernelSize: Int, dilation: Int) {
        var dict: [String: CAMDenseTDNNLayer] = [:]
        var keys: [String] = []
        for i in 0..<numLayers {
            let key = "tdnnd\(i + 1)"
            dict[key] = CAMDenseTDNNLayer(
                inChannels: inChannels + i * outChannels, outChannels: outChannels,
                bnChannels: bnChannels, kernelSize: kernelSize, dilation: dilation)
            keys.append(key)
        }
        self.layerDict = dict
        self.orderedKeys = keys
        super.init()
    }

    /// Expose each tdnnd layer as a direct child of the block (no `layers.` prefix).
    public override func items() -> ModuleItems {
        var items = ModuleItems()
        for (k, v) in layerDict {
            items[k] = .value(.module(v))
        }
        return items
    }

    /// x: `(B, C, T)`. Returns `(B, C + N*out, T)`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for k in orderedKeys {
            let out = layerDict[k]!(h)
            h = concatenated([h, out], axis: 1)   // channel-major cat
        }
        return h
    }
}

/// CAMDenseTDNNLayer: BN+ReLU -> Conv1d(in->128, k1) -> BN+ReLU -> CAMLayer.
public final class CAMDenseTDNNLayer: Module {
    @ModuleInfo(key: "nonlinear1") var nonlinear1: NonlinearBNReLU
    @ModuleInfo(key: "linear1") var linear1: Conv1d
    @ModuleInfo(key: "nonlinear2") var nonlinear2: NonlinearBNReLU
    @ModuleInfo(key: "cam_layer") var camLayer: CAMLayer

    public init(inChannels: Int, outChannels: Int, bnChannels: Int,
                kernelSize: Int, dilation: Int) {
        self._nonlinear1.wrappedValue = NonlinearBNReLU(inChannels)
        self._linear1.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: bnChannels, kernelSize: 1, bias: false)
        self._nonlinear2.wrappedValue = NonlinearBNReLU(bnChannels)
        self._camLayer.wrappedValue = CAMLayer(
            inChannels: bnChannels, outChannels: outChannels,
            kernelSize: kernelSize, dilation: dilation)
        super.init()
    }

    /// x: `(B, Cin, T)`. Returns `(B, out, T)`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let bn = conv1dCM(linear1, nonlinear1(x))   // (B, 128, T)
        return camLayer(nonlinear2(bn))             // (B, 32, T)
    }
}

/// CAMLayer (context-aware mask). Computes a per-channel gating mask from the
/// global + segmental context and multiplies it into the local conv output.
public final class CAMLayer: Module {
    @ModuleInfo(key: "linear_local") var linearLocal: Conv1d
    @ModuleInfo(key: "linear1") var linear1: Conv1d
    @ModuleInfo(key: "linear2") var linear2: Conv1d
    let segLen = 100

    public init(inChannels: Int, outChannels: Int, kernelSize: Int, dilation: Int) {
        let padding = (kernelSize - 1) / 2 * dilation
        self._linearLocal.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: kernelSize,
            stride: 1, padding: padding, dilation: dilation, bias: false)
        self._linear1.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: inChannels / 2, kernelSize: 1, bias: true)
        self._linear2.wrappedValue = Conv1d(
            inputChannels: inChannels / 2, outputChannels: outChannels, kernelSize: 1, bias: true)
        super.init()
    }

    /// x: `(B, Cin, T)`. Returns `(B, out, T)`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv1dCM(linearLocal, x)                    // (B, out, T)
        let globalMean = x.mean(axis: -1, keepDims: true)   // (B, Cin, 1)
        let seg = segPooling(x)                             // (B, Cin, T)
        let context = globalMean + seg                      // broadcast -> (B, Cin, T)
        var m = relu(conv1dCM(linear1, context))           // (B, Cin/2, T)
        m = sigmoid(conv1dCM(linear2, m))                  // (B, out, T)
        return y * m
    }

    /// Segmental average pooling: avg_pool1d(kernel=100, stride=100, ceil_mode=true)
    /// over time, then broadcast each segment value back across its 100 frames and
    /// crop to the original T. Piecewise-constant segmental mean. x: `(B, C, T)`.
    private func segPooling(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), C = x.dim(1), T = x.dim(2)
        let nSeg = (T + segLen - 1) / segLen           // ceil_mode
        let padded = nSeg * segLen
        var xp = x
        if padded > T {
            // ceil_mode pools the final partial window over the frames present;
            // pad with zeros AND track counts so the partial-window mean divides by
            // the real count (avg_pool1d ceil_mode counts only valid elements).
            let pad = MLXArray.zeros([B, C, padded - T])
            xp = concatenated([x, pad], axis: 2)
        }
        // reshape (B, C, nSeg, segLen) and sum over segLen, dividing by valid counts.
        let grid = xp.reshaped(B, C, nSeg, segLen)
        let segSum = grid.sum(axis: -1)                // (B, C, nSeg)
        // valid counts per segment: full segLen except possibly the last.
        var counts = [Float](repeating: Float(segLen), count: nSeg)
        let lastValid = T - (nSeg - 1) * segLen
        counts[nSeg - 1] = Float(lastValid)
        let countArr = MLXArray(counts).reshaped(1, 1, nSeg)
        let segMean = segSum / countArr                // (B, C, nSeg)
        // broadcast each segment value across segLen frames, then crop to T.
        let expanded = segMean.expandedDimensions(axis: -1)        // (B, C, nSeg, 1)
        let broadcast = MLX.broadcast(expanded, to: [B, C, nSeg, segLen])
        let flat = broadcast.reshaped(B, C, padded)               // (B, C, padded)
        return flat[0..., 0..., 0..<T]                            // crop to T
    }
}

// MARK: - statistics pooling

/// Mean + Bessel-corrected std over time, concatenated on the channel axis.
/// x: `(B, C, T)`. Returns `(B, 2C)`.
@inline(__always)
func statisticsPooling(_ x: MLXArray) -> MLXArray {
    let mean = x.mean(axis: -1)                  // (B, C)
    let sd = std(x, axis: -1, ddof: 1)           // (B, C) unbiased
    return concatenated([mean, sd], axis: -1)    // (B, 2C)
}
