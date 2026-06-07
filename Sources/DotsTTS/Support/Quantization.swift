import Foundation
import MLXNN

/// Per-component quantisation settings, read from the component's `config.json`
/// `quantization` block (mlx_lm format). The backbone ships int4 group-64; other
/// components are fp32 by default.
public struct QuantizationSettings: Sendable {
    public let enabled: Bool
    public let bits: Int
    public let groupSize: Int

    public init(enabled: Bool, bits: Int = 4, groupSize: Int = 64) {
        self.enabled = enabled
        self.bits = bits
        self.groupSize = groupSize
    }

    public static let none = QuantizationSettings(enabled: false)

    /// Decoded `quantization` block: `{ "group_size": Int, "bits": Int }`.
    public struct Config: Codable, Sendable {
        public var groupSize: Int?
        public var bits: Int?
        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
        }
    }

    public init(from config: Config?) {
        if let config, let bits = config.bits {
            self.enabled = true
            self.bits = bits
            self.groupSize = config.groupSize ?? 64
        } else {
            self.enabled = false
            self.bits = 4
            self.groupSize = 64
        }
    }
}

/// Builds a `Linear` (or `QuantizedLinear` when quantisation is enabled) so the
/// same module code loads both fp32 and mlx_lm-quantised checkpoints.
public enum QuantizedLayerFactory {
    public static func linear(
        _ inputDims: Int,
        _ outputDims: Int,
        bias: Bool = true,
        settings: QuantizationSettings
    ) -> Linear {
        if settings.enabled {
            return QuantizedLinear(
                inputDims, outputDims, bias: bias,
                groupSize: settings.groupSize, bits: settings.bits
            )
        }
        return Linear(inputDims, outputDims, bias: bias)
    }
}
