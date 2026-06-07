import Foundation
import MLX
import MLXNN

/// Loading helpers shared by every component module.
///
/// The HF model repo `smcleod/dots.tts-soar-mlx` is laid out as one directory per
/// component (backbone/, dit/, vocoder/, speaker/, patch_encoder/,
/// audiovae_encoder/), each holding a `model.safetensors` (optionally sharded with
/// a `model.safetensors.index.json`) and, where relevant, a `config.json`.
public enum WeightLoading {
    public enum LoadError: Error, CustomStringConvertible {
        case missingWeights(URL)
        case badIndex(URL)

        public var description: String {
            switch self {
            case .missingWeights(let url): return "no safetensors found under \(url.path)"
            case .badIndex(let url): return "could not parse \(url.path)"
            }
        }
    }

    /// Flat `[parameterPath: MLXArray]` for a component directory, merging shards.
    public static func arrays(in directory: URL) throws -> [String: MLXArray] {
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: index.path) {
            return try shardedArrays(directory: directory, index: index)
        }
        let single = directory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: single.path) else {
            throw LoadError.missingWeights(directory)
        }
        return try MLX.loadArrays(url: single)
    }

    private static func shardedArrays(directory: URL, index: URL) throws -> [String: MLXArray] {
        guard
            let data = try? Data(contentsOf: index),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let weightMap = root["weight_map"] as? [String: String]
        else { throw LoadError.badIndex(index) }

        var merged: [String: MLXArray] = [:]
        for shard in Set(weightMap.values) {
            let url = directory.appendingPathComponent(shard)
            for (k, v) in try MLX.loadArrays(url: url) { merged[k] = v }
        }
        return merged
    }

    /// Load a component dir's weights into `module` (keys must already match the
    /// module's parameter paths; converters produce matching dotted names).
    public static func load(_ module: Module, from directory: URL, verify: Bool = true) throws {
        let weights = try arrays(in: directory)
        let params = ModuleParameters.unflattened(weights)
        try module.update(parameters: params, verify: verify ? .all : .none)
        eval(module)
    }
}
