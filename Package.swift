// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "mlx-swift-dots-tts",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        .library(name: "DotsTTS", targets: ["DotsTTS"]),
    ],
    dependencies: [
        // mlx-swift 0.31.x (matches Cloney's pin).
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.4"),
        // swift-transformers for the Qwen2 BPE tokenizer + Hub download helpers.
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.3"),
    ],
    targets: [
        .target(
            name: "DotsTTS",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ]
        ),
        .testTarget(
            name: "DotsTTSTests",
            dependencies: ["DotsTTS"]
        ),
    ]
)
