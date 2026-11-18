SHELL := /bin/bash
# Asked of SwiftPM rather than hardcoded: the path moved between toolchains.
DEBUG_DIR := $(shell swift build --show-bin-path)

# mlx-swift ships its Metal kernels as a resource compiled by xcodebuild's Metal
# toolchain; `swift build` does NOT build them, so MLX ops crash under `swift
# test` with "Failed to load the default metallib". We copy a prebuilt
# default.metallib (same mlx-swift version) next to the test runner, where mlx's
# loader looks for a colocated `mlx.metallib`. Override with METALLIB=/path.
METALLIB ?= $(shell find $(HOME)/Library/Developer/Xcode/DerivedData $(HOME)/git/sammcj/cloney/build -name default.metallib -path '*Cmlx*' 2>/dev/null | head -1)

.PHONY: build test metallib clean
build:
	swift build

# Place the metallib colocated with each built .xctest runner binary.
metallib:
	@test -n "$(METALLIB)" || { echo "no default.metallib found - build one once via xcodebuild or set METALLIB=/path/to/default.metallib"; exit 1; }
	@swift build --build-tests >/dev/null
	@for x in $(DEBUG_DIR)/*.xctest/Contents/MacOS; do \
		cp "$(METALLIB)" "$$x/mlx.metallib" && echo "metallib -> $$x/mlx.metallib"; \
	done

# Run the full suite (or a subset: make test ARGS="--filter DiTParityTests").
test: metallib
	swift test $(ARGS)

clean:
	swift package clean
