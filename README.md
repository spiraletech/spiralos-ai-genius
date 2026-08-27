# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L0 — sovereign foundation

This branch establishes the first dependency-light primitives owned by Spiral:

- `SpiralTensor`: contiguous float tensor storage, shape validation, elementwise add, matrix multiply, ReLU, and softmax.
- `SpiralToken`: deterministic byte tokenizer with BOS/EOS control tokens.
- `spiral_genius`: tiny native executable proving both systems work together.
- `spiral_core_tests`: zero-dependency executable tests wired into CTest.

No llama.cpp, PyTorch, TensorFlow, diffusion runtime, or hosted-model API is required by this rung.

## Build

```bash
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## Data ladder

L0 tensor/runtime foundation → L1 tokenizer + model graph → L2 attention/transformer blocks → L3 training/autodiff → L4 inference/sampling → L5 memory/tools → L6 agents → L7 vision/image → L8 video/audio → L9 Spiral Units → L10 integrated Spiral intelligence.

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
