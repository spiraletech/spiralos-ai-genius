# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L1 — neural foundation

L0 established Spiral-owned tensor storage and tokenization. L1 turns that substrate into a small neural-compute runtime:

- `SpiralTensor`: explicit row-major strides plus mutable/const non-owning tensor views and zero-copy rank-2 transpose views.
- `SpiralRandom`: deterministic SplitMix64-based random stream with uniform/normal tensor initialization.
- `Parameter`: named, trainable weight ownership independent of any external ML framework.
- `Linear`: Xavier-initialized dense projection.
- `Embedding`: owned vocabulary lookup table for token representations.
- `RMSNorm` and `LayerNorm`: last-dimension normalization primitives.
- `Sequential`: first executable Spiral neural graph shell with parameter discovery.
- `spiral_nn_tests`: deterministic tests for views, RNG, embeddings, normalization, graph execution, and parameter enumeration.

No llama.cpp, PyTorch, TensorFlow, diffusion runtime, or hosted-model API is required by this rung.

## Build

```bash
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## Data ladder

- L0 ✅ tensor/runtime + tokenizer foundation
- L1 ✅ strides/views + RNG + parameters + embeddings + normalization + neural graph
- L2 ⬜ tensor reshape/batched ops + positional encoding + causal masking + attention
- L3 ⬜ transformer block + tiny autoregressive model
- L4 ⬜ autodiff + loss + optimizer + training loop
- L5 ⬜ inference cache + sampling + native text generation
- L6 ⬜ persistent memory + tools
- L7 ⬜ agents + verification
- L8 ⬜ vision + image generation
- L9 ⬜ video/audio + Spiral Units
- L10 ⬜ integrated Spiral intelligence

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
