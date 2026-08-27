# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L2 — causal attention foundation

L0 established Spiral-owned tensor storage and tokenization. L1 added neural parameters, layers, normalization, embeddings, and graph execution. L2 adds the first sequence-intelligence primitives required for an autoregressive language model:

- `tensor_ops`: reshape-copy validation, rank-3 batched matrix multiplication, last-dimension softmax, and causal-mask construction.
- `apply_rotary_inplace`: Spiral-owned rotary position encoding for per-head query/key tensors.
- `CausalSelfAttention`: owned Q/K/V/output projections, head split/merge, scaled dot-product attention, future-token masking, and multi-head context mixing.
- `spiral_attention_tests`: validates batched math, row-wise softmax, mask topology, rotary norm preservation, parameter discovery, and the causal invariant that future-token changes cannot affect token 0.

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
- L2 ✅ reshape/batched ops + RoPE + causal masking + multi-head self-attention
- L3 ⬜ transformer block + feed-forward network + tiny autoregressive model
- L4 ⬜ autodiff + loss + optimizer + training loop
- L5 ⬜ inference cache + sampling + native text generation
- L6 ⬜ persistent memory + tools
- L7 ⬜ agents + verification
- L8 ⬜ vision + image generation
- L9 ⬜ video/audio + Spiral Units
- L10 ⬜ integrated Spiral intelligence

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
