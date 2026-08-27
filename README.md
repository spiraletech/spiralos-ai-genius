# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L5 — native autoregressive generation

L0 established Spiral-owned tensors and tokenization. L1 added neural layers. L2 added causal attention. L3 assembled the language model. L4 gave it a native learning engine. L5 turns trained logits into a deterministic, streamable text-generation runtime:

- `SamplingConfig`: temperature, top-k, nucleus/top-p, repetition penalty, and deterministic Spiral RNG seeds.
- `sample_token`: stable candidate filtering and stochastic/greedy decoding without an external inference library.
- `build_context`: bounded context-window management with optional BOS preservation.
- `GenerationConfig`: new-token limits, context limits, stop-token control, and sampling policy.
- `TokenGenerator`: prompt-token → model → sampled-token autoregressive loop with cancellation callbacks.
- `ByteTextGenerator`: native byte-token prompt encoding, token streaming, and generated-text decoding for the current Spiral vocabulary.
- deterministic stop/cancel state surfaced through `GenerationResult`.
- `spiral_generate_tests`: validates greedy decoding, repetition penalty, seeded top-k behavior, context trimming, stop-token recognition, callback cancellation, and an end-to-end L4→L5 proof where a trained tiny model greedily emits its learned next token.

L5 intentionally uses full-context decoding as the correctness baseline. A true per-layer KV cache will be added as an optimization behind the same generation boundary rather than contaminating the first correct decoder with premature cache state.

No llama.cpp, PyTorch, TensorFlow, pretrained model runtime, or hosted-model API is required.

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
- L3 ✅ residual transformer blocks + gated feed-forward + tiny autoregressive language model
- L4 ✅ gradients + reverse-mode backprop + cross-entropy + AdamW + training + checkpoints
- L5 ✅ temperature/top-k/top-p + repetition control + bounded context + streaming native generation
- L6 ⬜ persistent memory + tools + retrieval
- L7 ⬜ agents + verification
- L8 ⬜ vision + image generation
- L9 ⬜ video/audio + Spiral Units
- L10 ⬜ integrated Spiral intelligence

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
