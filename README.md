# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L3 — first Spiral language model

L0 established Spiral-owned tensor storage and tokenization. L1 added neural parameters, layers, normalization, embeddings, and graph execution. L2 added causal multi-head self-attention. L3 assembles those pieces into the first complete autoregressive language-model graph:

- `GatedFeedForward`: two-input SiLU-gated feed-forward path with owned gate/up/down projections.
- `TransformerBlock`: pre-norm causal attention, residual connection, pre-norm gated feed-forward, and second residual connection.
- `SpiralLanguageModel`: token embeddings, configurable transformer stack, final RMS normalization, and vocabulary projection to autoregressive logits.
- `last_token_logits`: stable model boundary for the future sampler/inference-cache rung.
- parameter enumeration and exact scalar parameter counting across embeddings, transformer blocks, final norm, and language-model head.
- `spiral_model_tests`: validates model shape, deterministic initialization, parameter accounting, last-token extraction, empty-input rejection, and end-to-end causal isolation of token-0 logits from future tokens.

No llama.cpp, PyTorch, TensorFlow, diffusion runtime, pretrained model runtime, or hosted-model API is required by this rung.

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
- L4 ⬜ autodiff + cross-entropy loss + optimizer + training loop
- L5 ⬜ inference cache + sampling + native text generation
- L6 ⬜ persistent memory + tools
- L7 ⬜ agents + verification
- L8 ⬜ vision + image generation
- L9 ⬜ video/audio + Spiral Units
- L10 ⬜ integrated Spiral intelligence

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
