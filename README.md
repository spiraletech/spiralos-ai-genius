# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L6 — runtime intelligence

L0 established Spiral-owned tensors and tokenization. L1 added neural layers. L2 added causal attention. L3 assembled the language model. L4 added native training. L5 added autoregressive decoding. L6 makes the model fast enough to keep state and gives it the first persistent intelligence substrate:

- `InferenceSession`: true one-token incremental model execution using a per-layer K/V cache rather than re-running the full prefix for every token.
- cached Q/K/V execution preserves the exact current transformer math: per-head RoPE at the token position, scaled dot-product attention over cached keys/values, residuals, RMSNorm, gated feed-forward, final norm, and vocabulary logits.
- cache reset/prefill APIs provide a correctness-preserving rebuild path when a bounded context window drops old tokens.
- `LayerKVCache`: explicit Spiral-owned key/value storage and token-count inspection per transformer layer.
- model bundles: one native binary artifact containing `ModelConfig` plus validated parameter shapes/data, allowing a model to be reconstructed without a separately hard-coded architecture config.
- `MemoryStore`: owned persistent memory records, binary save/load, tags, stable IDs, and deterministic lexical cosine retrieval.
- `ToolRegistry`: named native tools with descriptions, callbacks, controlled failures, duplicate protection, enumeration, and invocation.
- `spiral_runtime_tests`: compares incremental-cache logits against full-context logits, verifies cache growth, round-trips a model bundle, persists/retrieves memory, and validates tool dispatch.

The L5 generation API remains valid. Full-context generation is still available as the reference path while `InferenceSession` becomes the native fast path; bounded-window generation can rebuild the cache only when trimming changes positional semantics.

No llama.cpp, PyTorch, TensorFlow, vector database, pretrained-model runtime, or hosted-model API is required.

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
- L6 ✅ per-layer KV cache + incremental inference + model bundles + persistent memory + retrieval + native tool registry
- L7 ⬜ agents + planning + verification + repair loops
- L8 ⬜ vision + image generation
- L9 ⬜ video/audio + Spiral Units
- L10 ⬜ integrated Spiral intelligence

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
