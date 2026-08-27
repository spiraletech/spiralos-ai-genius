# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L4 — native training engine

L0 established Spiral-owned tensor storage and tokenization. L1 added neural parameters and layers. L2 added causal attention. L3 assembled the first autoregressive language-model graph. L4 gives that model a native learning path:

- trainable `Parameter::grad` storage with deterministic zeroing and shape validation.
- full model-specific reverse-mode backpropagation through embeddings, RMSNorm, Q/K/V projections, RoPE, causal softmax attention, residual paths, gated SiLU feed-forward layers, final normalization, and vocabulary projection.
- numerically stable next-token cross-entropy loss and analytic vocabulary-logit gradients.
- `AdamW`: owned first/second moments, bias correction, decoupled weight decay, and deterministic step accounting.
- global gradient-norm measurement and clipping.
- `TokenDataset`: sliding next-token training windows from raw token streams.
- `LanguageModelTrainer`: evaluate, train-step, and train-epoch APIs.
- native binary model checkpoint save/load with parameter shape validation.
- `spiral_train_tests`: proves the tiny model actually learns, reducing a held example from roughly 2.53 cross-entropy to below 0.02 in the deterministic test, while preserving all earlier regression suites.

This rung uses explicit reverse-mode derivatives owned by Spiral rather than an external autograd framework. No llama.cpp, PyTorch, TensorFlow, pretrained model runtime, or hosted-model API is required.

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
- L5 ⬜ inference cache + temperature/top-k/top-p sampling + native text generation
- L6 ⬜ persistent memory + tools
- L7 ⬜ agents + verification
- L8 ⬜ vision + image generation
- L9 ⬜ video/audio + Spiral Units
- L10 ⬜ integrated Spiral intelligence

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
