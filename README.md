# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L11 — trainable latent transformer denoiser

L0–L10 established Spiral-owned language, training, agents, vision, learned text-to-image latents, and an iterative CFG/img2img/inpaint sampler. L11 replaces the compact denoiser ceiling with a scalable patch-transformer path while keeping the L10 sampling contract intact:

- `flow::LatentPredictor`: common native predictor interface so compact and transformer denoisers drive the same iterative generator.
- `prompt_token_features`: deterministic multi-token prompt representation for real cross-attention rather than one pooled conditioning vector.
- `MultiHeadAttention`: reusable trainable query/context attention supporting both latent self-attention and prompt cross-attention.
- `LatentTransformerBlock`: latent self-attention → residual → prompt cross-attention → residual → SiLU feed-forward → residual.
- `LatentTransformerDenoiser`: noisy latent patches + 2D positions + timestep features → stacked transformer blocks → predicted clean latent patches.
- explicit reverse-mode gradients through Q/K/V projections, attention softmax, values, output projections, cross-attention, residual paths, and feed-forward layers.
- `ImagePromptDataset` plus `LatentTransformerTrainer`: true batch gradient accumulation with Spiral AdamW and gradient clipping.
- native transformer checkpoint save/load with configuration validation.
- direct compatibility with L10 deterministic generation, classifier-free guidance, image-to-image, and inpainting through `LatentPredictor`.
- `spiral_latent_transformer_tests`: validates prompt-token determinism, nonlocal patch coupling, prompt coupling, batch denoising loss reduction, sampler determinism/prompt sensitivity, and exact checkpoint reload.

L11 is still a deliberately tiny CPU reference model. The milestone is architectural: increasing model dimension, heads, layers, training data, and compute now scales an owned attention-based visual model instead of requiring a different external generator runtime.

No llama.cpp, PyTorch, TensorFlow, OpenCV, PIL, Stable Diffusion/diffusion wrapper, pretrained-model runtime, vector database, external agent framework, or hosted-model API is required.

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
- L7 ✅ intent + task graph + memory injection + tool execution + critic + repair/retry + verification + traces + outcome memory
- L8 ✅ native RGB images + patches + bidirectional vision transformer + cross-modal projection + latent raster decoder
- L9 ✅ trainable image autoencoder + prompt-conditioned latent learning + native prompt-to-RGB generation + image bundles
- L10 ✅ noise scheduling + timestep conditioning + trainable latent denoiser + CFG + iterative sampling + img2img + inpainting
- L11 ✅ latent self-attention + prompt cross-attention + residual transformer denoiser + batch training + checkpoints
- L12 ⬜ dataset manifests/shuffling + transformer normalization/depth scaling + audio latent codec + temporal/video attention + Spiral Units

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
