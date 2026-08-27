# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L9 — multimodal learning + first trained image generator

L0–L7 built Spiral's native language, training, runtime-memory, tool, and agent stack. L8 added owned RGB image representation, bidirectional vision encoding, cross-modal projection, and latent raster decoding. L9 adds the first visual learning path that can actually fit image data and condition generation on text:

- `prompt_features`: deterministic native text conditioning vectors derived from prompt bytes without an external tokenizer/model runtime.
- `ImageAutoencoder`: patch-space image encoder + latent representation + decoder using Spiral-owned trainable parameters.
- `AutoencoderTrainer`: explicit MSE reverse pass through the patch encoder/decoder with Spiral AdamW and gradient clipping.
- `PromptLatentGenerator`: text features plus normalized 2D patch coordinates → per-patch image latent predictions.
- `PromptGeneratorTrainer`: learns prompt-to-latent mappings against frozen latents produced by the trained image encoder.
- `SpiralImageGenerator`: prompt → learned latent grid → learned decoder → RGB image.
- `mean_squared_error` and `cosine_similarity`: native multimodal loss/similarity primitives for reconstruction and later alignment work.
- image bundles: native binary save/load for autoencoder + prompt generator configuration-compatible weights.
- `spiral_multimodal_tests`: requires autoencoder reconstruction loss to fall substantially, requires prompt-latent loss to fall substantially, trains distinct `red signal` and `blue signal` outputs, verifies visible channel separation, and proves bundle reload reproduces generated pixels exactly.

L9 is intentionally a small, inspectable learned generator rather than a claim of frontier text-to-image quality. It proves the complete sovereign path `text → trained conditioning → trained latent → trained pixels`. A later rung can replace the linear latent predictor with deeper transformer/flow/denoising networks while preserving this owned data/training/runtime foundation.

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
- L10 ⬜ deeper flow/denoising + audio/video + Spiral Units + integrated intelligence

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
