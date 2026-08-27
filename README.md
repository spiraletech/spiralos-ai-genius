# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L12 — depth-stable transformer + scalable training runtime

L0–L11 established Spiral-owned language, training, agents, vision, iterative image generation, and a trainable latent transformer. L12 turns that transformer research path into a repeatable scale-training system instead of only a tiny in-memory experiment:

- `StableLatentTransformerBlock`: pre-RMSNorm latent self-attention, pre-RMSNorm prompt cross-attention, pre-RMSNorm SiLU feed-forward, and depth-scaled residual branches.
- `StableLatentTransformerDenoiser`: scalable pre-norm transformer variant that still implements the L10 `LatentPredictor` contract for CFG, img2img, and inpainting.
- automatic residual scaling `1/sqrt(num_layers)` when no explicit scale is supplied, plus a final RMSNorm before latent prediction.
- explicit reverse-mode gradients through RMSNorm, self-attention, cross-attention, softmax, residual scaling, and feed-forward layers.
- TSV image-prompt manifests using native PPM assets, deterministic Fisher-Yates shuffling, and deterministic train/validation splits.
- `ScaleTrainer`: configurable microbatches, gradient-accumulation windows, gradient clipping, deterministic per-epoch shuffle/noise seeds, and validation evaluation.
- `StatefulAdamW`: native AdamW moments and step state that can be serialized and restored.
- resumable training checkpoints containing model configuration, weights, trainer configuration, epoch, optimizer step, first moments, and second moments.
- CSV metric logging for epoch, optimizer steps, train loss, validation loss, and gradient norm.
- `spiral_scale_tests`: validates manifest loading/splitting, deterministic shuffling, a four-layer pre-norm stack, prompt sensitivity, real loss reduction, expected accumulation step counts, L10 sampler compatibility, and exact next-epoch equivalence after checkpoint resume.

L12 intentionally keeps the reference L11 transformer untouched. This gives Spiral an A/B architecture boundary: the minimal transformer remains available for correctness experiments while the L12 pre-norm/depth-scaled path can grow to deeper models and larger datasets.

No llama.cpp, PyTorch, TensorFlow, OpenCV, PIL, Stable Diffusion/diffusion wrapper, pretrained-model runtime, vector database, external agent framework, or hosted-model API is required.

## Build

```bash
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## L12 manifest format

One example per line, with paths resolved relative to the manifest file:

```text
image_0001.ppm<TAB>a red signal in a dark room
images/image_0002.ppm<TAB>a blue signal in fog
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
- L12 ✅ pre-RMSNorm/depth scaling + manifests + deterministic shuffle/split + microbatches + gradient accumulation + resumable optimizer state + metrics
- L13 ⬜ dataset streaming/cache + mixed precision/quantization foundations + audio latent codec + temporal/video attention + Spiral Units

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
