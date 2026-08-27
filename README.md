# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L10 — iterative latent generative engine

L0–L7 built Spiral's native language, training, runtime-memory, tool, and agent stack. L8 added owned vision infrastructure. L9 proved a fully native trained path from text conditioning to image latents and RGB pixels. L10 replaces one-shot latent prediction with an explicit iterative denoising runtime:

- `NoiseScheduler`: linear or cosine noise schedules, deterministic clean↔noise interpolation, and descending sampling timesteps.
- `timestep_features`: native sinusoidal timestep conditioning.
- `LatentDenoiser`: noisy latent + prompt features + 2D patch coordinates + timestep features → hidden SiLU network → predicted clean latent.
- `DenoiserTrainer`: explicit MSE reverse pass through the denoiser using Spiral AdamW and gradient clipping; no external autograd runtime.
- `guided_prediction`: classifier-free-guidance equation combining unconditional and prompt-conditioned latent predictions.
- `IterativeImageGenerator`: seeded Gaussian latent initialization followed by repeated denoising/refinement steps and native latent→RGB decode.
- image-to-image: source images are encoded, partially noised according to strength, then refined with the same prompt-conditioned sampler.
- latent-patch inpainting: masks preserve unedited source latent patches while prompt-driven refinement updates selected patches.
- `spiral_flow_tests`: validates noise-schedule endpoints, deterministic timestep features, classifier-free guidance math, visual autoencoder learning, denoiser loss reduction, seeded generation determinism, prompt separation, guidance effect, img2img strength-zero preservation, and inpaint-mask preservation/edit behavior.

L10 remains deliberately small and inspectable. The denoiser is a compact two-layer latent network, not a claim of Stable Diffusion/Flux-class visual quality. The architectural milestone is that Spiral now owns the complete iterative generation loop `noise → conditioned denoising steps → latent → pixels`, so later transformer/flow architectures can replace the compact denoiser without replacing the scheduler, trainer, sampler, img2img, mask, or image runtime contracts.

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
- L11 ⬜ deeper latent transformer/flow blocks + batched datasets + audio/video latent foundation + Spiral Units

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
