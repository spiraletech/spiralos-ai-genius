# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L17 — prompt-conditioned temporal media flow

L0–L16 established Spiral-owned language, agents, vision, image generation, scalable training, phase-preserving audio, causal temporal prediction, and learned PCM/RGB media decoders. L17 removes the seed-clip requirement by learning prompt-conditioned latent timelines directly from noise:

- `media_flow::PromptTemporalFlowModel`: noisy sequence + global sequence state + prompt features + temporal position + diffusion time → predicted clean latent timeline.
- learned prompt conditioning through trainable projections rather than fixed prompt bias.
- sequence-level coupling through the global noisy-state feature.
- temporal consistency loss on adjacent latent deltas.
- `PromptTemporalFlowTrainer`: native AdamW training across multiple noise levels with explicit Linear/SiLU backward.
- iterative seeded denoising with classifier-free-style unconditional/conditional guidance.
- `PromptAudioGenerator`: prompt → complex-audio latent sequence → L16 complex decoder → inverse STFT → PCM.
- `PromptVideoGenerator`: prompt → frame-embedding timeline → L16 frame decoder → RGB frames.
- native prompt-media-flow checkpoint save/load with architecture validation.
- `spiral_media_flow_tests`: requires two prompt trajectories to learn separately, deterministic same-seed sampling, prompt separation, checkpoint-exact reload, prompt→PCM generation, and prompt→RGB-frame generation.

L17 remains a compact CPU reference media-flow model. The temporal denoiser is a small sequence-coupled MLP rather than a large video/audio DiT, and the L16 media codecs remain intentionally tiny. The important new contract is real: Spiral can start from noise plus text and produce a complete native audio/video latent timeline without a seed clip or foreign generative runtime.

No llama.cpp, PyTorch, TensorFlow, OpenCV, PIL, FFmpeg, libsndfile, Stable Diffusion/diffusion wrapper, pretrained-model runtime, vector database, external agent framework, or hosted-model API is required.

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
- L13 ✅ lazy dataset shards/cache + FP16/BF16/int8 foundations + PCM/WAV + STFT + spectral patches + trainable audio latent codec
- L14 ✅ radix-2 FFT + streaming audio windows + shared temporal transformer + audio temporal encoder + video frame temporal encoder
- L15 ✅ causal temporal attention + explicit temporal backward + next-latent learning + audio latent synthesis + video embedding generation + temporal checkpoints
- L16 ✅ complex/phase-preserving audio + learned complex codec + frame embedding decoder + temporal audio→PCM + temporal embeddings→RGB
- L17 ✅ learned prompt-temporal conditioning + sequence denoising + temporal consistency + prompt→PCM + prompt→RGB + media-flow checkpoints
- L18 ⬜ native accelerated compute: SIMD + threading + GPU kernels + serious model scaling
- L19 ⬜ Spiral Units + unified multimodal runtime

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
