# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L14 — shared temporal engine for audio + video

L0–L13 established Spiral-owned language, training, agents, vision, iterative image generation, scale training, streaming datasets, numeric formats, WAV/STFT audio, and a trainable audio latent codec. L14 adds the first shared representation of ordered time across modalities:

- `dsp::fft_inplace`: native iterative radix-2 Cooley–Tukey FFT with inverse support.
- `dsp::stft_magnitude_fft`: Hann-window FFT STFT that numerically tracks the older transparent O(N²) reference DFT for power-of-two frames.
- `temporal::AudioWindowCursor`: deterministic overlapping audio windows for long-stream processing.
- `temporal::TemporalTransformerEncoder`: ordered features → input projection → temporal sinusoidal positions → pre-RMSNorm multi-head self-attention/FFN blocks → output tokens or pooled sequence embedding.
- depth-scaled residual branches using `1/sqrt(num_layers)`.
- `temporal::AudioTemporalEncoder`: PCM audio → FFT STFT → spectral patches → trained audio latents → shared temporal transformer.
- `temporal::VideoFrameSequence`: ordered RGB frames with explicit frame rate.
- `temporal::VideoTemporalEncoder`: each frame → Spiral VisionEncoder pooled embedding → shared temporal transformer.
- `spiral_temporal_tests`: validates FFT inverse round-trip, FFT-vs-reference STFT agreement, dominant-frequency detection, streaming-window coverage, deterministic temporal initialization, order sensitivity, audio temporal latents, and video frame-order sensitivity.

The temporal encoder is intentionally modality-neutral: audio and video now differ in how they become feature tokens, not in how Spiral reasons across time. L14 remains a CPU reference path; optimized SIMD/GPU attention and learned temporal objectives can replace internals without changing the public sequence contract.

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
- L15 ⬜ temporal training objectives + audio generation/synthesis + frame prediction + Spiral Units unified multimodal runtime

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
