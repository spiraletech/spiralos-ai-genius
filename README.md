# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L15 — causal temporal learning + latent generation

L0–L14 established Spiral-owned language, training, agents, vision, image generation, scalable datasets, audio latents, FFT/STFT DSP, and a shared bidirectional temporal encoder. L15 adds the first trainable autoregressive time model and generation adapters:

- `temporal_generation::CausalTemporalAttention`: strict causal multi-head self-attention; token `t` can only attend to positions `<= t`.
- `CausalTemporalPredictor`: ordered features → projection + temporal positions → causal attention/FFN residual blocks → next-latent predictions.
- explicit reverse-mode gradients through Q/K/V, masked softmax, value mixing, output projection, residuals, and SiLU feed-forward layers.
- `TemporalNextLatentTrainer`: next-step MSE training with native Spiral AdamW and gradient clipping.
- bounded autoregressive context plus deterministic multi-step latent generation.
- native predictor checkpoint save/load with architecture validation and exact weight restoration.
- `magnitude_to_audio_zero_phase`: inverse FFT + Hann overlap-add reference reconstruction from generated magnitude spectra.
- `AudioLatentGenerator`: PCM seed → FFT/STFT → spectral patches → AudioLatentCodec → causal latent continuation → decoder → PCM synthesis.
- `VideoEmbeddingGenerator`: ordered RGB frames → VisionEncoder embeddings → causal next-frame embedding prediction and multi-step latent continuation.
- `spiral_temporal_generation_tests`: validates exact future-token isolation, real next-latent loss reduction, deterministic autoregression, exact checkpoint output restoration, inverse-spectrum PCM synthesis, audio latent continuation, and video embedding continuation.

L14 remains the bidirectional "understand the whole sequence" path. L15 is deliberately separate and causal because generation has a different contract: no future leakage. Video generation at this rung is latent/embedding generation only; Spiral does not claim to decode predicted frame embeddings back into RGB video yet.

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
- L16 ⬜ learned phase/audio codec upgrade + frame latent decoder + temporal diffusion/flow + Spiral Units unified multimodal runtime

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
