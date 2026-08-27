# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L16 — generative media decoders

L0–L15 established Spiral-owned language, training, agents, vision, image generation, scalable datasets, audio latents, FFT/STFT DSP, shared temporal understanding, and causal next-latent generation. L16 closes the most important media gap: predicted future latents can now be decoded into actual PCM audio and RGB frames through Spiral-owned trainable decoders.

- `media_generation::complex_stft`: native complex-valued STFT represented as `[frames,bins,2]` real/imaginary tensors, preserving phase instead of discarding it.
- `inverse_complex_stft`: Hermitian reconstruction + inverse FFT + Hann overlap-add back to PCM audio.
- `complex_spectral_patches` / `complex_patches_to_spectrum`: reversible patch representation for temporal audio learning.
- `ComplexAudioCodec`: trainable real+imaginary spectral-patch encoder/decoder.
- `ComplexAudioCodecTrainer`: explicit reconstruction MSE backprop through both codec Linear layers with Spiral AdamW.
- `FrameEmbeddingDecoder`: trainable VisionEncoder pooled embedding → hidden SiLU → sigmoid RGB raster decoder.
- `FrameDecoderTrainer`: trains frame reconstruction while keeping the vision encoder frozen.
- `AudioMediaGenerator`: PCM seed → complex STFT → complex audio latents → causal temporal continuation → complex decoder → inverse STFT → PCM output.
- `VideoMediaGenerator`: RGB seed frames → VisionEncoder embeddings → causal temporal continuation → frame decoder → generated RGB frames.
- deterministic `prompt_media_bias` / `apply_prompt_bias` conditioning hooks for steering latent seeds before a learned prompt-temporal conditioner is introduced.
- `spiral_media_generation_tests`: validates complex spectral round-trip, real audio-codec loss reduction, frame-decoder loss reduction, temporal audio latent continuation into PCM, temporal frame continuation into RGB, and deterministic prompt steering.

L16 is still a compact CPU reference generator. Audio quality is limited by a tiny linear complex-spectrum codec, and video quality is limited by a tiny embedding-to-raster decoder. The important architectural threshold is now real: Spiral can preserve audio phase, learn media decoders, predict future media latents, and render those latents back into native media without FFmpeg, PyTorch, or a foreign generative runtime.

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
- L17 ⬜ learned prompt-temporal conditioning + temporal flow/denoising + richer audio neural codec + video latent flow + media checkpoints
- L18 ⬜ native accelerated compute: SIMD + threading + GPU kernels + serious model scaling
- L19 ⬜ Spiral Units + unified multimodal runtime

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
