# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L13 — streaming data + numeric formats + audio latent foundation

L0–L12 established Spiral-owned language, training, agents, vision, iterative image generation, latent transformers, and resumable scale training. L13 opens the next two scaling surfaces: feeding larger datasets without loading them all into RAM, and giving Spiral a native audio representation that can later share temporal modeling machinery with video.

- `data::ImagePromptShard`: indexed native shard format that scans record metadata once and lazily loads prompt/RGB payloads on demand.
- `data::ShardedImagePromptDataset`: multiple shards presented as one logical dataset with bounded resident cache and explicit prefetch.
- `precision::NumericFormat`: native Float32/Float16/BFloat16/Int8 format identifiers.
- IEEE-style FP16 conversion, BF16 conversion, and symmetric int8 tensor quantization/dequantization.
- `audio::AudioBuffer`: native interleaved PCM float buffer with PCM16 WAV load/save and mono conversion.
- `audio::stft_magnitude`: Hann-window reference STFT implemented in C++ with no external DSP runtime.
- `audio::spectral_patches`: converts time-frequency frames into fixed-size neural patch matrices.
- `audio::AudioLatentCodec`: trainable spectral-patch encoder/decoder using Spiral Linear layers.
- `audio::AudioCodecTrainer`: explicit reconstruction-loss backprop through encoder and decoder using Spiral AdamW.
- `spiral_data_audio_tests`: validates numeric conversion/quantization, lazy shard access/cache bounds, WAV round-trip, frequency-bin detection, spectral patching, and real audio-codec loss reduction.

L13 is still a CPU reference implementation. The STFT deliberately uses a transparent O(N²) DFT so correctness is owned first; optimized FFT/SIMD/GPU kernels can replace it behind the same API later.

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
- L14 ⬜ optimized dataset pipeline + train CLI + SIMD/FFT kernels + temporal audio transformer + video frame tensors/attention
- L15 ⬜ Spiral Units + unified multimodal runtime

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
