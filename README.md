# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L18 — native accelerated compute

L0–L17 established Spiral-owned language, agents, multimodal generation, temporal learning, media decoders, and prompt-conditioned audio/video latent flow. L18 adds the first explicit execution backend so those models no longer have to treat the original scalar Tensor implementation as the only compute path:

- `compute::ThreadPool`: persistent native worker threads, queued futures, and deterministic chunked `parallel_for`.
- `compute::ScratchArena`: aligned bump allocation for temporary kernel memory with reset/high-watermark accounting.
- SSE2 four-wide FP32 dot products on supported x86/x64 targets with a portable scalar fallback.
- `parallel_matmul`: RHS-transposed contiguous dot kernels distributed across worker threads.
- native symmetric INT8 matrix multiplication with int32 accumulation and quantization-scale restoration; it does not dequantize the full operands before multiplying.
- `ComputeBackend`: hardware-independent execution contract for matmul and quantized matmul.
- `CpuBackend`: threshold-based scalar/threaded dispatch and explicit SIMD backend identity.
- `benchmark`: steady-clock benchmark harness for local kernel measurements without encoding unstable CI speed claims into tests.
- `spiral_compute_tests`: proves multiple worker threads execute work, aligned scratch allocation/reset, SIMD/scalar dot equivalence, threaded FP32 matmul equivalence, INT8 accuracy bounds, backend polymorphism, and benchmark operation.

L18 deliberately does **not** claim a native GPU kernel yet. `ComputeBackend` is the boundary that future Spiral GPU backends will implement; this rung certifies the threaded/SIMD/quantized CPU reference first so later accelerators have an exact numerical contract to beat without changing model code.

No llama.cpp, PyTorch, TensorFlow, OpenCV, PIL, FFmpeg, libsndfile, Stable Diffusion/diffusion wrapper, pretrained-model runtime, vector database, external agent framework, hosted-model API, BLAS library, or external thread-pool runtime is required.

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
- L18 ✅ native thread pool + aligned arena + SSE2/scalar kernels + parallel FP32 matmul + native INT8 matmul + compute backend + benchmark harness
- L19 ⬜ Spiral Units + unified multimodal runtime + compute-backed model execution
- L20 ⬜ native GPU kernels + device memory + serious heterogeneous scaling

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
