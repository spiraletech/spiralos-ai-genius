# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L23 — GPU compute brain

L0–L22 established Spiral-owned language, agents, multimodal generation, temporal media models, live Spiral Units, visual self-repair, accelerated CPU execution, a Win32 host, and a real D3D11 device/presentation path. L23 makes neural tensor math execute through native D3D11 compute shaders while retaining deterministic CPU fallback:

- `gpu::GpuTensor`: move-only RAII wrapper around Spiral GPU buffer handles plus tensor shape metadata.
- `gpu::D3D11ComputeEngine`: shares the exact L22 D3D11 device/context and compiles native HLSL compute kernels at runtime with `D3DCompile`.
- FP32 rank-2 matrix multiplication executes as an 8x8-thread-group D3D11 compute kernel over GPU-resident raw buffers.
- ReLU and SiLU execute as 64-thread elementwise compute kernels.
- row-wise LayerNorm executes on GPU with configurable epsilon.
- numerically stable row-wise softmax executes on GPU using max subtraction before exponentiation.
- tensor upload/readback counters plus dispatch counters make residency and transfer behavior observable.
- GPU-resident chaining allows `upload → matmul → activation → download` without a CPU roundtrip between neural operations.
- `gpu::HybridComputeBackend` implements the existing L18 `ComputeBackend` interface and selects GPU FP32 matmul above a configurable operation threshold, otherwise using the certified threaded CPU backend.
- INT8 matmul intentionally remains on the existing CPU path at this rung rather than pretending a GPU quantized kernel exists.
- Windows links `d3dcompiler` directly; no third-party GPU abstraction or shader runtime is required.
- `spiral_gpu_compute_tests`: on Windows, requires numerical parity for GPU matmul, ReLU, SiLU, LayerNorm, and softmax; verifies resident multi-op execution and hybrid CPU/GPU routing. On Ubuntu, it requires explicit D3D11 unavailability and deterministic CPU fallback.

L23 does **not** yet claim optimized tiled GEMM, fused transformer kernels, GPU attention, FP16/BF16 GPU arithmetic, or a Linux GPU backend. The first kernels prioritize correctness, inspectability, shared residency, and scheduler integration. L24 can build higher-throughput/tiled kernels and attention primitives on this certified substrate.

No llama.cpp, PyTorch, TensorFlow, OpenCV, PIL, FFmpeg, libsndfile, Stable Diffusion wrapper, pretrained-model runtime, hosted-model API, BLAS library, external thread-pool runtime, SDL renderer, CUDA runtime, DirectML, or third-party GPU abstraction layer is required.

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
- L19 ✅ Spiral Unit graph + constraints + state + events + tool/agent actions + revisioned hot patching + media surfaces + compute routing + prompt-generation boundary
- L20 ✅ deterministic layout + frame tree + hit testing + dirty redraw + host adapters + native device/buffer/command queue reference
- L21 ✅ RGBA framebuffer + software rasterizer + PPM snapshots + snapshot hashing + visual critic + bounded hot-repair loop + EtherPlay/Hakui host bridges
- L22 ✅ Win32 native host + D3D11 hardware/WARP discovery + native GPU buffers + GPU copy/fill + readback + DXGI framebuffer presentation
- L23 ✅ GPU-resident tensors + HLSL FP32 matmul + ReLU/SiLU + LayerNorm + softmax + hybrid CPU/GPU scheduler
- L24 ⬜ tiled GPU GEMM + batched matmul + Q/K/V projections + scaled-dot-product attention + fused transformer inference path

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
