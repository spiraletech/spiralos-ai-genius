# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L22 — native window + first real GPU backend

L0–L21 established Spiral-owned language, agents, multimodal generation, temporal media models, accelerated CPU execution, live Spiral Units, deterministic layout, software rasterization, and visual self-repair. L22 crosses the headless boundary on Windows with a real Win32/D3D11 execution path while preserving explicit unsupported behavior on other platforms:

- `gpu::D3D11GpuDevice`: implements the existing L20 `device::Device` contract with a native D3D11 device.
- device creation tries `D3D_DRIVER_TYPE_HARDWARE` first and falls back to Microsoft WARP only when hardware D3D11 is unavailable; capabilities record whether the selected path is hardware-accelerated.
- native D3D11 buffers use owned `ID3D11Buffer` resources with logical-vs-physical byte accounting and raw UAV capability.
- uploads use `UpdateSubresource`; readback uses a staging buffer plus `Map`.
- `CopyBuffer` executes through `ID3D11DeviceContext::CopySubresourceRegion`, including a temporary native resource for same-buffer overlap safety.
- aligned `FillBuffer` regions execute through raw UAV `ClearUnorderedAccessViewUint`; unaligned edge bytes are handled with bounded uploads.
- `host::NativeWindowHost`: owns a real Win32 `HWND`, UTF-8 title conversion, hidden/visible modes, resize, message pumping, and close state.
- `gpu::D3D11FramebufferPresenter`: binds the existing D3D11 device to a DXGI swapchain and presents L21 RGBA8 framebuffer pixels directly to the native window backbuffer.
- swapchain size tracks framebuffer size; presentation records frame count and the exact L21 snapshot hash being displayed.
- `spiral_native_gpu_host_tests`: on Windows, requires D3D11 device creation, native GPU upload/fill/copy/readback, invalid-range rejection, hidden Win32 window creation, DXGI swapchain creation, and two framebuffer presents. On Ubuntu, the same suite requires the D3D11/Win32 path to report unsupported and reject creation cleanly.

L22 does **not** claim a Linux GPU backend, GPU neural-network kernel, or hardware acceleration on every Windows runner. Windows uses hardware D3D11 when available and WARP as an explicit compatibility fallback; the capability record distinguishes them. GPU matmul/attention and heterogeneous CPU/GPU scheduling belong to the next rung.

No llama.cpp, PyTorch, TensorFlow, OpenCV, PIL, FFmpeg, libsndfile, Stable Diffusion/diffusion wrapper, pretrained-model runtime, vector database, external agent framework, hosted-model API, BLAS library, external font rasterizer, external thread-pool runtime, SDL renderer, or third-party GPU abstraction layer is required.

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
- L23 ⬜ GPU matmul + GPU activation/normalization kernels + attention primitives + CPU/GPU scheduler + live-host visual self-repair loop

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
