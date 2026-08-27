# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L20 — renderer + device bridge

L0–L19 established Spiral-owned language, agents, multimodal generation, temporal media models, accelerated CPU execution, and the live Spiral Unit graph. L20 gives that live graph a deterministic screen/device contract without hardcoding one client renderer:

- `render::LayoutEngine`: resolves validated Spiral Units into concrete clipped frame trees.
- deterministic vertical flex layout plus `Grid` layout through a `columns` property; absolute children remain parent-relative and min/max constraints are honored.
- `FrameTree::hit_test`: back-to-front interactive hit testing against resolved bounds and inherited clip rectangles.
- `UnitRenderer`: host-neutral begin/draw/end rendering interface receiving the resolved frame tree, dirty regions, and optional native device binding.
- `RendererBridge`: mounts a `UnitRuntime`, dispatches pointer events, recomputes layout, tracks dirty geometry/property changes, applies revisioned hot patches, resizes, and performs full redraws.
- `HostSurfaceAdapter`: named adapter hook used by clients such as EtherPlay and Hakui to claim media/runtime component kinds without forking Spiral Units.
- `device::Device`: native hardware-neutral buffer/upload/download/submit contract.
- `device::CpuReferenceDevice`: real buffer storage plus validated fill/copy command execution used as the reference contract for later GPU implementations.
- `device::CommandList` / `CommandQueue`: explicit queued device work with submission accounting.
- `spiral_renderer_device_tests`: proves device buffer commands and bounds safety, deterministic grid/flex layout, hit testing, EtherPlay/Hakui adapter dispatch, event-driven dirty redraw, hot-layout propagation, stale/corrupt patch safety, resize/redraw, and renderer access to the native device.

L20 deliberately does **not** claim a GPU kernel yet. The device/buffer/queue API and CPU reference implementation define the semantics that a later Direct3D/Vulkan/Metal-style Spiral backend must match. Likewise, the renderer is host-neutral: Hakui/SDL and EtherPlay native clients can now bind concrete drawing/media behavior without changing the Unit/agent/model layers.

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
- L19 ✅ Spiral Unit graph + constraints + state + events + tool/agent actions + revisioned hot patching + media surfaces + compute routing + prompt-generation boundary
- L20 ✅ deterministic layout + frame tree + hit testing + dirty redraw + host adapters + native device/buffer/command queue reference
- L21 ⬜ concrete Hakui/EtherPlay host renderer + visual self-test/repair + first native GPU backend

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
