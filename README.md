# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L21 — native visual self-test

L0–L20 established Spiral-owned language, agents, multimodal generation, temporal media models, accelerated CPU execution, the live Spiral Unit graph, deterministic layout, dirty rendering, and a hardware-neutral device contract. L21 adds a deterministic software framebuffer and the first native render → inspect → repair → rerender loop:

- `raster::Framebuffer`: owned RGBA8 pixels with checked access, region clears, stable 64-bit snapshot hashing, non-background accounting, and binary P6 PPM snapshot output.
- `raster::SoftwareRenderer`: headless `UnitRenderer` implementation for containers/grids/canvas, text, buttons, deterministic image/video/avatar placeholders, audio surfaces, and waveform rendering.
- dirty-region rendering preserves the L20 bridge contract: only invalidated regions are cleared and repainted.
- built-in tiny 5x7 UI glyph rasterization keeps text snapshots deterministic without external font/runtime dependencies.
- `visual::VisualCritic`: inspects the resolved frame tree and rendered pixels for undersized interactive targets, heavy clipping, and nearly blank frames.
- `visual::RepairPolicy`: emits revision-safe `UnitPatch` repairs only for findings it knows how to fix; the first certified repair promotes undersized interactive controls to a 44x44 minimum.
- `visual::VisualSelfTest`: executes inspect → patch → hot render → reinspect with bounded repair passes and preserves every report/snapshot hash.
- `host::HostBridgeLog` plus `make_etherplay_adapter` / `make_hakui_adapter`: concrete presentation bridges that let EtherPlay and Hakui claim their supported runtime/media nodes through the same L20 adapter contract.
- `spiral_visual_self_test_tests`: renders an intentionally flawed live Unit, saves/reads a real PPM snapshot, proves deterministic hashes, detects a 20x14 button, hot-repairs it to the configured minimum, verifies the framebuffer changes, dispatches a repaired button event, and requires the final critic report to be healthy.

L21 is a real software rasterizer and self-test loop, but it deliberately does **not** claim a native OS window, SDL host, or GPU renderer. The framebuffer is headless so CI can certify exact cross-platform visual behavior first. Hakui/EtherPlay clients can consume the host bridge now; the next rung can bind a native window/GPU backend without changing the Unit, visual-critic, or repair contracts.

No llama.cpp, PyTorch, TensorFlow, OpenCV, PIL, FFmpeg, libsndfile, Stable Diffusion/diffusion wrapper, pretrained-model runtime, vector database, external agent framework, hosted-model API, BLAS library, external font rasterizer, or external thread-pool runtime is required.

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
- L22 ⬜ native window host + first real GPU backend + GPU buffers/commands + raster presentation
- L23 ⬜ GPU matmul/attention + heterogeneous model scheduling + visual AI self-repair in live hosts

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
