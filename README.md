# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L19 — Spiral Units unified runtime

L0–L18 established Spiral-owned language, agents, multimodal generation, temporal learning, media decoders, prompt-conditioned media flow, and accelerated CPU execution. L19 adds the live application/world representation that lets those subsystems operate through one common runtime instead of remaining isolated libraries:

- `units::SpiralUnit`: deterministic component/state document with stable component IDs, explicit roots/children, layout constraints, properties, media surfaces, and event bindings.
- `ComponentKind`: container/text/button plus image, audio, video, waveform, canvas, grid, avatar, and custom runtime nodes.
- strict structural validation: unique IDs, valid roots/children, finite layout values, reachability, duplicate-child rejection, and cycle detection.
- `UnitPatch`: revision-checked hot patch operations for components, roots, state, and properties.
- copy → patch → validate → commit semantics: stale or structurally invalid patches cannot partially corrupt the live unit.
- event actions: set state, set another component property, invoke `ToolRegistry`, invoke `AgentEngine`, and emit application events.
- `$payload` and `$state.<key>` value resolution for event-driven live data flow.
- `UnitRuntime::matmul`: compute-bound execution through L18 `ComputeBackend`, with the original Tensor path as a portable fallback.
- `UnitGenerator`: pluggable prompt → `SpiralUnit` generation boundary; generated units must pass the same native validator before becoming live.
- explicit image/audio/video/waveform surface kinds establish the renderer/media integration contract without pretending Hakui or EtherPlay rendering code already lives in this repository.
- `spiral_units_tests`: proves live state/property mutation, tool and agent execution, emitted events, compute routing, hot patching, stale patch rejection, invalid-patch rollback, prompt-generated unit loading, media surface identity, and graph-cycle rejection.

L19 is the convergence/runtime-schema rung, not a finished visual renderer. It gives Hakui, EtherPlay, and future SpiralOS clients one stable executable document and event model to render and control. The next integration work can bind these nodes to SDL/Hakui widgets, EtherPlay audio/video surfaces, or a future GPU renderer without changing the AI/agent/unit contract.

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
- L20 ⬜ renderer bridges + native GPU device/buffer/queue contract + first GPU kernels
- L21 ⬜ full Hakui/EtherPlay/SpiralOS live Unit host + visual self-test/repair loop

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
