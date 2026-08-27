# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L8 — native vision foundation

L0 established Spiral-owned tensors/tokenization, L1–L5 built the language model, training, and generation stack, L6 added runtime memory/tools, and L7 added native orchestration. L8 gives Spiral its first owned visual representation and image-generation substrate:

- `RgbImage`: owned RGB8 image storage with checked dimensions, pixel access, and native binary P6 PPM load/save without OpenCV, PIL, or another image runtime.
- `image_to_tensor` / `tensor_to_image`: deterministic RGB8 ↔ normalized `[height,width,3]` SpiralTensor conversion.
- `patchify`: native image-to-patch tokenization for vision-transformer inputs.
- deterministic 2D sine/cosine positional encoding across image patch grids.
- `VisionSelfAttention`: Spiral-owned bidirectional multi-head self-attention so every image patch can attend to every other patch rather than inheriting the language model's causal mask.
- `VisionBlock`: pre-norm bidirectional attention + residual + gated feed-forward + residual.
- `VisionEncoder`: patch projection, configurable vision-block stack, final normalization, per-patch visual embeddings, and pooled image embeddings.
- `CrossModalProjector`: independent text and vision projections into a shared feature space for later multimodal alignment/training.
- `LatentRasterDecoder`: trainable Spiral-owned latent-grid → RGB patch decoder plus deterministic Gaussian latent sampling; this is the first native image-generation substrate, not yet a trained text-to-image model.
- `spiral_vision_tests`: validates PPM round-trip, tensor conversion, patch ordering, deterministic vision weights, bidirectional patch influence, pooled embeddings, cross-modal projection, deterministic latent sampling, and latent-to-RGB reconstruction.

L8 deliberately distinguishes infrastructure from capability: the vision encoder and latent decoder are real native neural components, but their random initial weights do not yet constitute a useful trained image model. Future rungs can train/align them without replacing the owned runtime.

No llama.cpp, PyTorch, TensorFlow, OpenCV, PIL, diffusion wrapper, pretrained-model runtime, vector database, external agent framework, or hosted-model API is required.

## Build

```bash
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
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
- L9 ⬜ multimodal training + image generation + video/audio + Spiral Units
- L10 ⬜ integrated Spiral intelligence

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
