# SpiralOS AI Genius

Native, sovereign C++ intelligence infrastructure for SpiralOS, Hakui, EtherPlay, and future EtherTech systems.

## Current rung: L7 — native agent brain

L0 established Spiral-owned tensors and tokenization. L1 added neural layers. L2 added causal attention. L3 assembled the language model. L4 added native training. L5 added decoding. L6 added incremental inference, memory, retrieval, and tools. L7 turns those capabilities into an explicit execution architecture:

- `IntentParser`: deterministic intent classification plus discovery of candidate native tools from the current registry.
- `TaskGraph`: dependency-aware task nodes with duplicate-ID, unknown-dependency, self-dependency, and cycle validation.
- `AgentContext`: injects the current goal, parsed intent, retrieved persistent memories, and available tool descriptions into planning policy.
- `AgentEngine`: plan → execute → critique → repair/retry → verify → remember loop with bounded attempts per task.
- planner, critic, and repair policies are injectable native interfaces. Deterministic fallback policies work today; future Spiral language models can occupy these policy slots without replacing the orchestration engine.
- dependency failures skip downstream tasks rather than executing invalid work.
- successful tool calls can still be rejected by the critic, forcing a repaired attempt rather than treating API success as semantic success.
- `TraceEvent`: ordered intent, memory, planning, execution, critic, repair, skip, verification, and outcome-memory events for every run.
- successful and failed agent outcomes can be written back into `MemoryStore` as episodic records for later retrieval/evaluation.
- `spiral_agent_tests`: validates memory injection, deterministic tool planning, multi-step output chaining, critic-triggered repair, retry accounting, outcome memory, task-cycle rejection, and graceful no-action failure.

L7 does not claim that the current tiny/random model is already a capable planner. Instead, it establishes the owned agent architecture and explicit policy boundaries required to plug trained Spiral models into planning and criticism later without turning the runtime into a model-specific wrapper.

No llama.cpp, PyTorch, TensorFlow, external agent framework, vector database, pretrained-model runtime, or hosted-model API is required.

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
- L8 ⬜ vision + image generation
- L9 ⬜ video/audio + Spiral Units
- L10 ⬜ integrated Spiral intelligence

The rule is simple: external systems may be studied and benchmarked, but Spiral owns its core abstractions and can replace any optional dependency.
