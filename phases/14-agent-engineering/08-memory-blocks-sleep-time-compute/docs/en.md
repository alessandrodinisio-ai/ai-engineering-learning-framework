# Memory Blocks and Sleep-Time Compute (Letta)

> MemGPT became Letta in 2024. The 2026 evolution adds two ideas: discrete functional memory blocks that the model can directly edit, and a sleep-time agent that asynchronously consolidates memory while the primary agent is idle. This is how you scale memory beyond a single conversation.

**Type:** Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 07 (MemGPT)
**Time:** ~75 minutes

## Learning Objectives

- Name the three memory tiers Letta uses (core, recall, archival) and the role of each.
- Explain the memory block pattern: Human block, Persona block, and user-defined blocks as first-class typed objects.
- Describe what sleep-time compute is, why it runs off the critical path, and why it can run a stronger model than the primary agent.
- Implement a scripted dual-agent loop: a primary agent serving responses and a sleep-time agent consolidating blocks between turns.

## The Problem

MemGPT (Lesson 07) solved control flow for virtual memory. Three more production problems surfaced:

1. **Latency.** Every memory operation sits on the critical path. If the agent must prune, summarize, or reconcile while the user waits, tail latency explodes.
2. **Memory rot.** Writes accumulate. Superseded facts remain. Retrieval drowns in stale content.
3. **Lost structure.** A flat archival store cannot express "Human block is always in the prompt; Persona block is always in the prompt; Task block rotates per session."

Letta (letta.com) is the 2026 rewrite. Memory blocks make structure explicit; sleep-time compute moves consolidation off the critical path.

## The Concept

### Three Tiers

| Tier | Scope | Where it lives | Written by |
|------|-------|----------------|------------|
| Core | Always visible | Inside the main prompt | Agent tool calls + sleep-time rewrites |
| Recall | Conversation history | Retrievable | Automatic turn logging |
| Archival | Arbitrary facts | Vector + KV + graph | Agent tool calls + sleep-time ingestion |

Core is MemGPT's core. Recall is the conversation buffer and its evicted tail. Archival is external storage. This split untangles the responsibility overload in MemGPT's two tiers.

### Memory Blocks

A block is a typed, persistent, editable section inside the core tier. The original MemGPT paper defined two:

- **Human block** — Facts about the user (name, role, preferences, goals).
- **Persona block** — The agent's self-concept (identity, tone, constraints).

Letta generalizes to arbitrary user-defined blocks: a `Task` block for the current objective, a `Project` block for codebase facts, a `Safety` block for hard constraints. Each block has `id`, `label`, `value`, `limit` (character cap), `description` (so the model knows when to edit it).

Blocks are editable via the tool surface:

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` — Compress a block approaching its limit.

### Sleep-Time Compute

A 2025 Letta addition: run a second agent in the background, off the critical path. The sleep-time agent processes conversation transcripts and codebase context, writes `learned_context` into shared blocks, and consolidates or invalidates archival records.

Properties that follow:

- **No latency cost.** The primary response doesn't wait for memory operations.
- **Allows a stronger model.** The sleep-time agent can be a more expensive, slower model because it's not latency-bound.
- **Natural consolidation window.** Deduplicate, summarize, and invalidate superseded facts while the user isn't waiting.

The shape fits how humans work: you do the task, you sleep, long-term memory consolidates overnight.

### Letta V1 and Native Reasoning

Letta V1 (`letta_v1_agent`, 2026) deprecated `send_message`/heartbeat and inline `Thought:` tokens in favor of native reasoning. The Responses API (OpenAI) and Messages API with extended thinking (Anthropic) emit reasoning on a separate channel, passed through across turns (encrypted cross-vendor in production). The control loop is still ReAct. Thinking traces are structured, not prompt-shaped.

### Where This Pattern Breaks

- **Block bloat.** Unbounded `block_append` hits the limit quickly. Wire a block summarizer before the write that would exceed the cap.
- **Silent drift.** The sleep-time agent rewrites a block and the primary agent is none the wiser. Version blocks and expose diffs in the trace.
- **Poisoned consolidation.** The sleep-time agent processes attacker-reachable content into core. Lesson 27 applies equally to the sleep-time surface.

## Build It

`code/main.py` implements:

- `Block` — id, label, value, limit, description.
- `BlockStore` — CRUD + `near_limit(label)` helper.
- Two scripted agents — `PrimaryAgent` serves one turn, `SleepTimeAgent` consolidates between turns.
- A trace showing a three-turn conversation with block writes, plus a sleep-time pass that summarizes a block and invalidates a stale fact.

Run it:

```
python3 code/main.py
```

The transcript shows the split: primary turns are fast and produce raw writes; the sleep pass does compaction and cleanup.

## Use It

- **Letta** (letta.com) as the reference implementation. Self-hosted or managed cloud.
- **Claude Agent SDK skill** as block-shaped knowledge — a skill is a named, versioned, retrievable block of instructions that the agent loads on demand.
- **Custom builds** for teams that want to own the storage backend. Use Letta's API contract so you can migrate later.

## Ship It

`outputs/skill-memory-blocks.md` generates a Letta-shaped block system for any runtime with sleep-time hooks, including safety rules and citation wiring.

## Exercises

1. Add a `block_summarize` tool that replaces a block's value with a model-generated summary when `near_limit` returns true. What trigger threshold minimizes both summarization calls and block overflow?
2. Implement sleep-time deduplication on archival: two records with >90% token overlap are merged into one. Only during the sleep pass, never on the critical path.
3. Version blocks. Every write records the old value and a diff. Expose `block_history(label)` so operators can debug "why did the agent forget X."
4. Treat the sleep-time agent as an untrusted writer. When it touches Persona or Safety blocks, require a second agent to review before committing.
5. Port the example to use the Letta API (`letta_v1_agent`). What changes in the block schema, and how does native reasoning alter the trace shape?

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Memory block | "editable prompt section" | A typed, persistent, LLM-editable segment in core memory |
| Human block | "user memory" | Facts about the user, pinned in core |
| Persona block | "agent identity" | Self-concept, tone, constraints, pinned in core |
| Sleep-time compute | "async memory work" | A second agent consolidating off the critical path |
| Core / Recall / Archival | "tiers" | Three-tier memory split: always-visible / conversation / external |
| Block limit | "cap" | Per-block character limit; forces summarization |
| Native reasoning | "thinking channel" | Vendor-level reasoning output, not prompt-level `Thought:` |
| Learned context | "sleep output" | Facts the sleep-time agent writes into shared blocks |

## Further Reading

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — Block pattern
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute) — Async consolidation
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) — Native reasoning rewrite
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — The origin
