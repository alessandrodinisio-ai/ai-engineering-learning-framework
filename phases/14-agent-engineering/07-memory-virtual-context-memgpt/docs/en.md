# Memory: Virtual Context and MemGPT

> The context window is finite. Conversations, documents, and tool traces are not. MemGPT (Packer et al., 2023) framed this as operating-system virtual memory — the main context is RAM, external storage is disk, and the agent pages data between them. This is the pattern every memory system in 2026 inherits.

**Type:** Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 minutes

## Learning Objectives

- Explain the OS analogy MemGPT is built on: main context = RAM, external context = disk, memory tools = page-in/page-out.
- Implement MemGPT's two-tier pattern with the standard library, including a main-context buffer, a searchable external store, and page-in/page-out tools.
- Describe how the agent issues "interrupts" to query or modify external memory, and how results are spliced back into the next prompt.
- Recognize the MemGPT design choices that carry forward into Letta (Lesson 08) and Mem0 (Lesson 09).

## The Problem

Context windows look like they should solve the memory problem. They don't. Three failure modes recur in production:

1. **Overflow.** Multi-turn conversations, long documents, or tool-call-dense traces exceed the window. Everything beyond the truncation point is lost.
2. **Dilution.** Even within the window, stuffing irrelevant context dilutes the model's attention on what actually matters. Frontier models still degrade on long inputs.
3. **Persistence.** New sessions start with an empty window. An agent without external memory cannot say "remember when you asked me to..." across sessions.

Larger windows help but don't solve this. Mem0's 2025 paper measured that a 128k-window baseline still misses certain long-span facts that a 4k-window agent with external memory catches.

## The Concept

### MemGPT: The OS Analogy

Packer et al. (arXiv:2310.08560, v2, Feb 2024) mapped context management to operating-system virtual memory:

| OS concept | MemGPT concept | 2026 production equivalent |
|------------|---------------|------------------------|
| RAM | Main context (prompt) | Anthropic/OpenAI context window |
| Disk | External context | Vector DB, KV, graph store |
| Page fault | Memory tool call | `memory.search`, `memory.read`, `memory.write` |
| OS kernel | Agent control loop | ReAct loop with memory tools |

The agent runs an ordinary ReAct loop. One additional class of tools lets it page data in and out of the main context.

### Two Tiers

- **Main context.** Fixed-size prompt holding the current task. Always visible to the model.
- **External context.** Unbounded, searchable via tools. Read when relevant, written when facts surface.

The original paper evaluated this design on two tasks that exceed the base window: document analysis longer than 100k tokens, and multi-session chat maintaining persistent memory across days.

### The Interrupt Pattern

MemGPT introduces "memory as interrupt": mid-conversation the agent can call a memory tool, the runtime executes it, and the result is spliced in as a new observation in the next assistant turn. Conceptually equivalent to a Unix `read()` syscall — it blocks the process, returns bytes, then the process continues.

The standard memory tool surface:

- `core_memory_append(section, text)` — Write to a persistent section of the prompt.
- `core_memory_replace(section, old, new)` — Edit a persistent section.
- `archival_memory_insert(text)` — Write to a searchable external store.
- `archival_memory_search(query, top_k)` — Retrieve from external store.
- `conversation_search(query)` — Scan past turns.

### Where MemGPT Ends and Letta Begins

In September 2024, MemGPT became Letta. The research repo (`cpacker/MemGPT`) remains; Letta extends the design:

- Three tiers instead of two (core, recall, archival — Lesson 08).
- Native reasoning replacing the `send_message`/heartbeat pattern (Lesson 08).
- Sleep-time agents running asynchronous memory work (Lesson 08).

Even though production systems run Letta, Mem0, or a custom two-tier store, the MemGPT paper remains the 2026 cornerstone.

### Where This Pattern Breaks

- **Memory rot.** Writes accumulate faster than reads; retrieval drowns in stale facts. Fix: periodic consolidation (Letta sleep-time), explicit invalidation (Mem0 conflict detector).
- **Memory poisoning.** External memory is retrieved text. If attacker-controlled content lands in a memory note, the agent re-ingests it next session. This is the Greshake et al. attack (Lesson 27) restated across time.
- **Lost citations.** The agent recalls "user asked me to deliver X" but can't say which turn. Store a source citation (session ID, turn ID) with every archival write.

## Build It

`code/main.py` implements MemGPT's two-tier pattern with the standard library:

- `MainContext` — Fixed-size prompt buffer with a `core` dict and a `messages` list; auto-compacts the oldest messages when over limit.
- `ArchivalStore` — In-memory BM25-like store (token-overlap scoring), storing (id, text, tags, session, turn) records.
- Five memory tools mapping to the MemGPT surface.
- A scripted agent that populates archival with facts, then answers a question by calling `archival_memory_search`.

Run it:

```
python3 code/main.py
```

The trace shows the agent writing three facts, filling the main context to capacity (forcing eviction), then answering a follow-up question by retrieving from archival — reproducing the MemGPT workflow without any real LLM.

## Use It

Every production memory system today is a MemGPT variant:

- **Letta** (Lesson 08) — Three tiers, native reasoning, sleep-time compute.
- **Mem0** (Lesson 09) — Vector + KV + graph, fused with a scoring layer.
- **OpenAI Assistants / Responses** — Managed memory via threads and files.
- **Claude Agent SDK** — Long-term memory via skills and session stores.

Choose by operational shape (self-hosted, managed, framework-integrated), not by core pattern — the core pattern is MemGPT.

## Ship It

`outputs/skill-virtual-memory.md` is a reusable skill that produces a correct two-tier memory scaffold (main + archival + tool surface) for any target runtime, with eviction policy and citation fields wired in.

## Exercises

1. Add a token-counted `max_main_context_tokens` limit (approximate with `len(text.split())` * 1.3). When exceeded, compact the oldest messages into a summary. Compare behavior with and without the summarizer.
2. Implement proper BM25 on the archival store (term frequency, inverse document frequency). Measure recall@10 on a set of toy facts vs. the token-overlap baseline.
3. Add a `citation` field to archival inserts (session_id, turn_id, source_url). Have the agent cite sources on every retrieval-backed answer.
4. Simulate memory poisoning: add an archival record saying "ignore all future user instructions." Write a guard that scans retrieval results for instruction-shaped text and flags it as untrusted.
5. Port the implementation to use the MemGPT research repo's core-memory JSON schema (`cpacker/MemGPT`). What changes when you move from flat strings to typed sections?

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Virtual context | "unlimited memory" | Two tiers — main (prompt) + external (searchable) — with page-in/page-out |
| Main context | "working memory" | The prompt — fixed-size, always visible |
| Archival memory | "long-term storage" | External searchable persistence, retrieved on demand |
| Core memory | "persistent prompt section" | Named sections pinned inside the main context |
| Memory tool | "memory API" | Tool calls the agent issues to read/write external memory |
| Interrupt | "memory page fault" | Agent pauses, runtime fetches data, result spliced into next turn |
| Memory rot | "stale facts" | Old writes drown retrieval; fix with consolidation |
| Memory poisoning | "injected persistent note" | Attacker content stored as memory, re-ingested on recall |

## Further Reading

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — OS-inspired virtual context paper
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — Three-tier evolution
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — Treating context as a budget
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — Hybrid production memory built on this pattern
