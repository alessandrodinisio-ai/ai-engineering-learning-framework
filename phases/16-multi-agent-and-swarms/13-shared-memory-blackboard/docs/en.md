# Shared Memory and the Blackboard Pattern

> Two approaches coexist in 2026 multi-agent systems: **message pools** (everyone sees everyone's messages, like AutoGen GroupChat or MetaGPT) and **blackboards with subscriptions** (agents subscribe to relevant events, like Context-Aware MCP or the Matrix framework). Both are the only stateful part of a multi-agent system — which means both are where interesting bugs live. The standard failure mode is **memory poisoning**: one agent hallucinates a "fact," other agents accept it as verified, and accuracy degrades gradually in a way far harder to debug than an outright crash. This lesson builds both structures with the standard library, injects a poisoning attack, and demonstrates three mitigations that actually work in production.

**Type:** Learn + Build
**Languages:** Python (standard library, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 min

## The Problem

Multi-agent systems need a place for agents to share facts. One literal option is "pass everything via messages" — but that reinvents shared state with extra copies. Another is "give everyone a global log" — but global logs grow unbounded and are easy to poison. A third is "project a view per agent" — scalable but schema-heavy.

When one agent hallucinates and writes the hallucination into shared state, every downstream agent that reads that state accepts the hallucination as fact. By the time a human notices, the reasoning chain is five steps deep and the root cause is the third message ever written. Debugging multi-agent accuracy degradation is harder than debugging a crash.

This is memory poisoning. It's the second-most documented failure family in the MAST taxonomy (Cemri et al., arXiv:2503.13657), and it's structural: any shared memory design without provenance and without an unwritable verifier will eventually exhibit it.

## The Concept

### Two Main Topologies

**Full message pool.** Every agent reads every message. AutoGen GroupChat and MetaGPT use this. Simple, transparent, inspectable, but scales poorly past ~10 agents because every agent's context fills with other agents' work.

```
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**Blackboard with subscriptions.** Agents declare interest in specific topics; the substrate routes only relevant messages. CA-MCP (arXiv:2601.11595) and the Matrix decentralized framework (arXiv:2511.21686) use this. Scales further but requires upfront schema design for subscriptions to be meaningful.

```
                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘
```

### When Each Dominates

- **Full pool** dominates when agents are few (< 10), heterogeneous, and conversations are short. Reasoning about "who said what" is trivial when everyone sees everything.
- **Blackboard** dominates when agents are many, roles are homogeneous but instances are numerous (swarm), and conversations are long. Routing saves token cost and context pollution.

Production systems often mix: a small full pool at the top level (planning tier) with a blackboard below (worker tier).

### Memory Poisoning, in a Scenario

Three agents work on a research task. Agent A is a retrieval agent. Agent B is a summarizer. Agent C is an analyst.

1. A fetches a page and writes to shared state: "This study reports a 42% accuracy improvement."
2. The fetched page actually said "4.2% improvement." A hallucinated a decimal point.
3. B reads shared state and writes: "Significant 42% accuracy gain reported (source: A)."
4. C reads shared state and writes: "Recommend adoption — 42% improvement is transformative."
5. The final report cites a 42% number that never existed.

No agent crashed. No test failed. The system "worked correctly." The hallucination propagated from one agent's context through shared state into every downstream agent's reasoning.

### Why This Is Structural

Without shared state, Agent A's hallucination stays in A's context. Downstream agents re-fetch or re-derive, potentially catching the error. With naive shared state, A's context becomes everyone's context, and the hallucination gets laundered into fact.

The problem isn't shared state itself — it's **shared state without provenance and without an independent verifier**. Three mitigations target this:

1. **Tag every write with provenance.** Each record in shared state logs who wrote it, when, under what prompt, and (if applicable) what source the agent cited. Downstream agents read with calibrated skepticism based on provenance.
2. **Version writes; treat as append-only.** A correction is a new entry that overrides the old one, not an in-place update. Audit trail is preserved.
3. **Keep at least one agent that cannot write to shared state.** A read-only verifier agent samples entries, re-fetches sources, and flags inconsistencies. Because it can't write to the pool, it can't be poisoned by the pool.

### The Blackboard Precedent (Hayes-Roth, 1985)

The blackboard pattern predates LLM agents by four decades. Hayes-Roth (1985, "A Blackboard Architecture for Control") described specialized "Knowledge Sources" that observe a global blackboard, contribute partial solutions, and trigger other sources. The 2026 blackboard (CA-MCP, Matrix) is the same pattern with LLM agents as knowledge sources and JSON blobs as partial solutions. The old literature already documented solutions for write races, opportunistic control, and consistency that modern systems are rediscovering.

### Projection vs Full View

A pure blackboard gives each subscriber the same projection (filtered by topic). A more aggressive design is **per-agent projection**: each agent gets a view custom-shaped for its role. LangGraph's state reducer is the 2026 standard implementation — the reducer function folds global state into a role-specific slice.

Per-agent projection scales further but requires a schema. Without one, you're ad-hoc rebuilding the projection in each agent's prompt.

### Write-Race Patterns

Multiple agents writing concurrently is a concurrency problem, not just an LLM problem. Three patterns work:

- **Serial writer (single producer).** All writes go through a coordinator agent that serializes them. Simple but a bottleneck.
- **Optimistic concurrency with versioning.** Each record has a version; writers fail-and-retry when the version doesn't match. Classic database technique.
- **Topic partitioning.** Different agents own different topics. No cross-topic races. Requires well-designed partition boundaries.

Most 2026 frameworks default to serial writer because LLM calls are slow enough that races are rare and the bottleneck doesn't hurt.

### The Unwritable Verifier

The most load-bearing mitigation is the read-only verifier. Implementation rules:

- The verifier shares state with the team (reads blackboard or pool).
- The verifier has no write handle to shared state — it can only write to a separate verification channel.
- The verifier independently fetches sources cited in writes. Flags discrepancies.
- The verifier's output is routed to a human or an independent decision agent, never fed back into the pool.

Without this isolation, the verifier's output becomes another entry in the pool, and a poisoned pool poisons the verifier which poisons its verification.

## Build It

`code/main.py` implements both topologies plus a toy poisoning attack and three mitigations in standard-library Python.

- `MessagePool` — thread-safe append-only log with full-read.
- `Blackboard` — topic-keyed pub/sub with per-agent subscriptions.
- `ProvenanceEntry` — each write records (writer, timestamp, prompt_hash, source_uri).
- `PoisoningScenario` — runs a 3-agent research task where Agent A hallucinates a decimal point. Prints the final report.
- `Verifier` — a read-only agent that re-fetches sources and flags inconsistencies. Runs the same scenario with verifier present.

Run:

```
python3 code/main.py
```

Expected output:
- Run 1 (no verifier): the hallucinated 42% propagates to the final report.
- Run 2 (with verifier): the verifier flags the inconsistency, the pool is tagged "flagged," and the final report includes a retraction.

## Use It

`outputs/skill-memory-auditor.md` is a skill that audits any multi-agent system's shared memory design for provenance, versioning, and verifier isolation. Run it on a new multi-agent architecture before production.

## Ship It

For any shared memory design:

- Tag every write with provenance: `(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`.
- Make the log append-only. Corrections are new entries citing the overridden entry.
- Deploy at least one read-only verifier agent with independent source access.
- Route verifier output to a separate channel, not back into the shared pool.
- Monitor the proportion of writes that are "overrides" — that proportion rising is early evidence of hallucination patterns.

## Exercises

1. Run `code/main.py`. Confirm run 1 propagates the hallucination and run 2 catches it.
2. Add a second hallucination: Agent B fabricates a dataset size. The verifier should catch both without per-case tuning.
3. Replace the full pool with a blackboard with topic partitions (`prices`, `summaries`, `analyses`). Which poisoning scenarios does topic partitioning make harder, and which does it not help with?
4. Read Hayes-Roth (1985, "A Blackboard Architecture for Control"). Identify two control patterns from the paper that this lesson doesn't discuss and that 2026 systems would benefit from.
5. Read CA-MCP (arXiv:2601.11595). Map its Shared Context Store to either the MessagePool or Blackboard class in `code/main.py`. What primitives does CA-MCP add on top?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Message pool | "Shared chat history" | Append-only log every agent reads. Fully transparent, scales poorly. |
| Blackboard | "Shared workspace" | Topic-keyed pub/sub. Agents subscribe to relevant topics. Scales further. |
| Provenance | "Who wrote what" | Metadata on each write: writer, timestamp, prompt, source. |
| Memory poisoning | "Hallucination spreading" | One agent's error enters shared state, downstream agents accept it as fact. |
| Append-only | "No in-place updates" | Corrections are new entries overriding old ones. Preserves audit trail. |
| Unwritable verifier | "Independent auditor" | A read-only agent that re-fetches sources and flags inconsistencies. |
| Projection | "Filtered view" | A per-agent view computed from global state. LangGraph reducer is the standard case. |
| Knowledge Source | "Specialized agent" | Hayes-Roth 1985's term for a blackboard participant. |

## Further Reading

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy; memory poisoning is a coordination-failure subfamily
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595) — Shared Context Store coordinating multiple MCP servers
- [Matrix — decentralized multi-agent framework](https://arxiv.org/abs/2511.21686) — message-queue blackboard with no central orchestrator
- [LangGraph state and reducers](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — per-agent projection pattern in production
- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — provenance and verification notes from production deployment
