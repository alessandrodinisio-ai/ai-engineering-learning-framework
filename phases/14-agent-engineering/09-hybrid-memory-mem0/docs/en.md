# Hybrid Memory: Vector + Graph + KV (Mem0)

> Mem0 (Chhikara et al., 2025) treats memory as three parallel stores — vector for semantic similarity, KV for fast fact lookup, graph for entity-relationship reasoning. At retrieval time a scoring layer fuses all three. This is the 2026 production standard for external memory.

**Type:** Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 minutes

## Learning Objectives

- Explain why a single store (vector-only, graph-only, KV-only) is insufficient for agent memory.
- Name Mem0's three parallel stores and what each is optimized for.
- Describe Mem0's fusion scoring — relevance, importance, recency — and why it's a weighted sum rather than hierarchical.
- Implement a toy three-store memory with the standard library where `add()` writes to all three and `search()` fuses results.

## The Problem

A single store is wrong for one of three query classes:

- **Semantic similarity** — "What did we discuss about agent drift last week?" Vector wins; KV and graph miss.
- **Fact lookup** — "What is the user's phone number?" KV wins; vector is wasteful, graph is overkill.
- **Relationship reasoning** — "Which customers share the same billing entity?" Graph wins; vector and KV can't answer.

Production agents issue all three classes within a single session. Single-store memory is always wrong for two of them. Mem0's contribution is wiring all three behind a single `add`/`search` surface with a scoring function that fuses them.

## The Concept

### Three Parallel Stores

Mem0 (arXiv:2504.19413, April 2025) on `add(text, user_id, metadata)`:

1. Extracts candidate facts from text (an LLM-driven step).
2. Writes each fact to the vector store (embedding) for semantic search.
3. Writes each fact to the KV store keyed by (user_id, fact_type, entity) for O(1) lookup.
4. Writes each fact as typed edges to the graph store (Mem0g) for relationship queries.

On `search(query, user_id)`:

1. Vector store returns top-k by embedding cosine.
2. KV store returns direct hits for (user_id, type, entity) derived from the query.
3. Graph store returns subgraph reachable from query entities.
4. A scoring layer fuses all three.

### Fusion Scoring

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Relevance** — Vector cosine, KV exact match, graph path weight.
- **Importance** — Tagged at write time or learned (some facts matter more: names, IDs, policies).
- **Recency** — Exponential time decay since last write or read.

Weights are tuned per product. Chat agents boost `w_recency`; compliance agents boost `w_importance`; retrieval agents boost `w_relevance`.

### Mem0g and Temporal Reasoning

Mem0g adds a conflict detector. When a new fact contradicts an existing edge, the existing edge is marked invalid but not deleted. Temporal queries ("What city was the user in during March?") traverse the subgraph "valid at that time."

This is the compliance-grade behavior that generalizes Letta's invalidation pattern.

### Benchmark Numbers

Mem0 paper reports (2025):

- **LoCoMo** (long-conversation memory): 91.6
- **LongMemEval** (long-span episodic memory): 93.4
- **BEAM 1M** (1M-token memory benchmark): 64.1

Compared to baselines (full-context 128k LLM, flat vector store, flat KV), all trail by 10+ points. Benchmarks alone don't dictate selection — operational shape does — but these numbers indicate the fusion design is not a rounding error.

### Scope Classification

Mem0 splits memory by scope:

- **User memory** — Persistent across sessions, keyed by `user_id`.
- **Session memory** — Persistent within a thread.
- **Agent memory** — Per-agent-instance state.

Every write picks a scope. Retrieval can query across scopes, each with its own weight. Mixing scopes carelessly is how you get "assistant told Alice about Bob's project" incidents.

### Where This Pattern Breaks

- **Embedding drift.** Vector results that look correct for the first hundred queries degrade as the corpus grows. Add periodic re-embedding of the top-N most-accessed records.
- **KV schema sprawl.** `(user_id, type, entity)` looks simple until every team adds its own `type`. Audit the type set quarterly.
- **Graph explosion.** A noisy extractor adds 50 edges per message. Cap graph writes per `add` call; drop low-confidence edges.

## Build It

`code/main.py` implements the three-store pattern with the standard library:

- `VectorStore` — Naive token-overlap similarity standing in for embeddings.
- `KVStore` — Dict keyed by `(user_id, fact_type, entity)`.
- `GraphStore` — Typed edges (subject, relation, object, valid).
- `Mem0` — Top-level facade with `add()`, `search()`, fusion scoring, and scope-aware retrieval.
- A full trace on a multi-user, multi-session conversation.

Run it:

```
python3 code/main.py
```

Output shows three independent recall paths plus the fused top-k. Flip the scoring weights at the top of `main()` to see ranking changes.

## Use It

- **Mem0 (Apache 2.0)** — Production-ready. Self-host with Postgres + Qdrant + Neo4j, or use managed cloud.
- **Letta** — Three-tier core/recall/archival; ships with vector and graph backends.
- **Zep** — Commercial alternative with temporal KG and fact extraction.
- **Custom builds** — When you need precise control over the extractor (compliance) or fusion weights (recency-dominated voice agents).

## Ship It

`outputs/skill-hybrid-memory.md` generates a three-store memory scaffold with fusion scorer, scope classification, and temporal invalidation wired in.

## Exercises

1. Replace the toy vector similarity with a real embedding model (sentence-transformers, Ollama, OpenAI embeddings). Measure recall@10 on a synthetic long conversation. Do rankings drift after 1000 writes?
2. Add a temporal query: `search(query, as_of=timestamp)`. Return only records valid at or before that time. Which store requires the most changes?
3. Implement a conflict detector: if an incoming fact contradicts a graph edge, invalidate the old edge and log both. Test on "user lives in Berlin" -> "user lives in Lisbon."
4. Port the fusion scorer to include a `user_feedback` dimension (thumbs-up on retrieved records). How do you prevent gaming (the agent only returns records it already likes)?
5. Read the Mem0 docs (`docs.mem0.ai`). Port the toy to `mem0` client calls. Compare retrieval quality on the same 20 test queries.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Hybrid memory | "vector plus graph plus KV" | Three stores written in parallel, fused at retrieval |
| Fact extraction | "memory ingestion" | LLM step that splits text into (entity, relation, fact) triples |
| Fusion scoring | "relevance ranking" | Weighted sum of relevance, importance, and recency |
| Scope | "memory namespace" | user / session / agent — determines who sees what |
| Mem0g | "memory graph" | Typed edges with temporal validity for relationship queries |
| Temporal invalidation | "soft delete" | Marking superseded edges invalid; never deleting |
| Embedding drift | "retrieval rot" | Vector quality degrades as corpus grows; periodic re-embedding |

## Further Reading

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — Original paper
- [Mem0 docs](https://docs.mem0.ai/platform/overview) — Production API, SDK, managed cloud
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — Virtual context predecessor
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — Sibling three-tier design
