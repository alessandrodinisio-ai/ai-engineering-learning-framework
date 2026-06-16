# Capstone Project 02 — RAG Over Codebase (Cross-Repository Semantic Search)

> In 2026, every serious engineering organization runs an internal code search that understands meaning, not just strings. Sourcegraph Amp, Cursor's codebase Q&A, Augment's enterprise graph, Aider's repomap, Pinterest's internal MCP — same shape. Ingest many repositories, parse with tree-sitter, embed function-level and class-level chunks, hybrid search, rerank, answer with citations. This capstone requires you to build a system that can handle 10 repositories, 2 million lines of code, and survive incremental re-indexing on every git push.

**Type:** Capstone
**Languages:** Python (ingestion), TypeScript (API + UI)
**Prerequisites:** Phase 5 (NLP Foundations), Phase 7 (Transformers), Phase 11 (LLM Engineering), Phase 13 (Tools), Phase 17 (Infrastructure)
**Phases involved:** P5 · P7 · P11 · P13 · P17
**Time:** 30 hours

## The Problem

By 2026, every frontier coding agent ships with a codebase retrieval layer because context windows alone cannot solve cross-repository questions. Claude's 1M token context helps, but it does not eliminate the need for ranked retrieval. Naive cosine search on raw chunks breaks down on generated code, monorepo duplication, and the long tail of rarely-imported symbols. The production answer is hybrid search (dense + BM25) on AST-aware chunks, plus a reranker, backed by a symbol-reference graph.

You learn this by indexing a real fleet of repositories — not a tutorial's single repo — and measuring MRR@10, citation faithfulness, and incremental freshness. The failure modes are all infrastructure-level: a 100k-file monorepo, a push that touches half the files, a query that requires crossing four repositories to answer correctly.

## The Concept

An AST-aware ingestion pipeline uses tree-sitter to parse every file, extracts function and class nodes, and chunks at node boundaries rather than fixed token windows. Each chunk gets three representations: a dense embedding (Voyage-code-3 or nomic-embed-code), sparse BM25 terms, and a short natural-language summary. The summary adds a third retrievable modality — when a user asks "how does X do auth," the summary will mention "authz" even if the code only contains `check_permission`.

Retrieval is hybrid. A query fires both dense and BM25 searches in parallel, merges top-k, and passes the union to a cross-encoder reranker (Cohere rerank-3 or bge-reranker-v2-gemma-2b). The reranked list feeds a long-context synthesizer (Claude Sonnet 4.7 with prompt caching, or self-hosted Llama 3.3 70B) instructed to cite every assertion by file and line range. Answers without citations are rejected by a post-filter.

Incremental freshness is the infrastructure hard problem. A git push triggers a diff: which files changed, which symbols changed. Only affected chunks are re-embedded. Affected cross-file symbol edges (imports, method calls) are recomputed. The index stays consistent without reprocessing 2M lines on every commit.

## Architecture

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## Tech Stack

- Parsing: tree-sitter with 17 language grammars (Python, TS, Rust, Go, Java, C++, etc.)
- Dense embedding: Voyage-code-3 (hosted) or nomic-embed-code-v1.5 (self-hosted), bge-code-v1 as fallback
- Sparse index: Tantivy (Rust) with BM25F, field-weighted on symbol names and function bodies
- Vector store: Qdrant 1.12 with hybrid search, or pgvector + pgvectorscale for teams with fewer than 50M vectors
- Chunk summary model: Claude Haiku 4.5 or Gemini 2.5 Flash with prompt caching
- Reranker: Cohere rerank-3 or self-hosted bge-reranker-v2-gemma-2b
- Orchestration: LlamaIndex Workflows for ingestion, LangGraph for the query agent
- Synthesizer: Claude Sonnet 4.7 with prompt caching (1M context)
- Symbol graph: Neo4j (hosted) or kuzu (embedded), storing import and call edges
- Observability: one Langfuse span per retrieval + synthesis step

## Build It

1. **Ingestion walker.** Walk git history on each push hook. Collect changed files. For each file, parse with tree-sitter and extract function and class nodes along with full source ranges. Produce chunk records `{repo, path, start_line, end_line, symbol, body}`.

2. **Chunk summarizer.** Batch chunks into Haiku 4.5 calls with a prompt-cached system preamble. Prompt: "Summarize this function in one sentence, highlighting its public contract and side effects." Store the summary alongside the chunk.

3. **Embedding pool.** Two parallel queues: dense (Voyage-code-3, batch size 128) and summary (same model, but run on summary strings). Write vectors into Qdrant with payload `{repo, path, start_line, end_line, symbol, kind}`.

4. **BM25 index.** Field-weighted Tantivy index: symbol name weight 4, function body weight 1, summary weight 2. Let "find the function named X" queries coexist with "find the function that does X."

5. **Symbol graph.** For each chunk, record edges: import (this file uses symbol Y from repo Z), call (this function calls method M on class C), inheritance. Store in kuzu. Use at query time to expand retrieval across repository boundaries.

6. **Query agent.** A three-node LangGraph. `retrieve` fires dense + BM25 in parallel, deduplicates by (repo, path, symbol). `rerank` runs a cross-encoder on the top-50, keeps top-10. `synth` puts the reranked chunks in context and calls Claude Sonnet 4.7, caching the system prompt, requiring file:line citations.

7. **Citation enforcement.** Parse model output; any assertion without a `(repo/path:start-end)` anchor is flagged and either re-asked or discarded. Return only cited content to the user.

8. **Incremental re-indexing.** On each webhook, compute a symbol-level diff. Re-embed only chunks whose text changed. Recompute symbol edges for chunks whose imports changed. Target metric: on a 2M-line fleet, a 50-file push completes re-indexing within 60 seconds.

9. **Evaluation.** Annotate 100 cross-repo questions with gold-standard file:line answers. Measure MRR@10, nDCG@10, citation faithfulness (fraction of assertions with verifiable anchors), and p50/p99 latency.

## Use It

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## Ship It

The deliverable skill `outputs/skill-codebase-rag.md`. Given a set of repository corpora, it stands up the ingestion pipeline, hybrid index, and query agent, returning a cited answer to any cross-repository question. Grading rubric:

| Weight | Criterion | How to measure |
|:-:|---|---|
| 25 | Retrieval quality | MRR@10 and nDCG@10 on a 100-question holdout set |
| 20 | Citation faithfulness | Fraction of answer assertions with verifiable file:line anchors |
| 20 | Latency and scale | p95 query latency at 10k QPS on the indexed corpus size |
| 20 | Incremental index correctness | Time from git push to searchable for a 50-file commit |
| 15 | Experience and answer formatting | Clickable citations, snippet previews, actionable follow-ups |
| **100** | | |

## Exercises

1. Replace Voyage-code-3 with self-hosted nomic-embed-code. Measure the MRR@10 delta. Report whether the gap closes after reranking is enabled.

2. Inject 20% generated code (LLM-produced boilerplate) into the corpus and re-evaluate. Observe retrieval pollution. Add a "generated" tag to the payload and downweight those hits.

3. Benchmark Qdrant hybrid search vs pgvector + pgvectorscale on your corpus size. Report p99 at batch size 1.

4. Add a sampling-based drift check: re-run the 100-question evaluation weekly. Alert when MRR@10 drops > 5%.

5. Extend to cross-language symbol resolution: a Python function calls a Go service via gRPC. Use the symbol graph to connect them.

## Key Terms

| Term | What people call it | What it actually is |
|------|---------------------|---------------------|
| AST-aware chunking | "function-level splitting" | Splitting code at tree-sitter node boundaries rather than fixed token windows |
| Hybrid search | "dense + sparse" | Running BM25 and vector search in parallel, merging top-k, then reranking |
| Cross-encoder rerank | "second-stage ranking" | A model that scores (query, candidate) pairs together, more accurate than cosine |
| Prompt caching | "cached system prompt" | A 2026 Claude / OpenAI feature that discounts repeated prefix tokens by up to 90% |
| Symbol graph | "code graph" | Cross-file, cross-repo edges for imports, calls, and inheritance |
| Citation faithfulness | "grounded answer rate" | The fraction of assertions a user can verify by clicking the anchor and reading the cited range |
| Incremental re-index | "push-to-searchable time" | Wall-clock time from git push to when changed symbols become queryable |

## Further Reading

- [Sourcegraph Amp](https://ampcode.com) — production-grade cross-repo code intelligence
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — reference deep-dive for this capstone
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter ranked repo view
- [Augment Code enterprise graph](https://www.augmentcode.com) — commercial symbol-graph RAG
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/) — reference implementation
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 details
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — cross-encoder reference
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — internal platform reference
