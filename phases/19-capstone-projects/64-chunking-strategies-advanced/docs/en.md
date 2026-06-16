# Chunking Strategies: Head-to-Head Comparison

> Chunking determines what your retriever can actually surface. Once boundaries are cut wrong, no embedding model, no reranker, and no LLM can repair the damage downstream.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 Lesson 04 (embedding), Lesson 06 (RAG), Lesson 07 (advanced RAG); Phase 19 Track B foundations (Lessons 20-29)
**Time:** ~90 minutes

## Learning Objectives
- Implement five chunking strategies from scratch: fixed-window, sentence, recursive-split, semantic clustering, and structural markdown header.
- Measure recall@k on a fixture corpus with gold-annotated answer spans, and explain why one strategy wins on prose while another wins on technical documentation.
- Read chunk length distributions and identify the failure modes each strategy buries: orphan sentences, symbols cut mid-token, title-only chunks, semantic drift.
- Pick a default strategy for a new corpus without running a benchmark—just inspect three properties: document type, average paragraph length, and whether the format carries explicit structure.

## The Problem

The first step of every RAG pipeline is slicing source documents into pieces: small enough to fit an embedding model, yet large enough for each piece to carry a self-contained idea. Where you cut is not a hyperparameter—it is the ceiling on what the retriever can ever return.

A query asking "what does the budget abort threshold look like" can only succeed if the chunk containing that abort threshold is reachable. If a fixed-window splitter cuts the threshold value away from its surrounding context, the embedding drifts to a different cluster, the BM25 score drops, the reranker sees noise, and the LLM generates a wrong answer. The 2024 paper "LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs" measured a 35 percentage-point absolute gap in retrieval recall from chunking choice alone. A 2025 follow-up on contextual chunk headers narrowed that gap but did not close it.

This lesson builds five strategies side by side, runs them on a fixture corpus with gold-annotated answer spans, and lets you read the recall numbers yourself.

## The Concept

```mermaid
flowchart LR
  Doc[Source Document] --> S1[Fixed Window]
  Doc --> S2[Sentence]
  Doc --> S3[Recursive Split]
  Doc --> S4[Semantic Cluster]
  Doc --> S5[Structural Markdown]
  S1 --> Chunks1[Chunks]
  S2 --> Chunks2[Chunks]
  S3 --> Chunks3[Chunks]
  S4 --> Chunks4[Chunks]
  S5 --> Chunks5[Chunks]
  Chunks1 --> Index[Embedding Index]
  Chunks2 --> Index
  Chunks3 --> Index
  Chunks4 --> Index
  Chunks5 --> Index
  Index --> Eval[Recall@k vs Gold Span]
```

### Fixed-window

The brute-force baseline. Cut every N characters. Optionally add overlap, so that a sentence broken at position N appears in full in the chunk starting from N - overlap. Fast, deterministic, terrible at boundaries. Use it as a control, not a default.

### Sentence

Split at sentence boundaries using a regex or a simple state machine. Pack one or more sentences into a chunk until approaching the target character budget. It never cuts mid-word, but still cuts mid-paragraph or mid-section. This was the default for many early RAG pipelines and remains a reasonable choice for prose lacking other structure.

### Recursive split

The hierarchical strategy popularized by the 2023 wave of libraries. First try splitting on the strongest separator (double newline, paragraph), fall back to the next level (single newline), then to sentence, then to character. Recursion terminates when a chunk fits the budget. Strong on documents with inconsistent structure because it adapts per region.

### Semantic clustering

Embed every sentence. Cluster consecutive sentences sharing the same topic centroid. Cut when the sliding similarity to the centroid drops below a threshold. Boundaries reflect semantics, not characters. Slower to build and dependent on the embedding model, but robust on documents that switch topics mid-paragraph.

### Structural markdown header

For documents with explicit structure (markdown, reStructuredText, RFC-style numbered sections), cut at heading boundaries. Each chunk is the heading plus all content below it until the next same-level or higher-level heading. Minimal chunks per topic, but only usable when the corpus is well-formatted.

### How recall@k measures boundary choices

A gold-annotated query carries the exact character offsets of the answer span in the source document. After chunking, you ask: does any chunk in the retriever's top-k overlap the gold span? Yes gives recall@k = 1 for that query; no gives 0. Average across the query set. Run the same evaluation for each strategy, and the distributional difference tells you which boundary strategy survives on your corpus.

## Build It

`code/main.py` implements:

- `fixed_window(text, size, overlap)` — baseline.
- `sentence_chunks(text, target)` — simple sentence packer.
- `recursive_split(text, separators, target)` — hierarchical recursion.
- `semantic_chunks(text, similarity_threshold)` — centroid clustering based on deterministic mock embeddings.
- `structural_markdown(text)` — heading-aware splitter.
- `mock_embed(text, dim)` — a hash-based embedding so the entire loop runs offline.
- `DenseIndex` — same shape as the hybrid retrieval lesson in Phase 19 Track B.
- `eval_recall(strategy, corpus, queries, k)` — comparison loop.
- A `main()` that runs every strategy on the fixture corpus and prints a recall@k table.

Run:

```bash
python3 code/main.py
```

Output is a small table, one row per strategy, one column per k. Sentence loses on the structured fixture. Structural-markdown wins on the markdown fixture. Recursive holds steady on the mixed fixture because its recursion adapts. Semantic clustering wins on the prose fixture—where there are no structural cues to exploit.

## Failure Modes the Table Cannot Hide

**Orphan sentences.** Sentence packing produces chunks missing a topic sentence. The embedding points to the wrong cluster.

**Symbols cut mid-token.** Fixed-window splits an identifier in half inside code or YAML. Both halves embed as noise.

**Title-only chunks.** Structural markdown emits a chunk containing nothing but `## Title`. Filter them out or prepend the first paragraph of the next chunk.

**Semantic drift.** Semantic clustering cuts too infrequently when the corpus is topically uniform. A 5000-character chunk packs many specific answers into a single blurred embedding. Pair semantic with a hard character ceiling.

**Stale embeddings.** Semantic clustering uses an embedding model. Swap the model, swap the chunks. Pin the chunk model separately from the retrieval model, or rebuild the index together.

## Picking a Default Strategy Without Running a Benchmark

Three properties decide the default chunker for a new corpus.

| Property | Value | Default Strategy |
|----------|-------|---------|
| Document type | Unstructured prose | Recursive split, target 800 |
| Document type | Markdown / RFC / API docs | Structural markdown |
| Document type | Code | AST-aware (out of scope; see Phase 19 Lesson 02) |
| Paragraph length | Long, single-topic | Sentence, target 500 |
| Paragraph length | Short, mixed-topic | Semantic, threshold 0.6 |

When in doubt, choose recursive split. It is the strongest single-strategy baseline.

## Use It

Production practices:

- Run evaluation before shipping a new pipeline; do not blindly trust the strategy your library defaults to.
- Re-run evaluation whenever you swap embedding models or change corpus composition; the winner is corpus-dependent.
- Persist the strategy name in each chunk's metadata so regressions can be attributed later.

## Ship It

Lesson 69's Track F end-to-end RAG system uses the chunker selected here as its first stage. Lesson 68's evaluation framework reads the recall@k in the exact shape returned by `eval_recall` in this lesson. Pick the strategy that wins on your corpus and feed it downstream.

## Exercises

1. Add a sixth strategy: a token-window using `tiktoken` instead of character counts. Compare against fixed-window on the same fixture.
2. Inject 30% code blocks into the prose fixture. Re-run the table. Explain why every strategy except structural markdown drops recall.
3. Replace the deterministic embedding with the real provider from your project. Measure the change in semantic-clustering recall. Report whether the gap between strategies widens or narrows.
4. Add a `summary` field to each chunk: a one-sentence centroid description. Prepend the summary to the chunk body and re-run evaluation. Measure the recall improvement.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Recall@k | "Did we get the right chunk?" | Fraction of queries where any top-k chunk overlaps the gold answer span |
| Chunk overlap | "Sliding window" | Re-inserting the last N characters of the previous chunk into the next |
| Structural splitter | "Heading-aware chunking" | Cut at H1/H2/H3 boundaries; heading text counts toward the chunk |
| Semantic chunker | "Topic-aware chunking" | Embed sentences, cluster by centroid similarity, cut at drift |
| Centroid drift | "Topic switch" | Cosine similarity between the sliding mean and the next sentence drops below threshold |

## Further Reading

- [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs (arXiv 2406.15319)](https://arxiv.org/abs/2406.15319)
- [Anthropic, Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [LlamaIndex, Chunking strategies for production RAG](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
- Phase 11 Lesson 06 — RAG foundations
- Phase 11 Lesson 07 — advanced RAG
- Phase 19 Lesson 65 — hybrid retrieval that ranks the chunks produced here
- Phase 19 Lesson 68 — evaluation framework for scoring strategy choices in production
