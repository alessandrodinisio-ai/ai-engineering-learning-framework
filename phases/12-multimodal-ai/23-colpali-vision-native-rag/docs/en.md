# ColPali & Vision-Native Document RAG

> Traditional RAG parses PDFs to text, chunks them, embeds chunks, stores vectors. Every step loses signal: OCR drops chart data, chunking splits table rows, text embeddings ignore figures. ColPali (Faysse et al., July 2024) asked a simpler question: why extract text at all? Embed page images directly via PaliGemma, retrieve with ColBERT-style late interaction, preserving all the layout, figure, font, and formatting signals the document carries. Published benchmarks: 20-40% higher end-to-end accuracy than text RAG on visually rich documents. ColQwen2, ColSmol, and VisRAG extend the pattern. This lesson reads through the vision-native RAG argument and builds a tiny ColPali-style indexer.

**Type:** Build
**Languages:** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Prerequisites:** Phase 11 (LLM Engineering — RAG fundamentals), Phase 12 · 05 (LLaVA)
**Time:** ~180 minutes

## Learning Objectives

- Explain the difference between bi-encoder retrieval (one vector per document) and late-interaction retrieval (multiple vectors per document).
- Describe ColBERT's MaxSim operation and how ColPali generalizes it from text tokens to image patches.
- Build a tiny ColPali-style indexer: page → patch embeddings → MaxSim against query token embeddings → top-k pages.
- Compare ColPali + Qwen2.5-VL generator vs text RAG + GPT-4 on an invoice / financial report use case.

## The Problem

Text RAG on PDFs throws away most of the document. A financial report's Q3 revenue growth is usually in a chart; a medical report's finding is in an annotated image; a legal contract's signature block is a layout fact, not a text fact.

Text RAG pipeline:

1. PDF → text via OCR / pdftotext.
2. Text → 300-500 token chunks.
3. Chunks → bi-encoder embeddings (one vector).
4. User query → embed → cosine similarity → top-k chunks.
5. Chunks + query → LLM.

Five lossy steps. Charts aren't captured. Tables split across chunks. Multi-column layouts flatten. Captions disappear.

ColPali's answer: skip OCR, embed page images directly. Retrieve with ColBERT-style late interaction, letting the model attend to fine-grained patches at query time.

## The Concept

### ColBERT (2020)

ColBERT (Khattab & Zaharia, arXiv:2004.12832) is a text retrieval method. Instead of one vector per document, it produces one vector per token. At query time:

- Query tokens get their own embeddings (N_q vectors).
- Document tokens get embeddings (N_d vectors, typically cached).
- Score = for each query token, take the max cosine similarity with document tokens, then sum: Σ_i max_j cos(q_i, d_j).

This is the MaxSim operation. Each query token "picks" its best-matching document token. The final score is summed.

Upside: strong recall, handles token-level semantics. Downside: N_d vectors per document, expensive storage.

### ColPali

ColPali (Faysse et al., arXiv:2407.01449) applies the ColBERT pattern to images.

- Each page is encoded by PaliGemma (ViT + language) into patch embeddings: N_p vectors per page.
- Each user query (text) is encoded into query token embeddings: N_q vectors.
- Score = Σ_i max_j cos(q_i, p_j), i.e. MaxSim over query text tokens and page image patches.
- Retrieve top-k pages by total score.

At document ingestion: embed each page with PaliGemma, store all patch embeddings. At query time: embed query tokens, compute MaxSim against all stored page embeddings, return top-k pages.

Upside: 20-40% higher end-to-end than text RAG on visually rich documents. Each patch vector captures local layout and content.

Downside: N_p patches per page × 4 bytes float × D-dim vector = storage grows fast. Mitigate with PQ / OPQ quantization.

### ColQwen2 & ColSmol

ColQwen2 (illuin-tech, 2024-2025) swaps PaliGemma for Qwen2-VL. Better base encoder, better retrieval.

ColSmol is the smaller-scale variant for local / edge use. A ~1B parameter ColSmol retriever can run on consumer GPUs.

### VisRAG

VisRAG (Yu et al., arXiv:2410.10594) is a different variant: instead of MaxSim over patches, it pools each page into a single vector using a VLM, then does bi-encoder retrieval. Faster indexing + smaller storage, weaker recall.

Quality vs cost tradeoff: ColPali for quality, VisRAG for scale.

### M3DocRAG

M3DocRAG (Cho et al., arXiv:2411.04952) extends multimodal retrieval to multi-page multi-document reasoning. Retrieve pages across documents, assemble a multi-page context for the VLM.

### ViDoRe — The Benchmark

ColPali's companion benchmark. Visual Document Retrieval Evaluation. Tasks include financial reports, scientific papers, administrative documents, medical records, manuals. Metric: nDCG@5.

ColPali-v1 scores ~80% nDCG@5 on ViDoRe; text RAG on the same documents scores ~50-60%.

### End-to-End RAG Pipeline

For a vision-native RAG:

1. Ingestion: PDF → page images → PaliGemma encode → store all patch embeddings.
2. Query: user text → query token embeddings → MaxSim against all indexed pages → top-k pages.
3. Generation: top-k page images + query → VLM (Qwen2.5-VL or Claude) → answer.

No OCR anywhere. Figures, charts, fonts, layout all flow into the answer.

### Storage Math

A 50-page financial report, 729 patches per page, 128-dim embeddings:

- ColPali: 50 * 729 * 128 * 4 bytes = ~18 MB raw, ~4 MB after PQ.
- Text RAG: 50 chunks * 768 dims * 4 bytes = ~150 kB.

ColPali is ~30x more storage per document. At scale, OPQ / PQ brings it down to ~5-10x, usually acceptable.

### When Text RAG Still Wins

- Pure-text documents with no layout signal (wiki articles, chat logs). Text RAG is simpler, cheaper storage.
- Multi-million-page archives where storage dominates cost.
- Strict regulatory requirements demanding extractable OCR text beyond retrieval.

By 2026, everything else — financial reports, scientific papers, legal contracts, medical records, UX docs — vision-native RAG wins.

## Use It

`code/main.py`:

- Toy patch encoder: maps a "page" (small grid of feature vectors) to an array of patch embeddings.
- MaxSim scorer: computes ColBERT-style score between a set of query token embeddings and a page's patch set.
- Indexes 5 toy pages, runs 3 queries, returns top-k with scores.

## Ship It

This lesson produces `outputs/skill-vision-rag-designer.md`. Given a document RAG project, it picks ColPali / ColQwen2 / VisRAG / text RAG and specs the storage.

## Exercises

1. A 200-page annual report, 729 patches per page, 128-dim embeddings, 4-byte floats. Calculate raw storage and storage after PQ compression (8x).

2. MaxSim is Σ_i max_j cos(q_i, p_j). What does it capture that simple mean similarity cannot?

3. ColPali indexes pages as patch sets. What if we index at the word level instead (like ColBERT)? What are the tradeoffs?

4. Design an end-to-end pipeline for a 1M-page corpus, 500ms latency budget per query. Pick ColQwen2 / VisRAG and justify.

5. Read M3DocRAG (arXiv:2411.04952). Describe the multi-page attention pattern and how it differs from single-page ColPali retrieval.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Late interaction | "ColBERT-style" | Retrieval using per-token or per-patch embeddings + MaxSim, not a single document vector |
| MaxSim | "max over patches" | For each query token, pick the highest-similarity document token; sum across query |
| Bi-encoder | "single-vector" | One vector per document; faster but loses granularity |
| Multi-vector | "multiple vectors per doc" | Store N_p vectors per document/page; storage cost rises but recall improves |
| Patch embeddings | "page features" | One vector per image patch from the VLM encoder, cached per page |
| ViDoRe | "visual document benchmark" | ColPali's visual document retrieval benchmark suite |
| PQ quantization | "product quantization" | Compression that shrinks storage ~8x while preserving vector similarity |

## Further Reading

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
