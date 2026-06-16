# Capstone Project 04 — Multimodal Document QA (Vision-First PDF, Tables, Charts)

> The 2026 frontier of document QA has shifted from "OCR first, then process text" to vision-first late interaction. ColPali, ColQwen2.5, and ColQwen3-omni treat every PDF page as an image, embed it with multi-vector late interaction, and let queries attend directly to patches. On financial 10-Ks, scientific papers, and handwritten notes, this paradigm significantly outperforms OCR-first. Build this pipeline end-to-end on 10k pages, then publish a side-by-side comparison report against an OCR-first baseline.

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (viewer UI)
**Prerequisites:** Phase 4 (Computer Vision), Phase 5 (NLP), Phase 7 (Transformers), Phase 11 (LLM Engineering), Phase 12 (Multimodal), Phase 17 (Infrastructure)
**Phases involved:** P4 · P5 · P7 · P11 · P12 · P17
**Time:** 30 hours

## The Problem

Enterprises sit on piles of PDFs mangled by OCR pipelines: scanned 10-Ks with rotated tables, scientific papers dense with formulas, charts that only make sense as images, handwritten annotations. Treating these as text-first means losing half the signal. The 2026 answer is multi-vector retrieval with late interaction on raw page images. ColPali (Illuin Tech) proposed it; ColQwen2.5-v0.2 and ColQwen3-omni push accuracy further. On ViDoRe v3, vision-first retrieval leads OCR-first by a significant margin — and the gap widens on charts, tables, and handwriting.

The cost is storage and latency. A ColQwen embedding is approximately 2048 patch vectors per page, not a single 1024-dimensional vector. Naive storage explodes. DocPruner (2026) delivers 50% pruning with no measurable accuracy loss. You will index 10k pages, measure ViDoRe v3 nDCG@5, return answers within 2 seconds, and directly compare against an OCR-first baseline.

## The Concept

Late interaction means every query token scores against every patch token independently, then the maximum score per query token is summed. You get fine-grained matching without needing a single pooled vector. A multi-vector index (Vespa, Qdrant multi-vector, or AstraDB) stores the per-patch embeddings, and retrieval runs MaxSim.

The answerer is a vision-language model that receives the query plus the top-k retrieved page images and writes an answer with evidence regions (bounding boxes or page references). Qwen3-VL-30B, Gemini 2.5 Pro, and InternVL3 are the 2026 frontier choices. For formulas and scientific notation, an OCR fallback (Nougat, dots.ocr) serves as an optional text channel spliced in.

Evaluation is a two-dimensional matrix. One axis: content type (plain text paragraphs, dense tables, bar/line charts, handwritten notes, formulas). Other axis: retrieval method (vision-first late interaction vs OCR-first vs hybrid). Each cell gets nDCG@5 and answer accuracy. That report is the deliverable.

## Architecture

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## Tech Stack

- Page rendering: PyMuPDF (fitz), 180 DPI, portrait normalized
- Late interaction model: ColQwen2.5-v0.2 or ColQwen3-omni (vidore team on Hugging Face)
- Index: Vespa with multi-vector fields, or Qdrant multi-vector, or AstraDB with MaxSim
- Pruning: DocPruner 2026 strategy (keep high-variance patches, 50% compression, accuracy loss < 0.5%)
- OCR fallback (formulas / dense tables): dots.ocr or Nougat
- VLM answerer: self-hosted Qwen3-VL-30B or hosted Gemini 2.5 Pro; InternVL3 as fallback
- Evaluation: ViDoRe v3 benchmark, M3DocVQA for multi-page reasoning
- Viewer UI: Next.js 15 with canvas-overlaid evidence regions

## Build It

1. **Ingestion.** Walk a 10k-page PDF corpus spanning 10-Ks, scientific papers, and scanned documents. Render each page to a 1536x2048 PNG. Persist `{doc_id, page_num, image_path}`.

2. **Embedding.** Run ColQwen2.5-v0.2 on each page image. Output shape is approximately 2048 patch embeddings of dimension 128. Apply DocPruner to retain the most signal-rich half. Write to Vespa multi-vector field or Qdrant multi-vector.

3. **Query.** For each incoming query, embed with the query tower (token-level embeddings). Run MaxSim against the index: for each query token, take its maximum dot product against the page's patch embeddings, then sum. Return top-k pages.

4. **Synthesis.** Call Qwen3-VL-30B with the query and top-5 page images. Prompt: "Answer using only the provided pages. Cite every assertion by (doc_id, page) and indicate the region (figure, table, paragraph)."

5. **Evidence regions.** Post-process the answer to extract cited regions. If the VLM outputs bounding boxes (Qwen3-VL does), render them as overlays in the viewer.

6. **OCR fallback.** For pages identified as formula-dense (image-variance-based heuristic), run Nougat or dots.ocr and pass the OCR text as an additional channel alongside the image.

7. **Evaluation.** Run ViDoRe v3 (retrieval nDCG@5) and M3DocVQA (multi-page QA accuracy). Also run an OCR-first pipeline on the same corpus with the same synthesizer. Produce a content-type x method matrix.

8. **UI.** Start with a Streamlit prototype; then build a Next.js 15 production viewer with page-by-page evidence region overlays.

## Use It

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## Ship It

`outputs/skill-doc-qa.md` describes the deliverable: a vision-first multimodal document QA system tuned to a specific corpus and evaluated on ViDoRe v3 against an OCR-first baseline.

| Weight | Criterion | How to measure |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA accuracy | Benchmark numbers compared to the OCR text baseline and public leaderboard |
| 20 | Evidence region grounding | Fraction of cited regions that actually contain the answer span |
| 20 | Storage and latency engineering | DocPruner compression ratio, index p95, answer p95 |
| 20 | Multi-page reasoning | Accuracy on a hand-annotated 100-question multi-page set |
| 15 | Source verification experience | Viewer clarity, overlay fidelity, side-by-side comparison tool |
| **100** | | |

## Exercises

1. Measure ColQwen2.5-v0.2 vs ColQwen3-omni on the same corpus. Which pages does one get right that the other misses? Add a "content category" tag to the index and route by type.

2. Prune embeddings aggressively (75%, 90%). Find the compression cliff: the point where ViDoRe nDCG@5 drops below the OCR baseline.

3. Build a hybrid: run OCR-first and ColQwen in parallel, fuse with RRF, then rerank with a cross-encoder. Does hybrid beat either alone? Where does it help most?

4. Replace Qwen3-VL-30B with a smaller VLM (Qwen2.5-VL-7B). Measure the accuracy-per-dollar curve.

5. Add handwritten note support. Render handwriting corpus, embed with ColQwen, measure retrieval. Compare against a handwriting OCR pipeline.

## Key Terms

| Term | What people call it | What it actually is |
|------|---------------------|---------------------|
| Late interaction | "ColPali-style retrieval" | Query tokens independently score against page patches; MaxSim aggregates |
| Multi-vector | "per-patch embeddings" | Each document has many vectors rather than one pooled vector |
| MaxSim | "late interaction scoring" | For each query token, take the maximum similarity against document vectors; sum |
| DocPruner | "patch compression" | 2026 pruning that retains 50% of patches with negligible accuracy loss |
| ViDoRe v3 | "document retrieval benchmark" | The 2026 standard for measuring visual document retrieval |
| Evidence region | "cited bounding box" | A bbox on the source page locating the answer span |
| OCR fallback | "formula channel" | A text pipeline used alongside vision for formula- or table-dense pages |

## Further Reading

- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — reference for late-interaction document retrieval
- [ColPali paper (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — foundational methodology paper
- [ColQwen family on Hugging Face](https://huggingface.co/vidore) — production-ready checkpoints
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — multi-page multimodal RAG baseline
- [Vespa multi-vector tutorial](https://docs.vespa.ai/en/colpali.html) — reference serving stack
- [Qdrant multi-vector support](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — alternative index
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — alternative hosted index
- [Nougat OCR](https://github.com/facebookresearch/nougat) — formula-capable OCR fallback
