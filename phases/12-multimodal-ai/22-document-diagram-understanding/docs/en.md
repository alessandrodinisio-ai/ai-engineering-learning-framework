# Document and Diagram Understanding

> Documents are not photographs. A PDF, scientific paper, invoice, or handwritten form has layout, tables, diagrams, footnotes, headers, and semantic structure that pure image understanding doesn't capture. The pre-VLM stack was a pipeline: Tesseract OCR + LayoutLMv3 + table extraction heuristics. The VLM wave replaced it with OCR-free models — Donut (2022), Nougat (2023), DocLLM (2023) — that directly emit structured markup. By 2026, the frontier is simply "feed the page image at native 2576px to Claude Opus 4.7," and structured markup output comes for free. This lesson reads through the three-era arc of document AI.

**Type:** Build
**Languages:** Python (stdlib, layout-aware document parser skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**Time:** ~180 minutes

## Learning Objectives

- Explain the three eras of document AI: OCR pipeline, OCR-free, VLM-native.
- Describe LayoutLMv3's three input streams: text, layout (bbox), image patches, with unified masking.
- Compare Donut (OCR-free, image → markup), Nougat (scientific papers → LaTeX), DocLLM (layout-aware generative), PaliGemma 2 (VLM-native).
- Pick a document model for a new task (invoices, scientific papers, handwritten forms, Chinese receipts).

## The Problem

"Understand this PDF" is deceptively hard. Information resides in:

- Text content (90% of the signal).
- Layout (headers, footnotes, sidebars, two-column formatting).
- Tables (rows, columns, merged cells).
- Figures and diagrams.
- Handwritten annotations.
- Font and typography (heading vs body).

Raw OCR dumps text and discards the rest. A system that cares about invoices needs to know that "Total: $1,245" comes from the bottom-right corner, not from a footnote.

## The Concept

### Era 1 — OCR Pipeline (pre-2021)

The classic stack:

1. PDF → one image per page.
2. Tesseract (or commercial OCR) extracts text with per-word bounding boxes.
3. Layout analyzer identifies blocks (header, table, paragraph).
4. Table structure recognizer parses tables.
5. Domain rules + regex extract fields.

Works for clean printed text. Breaks on handwriting, skewed scans, complex tables, non-English scripts. Every failure mode requires a custom exception path.

### TrOCR (2021)

TrOCR (Li et al., arXiv:2109.10282) replaced Tesseract's classic CNN-CTC with a transformer encoder-decoder, trained on synthetic + real text images. A clean win on handwriting and multilingual text. Still a pipeline (detector then TrOCR then layout), but the OCR step improved dramatically.

### Era 2 — OCR-Free (2022–2023)

The first OCR-free models said: skip detection entirely, map image pixels directly to structured output.

Donut (Kim et al., arXiv:2111.15664):
- Encoder-decoder transformer, encoder is Swin-B.
- Output is JSON for form understanding, markdown for summarization, or any task-specific schema.
- No OCR, no layout, no detection.

Nougat (Blecher et al., arXiv:2308.13418):
- Trained specifically on scientific papers.
- Output is LaTeX / markdown.
- Handles equations, multi-column layout, figures.
- The model every arXiv parser calls.

They are specialists, not generalists. Donut fails on scientific papers; Nougat fails on invoices.

### LayoutLMv3 (2022)

A different track. LayoutLMv3 (Huang et al., arXiv:2204.08387) keeps OCR but adds layout understanding:

- Three input streams: OCR text tokens, per-token 2D bounding boxes, image patches.
- Masked training objective across all three modalities (mask text, mask patches, mask layout).
- Downstream: classification, entity extraction, table QA.

LayoutLMv3 is the apex of OCR-based document understanding. Strong on forms and invoices. Requires upstream OCR. Best pre-VLM accuracy on standardized document benchmarks.

### DocLLM (2023)

DocLLM (Wang et al., arXiv:2401.00908) is the generative sibling of LayoutLM. Generates free-form answers conditioned on layout tokens. Better at document QA; still relies on OCR input.

### Era 3 — VLM-Native (2024+)

2024 VLMs are good enough to replace the pipeline entirely. Feed the full page image at high resolution to a VLM, ask a question, get an answer.

- LLaVA-NeXT 336-tile AnyRes works for small documents.
- Qwen2.5-VL dynamic resolution natively handles 2048+ pixels.
- Claude Opus 4.7 supports 2576px documents.
- PaliGemma 2 (April 2025) trained specifically for documents + handwriting.

The gap between VLM-native and OCR pipeline is closing fast. By 2026, VLM-native wins on:

- Scene text (handwritten + printed, mixed scripts).
- Complex tables with merged cells.
- Math equations embedded in text.
- Diagrams with text annotations.

OCR pipeline still wins on:

- Massive-scale, pure-scan workloads where per-page latency matters.
- Pipeline reliability (deterministic failures vs VLM hallucinations).
- Regulated environments requiring auditable OCR output.

### Claude 4.7 / GPT-5 Frontier

At 2576-pixel native input, frontier VLMs do document understanding at near-human accuracy. 2026 benchmark numbers:

- DocVQA: Claude 4.7 ~95.1, PaliGemma 2 ~88.4, Nougat ~77.3, pipelined LayoutLMv3 ~83.
- ChartQA: Claude 4.7 ~92.2, GPT-4V ~78.
- VisualMRC: Claude 4.7 ~94.

The gap with closed models is mostly resolution and base LLM scale. Open 7B models trail by a few points but are catching up.

### Math Equations and LaTeX Output

Scientific papers need precise LaTeX output for equations. Nougat is trained on this. VLMs trained with LaTeX targets (Qwen2.5-VL-Math, Nougat derivatives) produce usable LaTeX. Without explicit LaTeX training, VLMs produce readable but imprecise transcriptions.

2026 scientific paper pipeline: Nougat first pass on the PDF, then VLM for tricky pages.

### Handwriting

Still the hardest subtask. Mixed printed + handwritten (doctor's notes, filled forms) is where OCR pipelines still beat VLMs on cost. Pure-handwriting VLMs are improving (Claude 4.7, PaliGemma 2).

### 2026 Recipe

For a new document AI project:

- Massive pure-printed invoices: LayoutLMv3 + rules, cost-effective.
- Mixed documents (scientific + handwritten + forms): VLM-native (PaliGemma 2 or Qwen2.5-VL).
- Full arXiv ingestion: Nougat for math, VLM for figures.
- Regulated scenarios: OCR pipeline + VLM verifier cross-check.

## Use It

`code/main.py`:

- A toy layout-aware tokenizer: given (text, bbox) pairs, produces LayoutLMv3-style input.
- A Donut-style task schema generator: JSON template for forms.
- A per-page token budget comparison across OCR pipeline, Donut, Nougat, VLM-native.

## Ship It

This lesson produces `outputs/skill-document-ai-stack-picker.md`. Given a document AI project (domain, scale, quality, regulation), it picks between OCR pipeline, OCR-free specialist, and VLM-native.

## Exercises

1. Your project is 10 million invoices per day. Which stack minimizes per-page cost without sacrificing accuracy?

2. Why does LayoutLMv3 beat a plain CLIP-VLM on form QA but lose on scene text? What does the bbox stream give up?

3. Nougat generates LaTeX. Propose a test case where VLM-native output beats Nougat on LaTeX fidelity, and one where Nougat wins.

4. Read the PaliGemma 2 paper (Google, 2024). What's the key training data addition that lifts document accuracy relative to PaliGemma 1?

5. Design a regulation-safe hybrid: OCR pipeline primary, VLM secondary cross-check. How do you resolve disagreements?

## Key Terms

| Term | How people say it | What it actually means |
|------|-----------------|------------------------|
| OCR pipeline | "Tesseract-style" | Staged stack: detect → OCR → layout → rules; deterministic, brittle |
| OCR-free | "Donut-style" | Image-to-output transformer skipping explicit OCR; single model |
| Layout-aware | "LayoutLM" | Input includes per-token bbox coordinates; cross-modal unified masking |
| VLM-native | "frontier VLM" | Feed page image at high resolution directly to Claude/GPT/Qwen VLM; no pipeline |
| DocVQA | "document benchmark" | Standard document VQA evaluation; most-cited score |
| Markup output | "LaTeX / MD" | Structured output format as alternative to free-form text; enables downstream automation |

## Further Reading

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
