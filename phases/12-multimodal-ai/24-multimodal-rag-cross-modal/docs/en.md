# Multimodal RAG & Cross-Modal Retrieval

> Vision-native document RAG is only one slice. Production-grade multimodal RAG goes wider — cross-text, image, audio, video retrieval serving workflows like trip planning ("find me a quiet brunch with natural light that's vegan"), medical triage ("what injury matches this photo + these notes"), e-commerce ("outfits similar to this selfie in my size"), and field service ("diagnose this engine sound plus this parts photo"). Three 2025 surveys — Abootorabi et al., Mei et al., Zhao et al. — codify the sub-problems: cross-modal retrieval, retrieval fusion, generation grounding, multimodal evaluation. This lesson reads through these surveys and designs a production pipeline.

**Type:** Build
**Languages:** Python (stdlib, cross-modal retriever with fusion + grounded generator)
**Prerequisites:** Phase 12 · 23 (ColPali), Phase 11 (RAG fundamentals)
**Time:** ~180 minutes

## Learning Objectives

- Design cross-modal retrieval: text → image, image → text, audio → video, etc.
- Compare three fusion strategies: score fusion, attention-based fusion, MoE fusion.
- Explain generation grounding: what "cite your sources" looks like when sources are a mix of modalities.
- Name the three canonical 2025 multimodal RAG surveys and their sub-problem taxonomies.

## The Problem

Single-modal RAG is a solved pattern: embed the query, embed chunks, retrieve, stuff into an LLM. Multimodal RAG requires:

1. Multiple retrieval heads (each modality needs embeddings in a compatible space).
2. Fusion of cross-modal retrieval results.
3. Generation grounding that cites sources across modalities.
4. Evaluation metrics that cover cross-modal signals.

The 2025 surveys all arrive at the same taxonomy.

## The Concept

### Cross-Modal Retrieval

Given a query in modality A, retrieve documents in modality B. Three modes:

1. Shared embedding space. CLIP and CLAP produce text + image / text + audio embeddings in a shared space. Cross-modal cosine similarity works directly. Limited to pairs CLIP was trained on.

2. Per-modality encoders + translator. Text encoder + image encoder + a small translator module that maps between spaces. Sen2Sen by Gupta et al. and other 2024 designs. Flexible but adds complexity.

3. VLM-as-encoder. Use the VLM's hidden states as retrieval representations. Any modality the VLM supports works. Higher quality, more expensive.

Choose: CLIP / SigLIP 2 for text+image; CLAP for text+audio; VLM hidden states for frontier-quality cross-modal.

### Fusion Strategies

You retrieve 10 results: 5 images, 3 text passages, 2 audio clips. How to merge?

Score fusion (cheapest). Each modality has its own retriever, each returns scores. Normalize scores within each modality and sum. Simple, often works.

Attention-based fusion. Concatenate all retrieved items, let a small attention network weight them. Requires training.

MoE fusion. A gating network routes to modality-specific experts. Different query types route differently — a visual question gives images higher weight.

Production default: score fusion with a slight bias toward the query's dominant modality. If A/B shows a clear winner in your domain, upgrade to MoE.

### Generation Grounding

The LLM should cite which retrieved item drove each claim. For multimodal:

- Text sources: standard citation `[1]`.
- Image sources: `[img 3]` with a brief caption.
- Audio: `[audio 2 at 0:34]`.

Train the generator with grounding-aware data: each claim in the training target is tagged with a source index. At inference, the model naturally emits citations.

### The 2025 Surveys

Abootorabi et al. (arXiv:2502.08826, "Ask in Any Modality"): taxonomy of multimodal RAG. Covers retrieval, fusion, generation. Broadest coverage.

Mei et al. (arXiv:2504.08748, "A Survey of Multimodal RAG"): focuses on sub-task benchmarks and failure modes. Useful for evaluation design.

Zhao et al. (arXiv:2503.18016): vision-focused survey. Strong on ColPali-family work.

Read all three and you have state-of-the-art as of spring 2025. Most sub-problems remain open.

### MuRAG — The Founding Paper

MuRAG (Chen et al., 2022) was the first multimodal RAG. Retrieved images + text from a multimodal knowledge base, generated answers. Showed feasibility before the VLM wave. Modern systems (REACT, VisRAG, M3DocRAG) build on it.

### A Production Trip-Planner Example

Query: "Find me a quiet brunch with natural light that's vegan."

Pipeline:

1. Decompose query. "quiet" → audio/review keywords; "vegan brunch" → menu items; "natural light" → image features.
2. Per-modality retrieval:
   - Text retrieval on reviews: "vegan brunch, quiet atmosphere."
   - Image retrieval on restaurant photos: "natural light, airy."
   - Audio retrieval on ambient sound clips: "low decibel, no music."
3. Fuse scores. Each restaurant gets a composite score.
4. Top-k restaurants → VLM generator with all evidence → answer with citations.

This goes far beyond text RAG. Each modality adds signal that text alone would miss.

### Agentic Multimodal RAG

Multi-hop: if the first retrieval doesn't return a high-confidence answer, the LLM reformulates and retrieves again. Phase 14's agentic RAG pattern applies here. Examples:

- Retrieve initial top-10 → LLM says "too noisy, filter <40 dB" → re-retrieve.
- Retrieve images → LLM sees one has a menu → retrieve menu text → answer.

Adds complexity but handles queries that single-shot retrieval cannot.

### Evaluation

Cross-modal evaluation is still immature. Common proxy metrics:

- Per-modality Recall@k.
- Fused top-k accuracy.
- Human-judged end-to-end satisfaction.
- Task-specific (bookings completed, purchases made).

No standard benchmark spans all modalities. Most papers evaluate on domain-specific tasks.

## Use It

`code/main.py`:

- Three simulated retrievers (text, image, audio) operating on a shared restaurant corpus.
- Score fusion with configurable weights combining per-modality scores.
- A generator stub that emits a final answer with citations.
- A simple agentic loop that reformulates the query when confidence is low.

## Ship It

This lesson produces `outputs/skill-multimodal-rag-designer.md`. Given a product spec with multimodal query flows, it designs the retriever, fusion, generator, and evaluation.

## Exercises

1. Propose a medical-triage multimodal RAG: query = injury photo + text symptoms. Which modalities retrieve from which knowledge base?

2. Score fusion is a simple weighted sum. What failure mode does it have that MoE fusion can avoid?

3. Read Abootorabi et al.'s taxonomy (Section 3). What are the three canonical sub-problems and how do they map to your chosen product?

4. Design an evaluation spec for a trip-planner multimodal RAG. What metrics cover image recall, audio recall, and composite correctness?

5. Agentic multi-hop RAG has a latency tax per round-trip. At what query difficulty does the accuracy gain justify the latency?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Cross-modal retrieval | "query one modality, retrieve another" | Text query retrieves images; image query retrieves text; needs shared space or translator |
| Score fusion | "combine scores" | Weighted sum of per-modality retrieval scores; simplest fusion |
| MoE fusion | "modality-routed experts" | Gating network picks which modality's scores to trust per query |
| Grounded generation | "cite your sources" | Each claim in the answer is tagged with a source index |
| MuRAG | "first multimodal RAG" | 2022 paper that established the multimodal RAG pattern |
| Agentic multi-hop | "reformulate and retry" | LLM re-queries the retriever when first-pass confidence is low |

## Further Reading

- [Abootorabi et al. — Ask in Any Modality (arXiv:2502.08826)](https://arxiv.org/abs/2502.08826)
- [Mei et al. — A Survey of Multimodal RAG (arXiv:2504.08748)](https://arxiv.org/abs/2504.08748)
- [Zhao et al. — Vision RAG Survey (arXiv:2503.18016)](https://arxiv.org/abs/2503.18016)
- [Chen et al. — MuRAG (arXiv:2210.02928)](https://arxiv.org/abs/2210.02928)
- [Liu et al. — REACT (arXiv:2301.10382)](https://arxiv.org/abs/2301.10382)
