# Capstone Project 08 — Production RAG Chatbot for a Regulated Vertical

> In 2026, Harvey, Glean, Mendable, and LlamaCloud all run the same production pattern. Ingest with docling or Unstructured; ColPali for visual content. Hybrid search. Rerank with bge-reranker-v2-gemma. Synthesize with Claude Sonnet 4.7 using prompt caching at 60-80% hit rate. Guard with Llama Guard 4 and NeMo Guardrails. Monitor with Langfuse and Phoenix. Score with RAGAS on a 200-question golden set. Build one in a regulated domain (legal, clinical, insurance)—this capstone project is passing the golden set, the red team, and the drift dashboard.

**Type:** Capstone
**Languages:** Python (pipeline + API), TypeScript (chat UI)
**Prerequisites:** Phase 5 (NLP), Phase 7 (Transformers), Phase 11 (LLM Engineering), Phase 12 (Multimodal), Phase 17 (Infrastructure), Phase 18 (Safety)
**Phases involved:** P5 · P7 · P11 · P12 · P17 · P18
**Time:** 30 hours

## The Problem

Regulated-domain RAG (legal contracts, clinical trial protocols, insurance policies) is the highest-shipping production pattern of 2026 because ROI is obvious and risks are concrete. Harvey (Allen & Overy) built it for legal. Mendable ships developer documentation. Glean covers enterprise search. The paradigm: high-fidelity ingestion, hybrid retrieval with reranking, synthesis with citation enforcement and prompt caching, multi-layer guardrails, and continuous drift monitoring.

The hard part is not the model. It is jurisdiction-aware compliance (HIPAA, GDPR, SOC2), citation-level auditability, cost control (prompt caching buys a 60-90% discount at high hit rates), hallucination detection via RAGAS faithfulness, and drift detection when source documents are updated but the index has not caught up. This capstone requires you to deliver all of this against a 200-question golden set alongside a red-team suite.

## The Concept

The pipeline has two sides. **Ingestion**: docling or Unstructured parses structured documents; ColPali handles visually rich ones; chunks get summaries, tags, and role-based access labels. Vectors go into pgvector + pgvectorscale (under 50M vectors) or Qdrant Cloud; sparse BM25 runs in parallel. **Conversation**: LangGraph handles memory and multi-turn; each query runs hybrid retrieval, reranks with bge-reranker-v2-gemma-2b, synthesizes with Claude Sonnet 4.7 (prompt cached), passes output through Llama Guard 4 and NeMo Guardrails, and emits a response with citation anchors.

The evaluation stack has four layers. **Golden set** (200 annotated Q/A with citations) checks correctness. **Red team** (jailbreaks, PII extraction attempts, out-of-domain questions) checks safety. **RAGAS** automatically scores faithfulness / answer relevance / context precision per turn. **Drift dashboard** (Arize Phoenix) monitors retrieval quality and hallucination scores weekly.

Prompt caching is the cost lever. Claude 4.5+ and GPT-5+ support caching the system prompt + retrieved context. At 60-80% hit rate, per-query cost drops 3-5x. The pipeline must be designed for stable prefixes (system prompt + reranked context up front) to achieve high cache hit rates.

## Architecture

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## Tech Stack

- Ingestion: Unstructured.io or docling for structured documents; ColPali for visually rich PDFs
- Vector store: pgvector + pgvectorscale for under 50M vectors; Qdrant Cloud otherwise
- Sparse: Tantivy BM25 with field weights
- Orchestration: LlamaIndex Workflows (ingestion) + LangGraph (conversation)
- Reranker: Self-hosted bge-reranker-v2-gemma-2b or managed Voyage rerank-2
- LLM: Claude Sonnet 4.7 with prompt caching; self-hosted Llama 3.3 70B as fallback
- Evaluation: RAGAS 0.2 online, DeepEval for hallucination and jailbreak suites
- Observability: Self-hosted Langfuse with annotation queue; Arize Phoenix for drift
- Guardrails: Llama Guard 4 input/output classifier, NeMo Guardrails v0.12 policies, Presidio PII scrub
- Compliance: Role-based access labels on chunks; jurisdiction tags for GDPR/HIPAA

## Build It

1. **Ingestion.** Parse your corpus (1,000-10,000 documents for a serious effort) with Unstructured or docling. Route scanned/visual-heavy pages through ColPali. Produce chunks with summaries, role labels, and jurisdiction tags.

2. **Indexing.** Dense embeddings (Voyage-3 or Nomic-embed-v2) into pgvector + pgvectorscale. Build a BM25 side index via Tantivy. Role and jurisdiction filters as payload.

3. **Hybrid retrieval.** Filter by role + jurisdiction first; then parallel dense + BM25; merge with Reciprocal Rank Fusion (RRF); top-20 to the reranker; top-5 to synthesis.

4. **Synthesis with prompt caching.** System prompt + static policies in the cache header; reranked context as cache extension; user question as the uncached suffix. Target 60-80% hit rate at steady state.

5. **Guardrails.** Run Llama Guard 4 on input; NeMo Guardrails rails block out-of-domain or policy-prohibited topics; Presidio scrubs accidental PII from output; citation-enforcement post-filter.

6. **Golden set.** 200 Q/A pairs annotated with (answer, citations) by domain experts. Score the agent on exact citation match, answer correctness, and faithfulness (RAGAS).

7. **Red team.** 50 adversarial prompts: jailbreaks (PAIR, TAP), PII exfiltration attempts, out-of-domain, cross-jurisdiction leakage. Score with pass/fail and severity.

8. **Drift dashboard.** Arize Phoenix tracks retrieval quality (nDCG, citation faithfulness) weekly. Alert on a 5% drop.

9. **Cost report.** Langfuse: prompt cache hit rate, tokens per query, $/query broken down by stage.

## Use It

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## Ship It

`outputs/skill-production-rag.md` describes the deliverable. A regulated-domain chatbot deployed with compliance tagging, passing the scoring rubric, and monitored with live drift observability.

| Weight | Criterion | How to Measure |
|:-:|---|---|
| 25 | RAGAS faithfulness + answer relevance | Online scores on golden set (200 Q/A) |
| 20 | Citation correctness | Fraction of answers with verifiable source anchors |
| 20 | Guardrail coverage | Llama Guard 4 pass rate + jailbreak suite results |
| 20 | Cost / latency engineering | Prompt cache hit rate, p95 latency, $/query |
| 15 | Drift monitoring dashboard | Phoenix live dashboard with weekly retrieval quality trends |
| **100** | | |

## Exercises

1. Build a second corpus partition under a different jurisdiction (e.g., HIPAA alongside GDPR). Demonstrate how role + jurisdiction filtering prevents cross-domain leakage on a 20-question cross-jurisdiction probe.

2. Measure prompt cache hit rate over one week of production traffic. Identify which queries break the cache prefix. Refactor them.

3. Add multi-turn memory with a 10k-token summary buffer. Measure whether faithfulness degrades as conversations grow longer.

4. Swap Claude Sonnet 4.7 for self-hosted Llama 3.3 70B. Measure $/query and faithfulness delta.

5. Add an "uncertain" mode: if the reranker top score is below a threshold, the agent says "I do not have a confident citation" instead of answering. Measure how much false confidence is reduced.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| Prompt caching | "cached system + context" | A Claude/OpenAI feature: cached prefix tokens get a 60-90% discount on hit |
| RAGAS | "RAG evaluator" | Automated scoring of faithfulness, answer relevance, and context precision |
| Golden set | "annotated eval" | 200+ expert-annotated Q/A with citations; the ground truth |
| Jurisdiction tag | "compliance label" | A GDPR/HIPAA/SOC2 scope attached to a chunk; enforced by retrieval filters |
| Citation faithfulness | "grounded answer rate" | The fraction of assertions backed by a retrievable source span |
| Drift | "retrieval quality decay" | Weekly change in nDCG or citation score; alert threshold 5% |
| Red team | "adversarial eval" | Pre-release jailbreak, PII extraction, and out-of-domain probes |

## Further Reading

- [Harvey AI](https://www.harvey.ai) — reference legal production stack
- [Glean enterprise search](https://www.glean.com) — enterprise-scale RAG reference
- [Mendable documentation](https://mendable.ai) — developer-docs RAG reference
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — managed ingestion
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — cost lever reference
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — standard RAG evaluation framework
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — reference drift observability
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 safety classifier
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — policy rails framework
