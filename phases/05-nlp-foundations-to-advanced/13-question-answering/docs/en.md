# Question Answering

> Three systems shaped modern QA. Extractive finds spans, retrieval-augmented grounds answers in documents, generative produces answers. Every modern AI assistant is a mix of all three.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 11 (Machine Translation), Phase 5 · 10 (Attention Mechanism)
**Time:** ~75 min

## The Problem

A user types "When did the first iPhone launch?" and expects "June 29, 2007." Not "Apple's history is long and varied." Not a bare "2007" with no context. A direct, grounded, correct answer.

Over the past decade, three architectures have dominated QA.

- **Extractive QA.** Given a question and a passage known to contain the answer, find the start and end indices of the answer span in the text. SQuAD is the classic benchmark.
- **Open-domain QA.** No passage given. First retrieve relevant passages, then extract or generate the answer. This is the foundation of every RAG pipeline today.
- **Generative / closed-book QA.** A large language model answers from its parametric memory. No retrieval. Fastest inference, least factually reliable.

The 2026 trend is hybrid: retrieve the best passages, then let a generative model answer grounded in them. That's RAG — lesson 14 goes deep on the retrieval half. This lesson builds the QA half.

## The Concept

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**Extractive.** Encode the question and passage together with a transformer (BERT family). Train two heads that predict the start and end token indices of the answer. Loss is cross-entropy over valid positions. Output is a span from the passage. Never hallucinates (by construction) and never handles unanswerable questions (by construction).

**Retrieval-augmented (RAG).** Two stages. A retriever finds top-`k` passages from the corpus. A reader (extractive or generative) uses those passages to produce the answer. The retriever-reader split lets both be trained and evaluated independently. Modern RAG often adds a reranker between the two.

**Generative.** A decoder-only LLM (GPT, Claude, Llama) answers from learned weights. No retrieval step. Excellent on common knowledge, catastrophically bad on rare or recent facts. Hallucination rate is inversely proportional to the fact's frequency in pretraining data.

## Build It

### Step 1: Extractive QA with a pretrained model

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` is trained on SQuAD 2.0, which includes unanswerable questions. By default the `question-answering` pipeline returns the highest-scoring span even when the model's null-answer score wins — it does *not* automatically return a null answer. To get explicit "no answer" behavior, pass `handle_impossible_answer=True` to the pipeline call: then the pipeline returns null only when the null-answer score exceeds all span scores. Either way, check the `score` field.

### Step 2: A retrieval-augmented pipeline (sketch)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

Two-stage pipeline. A dense retriever (Sentence-BERT) finds relevant passages by semantic similarity. An extractive reader (RoBERTa-SQuAD) pulls the answer span from the combined top passages. Works on small corpora. For millions of documents, use FAISS or a vector database.

### Step 3: Generative with RAG

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

The prompt pattern matters. Explicitly telling the model to ground in context and return "I don't know" when context is insufficient reduces hallucination rates by 40–60% compared to naive prompting. More sophisticated patterns add citations, confidence scores, and structured extraction.

### Step 4: Evaluation that reflects the real world

SQuAD uses **Exact Match (EM)** and **token-level F1**. EM is strict string match after normalization (lowercase, strip punctuation, strip articles) — the prediction either matches exactly or scores 0. F1 computes over token overlap between prediction and reference, giving partial credit. Both undercount paraphrases: "June 29, 2007" vs "June 29th, 2007" typically scores 0 EM (ordinal breaks normalization) but still earns decent F1 from overlapping tokens.

Production QA:

- **Answer accuracy** (LLM-judged or human-judged, because metrics can't capture semantic equivalence).
- **Citation accuracy.** Does the cited passage actually support the answer? String-match the generated citation against retrieved passages for an easy automated check.
- **Refusal calibration.** When the answer isn't in the retrieved passages, does the system correctly say "I don't know"? Measure false-confidence rate.
- **Retrieval recall.** Before evaluating the reader, measure whether the retriever put the correct passage in top-`k`. The reader can't fix a missing passage.

### RAGAS: the 2026 production evaluation framework

`RAGAS` is purpose-built for RAG systems and is the 2026 shipping default. It scores four dimensions without requiring gold-standard references:

- **Faithfulness.** Is every claim in the answer grounded in the retrieved context? Measured via NLI-based entailment. Your primary hallucination metric.
- **Answer relevance.** Does the answer address the question? Computed by generating hypothetical questions from the answer and comparing with the real question.
- **Context precision.** Of the retrieved chunks, how many are actually relevant? Low precision = noise in the prompt.
- **Context recall.** Does the retrieval set contain all needed information? Low recall = the reader can't succeed.

Reference-free scoring lets you evaluate on real production traffic without curated gold-standard answers. For open-ended questions, layer an LLM-as-judge on top — exact-match metrics are useless there.

`pip install ragas`. Wire up your retriever + reader. Get four scalars per query. Alert on regressions.

## Use It

The 2026 stack.

| Use case | Recommendation |
|---------|-------------|
| Given a passage, find the answer span | `deepset/roberta-base-squad2` |
| Over a fixed corpus, closed-book not acceptable | RAG: dense retriever + LLM reader |
| Real-time queries over a document store | RAG with hybrid (BM25 + dense) retriever + reranker (lesson 14) |
| Conversational QA (follow-ups) | LLM with conversation history + per-turn RAG |
| Highly factual regulated domain | Extractive on authoritative corpus; never generative alone |

Extractive QA is unfashionable in 2026 because RAG with an LLM handles more cases. It still appears where verbatim source citation is required: legal retrieval, regulatory compliance, audit tools.

## Ship It

Save as `outputs/skill-qa-architect.md`:

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## Exercises

1. **Easy.** Set up the SQuAD extractive pipeline on 10 Wikipedia paragraphs. Write 10 questions by hand. Test how often the answer is correct. With clean passages and questions, you should see 7–9 correct.
2. **Medium.** Add a refusal classifier. When the top retrieval score is below a threshold (e.g., 0.3 cosine), return "I don't know" instead of calling the reader. Tune the threshold on a held-out set.
3. **Hard.** Build a RAG pipeline on a 10,000-document corpus of your choice. Implement hybrid retrieval with RRF fusion (BM25 + dense, see lesson 14). Measure answer accuracy with and without the hybrid step. Document which question types benefit most.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| Extractive QA | Find the answer span | Predict start and end indices of the answer in a given passage. |
| Open-domain QA | QA over a corpus | No passage given; must retrieve then answer. |
| RAG | Retrieve then generate | Retrieval-Augmented Generation. Retriever + reader pipeline. |
| SQuAD | Classic benchmark | Stanford Question Answering Dataset. EM + F1 metrics. |
| Hallucination | Made-up answer | Reader output unsupported by retrieved context. |
| Refusal calibration | Knowing when to shut up | The system correctly says "I don't know" when it can't answer. |

## Further Reading

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — The benchmark paper.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, the classic dense retriever for QA.
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — The paper that named RAG.
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — Comprehensive RAG survey.
