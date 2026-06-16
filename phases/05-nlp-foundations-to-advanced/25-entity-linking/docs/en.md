# Entity Linking & Disambiguation

> NER found "Paris." Entity linking decides: Paris, France? Paris Hilton? Paris, Texas? Or the Trojan prince Paris? Without linking, your knowledge graph stays ambiguous.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**Time:** ~60 minutes

## The Problem

A sentence reads: "Jordan beat the press." Your NER tags "Jordan" as PERSON. Fine. But *which* Jordan?

- Michael Jordan (basketball)?
- Michael B. Jordan (actor)?
- Michael I. Jordan (Berkeley ML professor — yes, this confusion genuinely happens in ML papers)?
- Jordan (the country)?
- Jordan (a Hebrew name)?

Entity linking (EL) resolves each mention to a unique entry in a knowledge base: Wikidata, Wikipedia, DBpedia, or your domain KB. Two subtasks:

1. **Candidate generation.** Given "Jordan," which KB entries are plausible?
2. **Disambiguation.** Given context, which candidate is correct?

Both steps are learnable. Both have benchmarks. The combined pipeline has been stable for a decade — what changes is disambiguator quality.

## The Concept

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**Candidate generation.** Given a mention's surface form ("Jordan"), look up candidates in an alias index. A Wikipedia alias dictionary covers most named entities: "JFK" → John F. Kennedy, Jacqueline Kennedy, JFK Airport, JFK (film). A typical index returns 10–30 candidates per mention.

**Disambiguation: three approaches.**

1. **Prior + context (Milne & Witten, 2008).** `P(entity | mention) × context-similarity(entity, text)`. Works well, fast, no training needed.
2. **Embedding-based (ESS / REL / Blink).** Encode mention + context. Encode each candidate's description. Take max cosine. The default from 2020–2024.
3. **Generative (GENRE, 2021; LLM-based, 2023+).** Decode the entity's canonical name token by token. Constrain to a trie of valid entity names, guaranteeing output is always a valid KB id.

**End-to-end vs pipeline.** Modern models (ELQ, BLINK, ExtEnD, GENRE) run NER + candidate generation + disambiguation in one pass. Pipeline systems still dominate production because you can swap components.

### Two Metrics

- **Mention recall (candidate generation).** Fraction of gold mentions where the correct KB entry appears in the candidate list. The floor for the entire pipeline.
- **Disambiguation accuracy / F1.** Given the correct candidate is present, how often is top-1 correct.

Always report both. A system achieving 99% disambiguation on 80% candidate recall is an 80% pipeline.

## Build It

### Step 1: Build an alias index from Wikipedia redirects

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia alias data: ~18 million (alias, entity) pairs. Download from a Wikidata dump. Store as an inverted index.

### Step 2: Context-based disambiguation

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

This Jaccard overlap is a toy. Replace with cosine similarity over embeddings (see `code/main.py` Step 2 for the transformer version).

### Step 3: Embedding-based (BLINK-style)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

At index time, encode each KB entity once. At query time, encode mention + context once, dot-product against the candidate pool, take the max.

### Step 4: Generative entity linking (concept)

GENRE decodes the entity's Wikipedia title character by character. Constrained decoding (see Lesson 20) ensures only valid titles can be output. Tightly integrated with a KB-backed trie. Modern descendants are REL-GEN and LLM-prompted EL with structured output.

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

Paired with a whitelist (Outlines `choice`), this is the easiest EL pipeline to ship in 2026.

### Step 5: Evaluate on AIDA-CoNLL

AIDA-CoNLL is the standard EL benchmark: 1393 Reuters articles, 34k mentions, Wikipedia entities. Report in-KB accuracy (`P@1`) and out-of-KB NIL detection rate.

## Pitfalls

- **NIL handling.** Some mentions are not in the KB (emerging entities, obscure people). The system must predict NIL rather than guessing a wrong entity. Measure separately.
- **Mention boundary errors.** Upstream NER misses part of the span ("Bank of America" tagged only as "Bank"). EL recall drops.
- **Popularity bias.** Trained systems over-predict high-frequency entities. A mention of "Michael I. Jordan" in an ML paper often gets linked to basketball Jordan.
- **Cross-lingual EL.** Mapping mentions in Chinese text to English Wikipedia entities. Requires a multilingual encoder or a translation step.
- **KB staleness.** New companies, events, people are absent from last year's Wikipedia dump. Production pipelines need a refresh loop.

## Use It

The 2026 stack:

| Scenario | Choice |
|-----------|------|
| General English + Wikipedia | BLINK or REL |
| Cross-lingual, KB = Wikipedia | mGENRE |
| LLM-friendly, low daily mention volume | Prompt Claude/GPT-4 with candidate list + constrained JSON |
| Domain-specific KB (medical, legal) | Custom BERT with KB-aware retrieval + fine-tune on domain AIDA-style set |
| Ultra-low latency | Exact-match prior only (Milne-Witten baseline) |
| Research SOTA | GENRE / ExtEnD / Generative LLM-EL |

Production pattern shipping in 2026: NER → coreference → EL per mention → collapse clusters to one canonical entity per cluster. Output: one KB id per entity in the document, not per mention.

## Ship It

Save as `outputs/skill-entity-linker.md`:

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## Exercises

1. **Easy.** Implement the prior + context disambiguator in `code/main.py` on 10 ambiguous mentions (Paris, Jordan, Apple). Hand-label correct entities. Measure accuracy.
2. **Medium.** Use a sentence transformer to encode 50 ambiguous mentions. Embed each candidate's description. Compare embedding-based disambiguation against Jaccard context overlap.
3. **Hard.** Build a 1000-entity domain KB (e.g., your company's employees + products). Implement end-to-end NER + EL. Measure precision and recall on 100 held-out sentences.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| Entity Linking (EL) | Link to Wikipedia | Map a mention to a unique KB entry. |
| Candidate Generation | Who could it be? | Return a shortlist of plausible KB entries for a mention. |
| Disambiguation | Pick the right one | Score candidates using context, pick the winner. |
| Alias Index | That lookup table | A mapping from surface forms → candidate entities. |
| NIL | Not in KB | Explicitly predict that no KB entry matches. |
| KB | Knowledge base | Wikidata, Wikipedia, DBpedia, or your domain KB. |
| AIDA-CoNLL | The benchmark | 1393 Reuters articles with gold-standard entity links. |

## Further Reading

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — The foundational prior + context approach.
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — The embedding-based workhorse.
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — Generative EL with constrained decoding.
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — The benchmark paper.
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — Open-source production stack.
