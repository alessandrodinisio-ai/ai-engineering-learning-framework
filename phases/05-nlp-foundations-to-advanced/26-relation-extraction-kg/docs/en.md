# Relation Extraction & Knowledge Graph Construction

> NER found the entities. Entity linking anchored them. Relation extraction finds the edges between them. A knowledge graph is the sum of nodes, edges, and their provenance.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**Time:** ~60 minutes

## The Problem

An analyst reads: "Tim Cook became CEO of Apple in 2011." Four facts:

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

Relation extraction (RE) converts free text into structured triples `(subject, relation, object)`. Aggregate across a corpus and you have a knowledge graph. Aggregate and query, and you have a reasoning substrate for RAG, analytics, or compliance auditing.

The 2026 problem: LLMs extract relations eagerly—too eagerly. They hallucinate triples unsupported by the source text. Without provenance, you cannot distinguish real triples from plausible fabrications. The 2026 answer is AEVS-style anchor-verify pipelines.

## The Concept

![Text → Triples → Knowledge Graph](../assets/relation-extraction.svg)

**Triple form.** `(subject_entity, relation_type, object_entity)`. Relations come from a closed ontology (Wikidata properties, FIBO, UMLS) or an open set (OpenIE-style, anything goes).

**Three extraction approaches.**

1. **Rule/pattern-based.** Hearst patterns: "X such as Y" → `(Y, isA, X)`. Plus handcrafted regexes. Brittle, precise, interpretable.
2. **Supervised classifier.** Given two entity mentions in a sentence, predict a relation from a fixed set. Trained on TACRED, ACE, KBP. The standard from 2015–2022.
3. **Generative LLM.** Ask the model to emit triples. Works out of the box. Needs provenance, otherwise it hallucinates plausible garbage.

**AEVS (Anchor-Extraction-Verification-Supplement, 2026).** The current hallucination mitigation framework:

- **Anchor.** Mark each entity span and relation phrase span with exact positions.
- **Extract.** Generate triples linked to the anchored spans.
- **Verify.** Check each triple element back against the source text; reject anything unsupported.
- **Supplement.** A coverage check pass ensuring no anchored span was dropped.

Hallucination drops sharply. Costs more compute, but is auditable.

**Open vs closed trade-offs.**

- **Closed ontology.** Fixed property list (e.g., Wikidata's 11,000+ properties). Predictable, queryable, hard to fabricate.
- **Open IE.** Any verb phrase becomes a relation. High recall, low precision, messy to query.

Production KGs usually mix both: use open IE for discovery, canonicalize relations to a closed ontology, then merge into the main graph.

## Build It

### Step 1: Pattern-based extraction

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

See `code/main.py` for the full toy extractor. Hearst patterns are still used in domain-specific pipelines because they are debuggable.

### Step 2: Supervised relation classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL is a seq2seq relation extractor: text in, triples out, already mapped to Wikidata property IDs. Fine-tuned on distantly supervised data. Standard open-source baseline.

### Step 3: LLM prompt extraction with anchoring

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

Verify each returned span against the source. Reject any case where `text[start:end] != triple_entity`. This is the minimal form of the AEVS "verify" step.

### Step 4: Canonicalization to a closed ontology

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (subject/object inverted)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # Drop unmapped open relations, or route to human review
```

Canonicalization often accounts for 60–80% of engineering effort. Budget for it.

### Step 5: Build a small graph and query it

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

This is the atom of every RAG-over-KG system. Scale it up with RDF triple stores (Blazegraph, Virtuoso), property graphs (Neo4j), or vector-augmented graph stores.

## Pitfalls

- **Resolve coreference before RE.** "He founded Apple"—RE needs to know who "he" is. Run coreference first (Lesson 24).
- **Entity normalization.** "Apple Inc" and "Apple" must resolve to the same node. Do entity linking first (Lesson 25).
- **Hallucinated triples.** LLMs emit triples unsupported by the text. Enforce span verification.
- **Relation normalization drift.** Open IE relations are inconsistent ("was born in", "came from", "is a native of"). Collapse to canonical IDs, otherwise the graph is unqueryable.
- **Temporal errors.** "Tim Cook is CEO of Apple"—true now, false in 2005. Many relations are time-bounded. Use qualifiers (Wikidata `P580` start time, `P582` end time).
- **Domain mismatch.** REBEL is trained on Wikipedia. Legal, medical, and scientific text often requires domain-fine-tuned RE models.

## Use It

The 2026 stack:

| Scenario | Choice |
|-----------|------|
| Fast production, general domain | REBEL or LlamaPred with Wikidata canonicalization |
| Domain-specific (biomedical, legal) | SciREX-style domain fine-tuning + custom ontology |
| LLM prompting, auditable output | AEVS pipeline: Anchor → Extract → Verify → Supplement |
| High-throughput news IE | Pattern-based + supervised hybrid |
| Building a KG from scratch | Open IE + human canonicalization pass |
| Temporal KG | Extraction with qualifiers (start/end time, point-in-time) |

Integration pattern: NER → Coreference → Entity Linking → Relation Extraction → Ontology Mapping → Graph Ingestion. Each stage is a potential quality gate.

## Ship It

Save as `outputs/skill-re-designer.md`:

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## Exercises

1. **Easy.** Run the pattern extractor in `code/main.py` on 5 news article sentences. Manually check precision.
2. **Medium.** Use REBEL (or a small LLM) on the same sentences. Compare triples. Which extractor has higher precision? Higher recall?
3. **Hard.** Build an AEVS pipeline: LLM extraction + span verification against the source. On 50 Wikipedia-style sentences, measure hallucination rate before and after the verification step.

## Key Terms

| Term | How people say it | What it actually is |
|------|-----------------|-----------------------|
| Triple | subject-predicate-object | `(s, r, o)` tuple, the atomic unit of a KG. |
| Open IE | Extract everything | Open-vocabulary relation phrases; high recall, low precision. |
| Closed ontology | Fixed schema | A bounded set of relation types (Wikidata, UMLS, FIBO). |
| Canonicalization | Normalize everything | Map surface names/relations to canonical IDs. |
| AEVS | Grounded extraction | Anchor-Extraction-Verification-Supplement pipeline (2026). |
| Provenance | Link to truth source | Each triple carries a doc ID + char span pointing to its source. |
| Distant supervision | Cheap labels | Align text with an existing KG to create training data. |

## Further Reading

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — The distant supervision paper.
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — The seq2seq RE workhorse.
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — Joint IE.
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 hallucination mitigation design.
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — Classic graph querying.
