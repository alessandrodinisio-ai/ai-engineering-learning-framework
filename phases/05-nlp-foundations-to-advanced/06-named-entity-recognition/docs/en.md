# Named Entity Recognition

> Extract the names. Sounds simple until you hit ambiguous boundaries, nested entities, and domain jargon.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time:** ~75 min

## The Problem

"Apple sued Google over its iPhone search deal in the US." Five entities: Apple (ORG), Google (ORG), iPhone (PRODUCT), search deal (maybe), US (GPE). A good NER system extracts them all with correct types. A bad one misses iPhone, confuses Apple the fruit with Apple the company, and labels "US" as PERSON.

NER is the workhorse under every structured extraction pipeline. Resume parsing, compliance log scanning, medical record de-identification, search query understanding, fact grounding for chatbot responses, legal contract extraction. You almost never look at it directly yet always depend on it.

This lesson walks from the classical route (rule-based, HMM, CRF) through the modern route (BiLSTM-CRF, then transformers). Each step solves a specific limitation of the previous one. The evolution pattern itself is the point of this lesson.

## The Concept

**BIO tagging** (or BILOU) turns entity extraction into a sequence labeling problem. Label each token as `B-TYPE` (beginning of entity), `I-TYPE` (inside entity), or `O` (outside any entity).

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

Multi-token entities chain together: `New B-GPE`, `York I-GPE`, `City I-GPE`. A model that understands BIO can extract arbitrary spans.

The architectural evolution:

- **Rule-based.** Regex + gazetteer lookup. High precision on known entities, zero recall on novel ones.
- **HMM.** Hidden Markov Model. Emission probabilities of tokens given tags, transition probabilities between tags. Decoded with Viterbi. Trained on labeled data.
- **CRF.** Conditional Random Field. Like HMM but discriminative, so you can mix in arbitrary features (word shape, capitalization, neighboring words). Still a production workhorse for low-resource deployments in 2026.
- **BiLSTM-CRF.** Replace hand-crafted features with neural ones. An LSTM reads the sentence bidirectionally; a CRF layer on top enforces label sequence consistency.
- **Transformer-based.** Fine-tune BERT with a token classification head. Best accuracy, highest compute cost.

## Build It

### Step 1: BIO tagging helpers

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### Step 2: Hand-crafted features

For classical (non-neural) NER, features are the deciding factor. Useful ones:

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` returns `xXxxxx`. `word_shape("USA-2024")` returns `XXX-dddd`. Capitalization patterns are a high-signal feature for proper nouns.

### Step 3: A simple rule + gazetteer baseline

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

Production gazetteers have millions of entries crawled from Wikipedia and DBpedia. Coverage is decent, but disambiguation (Apple the company vs the fruit) is terrible. That's why statistical models won.

### Step 4: The CRF step (sketched, not fully implemented)

Writing a CRF from scratch in 50 lines without probability theory background isn't illuminating. Use `sklearn-crfsuite` instead:

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` and `c2` are L1 and L2 regularization. `all_possible_transitions=True` lets the model learn that illegal sequences (like `O` followed by `I-ORG`) are unlikely — this is how the CRF enforces BIO consistency without you writing hard constraints.

### Step 5: What BiLSTM-CRF adds

Features become learned. Input: token embeddings (GloVe or fastText). An LSTM reads the sentence left-to-right and right-to-left. Concatenated hidden states pass through a CRF output layer. The CRF still enforces label sequence consistency; the LSTM replaces hand-crafted features with learned ones.

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

The CRF layer uses `torchcrf.CRF` (pip install pytorch-crf). The gain over hand-crafted CRF features is measurable but smaller than you'd expect — unless you have tens of thousands of labeled sentences.

## Use It

spaCy provides production-grade NER out of the box.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

Note `iPhone` tagged as `ORG` rather than `PRODUCT` — spaCy's small model has weak product entity coverage. The large model (`en_core_web_lg`) is better. The transformer model (`en_core_web_trf`) is better still.

BERT-based NER with Hugging Face:

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` merges consecutive B-X, I-X tokens into a single span. Without it you get token-level labels and must merge yourself.

### LLM-based NER (the 2026 option)

Zero-shot and few-shot LLM NER now rivals fine-tuned models in many domains and significantly outperforms them when labeled data is scarce.

- **Zero-shot prompting.** Give the LLM an entity type list and an example schema; have it output JSON. Works out of the box; medium accuracy on novel domains.
- **ZeroTuneBio-style prompting.** Break the task into candidate extraction → meaning explanation → judgment → review. A multi-stage prompt (not one-shot) significantly boosts accuracy on biomedical NER. The same pattern works for legal, financial, and scientific domains.
- **Dynamic prompting with RAG.** At each inference call, retrieve the most similar labeled examples from a small annotated seed set to build a few-shot prompt on the fly. In 2026 benchmarks this improves GPT-4's biomedical NER F1 by 11–12% over static prompting.
- **Per-entity-type decomposition.** For long documents, a single call extracting all entity types loses recall as length grows. Run one extraction pass per entity type. Higher inference cost, significantly higher accuracy. This is the standard pattern for clinical notes and legal contracts.

2026 production advice: start with an LLM zero-shot baseline before collecting training data. Often the F1 is good enough and you never need fine-tuning.

### Where classical NER still wins

Even with LLMs available, classical NER wins when:

- Latency budget is under 50ms.
- You have thousands of labeled samples and need 98%+ F1.
- The domain has a stable ontology and a pretrained CRF or BiLSTM transfers well.
- Regulatory constraints require a locally deployed, non-generative model.

### Where it breaks down

- **Domain shift.** NER trained on CoNLL deployed on legal contracts performs worse than a gazetteer. Fine-tune on your domain.
- **Nested entities.** "Bank of America Tower" is simultaneously an ORG and a FACILITY. Standard BIO can't represent overlapping spans. You need nested NER (multi-pass or span-based models).
- **Long entities.** "United States Federal Deposit Insurance Corporation." Token-level models sometimes split it. Use `aggregation_strategy` or post-process.
- **Sparse types.** Medical NER labels like DRUG_BRAND, ADVERSE_EVENT, DOSE. General models are clueless. Scispacy and BioBERT are the starting points there.

## Ship It

Save as `outputs/skill-ner-picker.md`:

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## Exercises

1. **Easy.** Implement `bio_to_spans` (the inverse of `spans_to_bio`) and verify round-trip consistency on 10 sentences.
2. **Medium.** Train the sklearn-crfsuite CRF above on the CoNLL-2003 English NER dataset. Report per-entity F1 using `seqeval`. Typical result: ~84 F1.
3. **Hard.** Fine-tune `distilbert-base-cased` on a domain-specific NER dataset (medical, legal, or financial). Compare against the spaCy small model. Document your data leakage check and write up what surprises you.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| NER | Extract names | Label token spans with types (PERSON, ORG, GPE, DATE…). |
| BIO | Labeling scheme | `B-X` begins, `I-X` continues, `O` is outside. |
| BILOU | Better BIO | Adds `L-X` (last) and `U-X` (unit/single-token) for cleaner boundaries. |
| CRF | Structured classifier | Models both emissions and tag-to-tag transitions. Enforces valid sequences. |
| Nested NER | Overlapping entities | One span and its sub-span are different entities. BIO can't express this. |
| Entity-level F1 | The correct NER metric | Predicted span must match gold span exactly. Token-level F1 overestimates accuracy. |

## Further Reading

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — The BiLSTM-CRF paper. A classic.
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Introduced the token classification pattern that became standard.
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — Practical reference for `Doc.ents` and every attribute on `Span`.
- [seqeval](https://github.com/chakki-works/seqeval) — The correct metrics library. Always use it.
