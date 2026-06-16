# Multilingual NLP

> One model, 100+ languages, most with no training data. Cross-lingual transfer is the practical miracle of the 2020s.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 04 (GloVe, FastText, subword), Phase 5 · 11 (machine translation)
**Time:** ~45 minutes

## The Problem

English has billions of labeled examples. Urdu has a few thousand. Maithili has almost none. Any practical NLP system serving a global audience must work on the long tail of languages where no task-specific training data exists.

Multilingual models solve this by training one model on many languages simultaneously. Shared representations let the model transfer skills learned from high-resource languages to low-resource ones. Fine-tune it on English sentiment analysis and it produces surprisingly good sentiment predictions for Urdu out of the box. That is zero-shot cross-lingual transfer, and it reshaped how NLP goes global.

This lesson identifies the tradeoffs, the landmark models, and a decision that trips up multilingual newcomers: picking a source language for transfer.

## The Concept

![Cross-lingual transfer via shared multilingual embedding space](../assets/multilingual.svg)

**Shared vocabulary.** Multilingual models use a single SentencePiece or WordPiece tokenizer trained on text from all target languages. The vocabulary is shared: the same subword unit represents the same morpheme across related languages. `anti-` in English and Italian maps to the same token.

**Shared representations.** A transformer pre-trained with masked language modeling across multiple languages learns that semantically similar sentences in different languages produce similar hidden states. mBERT, XLM-R, and NLLB all exhibit this. The embedding for English "cat" clusters near French "chat" and Spanish "gato", and full-sentence embeddings do the same.

**Zero-shot transfer.** Fine-tune the model on labeled data in one language (usually English). At inference, run it on any other language the model supports. No target-language labels needed. Works well for typologically related languages, weaker for distant ones.

**Few-shot fine-tuning.** Add 100–500 labeled examples in the target language. On classification tasks, accuracy jumps to 95–98% of the English baseline. This is the highest-leverage move in multilingual NLP.

## Models

| Model | Year | Coverage | Notes |
|-------|------|----------|-------|
| mBERT | 2018 | 104 languages | Trained on Wikipedia. First practical multilingual LM. Weak on low-resource. |
| XLM-R | 2019 | 100 languages | Trained on CommonCrawl (far larger than Wikipedia). Set the cross-lingual baseline. Base 270M, Large 550M. |
| XLM-V | 2023 | 100 languages | XLM-R with a 1M-token vocabulary (vs 250K). Better on low-resource. |
| mT5 | 2020 | 101 languages | T5 architecture for multilingual generation. |
| NLLB-200 | 2022 | 200 languages | Meta's translation model; includes 55 low-resource languages. |
| BLOOM | 2022 | 46 languages + 13 programming languages | Open-source 176B LLM trained multilingually. |
| Aya-23 | 2024 | 23 languages | Cohere's multilingual LLM. Strong on Arabic, Hindi, Swahili. |

Pick by use case. For classification, XLM-R-base is a solid default. For generation, choose mT5 or NLLB depending on whether you need translation or open generation. For LLM-style work, use Aya-23 or Claude with explicit multilingual prompting.

## Source Language Decision (2026 Research)

Most teams default to English as the fine-tuning source. Recent research (2026) shows this is often wrong.

Language similarity predicts transfer quality more accurately than raw corpus size. For Slavic targets, German or Russian often beats English. For Indic targets, Hindi often beats English. The **qWALS** similarity metric (2026, based on World Atlas of Language Structures features) quantifies this. **LANGRANK** (Lin et al., ACL 2019) is an independent, earlier method that combines language similarity, corpus size, and genealogical relatedness to rank candidate source languages.

Practical rule: if your target language has a typologically close high-resource relative, try fine-tuning on that first, then compare against English fine-tuning.

## Build It

### Step 1: Zero-shot cross-lingual classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

One model, three languages, same API. XLM-R trained on NLI data transfers well to classification via the entailment trick.

### Step 2: Multilingual embedding space

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

Translations land close together in embedding space. A different English sentence lands farther away. This is what makes cross-lingual retrieval, clustering, and similarity work.

### Step 3: Few-shot fine-tuning strategy

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

For 100–500 target-language examples, `num_train_epochs=5` and `learning_rate=2e-5` are safe defaults. A higher learning rate collapses the multilingual alignment, leaving you with an English-only model.

## Evaluation That Actually Works

- **Per-language accuracy on held-out sets.** Don't aggregate. Aggregation hides the long tail.
- **Compare against monolingual baselines.** For languages with enough data, a monolingual model trained from scratch sometimes beats the multilingual model. Test it.
- **Entity-level testing.** Named entities in the target language. Multilingual models often tokenize poorly on characters far from Latin script.
- **Cross-lingual consistency.** The same meaning in two languages should produce the same prediction. Measure this gap.

## Use It

The 2026 stack:

| Task | Recommendation |
|------|----------------|
| Classification, 100 languages | Fine-tuned XLM-R-base (~270M) |
| Zero-shot text classification | `joeddav/xlm-roberta-large-xnli` |
| Multilingual sentence embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Translation, 200 languages | `facebook/nllb-200-distilled-600M` (see lesson 11) |
| Generative multilingual | Claude, GPT-4, Aya-23, mT5-XXL |
| Low-resource language NLP | XLM-V, or domain-specific fine-tuning on a related high-resource language |

When performance matters, always budget for target-language fine-tuning. Zero-shot is a starting point, not the finish line.

### The Tokenization Tax (What Goes Wrong for Low-Resource Languages)

Multilingual models share one tokenizer across all languages. That vocabulary is trained on a corpus dominated by English, French, Spanish, Chinese, and German. For any language outside the dominant set, three taxes silently stack up:

- **Fertility tax.** Low-resource language text tokenizes into far more tokens per word than English. A Hindi sentence may need 3–5× the tokens of an equivalent English sentence. That 3–5× eats your context window, training efficiency, and latency.
- **Variant recovery tax.** Every spelling error, diacritic variant, Unicode normalization mismatch, or casing variation becomes a cold-start, unrelated sequence in embedding space. The model fails to learn spelling correspondences that native speakers take for granted.
- **Capacity overflow tax.** Taxes 1 and 2 consume context positions, layer depth, and embedding dimensions. What remains for actual reasoning is systematically less than what high-resource languages get from the same model.

Practical symptoms: your model trains on Hindi normally, the loss curve looks right, evaluation perplexity looks reasonable, yet production output is subtly wrong. Morphology breaks mid-sentence. Rare inflections never recover. **You cannot data-brute-force your way past a broken tokenizer.**

Mitigation: pick a tokenizer with good coverage for your target language (XLM-V's 1M-token vocabulary is a direct fix); validate tokenization fertility on held-out target text before training; for truly long-tail scripts use byte-level fallback (SentencePiece `byte_fallback=True`, GPT-2-style byte-level BPE) so nothing goes OOV.

## Ship It

Save as `outputs/skill-multilingual-picker.md`:

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## Exercises

1. **Easy.** Run a zero-shot classification pipeline on 10 sentences each in English, French, Hindi, and Arabic. Report per-language accuracy. You should see French strong, Hindi decent, Arabic variable.
2. **Medium.** Build a cross-lingual retriever using `paraphrase-multilingual-MiniLM-L12-v2` on a small mixed-language corpus. Query in English, retrieve documents in any language. Measure recall@5.
3. **Hard.** For a Hindi classification task, compare English-source vs Hindi-source fine-tuning. Both use 500 target-language examples for few-shot fine-tuning. Report which source produces better Hindi accuracy and by how much. This is the LANGRANK argument in miniature.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| Multilingual model | One model, many languages | Shares vocabulary and parameters across languages. |
| Cross-lingual transfer | Train on one language, run on another | Fine-tune on source, evaluate on target without target-language labels. |
| Zero-shot | No target-language labels | Transfer without fine-tuning on the target language. |
| Few-shot | A few target labels | 100–500 target-language examples used for fine-tuning. |
| mBERT | First multilingual LM | 104-language BERT pre-trained on Wikipedia. |
| XLM-R | Standard cross-lingual baseline | 100-language RoBERTa pre-trained on CommonCrawl. |
| NLLB | Meta's 200-language MT | No Language Left Behind. Includes 55 low-resource languages. |

## Further Reading

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — The XLM-R paper.
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — The analysis paper that opened the cross-lingual transfer research line.
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — The NLLB-200 paper.
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya, Cohere's multilingual LLM.
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — The qWALS / LANGRANK source language paper.
