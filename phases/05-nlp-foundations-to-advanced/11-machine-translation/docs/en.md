# Machine Translation

> Translation is the task that fed NLP research for thirty years and keeps feeding it.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 10 (Attention Mechanism), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 min

## The Problem

A model reads a sentence in one language and produces it in another. Length changes, word order changes. Some source words map to multiple target words and vice versa. Idioms refuse one-to-one mapping. "I miss you" in French is "tu me manques" — literally "you are missing from me." No word-level alignment survives this.

Machine translation as a task forced NLP to invent the encoder-decoder, attention, the transformer, and ultimately the entire LLM paradigm. Each advance arrived because translation quality is measurable and the gap between human and machine was stubborn.

This lesson skips the history and goes straight to what runs in 2026: pretrained multilingual encoder-decoders (NLLB-200 or mBART), subword tokenization, beam search, BLEU and chrF evaluation, and the failure modes that still reach production undetected.

## The Concept

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

Modern MT is a transformer encoder-decoder trained on parallel text. The encoder reads the source sentence with the source language's tokenization. The decoder generates the target one subword at a time, using cross-attention (lesson 10) over the encoder's output. Decoding uses beam search to avoid greedy decoding traps. Output is detokenized, recased, and scored against references.

Three operational choices determine real-world MT quality.

- **Tokenizer.** SentencePiece BPE trained on mixed-language corpora. Shared vocabulary across languages is what makes zero-shot language pairs possible in NLLB.
- **Model size.** NLLB-200 distilled 600M fits on a laptop. NLLB-200 3.3B is the published production default. 54.5B is the research ceiling.
- **Decoding.** Beam width 4–5 for general content. Add length penalty to prevent short outputs. Constrained decoding when terminology consistency matters.

## Build It

### Step 1: A single pretrained MT call

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

Three things matter here. `src_lang` tells the tokenizer which script and segmentation to use. `forced_bos_token_id` tells the decoder which language to generate. Both are NLLB-specific conventions; mBART and M2M-100 use their own, and they're not interchangeable.

### Step 2: BLEU and chrF

BLEU measures n-gram overlap between output and reference. Four reference n-gram sizes (1–4), geometric mean of precisions, brevity penalty for short outputs. Scores are in [0, 100]. Widely used but annoying to interpret: 30 BLEU is "usable," 40 is "good," 50 is "excellent," differences under 1 BLEU are noise.

chrF measures character-level F-score. More sensitive to morphologically rich languages where BLEU undercounts matches. Usually reported alongside BLEU.

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

Always use `sacrebleu`. It standardizes tokenization normalization, making scores comparable across papers. Rolling your own BLEU computation is a source of misleading benchmarks.

### Three-tier evaluation (2026)

Modern MT evaluation uses three complementary metric families. Ship with at least two.

- **Heuristic** (BLEU, chrF). Fast, reference-based, interpretable, insensitive to paraphrase. Use for historical comparison and regression detection.
- **Learned** (COMET, BLEURT, BERTScore). Neural models trained on human judgments; compare translation to source and reference for semantic similarity. COMET has the highest correlation with MT research since 2023 and is the 2026 production default when quality matters.
- **LLM-as-judge** (reference-free). Prompt a large model to rate translations on fluency, adequacy, register, cultural appropriateness. GPT-4-as-judge correlates ~80% with humans when the rubric is well-designed. Use for open-ended content with no reference.

The practical 2026 stack: `sacrebleu` for BLEU and chrF, `unbabel-comet` for COMET, and a prompted LLM for the final human-facing signal. Calibrate any metric against 50–100 human-annotated samples before trusting it for production data.

Reference-free metrics (COMET-QE, BLEURT-QE, LLM-as-judge) let you evaluate translations without references, which matters for long-tail language pairs where references don't exist.

### Step 3: What breaks in production

The pipeline above translates fluently 80% of the time and silently fails the other 20%. Named failure modes:

- **Hallucination.** The model invents content not in the source. Common on unfamiliar domain vocabulary. Symptom: fluent output that asserts facts the source never stated. Mitigation: constrained decoding on domain terms, human review for regulated content, monitoring for outputs much longer than inputs.
- **Off-target.** The model translates into the wrong language. NLLB does this surprisingly often on rare language pairs. Mitigation: verify `forced_bos_token_id` and always run a language-ID check on outputs.
- **Terminology drift.** "Sign up" becomes "s'inscrire" in document 1 and "créer un compte" in document 2. For UI text and user-facing strings, consistency matters more than raw quality. Mitigation: glossary-constrained decoding or post-editing dictionary.
- **Politeness register mismatch.** French "tu" vs "vous," Japanese honorific levels. The model picks whichever form was more common in training. For customer-facing content, this is usually wrong. Mitigation: prompt prefix with a politeness-level token if the model supports it, or fine-tune a small model on formal-only corpora.
- **Length explosion on short inputs.** Very short source sentences often produce excessively long translations because length penalty breaks down below ~5 source tokens. Mitigation: hard max-length cap proportional to source length.

### Step 4: Fine-tuning for a domain

Pretrained models are generalists. Legal, medical, or game-dialogue translation improves measurably with fine-tuning on in-domain parallel data. The recipe is not exotic:

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

A few thousand high-quality parallel samples beat tens of thousands of noisy web-crawled ones. Data quality is the biggest lever in production.

## Use It

The 2026 production stack for MT:

| Use case | Recommended starting point |
|---------|---------------------------|
| Any-to-any, 200 languages | `facebook/nllb-200-distilled-600M` (laptop) or `nllb-200-3.3B` (production) |
| English-centric, high quality, 50 languages | `facebook/mbart-large-50-many-to-many-mmt` |
| Fast, cheap inference, EN-FR/DE/ES | Helsinki-NLP / Marian models |
| Latency-sensitive browser-side | ONNX-quantized Marian (~50 MB) |
| Maximum quality, willing to pay | GPT-4 / Claude / Gemini with translation prompt |

As of 2026, LLMs already beat dedicated MT models on several language pairs, especially on idiomatic content and long context. The tradeoff is per-token cost and latency. Pick LLMs when context length, style consistency, or domain adaptation via prompting matters more than throughput.

## Ship It

Save as `outputs/skill-mt-evaluator.md`:

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## Exercises

1. **Easy.** Use `nllb-200-distilled-600M` to translate a 5-sentence English paragraph to French and back. Measure how close the round-trip result is to the original. You should see semantics preserved with lexical drift.
2. **Medium.** Implement a language-ID check on translation output using `fasttext lid.176` or `langdetect`. Integrate it into the MT call so off-target generations get caught before returning.
3. **Hard.** Fine-tune `nllb-200-distilled-600M` on a 5000-pair domain corpus of your choice. Measure BLEU before and after on a held-out set. Report which sentence types improved and which regressed.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| BLEU | Translation score | N-gram precision with brevity penalty. [0, 100]. |
| chrF | Character F-score | Character-level F-score. More sensitive to morphologically rich languages. |
| NMT | Neural MT | Transformer encoder-decoder trained on parallel text. The default since 2017. |
| NLLB | No Language Left Behind | Meta's 200-language MT model family. |
| Constrained decoding | Controlled output | Force specific tokens or n-grams to appear/not appear in output. |
| Hallucination | Made-up content | Model output unsupported by the source sentence. |

## Further Reading

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — The NLLB paper.
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — Why `sacrebleu` is the only correct way to report BLEU.
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — The chrF paper.
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — Practical fine-tuning walkthrough.
