# Text Processing — Tokenization, Stemming, Lemmatization

> Language is continuous, models are discrete. Preprocessing is the bridge between them.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 · 14 (Naive Bayes)
**Time:** ~45 min

## The Problem

Models can't read "The cats were running." — they read integers.

Every NLP system starts by answering the same three questions: where does a word begin? What is this word's root? When should "run", "running", "ran" be treated as the same thing, and when should they stay distinct?

Get tokenization wrong and the model learns on garbage. If your tokenizer treats `don't` as one token but `do n't` as two, the training distribution splits. If your stemmer maps `organization` and `organ` to the same stem, topic modeling breaks. If your lemmatizer needs POS context but you don't pass it in, verbs get treated as nouns.

This lesson implements all three preprocessing steps from scratch, then shows how NLTK and spaCy do the same thing — so you can see the tradeoffs.

## The Concept

Three operations, each with a job and a failure mode.

**Tokenization** splits a string into tokens. The word "token" is deliberately vague because the right granularity depends on the task: classical NLP uses word-level, transformers use subword-level, languages without whitespace delimiters use character-level.

**Stemming** chops suffixes with rules. Fast, aggressive, dumb. `running -> run`, `organization -> organ`. That second one is its failure mode.

**Lemmatization** uses grammatical knowledge to reduce a word to its dictionary form. Slow, accurate, requires a lookup table or morphological analyzer. `ran -> run` (needs to know "ran" is past tense of "run"), `better -> good` (needs to understand comparatives).

Rule of thumb: use stemming when speed matters and you can tolerate noise (search indices, coarse classification). Use lemmatization when semantics matter (QA, semantic search, anything user-facing).

## Build It

### Step 1: A regex tokenizer

The simplest useful tokenizer splits on non-alphanumeric characters while keeping punctuation as separate tokens. Not perfect, not final, but runs in one line.

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

Three patterns in priority order: words with an optional internal apostrophe (`don't`, `it's`), pure digits, any single non-whitespace non-alphanumeric character as a standalone token (punctuation).

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

Note the failure mode: `3pm` gets split into `['3', 'pm']` because we alternate between digit and letter segments. Good enough for most tasks. URLs, emails, hashtags all break. For production, add dedicated patterns before the general ones.

### Step 2: A Porter stemmer (step 1a only)

The full Porter algorithm has five phases of rules. Step 1a alone covers the most common English suffixes and is enough to show the pattern.

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

Rules read top to bottom. The `ies -> i` rule is why `ponies -> poni` rather than `pony`. The real Porter has step 1b to fix that. Rules compete; earlier rules win. Order matters more than any single rule.

### Step 3: A lookup-based lemmatizer

Real lemmatization needs morphology. A teachable, runnable version uses a small lemma table plus a fallback.

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

That last example is the key teaching point. `watched` isn't in our table, and the fallback only handles `ing`. Real lemmatization must cover `ed`, irregular verbs, comparative adjectives, plurals with phonological changes (`children -> child`). That's why production systems use WordNet, spaCy's morphological analyzer, or a full morphological analyzer.

### Step 4: Putting it together

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

The missing piece is a POS tagger. Phase 5 · 07 (POS tagging) builds one. For now, default everything to `NOUN` and acknowledge the limitation.

## Use It

NLTK and spaCy ship production-grade versions, each in a few lines.

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` handles contractions, Unicode, and edge cases your regex will miss. `PorterStemmer` runs all five phases. `WordNetLemmatizer` requires translating POS tags from NLTK's Penn Treebank scheme to WordNet's abbreviation set. The wiring above is the part most tutorials skip.

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy hides the entire pipeline behind `nlp(text)` — tokenization, POS tagging, lemmatization in one pass. Faster than NLTK at scale and more accurate out of the box. The tradeoff: you can't easily swap individual components.

### Which to choose

| Scenario | Choice |
|-----------|------|
| Teaching, research, need to swap components | NLTK |
| Production, multilingual, speed matters | spaCy |
| Transformer pipeline (you'll tokenize with the model's own tokenizer anyway) | Use `tokenizers` / `transformers`, skip classical preprocessing |

### Two failure modes nobody warns you about

Most tutorials stop after teaching the algorithms. Two things bite real preprocessing pipelines, and they're almost never mentioned.

**Reproducibility drift.** NLTK and spaCy change tokenization and lemmatization behavior across versions. Input that produces `['do', "n't"]` in spaCy 2.x might become `["don't"]` in 3.x. Your model was trained on one distribution; inference now runs on another. Accuracy silently degrades and nobody knows why. Pin library versions in `requirements.txt`. Write a preprocessing regression test that freezes expected tokenization for 20 sample sentences. Run it on every upgrade.

**Train/inference mismatch.** You apply aggressive preprocessing at training time (lowercasing, stopword removal, stemming), then feed raw user input at inference time, and watch performance collapse. This is the single most common NLP failure mode in production. If you preprocess at training time, you must run the exact same functions at inference time. Package preprocessing as a function inside the model artifact, not as a notebook cell for the serving team to reimplement.

## Ship It

A reusable prompt to help engineers pick a preprocessing strategy without reading three textbooks.

Save as `outputs/prompt-preprocessing-advisor.md`:

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## Exercises

1. **Easy.** Extend `tokenize` to preserve URLs as single tokens. Test: `tokenize("Visit https://example.com today.")` should produce a URL token.
2. **Medium.** Implement Porter's step 1b. If a word contains a vowel and ends with `ed` or `ing`, strip the suffix. Handle the double-consonant rule (`hopping -> hop`, not `hopp`).
3. **Hard.** Build a lemmatizer that uses WordNet as its lookup table but falls back to your Porter stemmer when WordNet has no entry. Compare its accuracy against pure WordNet and pure Porter on an annotated corpus.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| Token | A word | Any unit a model consumes — could be a word, subword, character, or byte. |
| Stem | The root of a word | The result of rule-based suffix stripping. Not necessarily a real word. |
| Lemma | Dictionary form | The form you'd look up in a dictionary. Computing it correctly requires grammatical context. |
| POS tag | Part of speech | Categories like NOUN, VERB, ADJ. Accurate lemmatization depends on it. |
| Morphology | Word-form rules | How a word changes form with tense, number, case. Lemmatization relies on it. |

## Further Reading

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — The original paper, five pages, still the clearest explanation.
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — How a real pipeline fits together.
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — Tokenization edge cases you haven't thought of.
