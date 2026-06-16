# POS Tagging and Parsing

> Grammar fell out of fashion once. Then every LLM pipeline needed to validate structured extraction, and it came back.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 min

## The Problem

Lesson 01 promised: lemmatization needs POS tags. Without knowing `running` is a verb, the lemmatizer can't reduce it to `run`. Without knowing `better` is an adjective, it can't reduce it to `good`.

Behind that promise lies an entire subfield. POS tagging assigns grammatical categories to words. Parsing recovers the tree structure of a sentence: which word modifies which, which verb governs which arguments. Classical NLP spent twenty years refining both. Then deep learning collapsed them into a token classification task on top of a pretrained transformer, and the research community moved on.

But practitioners didn't. Every structured extraction pipeline still relies on POS and dependency trees underneath. LLM-generated JSON gets validated with grammar constraints. QA systems use dependency parsing to decompose queries. Machine translation quality estimators check alignment between parse trees.

Worth understanding. This lesson introduces the tagsets, various baselines, and the point at which you should stop building from scratch and call spaCy instead.

## The Concept

**POS tagging** assigns a grammatical category to each token. The **Penn Treebank (PTB)** tagset is the English default. 36 tags, some distinctions picky to a casual reader: `NN` singular noun, `NNS` plural noun, `NNP` singular proper noun, `VBD` verb past tense, `VBZ` verb 3rd-person singular present, and so on. The **Universal Dependencies (UD)** tagset is coarser (17 tags) and language-agnostic; it became the default for cross-lingual work.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**Parsing** produces a tree. Two main schools:

- **Constituency parsing.** Noun phrases, verb phrases, prepositional phrases nest inside each other. Output is a tree of non-terminal categories (NP, VP, PP) with words as leaves.
- **Dependency parsing.** Each word has exactly one head it depends on, labeled with a grammatical relation. Output is a tree where each edge is a (head, dependent, relation) triple.

Dependency parsing won out in the 2010s because it generalizes cleanly across languages, especially free-word-order ones.

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## Build It

### Step 1: Most-frequent-tag baseline

The dumbest tagger that works. For each word, predict the tag it appeared with most often in training.

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

On the Brown corpus, this baseline hits ~85% accuracy. Not good, but no serious model should fall below this floor.

### Step 2: Bigram HMM tagger

Model the joint probability of the sequence:

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

Two tables: transition probabilities (tag given previous tag), emission probabilities (word given tag). Both estimated from counts with Laplace smoothing. Decode with Viterbi (dynamic programming over the tag lattice).

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

The bigram HMM hits ~93% accuracy on Brown. The jump from 85% to 93% comes mostly from transition probabilities — the model learns that `DET NOUN` is common and `NOUN DET` is rare.

### Step 3: Why modern taggers beat it

Transition + emission probabilities are local. They can't capture that `saw` is a noun in "I bought a saw" but a verb in "I saw the movie." A CRF with arbitrary features (suffix, word shape, surrounding words, the word itself) hits ~97%. A BiLSTM-CRF or transformer hits ~98%+.

The ceiling on this task is set by annotator disagreement. Human annotators agree at about 97% on Penn Treebank. Models above 98% are likely overfitting the test set.

### Step 4: Dependency parsing sketch

Full dependency parsing from scratch is out of scope; see Jurafsky and Martin for the textbook treatment. Two classical schools to know:

- **Transition-based** parsers (arc-eager, arc-standard) work like shift-reduce parsers: read tokens, push them onto a stack, apply reduce actions that create arcs. Greedy decoding is fast. Classic implementation: MaltParser. Modern neural version: Chen and Manning's transition-based parser.
- **Graph-based** parsers (Eisner algorithm, Dozat-Manning biaffine) score every possible head-dependent edge and pick the maximum spanning tree. Slower but more accurate.

For most applied work, call spaCy:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

Read the `dep` column bottom-up and the grammatical structure emerges.

## Use It

Every production NLP library ships POS and dependency parsers as part of its standard pipeline.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). Fast, accurate, integrated with tokenization + NER + lemmatization. `token.tag_` (Penn), `token.pos_` (UD), `token.dep_` (dependency relation).
- **Stanford NLP (stanza)**. Stanford's successor to CoreNLP. State-of-the-art on 60+ languages.
- **trankit**. Transformer-based, good UD accuracy.
- **NLTK**. `pos_tag`. Works, slow, older. Fine for teaching.

### Where it still matters in 2026

- **Lemmatization.** Lesson 01 needs POS to lemmatize correctly. Always will.
- **Structured extraction from LLM output.** Validating that generated sentences obey grammatical constraints (subject-verb agreement, required modifiers).
- **Aspect-based sentiment.** Dependency parsing tells you which adjective modifies which noun.
- **Query understanding.** "movies directed by Wes Anderson starring Bill Murray" decomposes into structured constraints via parsing.
- **Cross-lingual transfer.** UD tags and relations are language-agnostic, enabling zero-shot structural analysis on new languages.
- **Low-compute pipelines.** Without a transformer, POS + dependency parsing + lexicon takes you surprisingly far.

## Ship It

Save as `outputs/skill-grammar-pipeline.md`:

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## Exercises

1. **Easy.** Run the most-frequent-tag baseline on a small annotated corpus (e.g., NLTK's Brown subset). Measure accuracy on held-out sentences. Verify the ~85% result.
2. **Medium.** Train the bigram HMM above and report per-tag precision/recall. Which tags does the HMM confuse most?
3. **Hard.** Use spaCy's dependency parser to extract subject-verb-object triples from a 1000-sentence sample. Evaluate on 50 hand-annotated triples. Document where extraction fails (typically passive voice, coordination, and dropped subjects).

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| POS tag | Word type | Grammatical category. PTB has 36; UD has 17. |
| Penn Treebank | Standard tagset | English-specific. Fine-grained on verb tense and noun number. |
| Universal Dependencies | Multilingual tagset | Coarser than PTB; language-agnostic; the default for cross-lingual work. |
| Dependency parse | Sentence tree | Each word has one head; each edge has a grammatical relation. |
| Viterbi | Dynamic programming | Finds the highest-probability tag sequence given emissions and transitions. |

## Further Reading

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — The textbook treatment of POS and parsing.
- [Universal Dependencies project](https://universaldependencies.org/) — The cross-lingual tagset and treebank collection every multilingual parser uses.
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — Practical reference for every attribute on `Token`.
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — The paper that brought neural parsers into the mainstream.
