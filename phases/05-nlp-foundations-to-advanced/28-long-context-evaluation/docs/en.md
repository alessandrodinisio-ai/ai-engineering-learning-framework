# Long-Context Evaluation: NIAH, RULER, LongBench, MRCR

> Gemini 3 Pro advertises 10 million token context. At 1 million tokens, 8-needle MRCR drops to 26.3%. Claimed ≠ usable. Long-context evaluation tells you the true capacity of the model you're shipping.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 23 (Chunking Strategies)
**Time:** ~60 minutes

## The Problem

You have a 200-page contract. The model claims 1 million token context. You paste the contract in and ask: "What is the termination clause?" The model answers—but it answers from the cover page, because the termination clause is buried 120k tokens deep, beyond the model's actual attention range.

This is the context capacity gap in 2026. Spec sheets say 1 million or 10 million. Reality says 60-70% of that is usable, and "usable" depends on the task.

- **Retrieval (single needle in a haystack):** Near-perfect all the way to the claimed limit on frontier models.
- **Multi-hop / aggregation:** Degrades sharply beyond ~128k on most models.
- **Reasoning over scattered facts:** The first task to break.

Long-context evaluation measures these dimensions. This lesson covers the benchmarks, what each actually tests, and how to build a custom needle test for your domain.

## The Concept

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle in a Haystack (NIAH, 2023).** Place a single fact ("the magic word is pineapple") at a controlled depth inside a long context. Ask the model to retrieve it. Sweep depth × length. The original long-context benchmark. Frontier models now saturate it; it is a necessary but insufficient baseline.

**RULER (Nvidia, 2024).** 13 tasks across 4 categories: retrieval (single-key/multi-key/multi-value), multi-hop tracing (variable tracking), aggregation (common word frequency), QA. Configurable context length (4k to 128k+). Exposes models that saturate NIAH but fail at multi-hop. In the 2024 release, only half of 17 models claiming 32k+ context maintained quality at 32k.

**LongBench v2 (2024).** 503 multiple-choice questions, 8k–2M word contexts, six task categories: single-document QA, multi-document QA, long in-context learning, long dialogue, code repository, long structured data. The production benchmark for real-world long-context behavior.

**MRCR (Multi-Round Coreference Resolution).** Coreference at scale across many turns. 8-needle, 24-needle, 100-needle variants. Reveals how many facts a model can juggle before attention degrades.

**NoLiMa.** "Non-literal needles." The needle and query share no lexical overlap; retrieval requires one step of semantic reasoning. Harder than NIAH.

**HELMET.** Concatenates many documents, asks a question from any one of them. Tests selective attention.

**BABILong.** Embeds bAbI reasoning chains inside irrelevant haystack. Tests reasoning within a haystack, not just retrieval.

### What to actually report

- **Claimed context window.** The number on the spec sheet.
- **Effective retrieval length.** The length at which NIAH passes at some threshold (e.g., 90%).
- **Effective reasoning length.** The length at which multi-hop or aggregation passes at that threshold.
- **Degradation curve.** Accuracy vs. context length, plotted separately by task type.

Give your spec sheet two numbers: effective retrieval and effective reasoning. Typically effective reasoning is 25-50% of the claimed window.

## Build It

### Step 1: Custom NIAH for your domain

See `code/main.py`. Skeleton:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

Sweep `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}. Plot a heatmap. That is your target model's NIAH card.

### Step 2: Multi-needle variant

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

A question like "What are the three magic words?" requires retrieving all three. Single-needle success does not predict multi-needle success.

### Step 3: Multi-hop variable tracking (RULER-style)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

The answer requires chaining three assignments. Frontier models commonly drop to 50-70% accuracy here at 128k.

### Step 4: Run LongBench v2 on your stack

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

Report per-category accuracy. Aggregate scores hide huge task-level variance.

## Pitfalls

- **NIAH-only evaluation.** Passing NIAH at 1 million tokens says nothing about multi-hop. Always run RULER or a custom multi-hop test.
- **Uniform depth sampling.** Many implementations only test depth=0.5. Test depth=0, 0.25, 0.5, 0.75, 1.0—the "lost in the middle" effect is real.
- **Lexical overlap with filler.** If the needle shares keywords with the filler, retrieval becomes trivial. Use NoLiMa-style non-overlapping needles.
- **Ignoring latency.** A 1 million token prompt prefill takes 30-120 seconds. Measure time-to-first-token alongside accuracy.
- **Vendor self-reported numbers.** OpenAI, Google, Anthropic all publish their own scores. Always rerun independently on your use case.

## Use It

The 2026 stack:

| Scenario | Benchmark |
|-----------|-----------|
| Quick sanity check | Custom NIAH, 3 depths × 3 lengths |
| Production model selection | RULER (13 tasks) at your target length |
| Real-world QA quality | LongBench v2 single-doc QA subset |
| Multi-hop reasoning | BABILong or custom variable tracking |
| Conversational | MRCR 8-needle at your target length |
| Model upgrade regression | Fixed internal NIAH + RULER harness, run on every new model |

Production rule of thumb: never trust a context window until you've run NIAH + 1 reasoning task at the length you intend to use.

## Ship It

Save as `outputs/skill-long-context-eval.md`:

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## Exercises

1. **Easy.** Build a NIAH with 3 depths (0.25, 0.5, 0.75) × 3 lengths (1k, 4k, 16k). Run on any model. Plot pass rate as a 3×3 heatmap.
2. **Medium.** Add a 3-needle variant. Measure all-three-retrieved rate at each length. Compare against single-needle pass rate at the same lengths.
3. **Hard.** Construct a variable tracking task (X1 → X2 → X3, 3 hops) embedded in 64k filler. Test accuracy on 3 frontier models. Report effective reasoning length for each.

## Key Terms

| Term | How people say it | What it actually is |
|------|-----------------|-----------------------|
| NIAH | Needle in a Haystack | Bury a single fact in filler, ask the model to retrieve it. |
| RULER | NIAH on steroids | 13 tasks across retrieval / multi-hop / aggregation / QA. |
| Effective context | True capacity | The length at which accuracy still holds above a threshold. |
| Lost in the middle | Depth bias | Models under-attend to content in the middle of long inputs. |
| Multi-needle | Multiple facts at once | Multiple insertions; tests attention juggling, not just retrieval. |
| MRCR | Multi-round coreference | 8, 24, or 100 needle coreference; reveals attention saturation. |
| NoLiMa | Non-literal needles | Needle and query share no literal token overlap; requires reasoning. |

## Further Reading

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — The original NIAH repo.
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — Multi-task benchmark.
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — Real-world long-context evaluation.
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — Harder needles.
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — Reasoning in a haystack.
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — The depth bias paper.
