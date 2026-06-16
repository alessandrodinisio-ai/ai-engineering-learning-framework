# Structured Outputs and Constrained Decoding

> Get an LLM to give you JSON. Most of the time it gives JSON. In production, "most of the time" is the problem. Constrained decoding edits logits before sampling, turning "most" into "always."

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 min

## The Problem

A classifier prompts an LLM: "Return one of {positive, negative, neutral}." The model returns "The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ...". Your parser crashes. Your classifier's F1 is 0.0.

Free-form generation is not a contract — it is a suggestion. Production systems need contracts.

Three layers exist in 2026.

1. **Prompting.** Ask nicely. "Return only the JSON object." Works ~80% of the time on frontier models, lower on small models.
2. **Native structured output APIs.** OpenAI `response_format`, Anthropic tool use, Gemini JSON mode. Reliable on supported schemas. Vendor lock-in.
3. **Constrained decoding.** Modifies logits at every generation step so the model *cannot* emit an illegal token. 100% valid by construction. Works on any local model.

This lesson builds intuition for all three and shows when to reach for which.

## The Concept

![Constrained decoding masks illegal tokens at each step](../assets/constrained-decoding.svg)

**How constrained decoding works.** At each generation step, the LLM produces a logit vector over the full vocabulary (~100k tokens). A *logit processor* sits between the model and the sampler. It computes which tokens are valid given the current position in the target grammar (JSON Schema, regex, context-free grammar), and sets all illegal token logits to negative infinity. Softmax over the remaining logits places probability mass only on valid continuations.

2026 implementations:

- **Outlines.** Compiles a JSON Schema or regex into a finite-state machine. Each token has O(1) valid-next-token lookup. FSM-based, so recursive schemas need flattening.
- **XGrammar / llguidance.** Context-free grammar engines. Handle recursive JSON Schemas. Near-zero decoding overhead. OpenAI named llguidance in their 2025 structured outputs implementation.
- **vLLM guided decoding.** Built-in `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` via Outlines, XGrammar, or lm-format-enforcer backends.
- **Instructor.** Pydantic-based wrapper over any LLM. Retries on validation failure. Cross-vendor, but does not modify logits — it relies on retries + structure-aware prompting.

### Counter-intuitive result

Constrained decoding is often *faster* than unconstrained generation. Two reasons. First, it shrinks the next-token search space. Second, smart implementations skip generation entirely for forced tokens (scaffolding like `{"name": "` — every byte is predetermined).

### The pitfall that will cost you

Field order matters. Put `answer` before `reasoning` and the model commits to an answer before thinking. The JSON is valid; the answer is wrong. No validation catches this.

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema field order is logic, not formatting.

## Build It

### Step 1: Regex-constrained generation from scratch

See `code/main.py` for a standalone FSM implementation. The core idea in 30 lines:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

The FSM tracks which part of the grammar we have satisfied so far. `valid_tokens(state, tokenizer)` computes which vocabulary tokens can advance the FSM without leaving an accepting path.

### Step 2: JSON Schema with Outlines

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

Zero validation errors. Ever. The FSM makes illegal outputs unreachable.

### Step 3: Vendor-agnostic Pydantic with Instructor

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

Different mechanism. Instructor does not touch logits. It writes the schema into the prompt, parses the output, and retries on validation failure (default 3 times). Works with any vendor. Retries add latency and cost. Cross-vendor portability is the selling point.

### Step 4: Native vendor APIs

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

Server-side constrained decoding. Matches Outlines reliability on supported schemas. No local model management. Locks you into a vendor.

## Pitfalls

- **Recursive schemas.** Outlines flattens recursion to a fixed depth. Tree-structured outputs (nested comments, ASTs) need XGrammar or llguidance (CFG-based).
- **Huge enums.** A 10,000-option enum compiles slowly or times out. Switch to a retriever: predict top-k candidates first, then constrain to those.
- **Overly strict grammars.** Enforcing a `date: "YYYY-MM-DD"` regex means the model cannot output `"unknown"` for a missing date. The model compensates by inventing one. Allow `null` or a sentinel value.
- **Premature commitment.** See the field-order pitfall above. Always put reasoning first.
- **Vendor JSON mode without schema.** Bare JSON mode guarantees valid JSON, not valid JSON *for your use case*. Always provide the full schema.

## Use It

2026 stack:

| Scenario | Choice |
|-----------|------|
| OpenAI/Anthropic/Google models, simple schema | Native vendor structured outputs |
| Any vendor, Pydantic workflow, retries acceptable | Instructor |
| Local model, need 100% validity, flat schema | Outlines (FSM) |
| Local model, recursive schema | XGrammar or llguidance |
| Self-hosted inference server | vLLM guided decoding |
| Batch processing where retries are acceptable | Instructor + cheapest model |

## Ship It

Save as `outputs/skill-structured-output-picker.md`:

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## Exercises

1. **Easy.** Prompt a small open-weight model (e.g., Llama-3.2-3B) without constrained decoding for `Review(sentiment, confidence, evidence_span)`. On 100 reviews, measure the fraction that parses as valid JSON.
2. **Medium.** Same corpus with Outlines JSON mode. Compare compliance rate, latency, and semantic accuracy.
3. **Hard.** Implement a regex-constrained decoder from scratch for phone numbers (`\d{3}-\d{3}-\d{4}`). Verify 0 illegal outputs on 1,000 samples.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| Constrained decoding | Forcing valid output | Masking illegal token logits at every generation step. |
| Logit processor | The thing that constrains | Function: `(logits, state) -> masked_logits`. |
| FSM | Finite-state machine | Compiled grammar representation; O(1) valid-next-token lookup. |
| CFG | Context-free grammar | Handles recursion; slower than FSM but more expressive. |
| Schema field order | Does it matter? | Yes — the first field commits; always put reasoning before answer. |
| Guided decoding | vLLM's name for it | Same concept, integrated into an inference server. |
| JSON mode | OpenAI's early version | Guarantees JSON syntax; does not guarantee schema conformance. |

## Further Reading

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — The Outlines paper.
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — Fast CFG-based constrained decoding.
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — Inference server integration.
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API reference + pitfalls.
- [Instructor library](https://python.useinstructor.com/) — Cross-vendor Pydantic + retries.
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — Benchmark of 6 constrained decoding frameworks.
