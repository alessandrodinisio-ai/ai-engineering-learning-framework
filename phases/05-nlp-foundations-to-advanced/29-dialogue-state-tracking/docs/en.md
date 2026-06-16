# Dialogue State Tracking

> "I want a cheap restaurant in the north... actually make it moderate... and add Italian." Three turns, three state updates. DST keeps the slot-value dictionary in sync so the booking doesn't go wrong.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**Time:** ~75 minutes

## The Problem

In task-oriented dialogue systems, the user's goal is encoded as a set of slot-value pairs: `{cuisine: italian, area: north, price: moderate}`. Each user turn may add, change, or remove a slot. The system must read the entire dialogue and correctly output the current state.

Get one slot wrong, and the system books the wrong restaurant, the wrong flight, or charges the wrong card. DST is the hinge between what the user says and what the backend executes.

Despite LLMs, why it still matters in 2026:

- Compliance-sensitive domains (banking, healthcare, flight booking) require deterministic slot values, not free generation.
- Tool-using agents still need slot resolution before calling APIs.
- Multi-turn corrections are harder than they look: "actually no, make it Thursday."

Modern pipeline: classic DST concepts + LLM extractor + structured output guardrails.

## The Concept

![DST: dialogue history → slot-value state](../assets/dst.svg)

**Task structure.** A schema defines domains (restaurant, hotel, taxi) and their slots (cuisine, area, price, people). Each slot can be empty, filled with a value from a closed set (price: {cheap, moderate, expensive}), or filled with a free value (name: "The Copper Kettle").

**Two DST formulations.**

- **Classification.** For each (slot, candidate_value) pair, predict yes/no. Works for closed-vocabulary slots. The standard before 2020.
- **Generation.** Given the dialogue, generate slot values as free text. Works for open-vocabulary slots. The modern default.

**Metrics.** Joint Goal Accuracy (JGA) — the proportion of turns where *every* slot is correct. All-or-nothing. The MultiWOZ 2.4 leaderboard tops out at ~83% in 2026.

**Architectures.**

1. **Rule-based (slot regex + keywords).** Strong baseline for narrow domains. Debuggable.
2. **TripPy / BERT-DST.** Copy-based generation with BERT encoding. The standard before LLMs.
3. **LDST (LLaMA + LoRA).** Instruction-tuned LLM with domain-slot prompts. Reaches ChatGPT-level quality on MultiWOZ 2.4.
4. **Ontology-free (2024–26).** Skips the schema; generates slot names and values directly. Handles open domains.
5. **Prompting + structured output (2024–26).** LLM with Pydantic schema + constrained decoding. 5 lines of code, production-ready.

### Classic Failure Modes

- **Cross-turn coreference.** "Let's stay with the first option." Requires resolving which option.
- **Override vs append.** User says "add Italian." Do you replace cuisine or append?
- **Implicit confirmation.** "OK cool" — does this accept the offered booking?
- **Correction.** "Actually make it 7 pm." Must update time without clearing other slots.
- **Coreference to prior system utterance.** "Yes, that one." Which "that"?

## Build It

### Step 1: Rule-based slot extractor

See `code/main.py`. Regex + synonym dictionaries cover 70% of canonical utterances in narrow domains:

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

Brittle outside the canonical vocabulary. Useful for deterministic slot confirmation.

### Step 2: State update loop

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

Three invariants:

- Never reset slots the user didn't touch.
- Explicit negation ("never mind the cuisine") must clear.
- User corrections ("actually...") must override, not append.

### Step 3: LLM-driven DST with structured output

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic guarantees a valid state object. No regex, no schema mismatch, no hallucinated slots.

### Step 4: JGA evaluation

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

Calibration: what fraction of turns does the system get all slots right? For MultiWOZ 2.4, 2026 top systems: 80–83%. Your in-domain system on your narrow vocabulary should beat that, or the LLM baseline wins.

### Step 5: Handling corrections

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

When a correction is detected, override the last-updated slot rather than appending. Hard to get right without an LLM. Modern pattern: always have the LLM regenerate the entire state from history rather than incrementally updating — this naturally handles corrections.

## Pitfalls

- **Full-history regeneration cost.** Having the LLM regenerate state every turn makes total token cost O(n²). Cap history or summarize older turns.
- **Schema drift.** Adding slots after the fact breaks old training data. Version your schemas.
- **Case sensitivity.** "Italian" vs "italian" vs "ITALIAN" — normalize everywhere.
- **Implicit carryover.** If the user previously specified "for 4 people," a new request changing time shouldn't clear people. Always pass full history.
- **Free values vs closed sets.** Names, times, addresses need free-value slots; cuisine and area are closed. Mix both in the schema.

## Use It

The 2026 stack:

| Scenario | Method |
|-----------|----------|
| Narrow domain (one or two intents) | Rule-based + regex |
| Wide domain, with labeled data | LDST (LLaMA + LoRA on MultiWOZ-style data) |
| Wide domain, no labels, production-ready | LLM + Instructor + Pydantic schema |
| Voice | ASR + normalizer + LLM-DST |
| Multi-domain booking flows | Schema-guided LLM with per-domain Pydantic models |
| Compliance-sensitive | Rule-based primary, LLM fallback with confirmation flow |

## Ship It

Save as `outputs/skill-dst-designer.md`:

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## Exercises

1. **Easy.** Build the rule-based state tracker in `code/main.py` for 3 slots (cuisine, area, price). Test on 10 hand-written dialogues. Measure JGA.
2. **Medium.** Use Instructor + Pydantic + a small LLM on the same dataset. Compare JGA. Inspect the hardest turns.
3. **Hard.** Implement both and route: rule-based primary, LLM fallback when rule-based confidently produces <2 slots. Measure combined JGA and per-turn inference cost.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| DST | Dialogue state tracking | Maintaining a slot-value dictionary across dialogue turns. |
| Slot | Unit of user intent | A named parameter the backend needs (cuisine, date). |
| Domain | Task area | restaurant, hotel, taxi — groups of slots. |
| JGA | Joint Goal Accuracy | Proportion of turns where every slot is correct. All-or-nothing. |
| MultiWOZ | The benchmark | Multi-domain WOZ dataset; standard DST evaluation. |
| Ontology-free DST | No schema | Generates slot names and values directly without a fixed list. |
| Correction | "Actually..." | A turn that overrides a previously filled slot. |

## Further Reading

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — The classic benchmark.
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA instruction tuning for DST.
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — Copy-based DST workhorse.
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — EM-based unsupervised TOD.
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — Classic DST results.
