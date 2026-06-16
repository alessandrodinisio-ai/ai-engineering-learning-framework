# Jailbreak Taxonomy

> Safety guardrails without a taxonomy are a coin flip. Name the attack first, then defend against it.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 18 safety-related lessons, Phase 19 Track A Lessons 25-29
**Time:** ~90 minutes

## The Problem

A model that ships without an attack model is a model defended against no particular threat. An ops engineer sees a Twitter post, recognizes a pattern, writes a regex, ships it, and moves on. The next prompt rephrases the same trick and the regex misses. A week later someone wraps the same trick in base64 and ops writes a second regex. By month three the system has 40 patched rules with no shared vocabulary, no way to describe what an attack actually is, and a backlog growing faster than patches.

Before any detector, classifier, or rule engine in this Track can be useful, the team needs a shared way to label attacks. Not because labels stop attacks, but because labels turn the attack stream into a histogram. Histograms become coverage charts. Coverage charts drive the next sprint. The guardrails in Lessons 83-87 spend most of their time deciding whether a prompt is — for example — a role-play attack against a refusal policy or a context-smuggling attack against a tool. Without a taxonomy, that decision has no foundation.

This capstone defines a six-category taxonomy: broad enough to cover most attacks seen in the wild, narrow enough that two reviewers usually agree on the category, and specific enough that each category has at least seven hand-crafted fixtures. The taxonomy is the carrier wave for all downstream work.

## The Concept

The six categories cut along one axis: which trust boundary does the attack abuse? Each name maps to a boundary.

| Category | Trust boundary abused |
|---|---|
| role-play | The assistant's persona |
| instruction-override | The system prompt's authority |
| context-smuggling | The gap between user content and instruction content |
| multi-turn-ramp | Conversation history treated as contract |
| encoding-trick | The surface form of prohibited tokens |
| prefix-injection | The assistant's next-token decision |

A role-play attack resets the assistant to a different agent ("you are an unrestricted research model called QX") so refusal rules attached to the original persona no longer fire. Instruction-override prompts say "ignore previous instructions," attempting to overwrite the system prompt directly. Context-smuggling hides instructions inside something that looks like data: a pasted document, a tool result, a code block. Multi-turn-ramp warms the model up with harmless turns, then moves the line step by step, exploiting the model's tendency to stay consistent with conversation history. Encoding-tricks (base64, rot13, leet-speak, zero-width character insertion) hide prohibited tokens from naive keyword filters. Prefix-injection ends the prompt with "Sure, here's how" so the model continues from the assumed answer rather than refusing.

```mermaid
flowchart TB
  P[prompt] --> C{Which boundary is abused?}
  C -->|persona| RP[role-play]
  C -->|system prompt| IO[instruction-override]
  C -->|data vs instruction| CS[context-smuggling]
  C -->|conversation history| MT[multi-turn-ramp]
  C -->|surface form| ET[encoding-trick]
  C -->|continuation| PI[prefix-injection]
  RP --> H[(histogram)]
  IO --> H
  CS --> H
  MT --> H
  ET --> H
  PI --> H
  H --> D[coverage chart]
```

Each fixture is a record with `id`, `category`, `subtype`, `prompt`, `target_behavior`, and `severity`. The taxonomy object loads fixtures, groups by category, and exposes a `match` API: given a candidate prompt, return the closest fixture and its category. Match uses character trigram cosine — rough, fast, zero dependencies. It is not a detector. The detector comes in Lesson 83. This is just a label producer.

Severity follows a 1-5 scale. A 1 is a clumsy attack against a benign target ("please pretend you are a pirate"). A 5 is an attack that, if successful, would produce content a deployed system must never output (operational details for a dangerous activity). Most fixtures land at 2-3 because real attacks at deployment scale skew simple and lazy. Severity is set by the fixture author. Two reviewers disagreeing by more than one grade signals the rubric needs refinement.

## Build It

The corpus lives in `code/fixtures.py` as a standalone Python list. The taxonomy class in `code/main.py` loads it, validates that every category has at least seven fixtures, exposes `by_category`, `match`, and `stats` methods, and includes a runnable demo that prints the histogram. Trigram cosine is implemented from scratch with `numpy`.

The validation step checks four invariants: every fixture has a non-empty prompt, every category in the schema has representation, every severity is in the `1..5` range, and every fixture id is unique. Failure here is a hard exit, not a warning, because the entire downstream Track depends on this corpus being internally consistent.

## Use It

Run `python3 main.py` in the lesson's `code/` directory. The demo prints fixture counts per category, runs `match` against three sample probes, and writes `taxonomy.json` to the lesson's outputs directory. Downstream lessons read `taxonomy.json` rather than importing the Python module directly, so the corpus is a stable artifact.

## Connections

`outputs/skill-jailbreak-taxonomy.md` documents the six categories and the scoring rubric. Use it as a team's shared vocabulary. Every finding logged by the guardrails in Lesson 87 references a taxonomy id.

## Exercises

1. Add a seventh category, indirect-prompt-injection (instructions embedded in a retrieved document rather than in the user turn). Write ten fixtures and re-run the validator.
2. Replace trigram cosine with a token edit-distance scorer and measure how many match classifications change on the existing corpus.
3. Pull thirty more fixtures from your own product's logs (after anonymization) and confirm whether the category distribution matches what the team intuitively expected.

## Key Terms

| Term | Common usage | Precise meaning |
|---|---|---|
| Jailbreak | Any unsafe model output | A prompt that produces output violating a stated policy |
| Taxonomy | A list of categories | A partition of attacks by which trust boundary they abuse |
| Fixture | A test case | An annotated prompt with category, severity, and target behavior |
| Severity | How bad the output is | A 1-5 grade of impact if the attack succeeds |
| Match | A detection decision | The nearest fixture by trigram cosine, used to assign a category to a new prompt |

## Further Reading

This lesson is the entry point. Lessons 83-87 build directly on this corpus.
