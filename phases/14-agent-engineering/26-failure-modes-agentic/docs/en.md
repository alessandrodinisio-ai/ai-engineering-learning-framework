# Failure Modes: Why Agents Break

> MASFT (Berkeley, 2025) cataloged 3 categories and 14 multi-agent failure modes. Microsoft's Taxonomy documents how existing AI failures are amplified in agentic scenarios. Industry field data converges on five recurring patterns: hallucinated actions, scope creep, cascading errors, context loss, and tool misuse.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 05 (Self-Refine & CRITIC), Phase 14 · 24 (Observability)
**Time:** ~60 minutes

## Learning Objectives

- Name MASFT's three failure categories with at least four specific modes per category.
- Explain why agent failures amplify existing AI failure modes (bias, hallucination).
- Describe five industry-recurring patterns and their mitigations.
- Implement a detector using the standard library that labels agent traces with failure mode tags.

## The Problem

Teams ship agents that work on 90% of traces. The 10% that fail aren't random noise — they fall into a handful of recurring categories. Once you can name them, you can monitor them and fix them.

## The Concept

### MASFT (Berkeley, arXiv:2503.13657)

Multi-Agent System Failure Taxonomy. 14 failure modes clustered into 3 categories. Inter-annotator Cohen's Kappa of 0.88 — these categories can be reliably distinguished.

Core claim: failures are fundamental design flaws of multi-agent systems, not LLM limitations fixable by better foundation models.

### Microsoft "Taxonomy of Failure Mode in Agentic AI Systems"

- Existing AI failures (bias, hallucination, data leakage) are amplified in agentic scenarios.
- New failures emerge from autonomy: unintended actions at scale, tool misuse, task drift.
- This whitepaper serves as a risk register for agent products.

### "Characterizing Faults in Agentic AI" (arXiv:2603.06847)

- Failures stem from orchestration, internal state evolution, and environment interaction.
- Not just "bad code" or "bad model output."

### LLM Agent Hallucination Survey (arXiv:2509.18970)

Two primary manifestations:

1. **Instruction-following deviation** — agent does not follow the system prompt.
2. **Long-range context misuse** — agent forgets or misuses context from earlier turns.

Sub-intention errors: Omission (skipping a step), Redundancy (repeating a step), Disorder (steps out of order).

### Five Industry-Recurring Patterns

Arize, Galileo, and NimbleBrain field analyses from 2024-2026 converge on:

1. **Hallucinated action.** Agent calls a non-existent tool or fabricates parameters.
2. **Scope creep.** Agent expands the task beyond what the user asked (creates extra PRs, sends extra emails).
3. **Cascading error.** One bad call triggers downstream effects. A hallucinated non-existent SKU triggers four API calls — a multi-system incident.
4. **Context loss.** Long-span tasks forget constraints from earlier turns.
5. **Tool misuse.** Calling the right tool with wrong parameters, or calling the entirely wrong tool.

Cascading is the killer. Agents can't distinguish "I failed" from "this task is impossible," and often hallucinate a success message on top of a 400 error to wrap up.

### Mitigation: Checkpoints at Every Step

Set automated verification checkpoints at every step in the reasoning chain, checking fact grounding against environment state. Specifically:

- Step-level safety classifiers (Lesson 21).
- Tool call parameter validation (Lesson 06).
- Cross-checking retrieved content against known facts (Lesson 05, CRITIC).
- Detecting success hallucination by re-probing state (was the file actually created?).

### Where Failure Monitoring Goes Wrong

- **Only tagging crashes.** Most agent failures produce valid-looking output. You need content-level checks.
- **No baseline.** Drift detection needs a last-known-good; without it you can't say "this is getting worse."
- **Over-alerting.** Paging on every failure. Cluster and rate-limit.

## Build It

`code/main.py` implements a failure-mode tagger using the standard library:

- A synthetic trace dataset covering all five patterns.
- Per-pattern detector functions (signature patterns on tool calls, outputs, repeated actions).
- A tagger that labels each trace and reports the pattern distribution.

Run it:

```
python3 code/main.py
```

Output: per-trace labels + aggregate distribution, cheaply reproducing what Phoenix's trace clustering reveals.

## Use It

- **Phoenix** for production drift clustering (Lesson 24).
- **Langfuse** for session replay + annotation.
- **Custom** for domain-specific signatures your observability platform won't detect.

## Ship It

`outputs/skill-failure-detector.md` generates a failure-mode detector tailored to your domain, wired to a trace store.

## Exercises

1. Add a "success hallucination" detector: agent returns success but target state didn't change.
2. Tag 100 real traces from a product you've built. Which pattern dominates? What's the cost to fix it?
3. Implement a "cascade radius" metric: given a failure at step N, how many downstream steps did it affect?
4. Read MASFT's 14 failure modes. Pick three that apply to your product. Write detectors.
5. Wire a detector into a CI job: fail the build if >=5% of traces hit a pattern.

## Key Terms

| Term | Common usage | What it actually is |
|------|----------------|------------------------|
| MASFT | "Multi-agent failure taxonomy" | Berkeley's 14-mode classification |
| Cascading error | "Ripple failure" | One early error propagates across N steps |
| Context loss | "Forgot constraints" | Long-span turns drop earlier-turn facts |
| Tool misuse | "Wrong tool / wrong args" | Valid call, incorrect invocation |
| Success hallucination | "Faking completion" | Agent claims success on a 400; state unchanged |
| Scope creep | "Going out of bounds" | Agent does more than asked |
| Instruction-following deviation | "Disobedience" | Ignoring system prompt or user constraints |
| Sub-intention errors | "Plan bugs" | Omission, redundancy, disorder in plan execution |

## Further Reading

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 failure modes, 3 categories
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — risk register
- [Arize Phoenix](https://docs.arize.com/phoenix) — drift clustering in practice
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — when simpler patterns avoid these modes entirely
