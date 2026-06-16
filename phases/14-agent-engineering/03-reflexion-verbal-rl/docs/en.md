# Reflexion: Verbal Reinforcement Learning

> Gradient-based RL needs thousands of trials and a GPU cluster to fix a single failure mode. Reflexion (Shinn et al., NeurIPS 2023) does it with natural language: after each failed trial, the agent writes a reflection, stores it in episodic memory, and builds the next trial on top of that memory. This is the pattern behind Letta's sleep-time compute, Claude Code's CLAUDE.md learnings, and pro-workflow's learn-rule.

**Type:** Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 01 (The Agent Loop), Phase 14 · 02 (ReWOO)
**Time:** ~60 minutes

## Learning Objectives

- Name Reflexion's three components (Actor, Evaluator, Self-Reflector) and the role of episodic memory.
- Implement a Reflexion loop with a binary evaluator, reflection buffer, and fresh retry using the standard library.
- Choose between scalar, heuristic, and self-eval feedback sources for a given task.
- Explain why verbal reinforcement catches errors that gradient-based RL needs thousands of trials to fix.

## The Problem

An agent botches a task. In standard RL you would run thousands more trials, compute gradients, and update weights. Expensive, slow, and most production agents cannot afford a training budget for every failure.

Reflexion (Shinn et al., arXiv:2303.11366) asks a different question: what if the agent just thinks about why it failed, then tries again carrying that thought? No weight updates. No gradients. Just natural language stored between trials.

The result: it beats ReAct and other untuned baselines on ALFWorld. It improves over ReAct on HotpotQA. It sets state of the art on code generation (HumanEval/MBPP) at the time. All without a single gradient step.

## The Concept

### Three Components

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

Plus a data structure:

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

A trial runs the Actor. The Evaluator scores it. If the score is low, the Self-Reflector produces a reflection ("I picked the wrong tool because I misread the question as asking about X when it actually asks about Y"). The reflection enters episodic memory. The next trial starts fresh but sees the reflection.

### Three Evaluator Types

1. **Scalar** — an external binary signal. ALFWorld success or failure. HumanEval test pass or fail. Simplest, strongest signal.
2. **Heuristic** — predefined failure signatures. "If the agent produces the same action twice in a row, flag as stuck." "If the trajectory exceeds 50 steps, flag as inefficient."
3. **Self-eval** — the LLM scores its own trajectory. Only needed when no ground truth exists. Weaker signal; pairs well with tool-anchored verification (Lesson 05 — CRITIC).

The 2026 default is to mix: use scalar when available, self-eval when not, heuristics as safety rails.

### Why It Generalizes

Reflexion is less a new algorithm than a named pattern. Nearly every production-grade "self-healing" agent runs some variant:

- Letta's sleep-time compute (Lesson 08): a separate agent reflects on past conversations, writes into a memory block.
- Claude Code's `CLAUDE.md` / "save memory" pattern: reflections are captured as learnings, prepended to future sessions.
- pro-workflow's `/learn-rule` command: corrections are captured as explicit rules.
- LangGraph's reflection node: a node scores output, routes to refine if needed.

They all derive from the same insight: natural language is a rich enough medium to carry "what I learned from failure" across runs.

### When It Works and When It Doesn't

Reflexion works when:

- There is a clear failure signal (test failure, tool error, wrong answer).
- The task class is reproducible (the same kind of question can be asked again).
- There is room for the reflection to improve the trajectory (enough action budget).

Reflexion does not help when:

- The agent succeeds on the first try.
- Failure is external (network down, tool broken) — reflecting "the network was down" does not help future runs.
- Reflections become superstitious — storing a narrative about a one-off flaky run.

The 2026 pitfall: memory rot. Reflections accumulate; some become outdated or wrong; retries slow down as the episodic buffer grows. Mitigations: periodic compaction (Lesson 06), TTL on reflections, or a separate sleep-time cleanup agent (Letta).

## Build It

`code/main.py` implements Reflexion on a toy puzzle: produce a 3-element list that sums to a target value. The Actor emits candidate lists; the Evaluator checks the sum; the Self-Reflector writes a one-line explanation of what went wrong. The reflection enters episodic memory for the next trial.

Components:

- `Actor` — a scripted policy that improves when it sees reflections.
- `Evaluator.binary()` — pass/fail against the target sum.
- `SelfReflector` — generates a one-line failure diagnosis.
- `EpisodicMemory` — a bounded list with TTL semantics.

Run it:

```
python3 code/main.py
```

The trace shows three trials. Trial 1 fails, stores a reflection, trial 2 sees the reflection and improves but still fails, trial 3 succeeds. Compare against a baseline run (no reflections) — it stays stuck on trial 1's answer.

## Use It

LangGraph offers reflection as a node pattern. Claude Code's `/memory` command and pro-workflow's `/learn-rule` externalize the episodic buffer into a markdown file. Letta's sleep-time compute runs the Self-Reflector off the hot path so the main agent stays latency-controlled. OpenAI Agents SDK does not offer Reflexion directly; you build it with a custom Guardrail that rejects trajectories by score plus a memory `Session` that survives across runs.

## Ship It

`outputs/skill-reflexion-buffer.md` creates and maintains an episodic buffer with reflection capture, TTL, and deduplication. Given a task class and a failure, it produces a reflection that actually helps the next trial (not a generic "be more careful").

## Exercises

1. Switch from a binary evaluator to a scalar evaluator that returns a distance metric (how far from the target). Does it converge faster?
2. Add a 10-trial TTL to reflections. Past that point, are older reflections helping or hurting?
3. Implement the heuristic evaluator: flag a trial as stuck if the same action repeats. How does it interact with the Self-Reflector?
4. Run Reflexion with an adversarial Actor that ignores reflections. What is the minimal reflection-prompt engineering needed to force the Actor to attend to them?
5. Read Reflexion paper Section 4 on AlfWorld. Conceptually reproduce the 130% success-rate improvement: what are the key differences vs vanilla ReAct?

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Reflexion | "self-correction" | Shinn et al. 2023 — Actor, Evaluator, Self-Reflector plus episodic memory |
| Verbal reinforcement | "gradient-free learning" | Natural-language reflections prepended to the next trial's prompt |
| Episodic memory | "per-task reflections" | A bounded buffer of past reflections for a given task class |
| Scalar evaluator | "binary success signal" | Pass/fail or numeric score from ground truth |
| Heuristic evaluator | "pattern-based detector" | Predefined failure signatures (e.g., stuck loops, too many steps) |
| Self-evaluator | "LLM judges its own trace" | Weak-signal fallback when no ground truth exists — use with tool-anchored verification |
| Memory rot | "stale reflections" | Episodic buffer fills with outdated entries; fix with compaction/TTL |
| Sleep-time reflection | "async self-reflection" | Moving the Self-Reflector off the hot path so the main agent stays fast |

## Further Reading

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — the canonical paper
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — async reflection in production
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — managing episodic buffers as part of context
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — reflection node pattern
