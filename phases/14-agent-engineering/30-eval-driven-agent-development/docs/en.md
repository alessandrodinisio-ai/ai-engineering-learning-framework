# Eval-Driven Agent Development

> Anthropic's advice: "Start with simple prompts, optimize them with comprehensive evaluations, and only add multi-step agent systems when needed." Evaluation is not the last step. It is the outer loop driving every other choice in Phase 14.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** All of Phase 14.
**Time:** ~60 minutes

## Learning Objectives

- Name three evaluation layers — static benchmarks, custom offline, online production — and what each is for.
- Explain the evaluator-optimizer tight loop.
- Describe 2026 best practices: evals co-located with code, run in CI, gating PRs.
- Connect every lesson in Phase 14 to the eval case it generates.

## The Problem

Agents pass demos. They fail in production in ways demos can't predict. Benchmarks answer "is this model broadly capable?" not "is this agent delivering the right patch for my product?" The answer: evaluation at three layers, running continuously, with every guardrail and learned rule mapped to an eval case.

## The Concept

### Three Evaluation Layers

1. **Static benchmarks** — SWE-bench Verified for code (Lesson 19), WebArena/OSWorld for browsing/desktop (Lesson 20), GAIA for generalist (Lesson 19), BFCL V4 for tool use (Lesson 06). Used for cross-model comparison and regression gating. Contamination is real: SWE-bench+ found 32.67% solution leakage. Always report Verified/audited scores.

2. **Custom offline evals** — your product's shape:
   - LLM-as-judge (Langfuse, Phoenix, Opik — Lesson 24).
   - Execution-based (run the patch, check tests).
   - Trajectory-based (compare action sequences against gold; OSWorld-Human shows top agents are 1.4-2.7x gold).

3. **Online evals** — production:
   - Session replay (Langfuse).
   - Guardrail-triggered alerts (Lessons 16, 21).
   - Per-step cost/latency tracking (Lesson 23 OTel spans).

### Evaluator-Optimizer (Anthropic)

Tight loop:

1. Proposer generates output.
2. Evaluator judges.
3. Refine until evaluator passes.

This is the generalization of Self-Refine (Lesson 05). Any agent flow you care about can be wrapped in evaluator-optimizer for reliability.

### 2026 Best Practices

- Evals co-located with code.
- Run in CI on every PR.
- Gate merges on eval scores (e.g., "no more than 5% regression vs. main").
- Every guardrail maps to an eval case.
- Every learned rule (Reflexion, pro-workflow learn-rule) maps to a failure case.

### Connecting Phase 14

Every lesson in Phase 14 generates eval cases:

| Lesson | Eval case it generates |
|--------|------------------------|
| 01 Agent Loop | Budget exhaustion, infinite loop guard |
| 02 ReWOO | Planner correctly re-plans on tool failure |
| 03 Reflexion | Learned reflections take effect on retry |
| 05 Self-Refine/CRITIC | Judge passes refined output |
| 06 Tool Use | Parameter coercion works; unknown tools rejected |
| 07-10 Memory | Retrieved citations match source; stale facts expire |
| 12 Workflow Patterns | Each pattern produces correct output |
| 13 LangGraph | Recovery reproduces state exactly |
| 14 AutoGen Actor | DLQ catches crashed handlers |
| 16 OpenAI Agents SDK | Guardrails trigger on correct inputs |
| 17 Claude Agent SDK | Sub-agent results return to orchestrator |
| 19-20 Benchmarks | SWE-bench Verified score, WebArena success rate, OSWorld efficiency |
| 21 Computer Use | Step-level safety catches injected DOM |
| 23 OTel | Spans emit required attributes |
| 26 Failure Modes | Detectors label known failures |
| 27 Prompt Injection | PVE rejects poisoned retrieval |
| 28 Orchestration | Supervisor routes to correct specialist |
| 29 Runtime Shapes | DLQ handles N% failures |

If your eval suite has cases for each of these, you cover Phase 14.

### Where Eval-Driven Development Fails

- **No baseline.** Evals without a last-known-good can't be interpreted. Store baselines.
- **Ungrounded LLM judge.** Judges hallucinate too. CRITIC pattern (Lesson 05) — ground the judge on external tools.
- **Overfitting to evals.** Optimizing for evals drifts from production usefulness. Rotate cases.
- **Flaky evals.** Non-deterministic cases create false positives. Pin seeds, snapshot state.

## Build It

`code/main.py` is a standard-library eval harness:

- A case registry with categories (benchmark, custom, online).
- A scripted agent under test.
- Evaluator-optimizer loop: propose, judge, refine until pass or max rounds.
- CI gate: aggregate pass rate + regression vs. baseline.

Run it:

```
python3 code/main.py
```

Output: per-case pass/fail, regression flags, CI gate verdict.

## Use It

- Write eval cases in the same repo as agent code.
- Run them via CI on every PR.
- Fail the build on regression.
- Track pass rates over time.
- Tie every production failure to a new eval case.

## Ship It

`outputs/skill-eval-suite.md` builds a three-layer eval suite for an agent product with CI gating and regression tracking.

## Exercises

1. Take one of your production failures. Write an eval case that reproduces it. Can your agent pass it now?
2. Build a three-dimension (factuality, tone, scope) LLM judge rubric for your domain. Score 50 sessions.
3. Wire the eval suite into CI. Fail the build on >=5% regression.
4. Add a trajectory efficiency metric: how many steps did the agent take vs. a gold trajectory?
5. Map every Phase 14 lesson to an eval case in your suite. Any missing? That's the gap to fill.

## Key Terms

| Term | Common usage | What it actually is |
|------|----------------|------------------------|
| Static benchmark | "Off-the-shelf eval" | SWE-bench, GAIA, AgentBench, WebArena, OSWorld |
| Custom offline eval | "Domain eval" | LLM-as-judge / execution / trajectory on your product shape |
| Online eval | "Production eval" | Session replay, guardrail alerts, cost/latency tracking |
| Evaluator-optimizer | "Propose-judge-refine" | Iterate until judge passes |
| CI gate | "Merge blocker" | Fail the build on eval regression |
| Baseline | "Last-known-good" | Reference score for detecting regression |
| Trajectory efficiency | "Steps vs. gold" | Agent steps divided by expert minimum steps |

## Further Reading

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — "start simple, optimize with evals"
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — curated benchmark
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — tool use benchmark
- [Langfuse docs](https://langfuse.com/) — evals + session replay in practice
