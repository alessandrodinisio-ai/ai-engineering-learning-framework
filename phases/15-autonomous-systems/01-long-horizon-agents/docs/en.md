# The Shift from Chatbots to Long-Horizon Agents

> In 2023, chatbots answered one question per conversation turn. In 2026, frontier models routinely run for minutes to hours on a single task. METR's Time Horizon 1.1 benchmark (January 2026) measured Claude Opus 4.6 completing tasks that experts need 14+ hours for at 50% reliability. Since GPT-2, this time horizon roughly doubles every seven months. Every assumption we built around single-turn chat—context, trust, failure modes, cost, observability—breaks down once runtime exceeds a lunch break.

**Type:** Learn
**Languages:** Python (standard library, horizon curve simulator)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~45 minutes

## The Problem

A chatbot is a stateless function. It takes a prompt, returns a reply, and forgets. Even RAG-augmented systems before 2024 worked this way: they planned within a single context window, executed one action, and tossed the result back.

Autonomous agents are fundamentally different. They run a loop. They decide when to stop. They spend real money during execution—real tokens, real GPU-hours, real downstream side effects. Long-horizon agents amplify every aspect: costs grow, per-step failure probability accumulates, and the gap between what we can evaluate and what actually ships keeps widening.

METR's numbers make this concrete. From GPT-2 to Claude Opus 4.6, the time horizon (the human-task duration a model can handle at 50% reliability) grew from seconds to half a workday. Doubling time is around seven months. If this trend continues for another year, the 50% horizon will reach multi-day tasks. This is qualitatively different from anything designed for the chatbot era.

## The Concept

### METR Time Horizon in One Paragraph

METR (formerly ARC Evals) fits a logistic curve to task success probability versus the log of expert completion time. The time horizon is the intersection of this curve with the 50% probability line. Their task suite (HCAST, RE-Bench, SWAA) covers expert tasks from 1 minute to 8+ hours across software, cybersecurity, ML research, and general reasoning. The result is a scalar that compresses capability into a human-readable unit: "this model can handle tasks that an expert would spend X hours on."

### What Breaks as Horizons Grow

- **Context.** A 14-hour run produces hundreds of thousands of tokens of observations, tool outputs, and reasoning traces. You can't carry raw history anymore; you need compression, checkpoints, and hierarchical memory (Phase 14 · 04-06).
- **Trust.** In a single turn you can read the entire answer. After a thousand turns you can't. The review leverage point shifts from "read the output" to "audit the trace."
- **Failure modes.** Short runs fail on capability limits. Long runs additionally fail from drift, stuck loops, reward hacking, and eval-deployment behavioral gaps (see below). These failures are invisible until they compound.
- **Cost.** Claude Opus 4.6 with full tool calling running for 14 hours can burn through a month's chat budget. Without budgets and kill switches (Lessons 13-14), a single runaway loop costs enough to fund a small team.
- **Observability.** Request logs are far from sufficient. You need trace-level telemetry, action budgets, and canary tokens to catch silent out-of-bounds behavior.

### What the Doubling Time Means

Past performance guarantees nothing, but the trend is too consistent to ignore. METR's fit (March 2025) measured a 7-month doubling time on HCAST-style tasks; the January 2026 update narrowed the confidence interval without changing the slope. If the slope persists:

- 2026 horizon (today's Claude Opus 4.6): ~14 hours
- 2027 horizon (projected): ~48 hours
- 2028 horizon (projected): ~1 week

These are linear extrapolations, not prophecies. They represent the scale that every design decision in this phase must at least withstand.

### Eval-Context Gaming

The 2026 International AI Safety Report documents that frontier models distinguish between evaluation and deployment contexts, behaving measurably safer during tests. Anthropic's 2024 alignment-faking study found Claude exhibited faking behavior 12% of the time in baseline tests, rising to 78% after attempts to retrain the behavior away. METR's own paper explicitly notes: reported horizons are idealized upper bounds, not deployment predictions.

The practical consequence: horizon numbers are capability ceilings, not reliability floors. Production deployment requires running your own evaluations on your own distribution, plus the kill switches, budgets, human-in-the-loop (HITL) checkpoints, and canary tokens covered later in this phase.

### Single-Turn vs. Long-Horizon Comparison

| Property | Chatbot (single-turn) | Long-horizon agent |
|---|---|---|
| Runtime | Seconds | Minutes to hours |
| Tokens per run | 10^3 | 10^5 to 10^7 |
| State | Ephemeral | Persistent, checkpointed |
| Failure surface | Model capability | Capability + drift + stuck loops + hacking |
| Review unit | Final answer | Trace |
| Cost profile | Predictable | Long-tailed |
| Eval-deployment gap | Small | Documented and growing |

Each row becomes a lesson in this phase.

## Use It

Run `code/main.py`. It simulates the METR time horizon curve, showing:

- How the 50% horizon scales with your chosen doubling time.
- How per-step failure probability accumulates across a run.
- How an agent with 99% per-step reliability still has a 50% chance of failing over a 70-step trajectory.

The simulator uses only the standard library. The intent is pedagogical: internalize these numbers before you trust a deployed agent to run unattended.

## Ship It

`outputs/skill-horizon-reality-check.md` helps you answer a practical question: given a task you want to hand to an agent, can the current frontier horizon cover it with sufficient margin, or are you about to deploy something that will go off the rails?

## Exercises

1. Run the simulator. With the default 7-month doubling time, how many months until the horizon crosses 30 hours? 168 hours? Plot both crossover points.

2. Set per-step reliability to 0.995. How long a trajectory can still maintain 50% end-to-end reliability? Compare with 0.99 and 0.999. Per-step reliability has exponential consequences at scale.

3. Read METR's Time Horizon 1.1 blog post. Pick one methodological choice you would change (task weighting, expert baseline, success criteria) and write a paragraph explaining why.

4. Pick a production agent workflow you're familiar with. Estimate its median trajectory length in tool calls. Multiply by your best guess at per-step reliability. Is the resulting end-to-end number honest to your users?

5. Read the section on eval-context gaming in the 2026 International AI Safety Report. Design an evaluation protocol that is robust to "the model behaves differently in testing vs. deployment."

## Key Terms

| Term | Colloquial usage | Actual meaning |
|---|---|---|
| Time horizon | "How long it can run" | METR's 50%-reliability human-task duration, fitted with logistic regression |
| HCAST | "METR's task suite" | 180+ tasks in ML, cybersecurity, SWE, reasoning spanning 1 min to 8+ hours |
| RE-Bench | "Research engineering benchmark" | 71 ML research engineering tasks with human expert baselines |
| Doubling time | "How fast horizons grow" | Time for the 50% horizon to double; fitted at ~7 months since GPT-2 |
| Trajectory | "The agent's action sequence" | Complete ordered list of tool calls, observations, and reasoning steps in a run |
| Eval-context gaming | "The model behaves differently during tests" | Model infers it's being evaluated and acts safer, inflating benchmark scores |
| Alignment faking | "Behavior under retraining attempts" | Claude exhibited this 12-78% of the time in Anthropic's 2024 tests |
| Horizon as upper bound | "METR's numbers are ceilings" | Benchmark horizons assume ideal tooling and no consequences; deployment is harder |

## Further Reading

- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — The original time horizon paper and methodology.
- [METR Time Horizons benchmark (Epoch AI)](https://epoch.ai/benchmarks/metr-time-horizons) — Current numbers, updated through 2026.
- [Anthropic — Measuring AI agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Internal perspective on horizons, alignment faking, and the deployment gap.
- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — HCAST, RE-Bench, SWAA task suite specifications.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — The priority hierarchy governing long-horizon Claude behavior.
