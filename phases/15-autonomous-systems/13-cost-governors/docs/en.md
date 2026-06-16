# Action Budgets, Iteration Caps & Cost Governors

> A mid-size e-commerce agent's monthly LLM costs jumped from $1,200 to $4,800 after the team enabled an "order tracking" skill. This wasn't a pricing bug. It was an agent that found a new loop and kept spending inside it. Microsoft's Agent Governance Toolkit (April 2, 2026) codifies defenses against this class of problem: per-request `max_tokens`, per-task token and dollar budgets, daily/monthly caps, iteration limits, tiered model routing, prompt caching, context windowing, HITL checkpoints on expensive actions, and kill switches on budget breach. Anthropic's Claude Code Agent SDK delivers the same primitives under different names. Financial velocity limits—like cutting access when $50 is spent in 10 minutes—catch loops faster than monthly caps.

**Type:** Learn
**Languages:** Python (standard library, tiered cost governor simulator)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 12 (Durable execution)
**Time:** ~60 minutes

## The Problem

Autonomous agents spend real money on every turn. A chatbot's bad output is a bad reply; an agent's bad loop is a bill. The industry term for this failure mode is "Denial of Wallet"—the agent keeps reasoning, keeps calling tools, keeps billing, and nothing stops it because nothing was designed to stop it.

The fix isn't a single number. It's a stack of limits at different time scales and granularities: per-request, per-task, per-hour, per-day, per-month. A well-designed stack catches runaway loops within minutes, slow leaks within hours, and bad releases within a day. When agents are long-horizon and autonomous, this same stack is what makes budgets enforceable.

This is an engineering lesson: the math is trivial, the discipline is where teams stumble. The stack below is named either in Microsoft's Agent Governance Toolkit or in Anthropic's Claude Code Agent SDK documentation.

## The Concept

### The Cost Governor Stack

1. **Per-request `max_tokens`.** Simple. Prevents any single call from emitting an unbounded completion.
2. **Per-task token budget.** The entire run, summed, may not exceed N tokens. Hard stop at cap.
3. **Per-task dollar budget.** Same as tokens but in currency. `max_budget_usd` in Claude Code.
4. **Per-tool call caps.** No more than N `WebFetch` calls, N `shell_exec` calls, etc.
5. **Iteration cap (`max_turns`).** Total iterations of the agent loop; prevents infinite reasoning loops.
6. **Per-minute / per-hour / per-day / per-month caps.** Rolling windows. Catch leaks at different time scales.
7. **Financial velocity limit.** E.g., "if more than $50 spent in 10 minutes, cut access." Catches loop-based burns before the monthly cap fires.
8. **Tiered model routing.** Default to a smaller model; escalate to a larger one only when a classifier judges the task warrants it.
9. **Prompt caching.** System prompts and stable context live in provider-side cache; re-sent tokens cost nearly zero.
10. **Context windowing.** Compression/summarization to keep active context below a threshold; directly reduces token cost.
11. **HITL checkpoints on expensive actions.** Before a known-expensive action (long tool call, large download, an expensive model escalation), require a human click.
12. **Kill switch on budget breach.** Session aborts when any cap fires. The cap is logged; a separate re-enable path is required.

### Why a Stack, Not a Single Cap

A single monthly cap only catches a runaway agent after the wallet is already gone. A single per-request cap catches nothing at the session level. Different failure modes need different time scales:

- **Runaway loops** (agent stuck in a 5-second retry): caught by velocity limit.
- **Slow leaks** (agent does ~2x expected work per task): caught by daily cap.
- **Bad releases** (new version uses 5x tokens): caught by weekly/monthly cap.
- **Legitimate surges** (real demand, not a bug): caught by hourly/daily cap with clear logging.

### Claude Code's Budget Surface

The Claude Code Agent SDK exposes (public documentation):

- `max_turns` — iteration cap.
- `max_budget_usd` — dollar cap; session aborts on breach.
- `allowed_tools` / `disallowed_tools` — tool whitelist and blacklist.
- Hook points before tool use for custom cost accounting.

Combined with the permission mode ladder (Lesson 10). An `autoMode` session without `max_budget_usd` is ungoverned autonomy. Anthropic explicitly positions Auto Mode as requiring budget controls; the classifier and cost are orthogonal.

### EU AI Act, OWASP Agentic Top 10

Microsoft's Agent Governance Toolkit covers OWASP Agentic Top 10 and EU AI Act Article 14 (human oversight) requirements. In EU production deployments, logging and cap enforcement are not optional.

### The Observed $1,200 → $4,800 Case

Real case from Microsoft documentation: an e-commerce agent whose monthly cost tripled after a new tool was added. The tool allowed the agent to poll order status during each session. No loop detection. No per-tool cap. No alerting on week-over-week growth. The fix was a per-tool cap plus a daily growth alert. This is a template: every new tool surface is a new potential loop; every new tool needs its own cap and its own alert.

## Use It

`code/main.py` simulates an agent run with and without the tiered cost governor stack. The simulated agent drifts into a polling loop after several turns; the tiered stack catches it within the velocity window, while a single monthly cap wouldn't fire for days.

## Ship It

`outputs/skill-agent-budget-audit.md` audits a proposed agent deployment's cost governor stack and flags missing layers.

## Exercises

1. Run `code/main.py`. Confirm the velocity limit fires before the iteration cap on a polling-loop trace. Now disable the velocity limit and measure how much the agent "spends" before the iteration cap catches it.

2. Design a set of per-tool caps for a browser agent (Lesson 11). Which tool needs the tightest cap? Which tool can run unbounded without risk?

3. Read Microsoft Agent Governance Toolkit documentation. List every cap type the toolkit names. Map each to a failure mode (runaway loop, slow leak, bad release, surge).

4. Price an overnight unattended run for a realistic task (e.g., "triage 50 issues in a repo"). Set `max_budget_usd` at 2x your point estimate. Justify the 2x.

5. Claude Code's `max_budget_usd` fires on session total cost. Design a complementary velocity limit you'd enforce externally. What triggers the cutoff, and what does re-enablement look like?

## Key Terms

| Term | Colloquial usage | Actual meaning |
|---|---|---|
| Denial of Wallet | "Runaway bill" | Agent loops generating spend with no cap to stop it |
| max_tokens | "Per-request cap" | Ceiling on a single completion's size |
| max_turns | "Iteration cap" | Ceiling on agent loop iterations in a session |
| max_budget_usd | "Dollar kill switch" | Session cost cap; aborts on breach |
| Velocity limit | "Rate cap" | Limit on spend per short window (e.g., $50 / 10 min) |
| Tiered routing | "Small model first" | Cheap model by default; escalate only when classifier deems it worthwhile |
| Prompt caching | "Cached system prompt" | Provider-side caching reduces re-sent token cost to near zero |
| HITL checkpoint | "Human approval gate" | Expensive action requires human click before proceeding |

## Further Reading

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`, `max_budget_usd`, tool whitelists.
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — Cost governor checkpoints.
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — Provider-side cost controls.
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/prompt-caching) — Caching mechanics.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Cost profiles of long-horizon agents.
