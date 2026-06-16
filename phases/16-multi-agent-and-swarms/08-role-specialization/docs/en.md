# Role Specialization — Planner, Critic, Executor, Verifier

> The most common 2026 multi-agent decomposition: one agent plans, one executes, one critiques or verifies. MetaGPT (arXiv:2308.00352) formalizes it as SOPs encoded into role prompts — product manager, architect, project manager, engineer, QA engineer — following `Code = SOP(Team)`. ChatDev (arXiv:2307.07924) uses a "chat chain" stringing designer, programmer, reviewer, tester with "communicative dehallucination" (agents explicitly request missing details). The verifier is load-bearing: Cemri et al. (MAST, arXiv:2503.13657) show every multi-agent failure traces back to verification missing or broken. PwC reports a 7x accuracy lift (10% → 70%) from structured validation loops in CrewAI.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 05 (Supervisor)
**Time:** ~60 min

## The Problem

Generic multi-agent systems produce generic output. Three coders in a group chat will write three flavors of the same mediocre code. You can add more agents, more rounds, and still not cross the quality threshold.

The fix isn't more agents — it's *different* agents. Assign clear roles. Give the critic tools the planner doesn't have. Give the verifier an objective test suite. Now the system has internal disagreement with solid error correction, not just parallel guessing.

## The Concept

### The Four Standard Roles

**Planner.** Reads the goal, produces a step list or spec. Tools: knowledge retrieval, documentation. Output: structured plan.

**Executor.** Reads one plan step at a time, produces artifacts. Tools: the real work tools (code compiler, shell, API clients). Output: artifacts.

**Critic.** Reads the executor's output against the planner's intent. Tools: read-only access to artifacts, static analysis. Output: accept/reject with rationale.

**Verifier.** Reads artifacts and runs a deterministic check. Tools: test runner, type checker, schema validator. Output: pass/fail with evidence.

The critic is subjective, opinionated, often LLM-based. The verifier is objective, deterministic, often code-based. They are not the same role.

### MetaGPT's SOP Pattern

MetaGPT (arXiv:2308.00352) encodes software engineering SOPs into role prompts:

- **Product Manager** writes the PRD.
- **Architect** produces system design.
- **Project Manager** splits tasks.
- **Engineer** implements.
- **QA Engineer** runs tests.

Each role has a strict input/output schema. The role prompt states what this role *is* and what it *must produce*. The expression `Code = SOP(Team)` — a deterministic SOP turns a team of LLMs into a predictable pipeline.

### ChatDev's Communicative Dehallucination

ChatDev adds a key move: when the executor needs a specific detail not in the plan, it explicitly asks the designer before continuing. This prevents the classic LLM failure of confidently making things up.

Implementation: the role prompt includes "when you need specific information you weren't told, ask the relevant role by name before producing output."

### Why the Verifier Matters Most

Cemri et al. (MAST) traced 1642 multi-agent execution failures. 21.3% were verification gaps — the system delivered an answer no one checked. The remaining 79% often also traced back to "a check that silently failed or was never run." Verification is the load-bearing role.

PwC's report (CrewAI deployment, 2025) states that adding a structured validation loop lifted accuracy from 10% to 70%. One role, 7x improvement.

### Critic vs Verifier

- A critic is an LLM reviewing an artifact for quality. Subjective. Can be fooled by plausible wording.
- A verifier is a deterministic program running against artifacts. Objective. Gives pass/fail with evidence.

Use both. The critic catches taste issues the verifier can't articulate. The verifier catches bugs the critic can't see — because they only manifest at runtime.

### Anti-pattern

Every role in your system is an LLM, and every role's output is "looks good to me." Classic MAST failure mode. Add at least one verifier whose pass/fail is determined by code, not an LLM.

### Framework Mapping

- **CrewAI** — `Agent(role, goal, backstory)` is the textbook specialization interface.
- **LangGraph** — nodes can have specialized prompts; edges enforce the pipeline.
- **AutoGen** — named role-specific ConversableAgents in a GroupChat.
- **OpenAI Agents SDK** — handoff tools between role-specialized Agents.

## Build It

`code/main.py` implements a 4-role pipeline building a simple Python function:

- **Planner** produces a spec.
- **Executor** generates a code string.
- **Critic** (LLM simulation) flags obvious issues.
- **Verifier** runs the generated code in a sandbox (`exec`) against a test case.

The demo runs twice: once where the executor produces correct code (critic + verifier both pass), and once where the executor produces code that deviates from spec (critic misses the bug because it looks plausible, verifier catches it because the test fails).

Run:

```
python3 code/main.py
```

## Use It

`outputs/skill-role-designer.md` takes a task and produces a role roster (3-5 roles), each role's input/output schema, and verifier checks. Use it before wiring agents into a framework.

## Ship It

Checklist:

- **At least one deterministic verifier.** Never all-LLM.
- **Explicit I/O schema per role.** Planner returns a spec, not prose; executor reads that schema.
- **Communicative dehallucination.** Executor must ask planner when it lacks information; never invent.
- **Critic/verifier ordering.** Run critic first (cheap, catches design issues), then verifier (slow, catches bugs).
- **Loop budget.** Critic-executor revision max 2 rounds, then escalate to human.

## Exercises

1. Run `code/main.py`, observe how the verifier catches a bug the critic missed. Add a static analysis check (count the occurrences of `return`) as an additional verifier. What does it catch that the runtime test misses?
2. Add a 5th role: "requirements analyst" that translates user wishes into planner-ready specs. Which communicative dehallucination requests should flow up to it?
3. Read MetaGPT Section 3 ("Agents"). List the input/output schema for each of MetaGPT's 5 roles.
4. Read ChatDev's chat-chain diagram (arXiv:2307.07924 Figure 3). Identify where communicative dehallucination breaks a loop that would otherwise be infinite.
5. PwC's 7x accuracy lift comes from a validation loop. Name three tasks where adding a verifier wouldn't help — tasks where deterministically checking correctness is impossible or prohibitively expensive.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Role specialization | "Different agents, different jobs" | Distinct system prompts tuned for planner/executor/critic/verifier roles. |
| SOP pattern | "Encoded standard operating procedure" | MetaGPT's formulation: strict I/O schemas per role turn a team into a pipeline. |
| Communicative dehallucination | "Ask before making up" | ChatDev's pattern: executor asks planner for missing details rather than inventing them. |
| Critic | "LLM reviewer" | Subjective, opinionated review. Catches taste issues. Can be fooled by plausible wording. |
| Verifier | "Deterministic check" | Code-based pass/fail. Test runner, type checker, schema validator. Can't be fooled. |
| Verification gap | "Nobody checked" | 21.3% of MAST failures. An answer shipped without a check that would have caught the bug. |
| Revision loop | "Critic sends it back" | Critic rejects → executor re-runs with feedback. Needs a budget. |
| All-LLM anti-pattern | "Looks good to me" | Every role is an LLM with no deterministic check. Classic MAST failure. |

## Further Reading

- [Hong et al. — MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) — SOP-as-role-prompt reference paper
- [Qian et al. — Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924) — chat chain + communicative dehallucination
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy; verification gaps account for 21.3% of failures
- [CrewAI docs — Agent roles](https://docs.crewai.com/en/introduction) — production-grade role description interface
