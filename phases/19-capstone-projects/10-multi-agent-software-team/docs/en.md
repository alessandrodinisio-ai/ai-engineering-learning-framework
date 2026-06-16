# Capstone Project 10 — Multi-Agent Software Engineering Team

> SWE-AF's factory architecture, MetaGPT's role-based prompting, AutoGen 0.4's typed actor graph, Cognition's Devin, and Factory's Droids all converge on the same 2026 pattern: an architect plans, N coders work in parallel worktrees, a reviewer gates, a tester verifies. Parallel worktrees trade wall-clock time for throughput. Shared state and handoff protocols become the failure surface. This capstone project builds that team, evaluates it on SWE-bench Pro, and reports which handoffs break and how often.

**Type:** Capstone
**Languages:** Python / TypeScript (agents), Shell (worktree scripts)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools), Phase 14 (Agents), Phase 15 (Autonomous Systems), Phase 16 (Multi-Agent), Phase 17 (Infrastructure)
**Phases involved:** P11 · P13 · P14 · P15 · P16 · P17
**Time:** 40 hours

## The Problem

Single-agent coding shells hit a ceiling on large tasks. Not because any individual agent is weak, but because a 200k-token context cannot hold an architecture plan plus four parallel codebase slices plus reviewer comments plus test output. Multi-agent factories decompose the problem: the architect owns the plan, coders own implementation in parallel worktrees, a reviewer gates, a tester verifies. SWE-AF's "factory" architecture, MetaGPT's roles, and AutoGen's typed actor graph—three frameworks describing the same pattern.

The failure surface is in the handoffs. The architect plans something a coder cannot implement. Coders produce conflicting diffs. The reviewer approves a hallucinated fix. The tester races a coder who is still writing. You will build such a team, run it on 50 SWE-bench Pro issues, track every handoff, and publish a retrospective.

## The Concept

Roles are typed agents. **Architect** (Claude Opus 4.7) reads the issue, writes a plan, and decomposes it into subtasks with explicit interfaces. **Coders** (Claude Sonnet 4.7, N parallel instances, each in a `git worktree` + Daytona sandbox) implement subtasks independently. **Reviewer** (GPT-5.4) reads the merged diff and either approves or requests specific changes. **Tester** (Gemini 2.5 Pro) runs the test suite in an isolated environment and reports pass/fail with artifacts.

Communication goes through a shared task board (file-backed or Redis). Each role consumes tasks it is allowed to handle. Handoffs are typed messages on the A2A protocol. Coordination concerns: merge conflict resolution (a coordinator role or automatic three-way merge), shared-state synchronization (plan freezes once coders start; replanning is a separate event), and reviewer gating (reviewer cannot approve its own changes or changes it proposed).

Token amplification is the hidden cost. Every role boundary adds summarization prompts and handoff context. A 40-turn single-agent run becomes 160 total turns across four roles. The scoring rubric explicitly measures token efficiency versus the single-agent baseline, because the question is not "does multi-agent work" but "does it win per dollar."

## Architecture

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## Tech Stack

- Orchestration: LangGraph with shared state + per-agent subgraphs
- Messaging: A2A protocol (Google 2025) for typed inter-agent messages
- Models: Opus 4.7 (architect), Sonnet 4.7 (coders), GPT-5.4 (reviewer), Gemini 2.5 Pro (tester)
- Worktree isolation: `git worktree add` per coder + Daytona sandbox
- Merge coordinator: Custom three-way merge + LLM-mediated conflict resolution
- Evaluation: SWE-bench Pro (50 issues), SWE-AF scenarios, HumanEval++ for unit tests
- Observability: Langfuse with role-tagged spans, per-agent token accounting
- Deployment: K8s with separate Deployment per role + HPA on backlog

## Build It

1. **Task board.** File-backed JSONL with typed messages: `plan_request`, `subtask`, `diff_ready`, `review_needed`, `test_needed`, `approved`, `rejected`, `replan_needed`. Agents subscribe by tag.

2. **Architect.** Read GitHub issue, run Opus 4.7 with a planning template that requires explicit subtask interfaces (files touched, public functions, test impact). Produce a `plan_request` with a subtask DAG.

3. **Coders.** N parallel workers, each claiming a subtask from the board. Each spins up a fresh `git worktree add` branch plus a Daytona sandbox. Implements the subtask. Produces a `diff_ready` with patch + test delta.

4. **Merge coordinator.** After all coders finish, three-way merge the N branches into a staging branch. LLM-mediated conflict resolution only where file-level overlap exists.

5. **Reviewer.** GPT-5.4 reads the merged diff. Cannot approve a diff it wrote itself. Produces `approved` (no-op) or `review_feedback` with specific change requests, routed to the relevant coder.

6. **Tester.** Gemini 2.5 Pro runs the test suite in a clean sandbox. Captures artifacts. Produces `test_passed` or `test_failed` with stack traces. Failed tests loop back to the coder that owns the failing subtask.

7. **Handoff accounting.** Every message that crosses a role boundary gets a span in Langfuse with payload size and model used. Compute token amplification per subtask: (coder_tokens + reviewer_tokens + tester_tokens + architect_share) / coder_tokens.

8. **Evaluation.** Run on 50 SWE-bench Pro issues. Compare pass@1 and cost per resolved issue against a single-agent baseline (one Sonnet 4.7 in one worktree).

9. **Retrospective.** For each failed issue, identify the broken handoff (plan too vague, merge conflict, reviewer false approval, tester flake). Produce a handoff-failure histogram.

## Use It

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## Ship It

`outputs/skill-multi-agent-team.md` is the deliverable. Given an issue URL and parallelism level, the team produces a mergeable PR with per-role token accounting.

| Weight | Criterion | How to Measure |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | pass@1 on matched 50-issue subset |
| 20 | Parallel speedup | Wall clock vs single-agent baseline |
| 20 | Review quality | False-approval rate on injected-bug probes |
| 20 | Token efficiency | Total tokens per resolved issue vs single agent |
| 15 | Coordination engineering | Merge conflict resolution, handoff-failure histogram |
| **100** | | |

## Exercises

1. Inject an obvious bug into a diff mid-run (add a `return None` before the body). Measure the reviewer's false-approval rate. Tune the reviewer prompt until false approvals drop below 5%.

2. Reduce to two coders (architect + coder + reviewer + tester, with the coder running two subtasks sequentially). Compare wall clock and pass rate.

3. Replace the merge coordinator with a single-writer constraint (subtasks touch disjoint file sets). Measure the planning burden on the architect.

4. Swap the reviewer from GPT-5.4 to Claude Opus 4.7. Measure false-approval rate and token cost delta.

5. Add a fifth role: Documenter (Haiku 4.5). After review, it produces a changelog entry. Measure whether documentation quality justifies the extra tokens.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| Parallel worktree | "isolated branch" | `git worktree add` gives each coder a fresh working tree |
| Task board | "shared message bus" | A file or Redis store of typed messages that agents subscribe to |
| Handoff | "role boundary" | Any message crossing from one role's context to another's |
| Token amplification | "multi-agent overhead" | Total tokens across roles on the same task / single-agent tokens |
| A2A protocol | "agent-to-agent" | Google's 2025 spec for typed inter-agent messaging |
| Merge coordinator | "integrator" | The component that runs three-way merge and mediates conflicts |
| False approval | "reviewer hallucination" | A reviewer approves a diff containing a known bug |

## Further Reading

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — reference 2026 multi-agent factory
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — role-based multi-agent framework
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoft's typed actor framework
- [Cognition AI (Devin)](https://cognition.ai) — reference product
- [Factory Droids](https://www.factory.ai) — alternative reference product
- [Google A2A protocol](https://developers.google.com/agent-to-agent) — inter-agent messaging spec
- [git worktree documentation](https://git-scm.com/docs/git-worktree) — isolation substrate
- [SWE-bench Pro](https://www.swebench.com) — evaluation target
