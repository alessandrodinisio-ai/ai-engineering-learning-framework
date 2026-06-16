# Capstone Project 01 — Terminal-Native Coding Agent

> By 2026, the shape of coding agents has crystallized. A TUI shell, a stateful plan, a sandboxed tool surface, and a loop responsible for planning, acting, observing, and recovering. From fifty meters away, Claude Code, Cursor 3, and OpenCode all look the same. This capstone requires you to build one end-to-end — input is CLI, output is a pull request — and benchmark it on SWE-bench Pro against mini-swe-agent and Live-SWE-agent. You will discover that the hard part is not the model call, but the tool loop, the sandbox, and the cost ceiling of a 50-turn run.

**Type:** Capstone
**Languages:** TypeScript / Bun (harness), Python (evaluation scripts)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools & Protocols), Phase 14 (Agents), Phase 15 (Autonomous Systems), Phase 17 (Infrastructure)
**Phases involved:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**Time:** 35 hours

## The Problem

In 2026, coding agents became the dominant AI application category. Claude Code (Anthropic), Cursor 3 with Composer 2 and Agent Tabs (Cursor), Amp (Sourcegraph), OpenCode (112k stars), Factory Droids, Google Jules — all are variants of the same architecture: a terminal shell, a permissioned tool surface, a sandbox, and a plan-act-observe loop wrapped around a frontier model. The frontier is narrow — Live-SWE-agent with Opus 4.5 hits 79.2% on SWE-bench Verified — but the engineering craft space is wide. Most failure modes are not model errors. They are tool-loop instability, context poisoning, runaway token costs, and destructive filesystem operations.

You cannot reason about these agents from the outside. You must build one, watch it crash its loop at turn 47 because ripgrep returned 8MB of matches, and then rebuild the truncation layer. That is the point of this capstone.

## The Concept

The harness has four surfaces. **Plan** maintains a TodoWrite-style state object that the model rewrites every turn. **Act** dispatches tool calls (read, edit, run, search, git). **Observe** captures stdout / stderr / exit codes, truncates, and feeds summaries back. **Recover** handles tool errors without blowing up the context window or entering infinite loops. The 2026 shape adds one more thing: **hooks**. `PreToolUse`, `PostToolUse`, `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `Notification`, `Stop`, `PreCompact` — a set of configurable extension points where operators inject policies, telemetry, and guardrails.

The sandbox is E2B or Daytona. Each task runs in a fresh devcontainer with a read-write git worktree mounted. The harness never touches the host filesystem. On task success or failure, the worktree is torn down. Cost control is enforced at three layers: per-turn token ceiling, per-session dollar budget, and a hard turn limit (typically 50). The observability layer is OpenTelemetry spans with GenAI semantic conventions, shipped to a self-hosted Langfuse.

## Architecture

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## Tech Stack

- Shell runtime: Bun 1.2 + Ink 5 (React in the terminal)
- Model access: OpenRouter unified API, connecting Claude Sonnet 4.7, GPT-5.4-Codex, Gemini 3 Pro, Opus 4.5 (for the hardest tasks)
- Tool transport: Model Context Protocol StreamableHTTP (MCP 2026 revision)
- Code search: ripgrep subprocess, pre-compiled tree-sitter parsers for 17 languages
- Isolation: one `git worktree add` per task, cleaned up on success / failure
- Evaluation harness: SWE-bench Pro (verified subset) + Terminal-Bench 2.0 + your own 30-task holdout set
- Observability: OpenTelemetry SDK with `gen_ai.*` semantic conventions → self-hosted Langfuse
- PR publishing: GitHub App with fine-grained tokens, scoped to the target repository only

## Build It

1. **TUI and command loop.** Set up a Bun project with Ink. Accept `agent run <repo> "<task>"`. Render a split-pane view: plan panel (top), tool-call stream (middle), token budget (bottom). Add Ctrl-C cancellation that fires the `SessionEnd` hook before exiting.

2. **Plan state.** Define a typed TodoWrite schema (entries with pending / in_progress / done states, with notes). The model rewrites the entire state as a tool call every turn — do not let it mutate in place incrementally. Persist the plan to `.agent/state.json` so it can resume after crashes.

3. **Tool surface.** Define six tools: `read_file`, `edit_file` (with diff preview), `ripgrep`, `tree_sitter_symbols`, `run_shell` (with timeout), `git` (status / diff / commit / push). Expose them via MCP StreamableHTTP so the harness is transport-agnostic. Each tool returns truncated output (4k token cap per call).

4. **Sandbox wrapper.** Spin up an E2B sandbox for each task. `git worktree add -b agent/$TASK_ID` checks out a fresh branch. All tool calls execute inside the sandbox. The host filesystem is unreachable.

5. **Hooks.** Implement all eight 2026 hook types. Wire up at least four user-authored hooks: (a) `PreToolUse` destructive-command guard that blocks `rm -rf` outside the worktree; (b) `PostToolUse` token accounting; (c) `SessionStart` budget initialization; (d) `Stop` that writes a final trace bundle.

6. **Evaluation loop.** Clone a 30-issue subset of SWE-bench Pro Python. Run your harness on each. Compare pass@1, turns-per-task, and dollars-per-task against mini-swe-agent (minimal baseline). Write results to `eval/results.jsonl`.

7. **Cost control.** Hard cutoffs: 50 turns, 200k context, $5 per task. The `PreCompact` hook at the 150k mark summarizes earlier turns into a prior-state block, freeing room for new observations without losing the plan.

8. **PR publishing.** On success, the final step is `git push` plus a GitHub API call to open a PR with the plan and diff summary in the body.

## Use It

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## Ship It

The deliverable skill lives at `outputs/skill-terminal-coding-agent.md`. Given a repository path and a task description, it runs the full plan-act-observe loop inside a sandbox and returns a PR URL plus a trace bundle. Grading rubric for this capstone:

| Weight | Criterion | How to measure |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs baseline | Your harness vs mini-swe-agent on 30 matched Python tasks |
| 20 | Architectural clarity | Plan/act/observe separation, hook surface, tool schemas — reviewed against Live-SWE-agent layout |
| 20 | Safety | Sandbox escape tests, permission prompts, destructive-command guards survive red-teaming |
| 20 | Observability | Trace completeness (100% of tool calls have spans), per-turn token accounting |
| 15 | Developer experience | Cold start < 2s, crash recovery resumes the plan, Ctrl-C cleanly cancels mid-tool-execution |
| **100** | | |

## Exercises

1. Swap the backing model from Claude Sonnet 4.7 to Qwen3-Coder-30B served on vLLM. Compare pass@1 and dollars-per-task. Report where the open-source model falls short.

2. Add a `reviewer` sub-agent that reads the diff before opening a PR and can trigger a revision loop. Test whether false-positive reviews push SWE-bench pass rate below the single-agent baseline (hint: they usually do).

3. Stress-test the sandbox: write a task that attempts to `curl` an external URL, and another that attempts to write outside the worktree. Confirm both are blocked by the PreToolUse hook. Log the attempts.

4. Implement `PreCompact` summarization using a smaller model (Haiku 4.5). Measure plan fidelity loss at 3x compression.

5. Replace the MCP StreamableHTTP transport with stdio. Benchmark cold-start and per-call latency. Pick a winner for purely local use.

## Key Terms

| Term | What people call it | What it actually is |
|------|---------------------|---------------------|
| Harness | "the agent loop" | The code wrapping the model that dispatches tools, maintains plan state, and enforces budgets |
| Hook | "agent event listener" | A user-authored script run by the harness on one of eight lifecycle events |
| Worktree | "git sandbox" | A linked git checkout at another path; disposable without touching the main clone |
| TodoWrite | "plan state" | A typed list of pending/in-progress/done entries rewritten by the model every turn |
| StreamableHTTP | "MCP transport" | The 2026 MCP revision: long-lived HTTP with bidirectional streaming; replaces SSE |
| Token ceiling | "context budget" | A per-turn or per-session cap on input+output tokens; triggers compaction or termination |
| pass@1 | "single-attempt pass rate" | The fraction of SWE-bench tasks solved on the first run, no retries, no peeking at the test suite |

## Further Reading

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — Anthropic's reference harness
- [Cursor 3 changelog](https://cursor.com/changelog) — Agent Tabs and Composer 2 product notes
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — minimal baseline for SWE-bench harness comparison
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 79.2% on SWE-bench Verified with Opus 4.5
- [OpenCode](https://opencode.ai) — open-source harness, 112k stars
- [SWE-bench Pro leaderboard](https://www.swebench.com) — the benchmark this capstone targets
- [Model Context Protocol 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP, capability metadata
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — span schema for tool calls and token usage
