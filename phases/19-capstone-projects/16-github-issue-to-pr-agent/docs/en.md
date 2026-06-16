# Capstone Project 16 — GitHub Issue-to-PR Autonomous Agent

> AWS Remote SWE Agents, Cursor Background Agents, OpenAI Codex cloud, Google Jules — they all shipped the same 2026 product form factor: label an issue, get a PR. Run an agent in a cloud sandbox, verify tests pass, then open a reviewable PR with rationale. The hard parts are automatically reproducing the repo's build environment, preventing credential leaks, enforcing per-repo budgets, and ensuring the agent cannot force-push. This capstone builds the self-hosted version and benchmarks cost and pass rate against managed alternatives.

**Type:** Capstone
**Languages:** Python (agent), TypeScript (GitHub App), YAML (Actions)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools), Phase 14 (Agents), Phase 15 (Autonomous Systems), Phase 17 (Infrastructure)
**Phases covered:** P11 · P13 · P14 · P15 · P17
**Time:** 30 hours

## The Problem

Async cloud coding agents are a different product category from interactive coding agents (Capstone 01). The UX is a GitHub label. You tag an issue with `@agent fix this`, a worker spins up in a cloud sandbox, clones the repo, runs tests, edits files, verifies, and opens a PR whose body contains the agent's rationale. No interactive loop, no terminal. AWS Remote SWE Agents, Cursor Background Agents, OpenAI Codex cloud, Google Jules, and Factory Droids have all converged on this.

The engineering challenges are specific: environment reproduction (the agent must build the repo from scratch without cached dev images), flaky tests (must be rerun or isolated), credential scoping (a GitHub App with minimal fine-grained permissions), per-repo per-day budget enforcement, and a no-force-push policy. This capstone measures pass rate, cost, and security against managed alternatives.

## The Concept

The trigger is a GitHub webhook (issue label or PR comment). A dispatcher enqueues work to ECS Fargate or Lambda. The worker pulls the repo into a Daytona or E2B sandbox using a generic Dockerfile inferred from the repo (language, framework). The agent runs a mini-swe-agent or SWE-agent v2 loop against Claude Opus 4.7 or GPT-5.4-Codex. It iterates: read code, propose fix, apply patch, run tests.

Verification is the gate step. Before the PR opens, full CI must pass inside the sandbox. Coverage delta is computed; if it goes negative beyond a threshold, the PR still opens but gets labeled `needs-review`. The agent publishes its rationale as the PR description, plus an `@agent` thread that reviewers can @ for follow-ups.

Security is scoped via two distinct GitHub surfaces: the App provides a short-lived installation token with `workflows: read` and narrow repo contents/PR scope; branch protection (not app permissions) enforces "no direct writes to `main`" and "no force-push" — the app is never added to the bypass list. Path-level read-only access to `.github/workflows` is not a real GitHub App primitive, so the agent's file-edit whitelist must enforce this at the worker level. Per-repo per-day budget caps are enforced at the dispatcher (e.g., max 5 PRs per repo per day, $20 per PR).

## Architecture

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## Tech Stack

- Trigger: GitHub App with fine-grained tokens; webhook receiver via Lambda or Fly.io
- Worker: ECS Fargate task (or GitHub Actions self-hosted runner)
- Sandbox: one Daytona devcontainer or E2B sandbox per task
- Agent loop: mini-swe-agent baseline or SWE-agent v2 powered by Claude Opus 4.7 / GPT-5.4-Codex
- Retrieval: tree-sitter repo-map + ripgrep
- Verification: full in-sandbox CI + coverage delta gate
- Observability: Langfuse, one trace archive per PR, linked from the PR body
- Budget: per-repo per-day dollar cap; max PRs per repo per day

## Build It

1. **GitHub App.** Fine-grained installation token: issues read+write, pull_requests write, contents read+write, workflows read. Branch protection (the only surface that can do this) enforces "no direct push to `main`" and "no force-push"; the app is not on the bypass list. The worker enforces "no writes to files under `.github/workflows`" as a whitelist check on proposed diffs, since GitHub App permissions are not path-level.

2. **Webhook receiver.** Lambda function receives issue label / PR comment webhooks. Filters by label `@agent fix this`. Enqueues to SQS.

3. **Dispatcher.** Pops tasks from SQS. Enforces per-repo per-day budget. Launches an ECS Fargate task with repo URL, issue body, and a fresh Daytona sandbox.

4. **Environment inference.** Detects language (Python, Node, Go, Rust) and package manager (uv, pnpm, go mod, cargo). Generates a Dockerfile on the fly if none exists.

5. **Agent loop.** mini-swe-agent or SWE-agent v2 with Claude Opus 4.7. Tools: ripgrep, tree-sitter repo-map, read_file, edit_file, run_tests, git. Hard limits: $20 cost, 30-minute wall clock, 30 agent turns.

6. **Verification.** After the loop finishes, run the full test suite in-sandbox. Compute coverage delta via jacoco / coverage.py. If CI is red: stop, do not open a PR. If coverage drops more than 2%: open the PR and label it `needs-review`.

7. **PR publishing.** Push the agent branch. Open a PR via GitHub API with: title, rationale, diff summary, trace URL, cost, turn count.

8. **Credential hygiene.** The worker runs with a short-lived GitHub App installation token. Logs are scrubbed of secrets before archiving.

9. **Evaluation.** 30 internal seed issues of varying difficulty. Measure pass rate, PR quality (diff size, style, coverage), cost, latency. Benchmark against Cursor Background Agents and AWS Remote SWE Agents on the same issues.

## Use It

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## Ship It

`outputs/skill-issue-to-pr.md` is the deliverable. A GitHub App + async cloud worker that turns labeled issues into reviewable PRs with bounded cost and scoped credentials.

| Weight | Criterion | How to measure |
|:-:|---|---|
| 25 | Pass rate on 30 issues | End-to-end success (CI green + coverage OK) |
| 20 | PR quality | Diff size, coverage delta, style consistency |
| 20 | Cost and latency per resolved issue | Dollars and wall clock per PR |
| 20 | Security | Scoped tokens, per-repo budget, no force-push, credential hygiene |
| 15 | Operator experience | Rationale comments, retry operability, @-mention follow-ups |
| **100** | | |

## Exercises

1. Add a "fix flaky test" mode: label `@agent stabilize-flake TestX` runs that test 50 times in the sandbox and proposes a minimal change to stabilize it.

2. Benchmark cost vs Cursor Background Agents on three shared issues. Report which tool wins where.

3. Implement a budget dashboard: per-repo per-day cost, per-user cost. Alert on anomalies.

4. Build a "dry-run" mode that opens a draft PR without running CI, letting reviewers cheaply inspect the plan.

5. Add a retention policy: PR branches not merged within 7 days are automatically deleted.

## Key Terms

| Term | What people call it | What it actually is |
|------|---------------------|------------------------|
| GitHub App | "Scoped bot identity" | An App with fine-grained permissions + short-lived installation tokens |
| Async cloud agent | "Background agent" | A non-interactive worker running in a cloud sandbox, not a terminal |
| Environment inference | "Dockerfile synthesis" | Detecting language + package manager and generating a Dockerfile if none exists |
| Verification | "In-sandbox CI" | Running the full test suite inside the worker before opening a PR |
| Coverage delta | "Coverage hold" | The percentage change in test coverage from baseline to agent branch |
| Per-repo budget | "Daily cap" | Dollar and PR-count limits enforced at the dispatcher |
| Rationale | "PR body explanation" | The agent's summary of what changed and why; mandatory in the PR body |

## Further Reading

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — Standard async cloud agent reference
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI reference
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — Commercial alternative
- [OpenAI Codex (cloud)](https://openai.com/codex) — Managed competitor
- [Google Jules](https://jules.google) — Google's managed version
- [Factory Droids](https://www.factory.ai) — Alternative commercial reference
- [GitHub App documentation](https://docs.github.com/en/apps) — Scoped bot identity
- [Daytona cloud sandboxes](https://daytona.io) — Reference sandbox
