# Autonomous Coding Agent Landscape (2026)

> SWE-bench Verified went from 4% to 80.9% in under three years. The same Claude Sonnet 4.5 scores 43.2% with SWE-agent v1 and 59.8% with Cline autonomous—scaffolding around the model now matters as much as the model itself. OpenHands (formerly OpenDevin) is the most active MIT-licensed platform; its CodeAct loop executes Python actions directly in a sandbox rather than JSON tool calls. These headline numbers mask a methodological issue: 161 of SWE-bench Verified's 500 tasks require only 1-2 line changes, and SWE-bench Pro (10+ line tasks) puts the same frontier models at 23-59%.

**Type:** Learn
**Languages:** Python (standard library, CodeAct vs JSON tool call comparison)
**Prerequisites:** Phase 14 · 07 (Tool use), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45 minutes

## The Problem

"Which coding agent is best" is the wrong question. The right question: on a task distribution that matches my work, with the scaffolding I'd run in production, what end-to-end reliability can I get?

Between 2022 and 2026, the field learned that scaffolding—the retrieval layer, the planner, the sandbox, the edit-verify loop, the feedback format—is load-bearing. Claude Sonnet 4.5 scores 43.2% on SWE-bench Verified with SWE-agent v1; the same model inside Cline's autonomous scaffolding scores 59.8%. Same weights, 16.6 absolute percentage points apart. The base model is a component; the loop is the product.

The companion issue is that benchmark saturation masks regressions. SWE-bench Verified is near saturation, and the easy-task tail (161 of 500 tasks need ≤2 lines) pulls top scores up. Real-world quality is better measured on distributions like SWE-bench Pro (10+ line changes), where the same leaders still land at 23-59%.

## The Concept

### SWE-bench in One Paragraph

SWE-bench (Jimenez et al.) takes real GitHub issues with real patches and asks the agent to produce a patch that passes the test suite. SWE-bench Verified (OpenAI, 2024) is a human-curated 500-task subset that removes ambiguous and broken tasks. SWE-bench Pro is the harder successor—tasks requiring 10+ line changes, where current frontier agents land at 23-59%.

### What the 2022 → 2026 Curve Actually Shows

- **2022**: Research models at ~4% on raw SWE-bench.
- **2024**: GPT-4 + Devin-style scaffolding at ~14%; SWE-agent at ~12%.
- **2025**: Claude 3.5/3.7 Sonnet inside Aider and SWE-agent pushing to 40-55% range.
- **2026**: Claude Sonnet 4.5 and frontier competitors at 70-80%+ on SWE-bench Verified. Epoch AI leaderboard tracks this live.

The slope comes from three compounding sources: better base models, better scaffolding (CodeAct, reflection, verifier loops), and better benchmarks (Verified removes noise).

### CodeAct vs. JSON Tool Calls

OpenHands (All-Hands-AI, arXiv:2407.16741, formerly OpenDevin) makes a specific architectural bet: instead of the model emitting JSON tool calls that the host decodes and executes, the model emits Python code and a Jupyter-style kernel runs it inside a sandbox. The agent can traverse files, chain tools, and catch its own exceptions in a single action.

Trade-offs:

- **JSON tool calls**: one action per turn; easy to audit; limited composability; safe by default because each call passes through an explicit validator.
- **CodeAct**: one action can be an entire program; composable; needs a hardened sandbox (OpenHands uses Docker isolation); failure modes include anything the sandbox runtime allows.

Both architectures are in production. CodeAct dominates in open platforms (OpenHands, smolagents). JSON tool calls still dominate in managed services (Anthropic Managed Agents, OpenAI Assistants) where the executor is controlled by the provider.

### Scaffolding in the 2026 Landscape

| Scaffold | License | Execution model | Notable property |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | CodeAct in Docker | Most active open platform; event stream replayable |
| SWE-agent | MIT | Agent-Computer Interface (ACI) | First end-to-end SWE-bench scaffold |
| Aider | Apache-2 | Diff edits in local repo | Minimal scaffolding, strong fallback stability |
| Cline | Apache-2 | VS Code agent with tool policy | Highest-scoring open scaffold on Sonnet 4.5 |
| Devin (Cognition) | Proprietary | Hosted VM + planner | First "AI software engineer" product category |
| Claude Code | Proprietary | Permission modes + routines | Lesson 10 covers its agent loop in detail |

### Why Scaffolding Dominates

A coding run is a long-horizon trajectory (Lesson 1). Reliability compounds across steps. The three places scaffolding buys points:

1. **Retrieval**: Finding the right files to read is the silent bottleneck. SWE-agent's ACI, OpenHands' file index, Aider's repo-map all attack this.
2. **Verifier loop**: Running tests, reading the stack trace, and retrying is 10+ percentage points on SWE-bench.
3. **Failure containment**: A sandbox that rolls back on error prevents damage accumulation. Same model with vs. without a verifier loop looks like two different products.

### Benchmark Saturation and Real Distributions

OpenHands' authors and Epoch AI both note that SWE-bench Verified has an easy tail: 161 of 500 tasks require only 1-2 line changes. High scores are partly driven by this tail. SWE-bench Pro restricts to 10+ line changes and returns 23-59% range scores even for frontier systems. Your production distribution is almost certainly closer to Pro than to Verified.

Implication for choosing an agent: run a Pro-like subset on your own bug backlog. The score that matters is the one on tasks representing what you actually ship.

## Use It

`code/main.py` compares two toy agent scaffolds on a fixed mini-task distribution:

1. A **JSON tool call** scaffold with one action per turn.
2. A **CodeAct** scaffold where each action can emit a small Python program.

Both use a stub "model" (deterministic rules) so the comparison isolates scaffold from model quality. Output shows the CodeAct scaffold solving more tasks in fewer rounds at the cost of larger blast radius per action.

## Ship It

`outputs/skill-scaffold-audit.md` helps you audit a coding agent scaffold before adoption: retrieval quality, verifier presence, sandbox isolation, and benchmark-to-distribution fit.

## Exercises

1. Run `code/main.py`. How many rounds does each scaffold spend on the same task set? What's the blast radius per action for each?

2. Read the OpenHands paper (arXiv:2407.16741). The paper argues CodeAct beats JSON tool calls on complex tasks. Identify one failure mode the paper acknowledges and state in one sentence when that mode would dominate in production.

3. Pick a task from your bug backlog that requires 10+ line changes across two files. Estimate end-to-end success probability with a frontier model under (a) JSON tool calls and (b) CodeAct. Argue the gap.

4. SWE-bench Verified has 161 single-file, 1-2 line tasks. Construct a scoring that excludes them. How does the leaderboard re-rank?

5. Read "Introducing SWE-bench Verified" (OpenAI). Explain the specific methodology used to remove ambiguous tasks and name one category that curation would miss.

## Key Terms

| Term | Colloquial usage | Actual meaning |
|---|---|---|
| SWE-bench | "The coding benchmark" | Real GitHub issues with real patches and test suites |
| SWE-bench Verified | "The cleaned subset" | 500 human-curated tasks, still containing an easy tail |
| SWE-bench Pro | "The harder subset" | 10+ line changes; frontier at 23-59% |
| CodeAct | "Code as action" | Agent emits Python; Jupyter-style kernel executes in sandbox |
| JSON tool call | "Function calling" | Each action is a structured JSON payload validated before execution |
| Scaffold | "Agent framework" | Retrieval + planner + executor + verifier loop around base model |
| ACI (Agent-Computer Interface) | "SWE-agent's format" | Command set designed for LLM ergonomics, not human shell |
| Verifier loop | "Test then retry" | Runs tests, reads output, revises patch; biggest non-model reliability gain |

## Further Reading

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — Original benchmark and methodology.
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — How the curated subset was made.
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct architecture and event-stream design.
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — Live-tracked scores.
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — Framing for long-horizon coding agent reliability.
