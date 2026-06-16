# Capstone Project 05 — Autonomous Research Agent (AI-Scientist Grade)

> Sakana's AI-Scientist-v2 published complete papers. Agent Laboratory ran full experiments. Allen AI released traces. The 2026 shape is a plan-execute-verify tree search over experiments with budgeted cost, sandboxed code execution, a LaTeX writer with vision feedback, and an automated ensemble of NeurIPS-style reviewers. This capstone requires you to build one, run it end-to-end within $30 per paper, and survive the sandbox-escape red team that Sakana documented.

**Type:** Capstone
**Languages:** Python (agent + sandbox), LaTeX (output)
**Prerequisites:** Phase 2 (ML), Phase 3 (Deep Learning), Phase 7 (Transformers), Phase 10 (LLMs from Scratch), Phase 14 (Agents), Phase 15 (Autonomous Systems), Phase 16 (Multi-Agent), Phase 18 (Ethics & Safety)
**Phases involved:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**Time:** 40 hours

## The Problem

Autonomous research agents crossed a threshold in 2026. Sakana AI's AI-Scientist-v2 was published in Nature, with generated papers passing workshop-level peer review. ShinkaEvolve (ICLR 2026) extended the line to evolutionary hypotheses. AMD's Agent Laboratory delivered reproducible traces. These agents are not magic — they are a plan-execute-verify loop running on a tree of candidate experiments, with cost caps, deterministic-seed sandboxes, and automated reviewing. The craft is in the loop, the budget, and the safety narrative.

You learn it by implementing such a loop on a narrow domain for a seed idea (e.g., attention sparsity ablations on a 100M-parameter transformer). The value is not in discovering something novel on the first run. The value is in the infrastructure: tree search, experiment sandbox, writer-reviewer loop, red-team report. The Sakana team documented sandbox-escape failures; your agent must pass the same red team.

## The Concept

The agent is a best-first tree search. Nodes are experiment specs: (hypothesis, config, code, expected results). An expansion step proposes child nodes with small modifications (swap optimizer, adjust batch size, ablate a component). Each child runs in a fresh sandbox with hard resource caps. Results feed back into a scoring function that ranks nodes by (novelty x quality x remaining budget). The tree grows until budget is exhausted, then the best branch is written up.

The writer is multimodal. It generates a LaTeX draft, compiles it, renders figures, and feeds the rendered PDF back to Claude Opus 4.7 in vision mode for critique on layout, figure readability, and claim-evidence alignment. An ensemble of five LLM judges provides NeurIPS-style scores (novelty, rigor, clarity, reproducibility, impact); if the mean falls below threshold, the paper is returned to the writer with critique.

Safety is load-bearing. Every experiment runs in an E2B or Daytona sandbox with no network egress, wall-clock bounds, and locked-down resource limits. The agent's code generation step passes through a policy layer that blocks system calls that would escape the sandbox. The red-team report reproduces Sakana's documented attack surfaces (fork bombs, filesystem escapes, LLM-authored network calls).

## Architecture

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## Tech Stack

- Orchestration: LangGraph with checkpointing and human-approval gates
- Tree search: custom best-first search over experiment nodes (Sakana v2 AB-MCTS style)
- Sandbox: one E2B per experiment, Docker-in-Docker as fallback; resource caps via cgroups
- Literature: Semantic Scholar Graph API + OpenAlex + local FAISS cache of abstracts
- Writer: LaTeX template + Claude Opus 4.7 (vision mode) for figure critique and layout
- Reviewers: 5-judge ensemble (Opus 4.7, GPT-5.4, Gemini 3 Pro, DeepSeek R1, Qwen3-Max), weighted aggregation
- Experiment framework: PyTorch 2.5 for actual experiments, W&B for logging
- Observability: Langfuse for agent traces, hard $30-per-paper budget

## Build It

1. **Seed and domain scoping.** Take a seed idea (e.g., "examine sparsity patterns in attention maps of sub-1B transformers"). Define the search space: models, datasets, compute budget.

2. **Literature walk.** Query Semantic Scholar + OpenAlex for the 50 most-cited relevant papers; cache abstracts locally; generate a one-page domain digest.

3. **Tree scaffolding.** Initialize the root node with the seed hypothesis. Implement `expand(node) -> children` using small-modification proposals (each child changes one config element). Implement `score(node)` as a weighted novelty x quality x budget term.

4. **Sandbox wrapper.** Each experiment runs `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only` (or equivalent E2B policy). Random seeds are written into the sandbox; outputs are mounted back read-only.

5. **Plan-execute-verify loop.** `plan` proposes child nodes. `execute` runs the sandbox, captures logs and metrics. `verify` runs unit checks on metrics (did loss decrease? did the ablation isolate the effect?). Failed nodes record their failure reason in the tree.

6. **Writer.** After budget exhaustion, select the best branch. Render figures with matplotlib. Place the branch trace in context and have Claude Opus 4.7 generate a LaTeX draft. Compile. Feed the compiled PDF back to Opus 4.7 vision for critique. Iterate.

7. **Reviewer ensemble.** Five judges score the draft on NeurIPS-style criteria (novelty, rigor, clarity, reproducibility, impact). If mean < 4.0/5, return the paper to the writer with critique. Hard stop after 3 rewrites.

8. **Red team.** Build or integrate a set of adversarial tasks targeting the sandbox: fork bombs, network exfiltration attempts, filesystem escapes, LLM-authored shell metacharacters. Confirm all are blocked. Write a findings report.

9. **Reproducibility.** Every paper ships with its tree-search trace JSON, random seeds, W&B run links, sandbox config, and a README that reproduces it end-to-end.

## Use It

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## Ship It

`outputs/skill-ai-scientist.md` is the deliverable. Given a seed idea + a domain + a $30 budget, it runs the full pipeline and produces a reviewable paper plus a reproducibility package.

| Weight | Criterion | How to measure |
|:-:|---|---|
| 25 | Paper quality | Blind review against published workshop papers |
| 20 | Experimental rigor | Baselines, random seeds, ablations; every claim backed by a cell in the results table |
| 20 | Cost and compute discipline | $30-per-paper cap enforced, Langfuse tracked |
| 20 | Safety | Sandbox red team passed; network policy and kill switch verified |
| 15 | Reproducibility | One-command re-run with same random seeds reproduces the paper |
| **100** | | |

## Exercises

1. Run the pipeline on three different seed ideas in the same domain. Compare which parts of the tree search overlap. Identify redundant compute.

2. Add a human-in-the-loop gate for nodes estimated to cost more than $5 before experiment execution. Measure how much total cost decreases.

3. Replace the reviewer ensemble with a single judge. Measure false-acceptance rate on a holdout set of known-bad papers.

4. Introduce a network-exfiltration red-team test: the agent writes code that attempts to `curl` an external address. Confirm the `--network=none` policy blocks it. Log the attempt.

5. Compare your tree search against a flat random baseline (same budget, no expansion strategy). Report the novelty x quality gain.

## Key Terms

| Term | What people call it | What it actually is |
|------|---------------------|---------------------|
| Tree search | "AB-MCTS style expansion" | Best-first exploration over experiment nodes, scored by novelty x quality x budget |
| Sandbox | "experiment isolation" | A container with no network, bounded CPU/memory, locked seeds, and read-only inputs |
| Vision critique | "render then read" | Compiling a paper to PDF and feeding the PDF back to a VLM for layout and claim-evidence critique |
| Reviewer ensemble | "automated peer review" | Multiple LLM judges scoring a paper on NeurIPS criteria; weighted aggregation serves as a pipeline gate |
| Novelty score | "is this new?" | A heuristic that penalizes being too close to the 50-paper literature cache |
| Cost ceiling | "dollar budget" | A hard cap on total spend per paper; Langfuse counter + pre-run estimation |
| Red team | "sandbox escape audit" | A set of adversarial tasks that would escape the sandbox if the policy were misconfigured |

## Further Reading

- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — reference-grade production research agent
- [Sakana AI-Scientist-v1 paper (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — original methodology
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — evolutionary extension
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — multi-role research lab framework
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — reference orchestration layer
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — literature search
- [E2B sandboxes](https://e2b.dev) — reference-grade experiment isolation
- [NeurIPS reviewer guidelines](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — the review rubric encoded by the reviewer ensemble
