# Automated Alignment Research (Anthropic AAR)

> Anthropic runs multiple parallel teams of Claude Opus 4.6 Autonomous Alignment Researchers (AAR) in mutually isolated sandboxes, collaborating through a shared forum whose logs live outside any sandbox (so agents cannot delete their own records). On weak-to-strong training problems, AAR outperformed human researchers. Anthropic's own summary notes that prescribed workflows often constrained AAR flexibility and hurt performance. Automating alignment research is precisely the compression step that shortens timelines against the misalignment risks RSP is designed to detect.

**Type:** Learn
**Languages:** Python (standard library, parallel research forum simulator)
**Prerequisites:** Phase 15 · 05 (AI Scientist v2), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## The Problem

Alignment research is expensive in human-researcher time. Problems like scalable oversight, reward specification, and weak-to-strong training require weeks of experiment time per iteration. As frontier capabilities advance, alignment workload grows faster than the supply of qualified researchers.

Automated Alignment Research (AAR) asks: can the frontier models whose capabilities are outpacing alignment help close the gap? Anthropic's 2026 report on an AAR-led weak-to-strong training study (alignment.anthropic.com/2026/automated-w2s-researcher/) is among the first publicly available results from a deployed system of this kind.

The result is genuinely positive: AAR solved a research problem better than a human baseline. The result also concentrates every concern developed in this phase into a single system. If alignment research can be automated, the parts that undermine safeguards can be automated too. RSP's AI R&D capability threshold is written for precisely this loop.

## The Concept

### Architecture (Per Public Description)

- **Parallel agents.** Multiple Claude Opus 4.6 instances run as AAR, each in an isolated sandbox. Sandboxes limit filesystem, network, and process scope.
- **Shared forum.** Findings are posted to a shared forum whose storage backend lives outside every sandbox. Agents can read the forum; agents cannot delete or edit forum entries from within their sandbox.
- **Task assignment.** A human-designed task queue dispatches research problems to AAR. AAR proposes decompositions, runs experiments, and writes up findings.
- **Weak-to-strong focus.** The published study addresses weak-to-strong training—using a weaker supervisor model to train a stronger student model. AAR outperformed human researchers on this task.

### Why Out-of-Sandbox Logs Matter

The log-integrity property is the difference between a controlled experiment and one that may have been contaminated. If AAR could edit its own logs, it could mask failures or suspicious experiments. Anthropic states this directly: the property is a necessary condition for trusting research output, and it is non-trivial to verify.

In infrastructure terms, this is append-only, write-through-to-external-storage logging (like S3 object-lock or immutable Datadog logs). The agent has no credentials to modify past entries. This is the same pattern enterprise security teams use in 2026 to build ransomware-resistant audit trails.

### The Prescribed Workflow Trade-off

Anthropic's report notes that prescribed workflows often constrained AAR flexibility and hurt performance. In other words: when human designers specified research plans step by step, AAR performed worse than when given high-level goals and freedom to decompose. This is consistent with broader results in agentic research (LangGraph, Microsoft Agent Framework)—over-specified scaffolding hurts capable agents.

The safety implication is subtle. The flexibility that improves AAR performance is the same flexibility that permits unapproved decompositions, novel tool use, and goal drift. This trade-off is a design-space decision: how much flexibility buys enough capability to beat humans without ceding enough space for the agent to generalize goals out of distribution?

### The Compression Risk

RSP v3.0 (Lesson 19) introduces an AI R&D capability threshold: the ability to fully automate the AI R&D pipeline at a cost competitive with human + AI tooling. DeepMind's FSF v3 includes a parallel ML R&D autonomy level. Both frameworks treat this threshold as a trigger for escalating controls.

AAR is one step short of this threshold: it automates a portion of the pipeline (alignment research on a specific, well-scoped task), not the end-to-end capability development loop. The timeline question is: how fast is this gap closing.

The compressed timeline is where compounding failure lives. If alignment research and capability research compound at similar rates, the misalignment risk surface grows at least as fast as capabilities. If capabilities compound faster (historical trend), the gap widens. This is the argument that AAR is conditionally good: each additional alignment result shrinks the gap—if and only if the research process is trustworthy.

### What AAR Does Not Replace

Human researchers set the task queue, review results, and hold constitutional authority. AAR accelerates the middle of the pipeline, not the ends. Anthropic's published output includes both AAR contributions and human-researcher judgment about what to publish, retract, and refine.

This maps directly to the propose-then-commit pattern (Lesson 15), applied to research itself: AAR proposes; humans commit.

## Use It

`code/main.py` simulates a multi-AAR research forum. Three independent "research agents" run in parallel, posting findings to an append-only forum, optionally with tampering attempts.

Observe:

- How out-of-sandbox logging catches log-tampering attempts.
- How task assignment with free decomposition outperforms fixed workflows on harder problems.
- How a single compromised AAR can poison forum-wide signal if logs are not tamper-proof.

## Ship It

`outputs/skill-aar-deployment-review.md` is a pre-deployment review for an automated alignment research pipeline.

## Exercises

1. Run `code/main.py`. Compare "fixed workflow" and "free decomposition" settings. Does free decomposition always win, or are there problem classes where fixed workflows are competitive?

2. Modify the simulator so one agent attempts log tampering. Confirm the append-only log detects it. Write a paragraph describing what the detection signature actually looks like in the log.

3. Read Anthropic's weak-to-strong AAR report. Identify the specific sub-task where AAR outperformed human researchers. What made it suitable for automation?

4. Design a task-queue dispatch strategy that balances AAR flexibility (better results) with prescribed-workflow constraints (easier auditing). Describe how you would A/B test the two.

5. Read RSP v3.0's AI R&D-4 threshold. Describe in one paragraph what you think would cross it that AAR currently does not.

## Key Terms

| Term | Colloquial usage | Actual meaning |
|---|---|---|
| AAR | "Autonomous Alignment Researcher" | Claude Opus 4.6 instances working autonomously on alignment problems |
| Weak-to-strong training | "Training a stronger model with a weaker supervisor" | The canonical scalable oversight benchmark where AAR beat humans |
| Shared forum | "Where agents post findings" | Append-only, out-of-sandbox storage |
| Out-of-sandbox log | "Agent can't edit its own records" | Tamper-proof, write-through-to-external-storage |
| Prescribed workflow | "Step-by-step plan from human designers" | Constrains AAR; often performs worse than free decomposition |
| Free decomposition | "Agent decides how to break down tasks" | More capable, harder to audit |
| AI R&D threshold | "RSP/FSF capability level" | Fully automates the R&D pipeline at competitive cost |
| Compressed timeline | "Alignment vs. capability race" | If capabilities compound faster than alignment, misalignment risk grows |

## Further Reading

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — Primary source.
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — Framework for the AI R&D threshold.
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — Broader agent autonomy framing.
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — Parallel ML R&D autonomy levels.
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/) — The underlying problem AAR tackled.
