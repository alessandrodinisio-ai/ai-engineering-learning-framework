# Benchmarks: WebArena and OSWorld

> WebArena tests web agents across four self-hosted applications. OSWorld tests desktop agents on Ubuntu, Windows, and macOS. At launch (2023–2024) both showed massive gaps between top agents and humans. The gaps are closing; the failure modes haven't changed.

**Type:** Learn
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 19 (SWE-bench, GAIA)
**Time:** ~60 minutes

## Learning Objectives

- Describe WebArena's four self-hosted applications and why execution-based evaluation matters.
- Explain why OSWorld uses real OS screenshots rather than accessibility APIs.
- Name OSWorld's two primary failure modes: GUI grounding and operational knowledge.
- Summarize what OSWorld-G and OSWorld-Human add on top of the base benchmark.

## The Problem

Generalist agents can call tools. Can they drive a browser and complete a shopping checkout across 20 clicks? Can they configure a Linux machine using only keyboard and mouse? These are the questions WebArena and OSWorld answer.

## The Concept

### WebArena (Zhou et al., ICLR 2024)

- 812 long-horizon tasks across four self-hosted web applications: a shopping site, a forum, a GitLab-like dev tool, and a business CMS.
- Plus utilities: maps, calculator, scratchpad.
- Evaluation is execution-based via a gym API — did the order get placed, did the issue get closed, did the CMS page get updated?
- At launch: best GPT-4 agent 14.41% success rate, humans 78.24%.

The self-hosted framing matters — the benchmark doesn't jitter because target applications are pinned and reproducible.

### Extensions

- **VisualWebArena** — visual-grounding tasks where success depends on interpreting images (screenshots as first-class observations).
- **TheAgentCompany** (December 2024) — adds terminal + coding; more like a real remote-work environment.

### OSWorld (Xie et al., NeurIPS 2024)

- 369 real computer tasks across Ubuntu, Windows, and macOS.
- Free-form keyboard-and-mouse control over real applications.
- 1920×1080 screenshots as observations.
- At launch: best model 12.24%, humans 72.36%.

### Primary Failure Modes

1. **GUI grounding.** Pixel → element mapping. Models struggle to reliably locate UI elements within a 1920×1080 frame.
2. **Operational knowledge.** Which menu has that setting, which keyboard shortcut, which preference pane. The long tail of knowledge humans accumulate over years.

### Follow-up Work

- **OSWorld-G** — a 564-sample grounding suite + Jedi training set. Factors grounding out from planning so you can measure each separately.
- **OSWorld-Human** — human-curated gold action trajectories. Shows that top agents use 1.4–2.7× the steps needed (trajectory efficiency gap).

### Why This Matters

Claude computer use, OpenAI CUA, Gemini 2.5 Computer Use (Lesson 21) all train on workloads shaped by WebArena and OSWorld. Benchmarks are the target; production models are the delivered answer.

### Where Benchmarking Breaks Down

- **Screenshot-only evaluation.** OSWorld is screenshot-driven; evaluating an agent that uses DOM or accessibility APIs on OSWorld misses the grounding challenge.
- **Ignoring trajectory length.** Scoring only success rate misses the 1.4–2.7× step inefficiency OSWorld-Human reveals.
- **Stale self-hosted apps.** WebArena's applications are pinned to specific versions; updating without re-curating breaks comparability.

## Build It

`code/main.py` implements a toy web-agent harness:

- A minimal "shopping app" state machine: list_items, add_to_cart, checkout.
- Gold trajectories for 3 tasks.
- A scripted agent that attempts each task.
- Execution-based evaluator (state checks) and trajectory efficiency metric (steps vs gold).

Run it:

```
python3 code/main.py
```

Output: per-task success rate and trajectory efficiency, mirroring OSWorld-Human's methodology.

## Use It

- **WebArena Verified** self-hosted on an internal cluster for continuous evaluation.
- **OSWorld** on a VM fleet for desktop agents.
- **Computer-use agents** (Lesson 21) — Claude, OpenAI CUA, Gemini — all train on these kinds of workloads.
- **Your own product flows** — capture gold trajectories for your top 20 tasks; run agents against them weekly.

## Ship It

`outputs/skill-web-desktop-harness.md` builds a web/desktop agent harness with execution-based evaluation and trajectory efficiency metrics.

## Exercises

1. Extend the toy harness with a second application (a forum). Write 3 tasks plus gold trajectories.
2. Add per-task trajectory efficiency reporting. On your toy, is the agent 1×, 2×, or 3× gold?
3. Implement a "distractor" tool — one the gold trajectory never uses. Does the scripted agent get tempted?
4. Read OSWorld-G. How would you separate grounding failures from planning failures in your own evals?
5. Read WebArena's application README. What breaks when you upgrade a pinned application version?

## Key Terms

| Term | Common description | What it actually is |
|------|----------------|------------------------|
| WebArena | "web agent benchmark" | 812 tasks across 4 self-hosted apps; gym-style evaluation |
| VisualWebArena | "visual WebArena" | Visual-grounding WebArena; screenshots are observations |
| OSWorld | "desktop agent benchmark" | 369 tasks on real Ubuntu/Windows/macOS |
| GUI grounding | "pixel-to-element mapping" | Model locating UI elements within 1920×1080 |
| Operational knowledge | "OS know-how" | Which menu, which shortcut, which preference pane |
| OSWorld-G | "grounding suite" | 564 grounding-only samples + training set |
| OSWorld-Human | "gold trajectories" | Human expert action sequences for measuring efficiency |
| Trajectory efficiency | "steps relative to gold" | Agent steps divided by human minimum steps |

## Further Reading

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — the four-app web benchmark
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — the cross-OS desktop benchmark
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude's benchmark-shaped capability
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) — OSWorld and WebArena numbers
