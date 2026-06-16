# AI Scientist v2 — Workshop-Level Autonomous Research

> Sakana's AI Scientist v2 (Yamada et al., arXiv:2504.08066) runs the full research cycle: hypothesis, code, experiments, figures, writing, submission. It is the first system to have a generated paper pass peer review at an ICLR 2025 workshop. An independent evaluation (Beel et al.) found 42% of experiments failed due to coding errors, and the literature review often mislabeled established concepts as novel. Sakana's own documentation warns the code executes LLM-written code and recommends Docker isolation. Both halves of this picture are the point.

**Type:** Learn
**Languages:** Python (standard library, research loop state machine toy)
**Prerequisites:** Phase 15 · 03 (AlphaEvolve), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## The Problem

Research is an open-ended task. Unlike AlphaEvolve's algorithmic search or DGM's benchmark-constrained self-modification, a research result has no machine-checkable correctness criterion. A paper is judged by reviewers, not unit tests. This makes the loop harder to close—and more valuable once closed, because research is where compounding progress lives.

AI Scientist v1 (Sakana, 2024) closed the loop by starting from human-written templates. The LLM filled in experiments within fixed scaffolding. AI Scientist v2 (Yamada et al., 2025) removes the template requirement with agentic tree search plus a vision-language model critique loop. The system generates ideas, implements experiments, produces figures, writes the paper, and iterates on reviewer feedback.

The peer review verdict: one v2-generated paper was accepted at an ICLR 2025 workshop (disclosed). The independent evaluation verdict: the system is far from reliable. Both are true.

## The Concept

### Architecture

1. **Idea generation.** The LLM proposes research ideas conditioned on a topic and prior literature. v1 used templates; v2 does agentic search over the hypothesis space.
2. **Novelty check.** A literature retrieval step checks whether the idea has been published. This is the step where Beel et al.'s evaluation found mislabeling—established methods are often classified as novel.
3. **Experiment plan.** The agent drafts an experiment protocol and writes code.
4. **Execution.** Code runs in a sandbox. Failures are fed back to a retry loop. This stage has a 42% experiment failure rate in Beel et al.'s measurements.
5. **Figure generation.** A vision-language model reads generated plots and rewrites them for clarity. This is v2's key technical addition.
6. **Writing.** The LLM drafts the paper, iterating with an internal reviewer.
7. **Optional: Submission.** The paper is submitted to a venue.

### What the Workshop Acceptance Means

One v2-generated paper passed peer review at an ICLR 2025 workshop. The authors disclosed the paper's provenance to the program committee. The acceptance is a data point; it is not a license to claim the system "does research."

Important context: workshop papers have a lower bar than main conference papers. Peer review is noisy; only a fraction of submissions are accepted on any given day. One success is proof of concept, not a reliability claim. The Nature 2026 paper documents the end-to-end loop and is itself co-authored by human researchers; this is not "the system wrote a Nature paper."

### What the Independent Evaluation Found

Beel et al. (arXiv:2502.14297) conducted an external evaluation. Headline findings:

- **Experiment failures.** 42% of experiments failed due to coding errors (wrong imports, shape mismatches, undefined variables). The retry loop caught some but not all.
- **Novelty mislabeling.** The literature retrieval step often labeled established concepts as novel. This is hallucination for research.
- **Presentation quality gap.** The vision-language figure critique produced publication-grade visuals that masked underlying experimental weakness.

The last finding is the one that matters for this phase. A system that produces convincing outputs without doing convincing research is more dangerous, not safer, than one that obviously fails. Evaluation must reach the underlying claims, not stop at the figures.

### Sandbox Escape Concerns

Sakana's own repository README warns:

> Since this software executes LLM-generated code, by nature we cannot guarantee safety. There is risk of dangerous packages, uncontrolled network access, and generating unexpected processes. Use at your own risk, and consider Docker isolation.

This is what autonomy in an unverified domain looks like in practice. The LLM writes code; the code runs; the code can do anything the process is allowed to do. Without a sandbox that hard-limits filesystem, network, and process actions, any autonomous research agent can exfiltrate data, burn compute, or rewrite itself.

AlphaEvolve's sandbox story is easier because its evaluator is tight. AI Scientist v2's loop runs open-ended code with an open-ended goal. This is why it needs stronger isolation (Docker at minimum; seccomp / gVisor preferred), plus human review before any submission leaves the system.

### Where v2 Fits in the Frontier Stack

| System | Target | Output type | Evaluator | Known failure |
|---|---|---|---|---|
| AlphaEvolve | Algorithms | Code | Unit + benchmark | Bounded by evaluator rigor |
| DGM | Agent scaffold | Code | SWE-bench | Reward hacking |
| AI Scientist v2 | Research papers | Text + code + figures | Peer review (weak) | Experiment failure, mislabeling, polish masking weakness |

Of the three, v2 has the weakest automatic evaluator, the broadest output surface, and the shortest path to public artifacts. Operational controls (sandbox, review, disclosure) carry most of the safety work.

## Use It

`code/main.py` simulates the v2 loop as a state machine: idea → novelty check → experiment → figures → writing → review → accept or iterate. Each state has a configurable failure probability drawn from Beel et al.'s findings. Run the simulator for N loops and tally:

- How many ideas make it to submission.
- How many submissions carry a critical experimental flaw hidden by a polished paper.
- How retry budget trades off quality vs. throughput.

## Ship It

`outputs/skill-ai-scientist-sandbox-review.md` is a two-gate review checklist for anything produced by a research-loop agent before it leaves the sandbox.

## Exercises

1. Run `code/main.py` with default parameters. What fraction of loop runs produce a "clean" paper? What fraction produce one with an experiment-failure flaw masked by figure critique polish?

2. Defaults already use Beel et al.'s 42%/25%. Run again with `--experiment-failure 0.20 --novelty-mislabel 0.10`, then with `--experiment-failure 0.60 --novelty-mislabel 0.40`. How does the fraction of polished-but-flawed papers change between the two runs?

3. Read the Sakana AI Scientist v2 repository README on sandbox requirements. Name two constraints you would add beyond Docker for a multi-day autonomous run.

4. Read Beel et al. Section 4 on the presentation quality gap. Design an additional evaluator that catches papers that look polished but have flawed experiments.

5. Propose a human review protocol for research agent outputs that scales better than "one PhD reads each paper." Identify the bottleneck and design around it.

## Key Terms

| Term | Colloquial usage | Actual meaning |
|---|---|---|
| AI Scientist v1 | "Sakana's template-based research agent" | Fills experiments into fixed scaffolding |
| AI Scientist v2 | "Template-free research agent" | Agentic tree search with VLM figure critique |
| Agentic tree search | "Branching research agent" | Expands multiple experiment plans in parallel; prunes by internal critic |
| Vision-language critique | "VLM polish on figures" | Multimodal model reads plots and rewrites them for clarity |
| Literature retrieval | "Novelty check" | Searches prior work to confirm idea novelty—documented to mislabel |
| Polish masking | "Paper looks good, research is broken" | Presentation quality exceeds experiment quality; masks weakness |
| Sandbox escape | "LLM code got out" | Code executed by the agent does things the loop designer didn't intend |

## Further Reading

- [Yamada et al. (2025). The AI Scientist-v2](https://arxiv.org/abs/2504.08066) — The paper.
- [Sakana blog on the Nature 2026 publication](https://sakana.ai/ai-scientist-nature/) — Vendor summary with peer review context.
- [Beel et al. (2025). Independent evaluation of The AI Scientist](https://arxiv.org/abs/2502.14297) — External evaluation numbers.
- [Sakana AI Scientist v1 paper](https://arxiv.org/abs/2408.06292) — The template-based predecessor.
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — Broader framing for open-ended research agents.
