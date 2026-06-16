# Society of Mind and Multi-Agent Debate

> Minsky's 1986 premise — intelligence is a society of interacting experts — gets rediscovered every decade. In 2023 Du et al. turned it into a concrete algorithm: multiple LLM instances propose answers, read each other's answers, critique, and update. After N rounds they converge to a consensus that beats zero-shot CoT and reflection on six reasoning and factuality tasks. Two findings are key: **multiple agents** and **multiple rounds** each contribute independently. The society beats a single agent's monologue; the multi-round exchange beats one-shot voting.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 min

## The Problem

Self-consistency — sampling a model multiple times and taking the majority answer — is the cheapest reasoning boost you can add. It works, but it saturates quickly. You can double the sample count and no longer see a meaningful jump.

Debate breaks this saturation. Instead of N independent samples from one model, N agents read each other's reasoning and revise. The correlation between samples drops (they're no longer i.i.d.), and the convergence point is often correct where i.i.d. voting confidently gets it wrong.

## The Concept

### Du et al. 2023 Algorithm

From arXiv:2305.14325 (ICML 2024):

1. N agents each produce an initial answer to the question.
2. For rounds r = 2..R: show each agent the other agents' round r-1 answers and ask "Given these, give your updated answer."
3. After R rounds, take a majority vote on final answers.

The paper tests on MMLU, GSM8K, biography, MATH, and factuality benchmarks. Debate consistently beats CoT and Self-Reflection.

### Two Independent Knobs

Ablations from the same paper:

- **Adding only agents** (1 round, majority vote over N) beats a single agent on most tasks but plateaus.
- **Adding only rounds** (1 agent reading its own prior reasoning) barely helps — this is the known weakness of reflection.
- **Both together** produces the large jump. Multi-round exchange across multiple agents drives the gain.

### Why It Works

Two mechanisms:

1. **Exposure to disagreement.** When an agent sees another agent's reasoning chain reaching a different conclusion, it must either defend or update. Either way, the context in round r+1 is richer than round r.
2. **Decorrelated error.** In self-consistency, all samples come from the same model, so errors correlate — you average them into a confident wrong answer. Different models or different seeds decorrelate. Different *debated perspectives* decorrelate further.

### Heterogeneous Debate

A-HMAD and related follow-ups use *different base models* for different agents. Llama + Claude + GPT debating together mitigates monoculture collapse (Lesson 26) because one model family's correlated errors aren't shared by the others.

Downside: a weak model participating in debate can drag the consensus toward its wrong answer (see "Should we be going MAD?", arXiv:2311.17371).

### NLSOM — Scaling to 129 Agents

Zhuge et al. ("Mindstorms in Natural Language-Based Societies of Mind," arXiv:2305.17066) scaled the idea to 129-member societies. Result: as scale increases, specialization and self-organization emerge, and the system outperforms single agents on tasks like visual QA.

### Failure Modes

- **Sycophancy cascade.** All agents collapse toward the most confident-sounding one. Debate degenerates into the loudest voice. Prompting agents to play adversarial roles ("one agent must always argue the opposite") mitigates.
- **Topic drift.** Multi-round debate drifts from the original question. Mitigation: re-inject the question every round.
- **Compute explosion.** N agents × R rounds = N·R LLM calls, each with growing context. A 5-agent, 5-round debate is 25 calls with growing context. Cost per question can exceed 10x a single CoT call.

## Build It

`code/main.py` runs a 3-agent × 3-round debate on a math problem, each agent starting with a different (possibly wrong) answer. Agents are scripted — each agent "updates" by doing a weighted average of neighbor answers with scripted confidence. The convergence process is visible in the round-by-round log.

The demo shows two key effects:

- A single exchange round already pushes agents closer to the correct answer.
- Additional rounds after round 2 show diminishing returns (consistent with Du et al.'s plateau).

Run:

```
python3 code/main.py
```

## Use It

`outputs/skill-debate-configurator.md` configures a debate for a new task: agent count, round count, heterogeneity (same model vs mixed), role assignment (symmetric vs one adversary). It also estimates token cost before you run.

## Ship It

If you're going with debate:

- **Cap rounds at 3.** Du et al. show 3 rounds captures most of the benefit. More is cost, not quality.
- **Cap agents at 5.** Beyond 5, context bloat and cost dominate.
- **Default to heterogeneous.** At least two different base models in the pool.
- **Adversarial seat.** Prompt one agent to always argue the contrary. Breaks sycophancy.
- **Log per-round.** Debate systems that hide intermediate rounds are un-debuggable and un-auditable.

## Exercises

1. Run `code/main.py`, then set rounds to 5 and observe diminishing returns. At which round does additional convergence stop?
2. Add a fourth agent with an adversarial role: always argues against the current majority. Does this break or improve convergence?
3. Print (plot) the agreement score per round (fraction of agents on the majority answer). When does it hit 1.0, and does that equal "correct"?
4. Read Du et al. Section 4 ablations. Replicate the "agents only" vs "rounds only" vs "both" result with this code.
5. Read "Should we be going MAD?" (arXiv:2311.17371) and list two debate variants beyond round-robin — e.g., judge-led, chain-of-debate, adversarial.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Society of Mind | "Minsky's idea" | Intelligence as interacting experts; the 1986 premise now instantiated via LLM debate. |
| Multi-agent debate | "Agents arguing" | N agents propose, critique each other across R rounds, majority vote. |
| Consensus | "They agree" | Not epistemological truth — just the fraction on the majority answer. Can be confidently wrong. |
| Rounds | "Exchange steps" | One round = each agent reads the others and updates once. |
| Heterogeneous debate | "Mixed model families" | Using different base models to decorrelate errors. |
| Sycophancy cascade | "Everyone agrees with the loudest" | Debate failure where agents collapse toward the most confident agent regardless of correctness. |
| NLSOM | "129-agent society" | Natural Language Society of Mind; Zhuge et al.'s scaled version. |
| Correlated error | "Same model, same bug" | Why self-consistency saturates; debate across different perspectives decorrelates. |

## Further Reading

- [Du et al. — Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) — the reference paper, ICML 2024
- [Zhuge et al. — Mindstorms in Natural Language-Based Societies of Mind](https://arxiv.org/abs/2305.17066) — 129-agent NLSOM
- [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371) — benchmarks various debate variants
- [Debate project page](https://composable-models.github.io/llm_debate/) — Du et al.'s code, demos, and ablation details
