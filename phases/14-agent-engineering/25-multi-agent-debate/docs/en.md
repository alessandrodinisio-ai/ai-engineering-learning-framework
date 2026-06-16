# Multi-Agent Debate and Collaboration

> Du et al. (ICML 2024, "Society of Minds") run N model instances that each independently propose an answer, then cross-critique over R rounds to converge. Improves factuality, rule adherence, and reasoning. Sparse topologies beat full-mesh on token cost.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 05 (Self-Refine and CRITIC)
**Time:** ~60 min

## Learning Objectives

- Explain the debate protocol: N proposers, R rounds, convergence to a shared answer.
- Describe why debate improves factuality, rule adherence, and reasoning.
- Explain sparse topologies: not every debater needs to see every other one.
- Implement debate against a scripted LLM using the standard library with both full-mesh and sparse variants; measure token cost vs accuracy.

## The Problem

Self-Refine (Lesson 05) is one model critiquing itself — risking groupthink. CRITIC (Lesson 05) grounds critique in external tools — not always available. Debate introduces a third mode: multiple instances, cross-critique, convergence through disagreement.

## The Concept

### Society of Minds (Du et al., ICML 2024)

- N model instances each independently propose an answer to the same question.
- Over R rounds, each model reads others' proposals and critiques them.
- Models update their answers based on critiques.
- After R rounds, return the converged answer.

Original experiments used N=3, R=2 for cost reasons. On hard problems (MMLU, GSM8K, chess move legality, biography generation), accuracy improves with more agents and more rounds.

Cross-model combinations beat single-model debate: ChatGPT + Bard together > either alone.

### Sparse topologies

"Improving Multi-Agent Debate with Sparse Communication Topology" (arXiv:2406.11776, 2024-2025) shows full-mesh debate is not always optimal. Sparse topologies (star, ring, hub-and-spoke) can match accuracy at lower token cost. Each debater only sees a subset of peers.

Implications:

- Full mesh N=5, R=3 = 5 × 3 = 15 proposals, each reading 4 peers = 60 critique operations.
- Star N=5, R=3 (one hub + 4 spokes) = 15 proposals, spokes only read hub = 12 critique operations.

### When debate helps

- **Factuality.** N independent proposals, cross-checking reduces hallucinations.
- **Rule adherence.** Chess move legality — one model misses a rule, others catch it.
- **Open-ended reasoning.** Multiple framings narrow toward the correct answer.

### When debate hurts

- **Latency-sensitive UX.** N × R serial rounds is latency you may not have.
- **Cost-sensitive scale.** N × R tokens per question.
- **Simple factual lookups.** One lookup is cheaper than five debates.

### Real-world instantiations in 2026

- **Anthropic orchestrator-workers** (Lesson 12) — a variant of debate with a synthesis step.
- **LangGraph supervisor** (Lesson 13) — central router + specialist agents; can implement debate as a node.
- **OpenAI Agents SDK** (Lesson 16) — agents hand off back and forth for iterative critique.
- **Multi-agent evaluation** — debate + evaluator-optimizer pairing for evaluation signal.

### Where this pattern breaks

- **Convergence collapse.** All agents converge on the first wrong answer. Mitigate with forced divergence rounds.
- **Hub failure.** In star topology, a bad hub poisons everyone. Rotate or use multiple hubs.
- **Prompt homogenization.** All agents use the same prompt; they produce the same answer. Use diverse prompts and/or models.

## Build It

`code/main.py` implements debate using the standard library:

- `Debater` class (scripted LLM with per-debater opinion drift).
- `FullMeshDebate` and `SparseDebate` runners.
- Three questions: one factual, one rule-based, one reasoning.
- Metrics: converged answer, rounds to convergence, total critique operations.

Run it:

```
python3 code/main.py
```

Output: accuracy and cost per protocol; sparse matches full-mesh at lower cost on 2/3 questions.

## Use It

- **Anthropic orchestrator-workers** for simple 2–3 worker debates.
- **LangGraph** for stateful multi-round debate with checkpointing.
- **Custom** for research or specialized correctness guarantees.

## Ship It

`outputs/skill-debate.md` scaffolds a multi-agent debate with configurable topology, N, R, and convergence rules.

## Exercises

1. Implement a "forced divergence" rule: in round 1, each debater must produce a different proposal. Measure impact on convergence speed.
2. Add confidence-weighted aggregation: debaters return (answer, confidence); aggregator weights by confidence. Does it help?
3. Swap one "agent" for a different scripted LLM with a different opinion. Does heterogeneity improve accuracy?
4. Measure token cost for full-mesh vs sparse on your 3 questions. Plot cost vs accuracy.
5. Read the Society of Minds paper. Port your toy to N=5, R=3. What breaks? What improves?

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Debate | "Multi-agent critique" | N proposers, R rounds of cross-critique, convergence |
| Full mesh | "Everyone reads everyone" | Every debater reads every peer each round |
| Sparse topology | "Restricted peer visibility" | Debaters only read a subset of peers |
| Hub-and-spoke | "Star topology" | One central debater, N-1 spokes only read hub |
| Convergence | "Reaching agreement" | Debaters converge to a shared answer |
| Society of Minds | "Du et al. debate paper" | ICML 2024 multi-agent debate method |

## Further Reading

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — canonical multi-agent debate
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — sparse topology results
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — orchestrator-workers as a debate variant
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — single-model self-critique counterpart
