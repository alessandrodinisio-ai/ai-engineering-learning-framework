# Voting, Self-Consistency, and Debate Topologies

> The cheapest aggregation: sample N independent agents, majority vote. Wang et al. 2022's self-consistency did this with one model sampled N times. Multi-agent extends it with **heterogeneous** agents to escape monoculture — different models, different prompts, different temperatures, different contexts. Beyond majority vote, debate topology matters: MultiAgentBench (arXiv:2503.01935, ACL 2025) benchmarked star / chain / tree / graph coordination, finding **graph best for research** and a "coordination tax" emerging beyond ~4 agents. AgentVerse (ICLR 2024) documented two emergent patterns — volunteer behavior and conformity behavior — and conformity is both a feature (finding consensus) and a risk (groupthink, Lesson 24). This lesson maps the topology space, builds each variant, and measures coordination tax.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 16 · 07 (Society of Mind & Debate), Phase 16 · 14 (Consensus & BFT)
**Time:** ~75 min

## The Problem

Debate can improve accuracy (Du et al., arXiv:2305.14325). It can also hurt accuracy. Whether debate helps depends on four structural choices:

1. Who talks to whom (topology).
2. How many rounds (Du 2023: rounds and agent count each matter independently).
3. Whether agents are heterogeneous (different base models can break monoculture).
4. Whether there's an adversarial voice (steel-manning vs straw-manning).

Teams that shove "run 5 agents and vote" onto a task often regress below single-agent performance. These failures aren't random. They correlate with topology and heterogeneity. This lesson is that topology map.

## The Concept

### Self-Consistency, the Single-Model Baseline

Wang et al. 2022 ("Self-Consistency Improves Chain of Thought Reasoning") sampled the same model N times at temperature > 0, then majority-voted over the answers from reasoning paths. Results on GSM8K: N=40 samples significantly improved over single greedy decode. Self-consistency is the single-agent precursor to multi-agent voting.

Limitation: self-consistency uses one base model. Errors are inherently correlated. If the model has a systematic bias, all N samples share it.

### Multi-Agent Voting, the Heterogeneous Extension

Replace N samples with N *different* agents. Different base models (Claude, GPT, Llama), different prompts, different tool access. Benefit: uncorrelated errors. Cost: different agents cost differently; coordinating them adds overhead.

The 2026 standard name for heterogeneous debate is **A-HMAD** — Adversarial Heterogeneous Multi-Agent Debate. Not universally adopted, but papers use the term to mean "different models debating to reduce correlated errors from monoculture collapse."

### Four Topologies

```
star                chain               tree                graph

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (fully connected)
```

Star: one hub, all other agents talk only to the hub. Equivalent to supervisor-worker without a back-channel.
Chain: linear, each agent sees the previous one's output. Pipeline-style.
Tree: hierarchical, used by hierarchical agent systems (Lesson 06).
Graph: any-to-any. Includes fully connected cliques and arbitrary DAGs.

### Coordination Tax (MultiAgentBench)

MultiAgentBench (MARBLE, ACL 2025, arXiv:2503.01935) benchmarked star, chain, tree, graph on a task set including research, coding, and planning. Key measured result:

- **Graph** topology wins on research tasks. Information flows any-to-any; agents can critique each other.
- **Star** wins on fast factual tasks. Hub filters and integrates.
- **Chain** wins on step-by-step pipelines (staged refinement).
- **Coordination tax** appears on graph topology beyond ~4 agents. Wall-clock time and token cost grow faster than quality.

The 4-agent ceiling is empirical, not fundamental. It reflects 2026 LLMs' context capacity: each agent's context fills with peers' outputs, and the marginal value of adding agent N+1 drops once everyone sees everyone.

### Multi-Agent Debate Strategies ("Should we be going MAD?")

arXiv:2311.17371 is the 2023 survey on MAD strategies. Key finding replicated by others: MAD variants that are *structurally similar* to self-consistency (independent sampling + aggregation) often don't beat self-consistency at equal budget. MAD is most useful when agents are genuinely heterogeneous and debate has adversarial structure (one agent plays devil's advocate).

### AgentVerse Emergent Patterns

AgentVerse (ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) documented two behaviors that emerge from multi-agent debate even without explicit design:

- **Volunteer.** One agent offers help without being asked ("I'll take the next step"). Useful: it routes work toward the most capable agent for a subtask.
- **Conformity.** One agent shifts its position to agree with a critic, even when the critic is wrong. This is debate-mode sycophancy (Lesson 14).

Conformity is why "debate until agreement" rewards bullies. Bounded rounds plus an independent judge mitigate.

### Heterogeneity: the Knob That Actually Moves Accuracy

A pattern across 2024-2026 practitioner literature: swapping one of your N agents for a different base model improves accuracy more than adding one more agent of the same kind. The intuition is monoculture — each new independent error source is worth more than an additional correlated sample.

At the limit, heterogeneity beats quantity. Three different models beat five copies of the same model on most tasks with clean ground truth.

### The Jury Approach

The Sibyl framework (cited in the Minsky-LLM literature) formalizes the "jury" — a small panel of specialized agents that refine an answer via voting at each stage. Unlike pure majority vote, the jury has roles: one agent cross-examines, one provides context, one scores plausibility. The jury approach is the midpoint between pure voting (cheap, monoculture-prone) and full MAD (expensive, conformity-prone).

### When Voting + Debate Dominates

- Problem has ground truth (factual, math, code behavior). Voting convergence is meaningful.
- Agents have access to different sources or tools (heterogeneity present).
- Rounds are bounded (usually 2-3) with an independent judge or verifier.
- Budget allows 3-5 agents. Beyond 5-7 on graph topology, coordination tax dominates.

### When Voting + Debate Hurts

- Problem is opinion-shaped. Agents converge to the most confident-sounding, not the most correct, answer.
- All agents share one base model. Monoculture makes consensus meaningless.
- Rounds are unbounded. Conformity wins every time.
- Task is simple. A single agent with N=5 self-consistency is cheaper and equally accurate.

## Build It

`code/main.py` implements:

- `run_star(agents, hub, question)` — hub polls each worker, aggregates.
- `run_chain(agents, question)` — sequential refinement.
- `run_tree(root, children, question)` — hierarchical with depth-2 aggregation.
- `run_graph(agents, question, rounds)` — all-to-all debate, bounded rounds.
- A scripted heterogeneity knob: each agent has an `error_bias` representing its systematic error rate.
- A measurement harness that runs each topology at N=3, 5, 7 and reports (accuracy, total_tokens, wallclock_simulated).

Run:

```
python3 code/main.py
```

Expected output: a table of topology × N → (accuracy, tokens, latency). Graph wins on research-type tasks at N=3-5; star wins on fast factual tasks; graph shows coordination tax at N=7 (latency grows faster than accuracy).

## Use It

`outputs/skill-topology-picker.md` is a skill that reads a task description and recommends a topology (star / chain / tree / graph), an N (agent count), a heterogeneity configuration (which base models), and a round bound.

## Ship It

For any ensemble:

- Start with **N=5 self-consistency** on a strong base model. This is the cheap baseline.
- If accuracy matters, upgrade to **N=3 heterogeneous voting**. Measure the increment.
- Upgrade to **debate topology** only when the task has structure (research, multi-step) and bounded rounds are feasible.
- Always log minority clusters. When minorities are persistently correct, you have a diversity signal.
- Benchmark wall-clock time and tokens alongside accuracy. "10x cost for better accuracy" is a business decision.

## Exercises

1. Run `code/main.py`. Plot the coordination tax curve for graph topology: accuracy vs N, tokens vs N. Where does the curve bend?
2. Implement A-HMAD: three agents with deliberately different biases. How does a uniform-bias baseline compare to A-HMAD on Lesson 14's monoculture attack?
3. Add a "judge" role to graph topology that doesn't vote, only scores the final consensus. Does this change the emergent conformity behavior?
4. Read the AgentVerse paper (ICLR 2024). Identify which emergent behavior is strongest in your implementation. Can you elicit the opposite behavior by changing prompts?
5. Read MultiAgentBench (arXiv:2503.01935) Section 4 (topology experiments). Reproduce the "graph wins research" result on one task from the paper using your measurement harness.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Self-consistency | "Sample N times, vote" | Wang 2022. Single model, N temperature>0 samples, majority vote over reasoning paths. |
| Heterogeneity | "Different models" | An ensemble of different base models or prompt families. Breaks monoculture. |
| MAD | "Multi-agent debate" | Generic term for agents exchanging critiques across rounds. See Du 2023. |
| A-HMAD | "Adversarial heterogeneous MAD" | MAD variant emphasizing different models + adversarial structure. |
| Topology | "Who talks to whom" | Star, chain, tree, graph. Determines information flow. |
| Coordination tax | "Diminishing returns" | Beyond ~4 agents on graph, cost grows faster than quality. |
| Volunteer behavior | "Helping without being asked" | AgentVerse emergent pattern: one agent takes a step unprompted. |
| Conformity behavior | "Agreeing under pressure" | AgentVerse emergent pattern: one agent shifts to agree with a critic. |
| Jury | "Small specialized panel" | Sibyl-style ensemble with roles (cross-examiner, context, scorer). |

## Further Reading

- [Wang et al. — Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — single-model baseline
- [Du et al. — Improving Factuality and Reasoning via Multiagent Debate](https://arxiv.org/abs/2305.14325) — agent count and rounds each matter independently
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — topology benchmark showing graph best for research, chain for pipelines
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371) — MAD strategy survey; finds MAD often loses to self-consistency at equal budget
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) — volunteer and conformity emergent patterns
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE) — reference benchmark implementation
