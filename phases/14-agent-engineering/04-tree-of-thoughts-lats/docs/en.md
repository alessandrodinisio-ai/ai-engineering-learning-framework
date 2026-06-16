# Tree of Thoughts and LATS: Deliberate Search

> A single chain-of-thought trajectory has no room to backtrack. ToT (Yao et al., 2023) turns reasoning into a tree with self-evaluation at each node. LATS (Zhou et al., 2024) unifies ToT, ReAct, and Reflexion under Monte Carlo Tree Search. Game of 24 goes from 4% (CoT) to 74% (ToT); LATS achieves 92.7% pass@1 on HumanEval.

**Type:** Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 01 (The Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~75 minutes

## Learning Objectives

- Frame reasoning as search: nodes are "thoughts," edges are "expansions," values are "how promising."
- Implement a ToT-style BFS tree search with self-evaluation scoring using the standard library.
- Extend to a toy LATS MCTS loop with select / expand / simulate / backpropagate.
- Judge when search is worth the token multiplier (Game of 24, code generation) and when a single trajectory suffices (simple Q&A).

## The Problem

Chain-of-thought is a single linear walk. If the first step is wrong, every subsequent step operates on a bad premise. On Game of 24 (use four numbers with + − × ÷ to make 24), GPT-4 CoT achieves only 4% accuracy. The model picks a bad sub-expression early and never recovers.

What reasoning needs is the ability to propose multiple candidates, evaluate them, pick the promising ones, and backtrack from dead ends. That is search. Tree of Thoughts and LATS are two canonical formulations.

## The Concept

### Tree of Thoughts (Yao et al., NeurIPS 2023)

Each node is a coherent intermediate step ("a thought"). Each node can expand into K child thoughts. The LLM self-evaluates each node with a scoring prompt. Search traverses the tree — BFS, DFS, or beam.

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

Self-evaluation is the load-bearing component. The paper offers three variants: `sure / likely / impossible` classification, `1..10` numeric scoring, and voting among candidates. All three dramatically beat CoT on Game of 24 (GPT-4 from 4% → 74%).

### LATS (Zhou et al., ICML 2024)

LATS unifies ToT, ReAct, and Reflexion under MCTS. The LLM plays three roles:

- **Policy**: proposes candidate next actions (ReAct-style).
- **Value function**: scores a partial trajectory (ToT-style self-eval).
- **Self-reflector**: on failure, writes a natural-language reflection (Reflexion-style) to re-seed future rollouts.

Environment feedback (observations) mixes into the value function, so search is guided by real tool results, not just the model's opinion. Results at publication: HumanEval pass@1 92.7% with GPT-4 (SOTA), WebShop average 75.9 with GPT-3.5 (approaching gradient-based fine-tuning).

### MCTS, Minimal Version

Four phases per iteration:

1. **Select** — walk from root to a leaf using UCT (Upper Confidence bound for Trees).
2. **Expand** — generate K child nodes via the policy.
3. **Simulate** — roll out from a child using the policy, score the leaf with the value function (or environment reward).
4. **Backpropagate** — update visit counts and value estimates up the path.

UCT formula: `Q(s, a) + c * sqrt(ln N(s) / N(s, a))`. First term is exploitation; second is exploration. Tune `c` per task.

### The Cost Reality

Search causes token explosion. ToT uses 100–1000× the tokens of CoT on Game of 24. LATS is similar. This is not free; reserve search for:

- Tasks where a single trajectory is clearly insufficient (Game of 24, complex code).
- Tasks where wall-clock time matters less than correctness.
- Tasks with a cheap, reliable value function (unit tests for code, explicit targets for math).

If your task has only one correct answer and the evaluator is noisy, search often makes things worse — it finds a "high-scoring" wrong answer.

### Positioning in 2026

Most production agents do not run LATS. They run ReAct with tool-anchored verification (CRITIC, Lesson 05). Search appears in specialized niches:

- Coding agents that run tests as a value function (HumanEval-style).
- Deep-research agents exploring multiple query paths.
- Planning-heavy workflows inside LangGraph subgraphs.

AlphaEvolve (Lesson 11) is the 2025 extreme: evolutionary search over code with machine-verifiable fitness, frontier-level gains (first 4×4 matrix multiplication improvement in 56 years).

## Build It

`code/main.py` implements:

- A mini ToT BFS on a stylized "pick arithmetic operations" task.
- A toy LATS MCTS loop (Select / Expand / Simulate / Backpropagate) on the same task, with UCT selection.
- A value function that combines symbolic score with self-evaluation score.

Run it:

```
python3 code/main.py
```

The trace shows ToT expanding three candidates per node with BFS, compared to LATS converging to the best rollout via MCTS. Both print token counts.

## Use It

LangGraph offers ToT-style exploration as a subgraph pattern; the LangChain team's blog on LATS (May 2024) is the reference tutorial. LlamaIndex provides a `TreeOfThoughts` agent. For most production agents in 2026, the pattern lives behind an `if task_complexity > threshold: use_search()` gate — see the evaluator-optimizer pattern in Lesson 05.

## Ship It

`outputs/skill-search-policy.md` chooses between linear ReAct, ToT, LATS, and evolutionary search given the task shape, budget, and evaluator fidelity.

## Exercises

1. Run the toy LATS with UCT c=0.1 vs c=2.0. What changes in the trace?
2. Replace the value function with a noisier scorer (add random jitter). Can MCTS still find the best leaf? What is the minimum signal-to-noise ratio it tolerates?
3. Implement beam-search ToT (keep top-k per level) and compare against BFS. Which is better under a tight token budget?
4. Read LATS Section 5.1. Reproduce the HumanEval trajectory count: how many rollouts does it take to reach the reported pass@1?
5. Read the LATS paper's discussion on "when LATS helps less." Write a decision rule mapping task shapes to search strategies.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Tree of Thoughts | "branching CoT" | Yao et al. — tree of thought nodes with self-evaluation |
| LATS | "MCTS for LLMs" | Zhou et al. — unifies ToT + ReAct + Reflexion under MCTS |
| UCT | "Upper Confidence bound for Trees" | Selection formula balancing exploitation (Q) and exploration (ln N / n) |
| Value function | "how good is this state" | LLM-prompted score or environment reward; feeds backpropagation |
| Policy | "action proposer" | ReAct-style generator; emits candidate next thoughts/actions |
| Rollout | "simulated trajectory" | Walking from a node to a leaf using the policy, scored by value function |
| Backpropagate | "update ancestors" | Pushing leaf reward up the path, updating visit counts and Q |
| Search cost | "token explosion" | 100–1000× CoT tokens on Game of 24; budget before adopting |

## Further Reading

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — the canonical paper
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) — MCTS with Reflexion feedback
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — subgraph patterns for search
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — evolutionary search with programmatic evaluators
