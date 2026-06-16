# Theory of Mind and Emergent Coordination

> Li et al. (arXiv:2310.10701) show that LLM agents in cooperative word games exhibit **emergent higher-order Theory of Mind (ToM)**—reasoning about "what one agent thinks about a third agent's beliefs"—but fail at long-horizon planning due to context management and hallucination. Riedl (arXiv:2510.05174) measures higher-order synergy across a population and finds that identity-linked differentiation and goal-directed complementarity emerge **only** under the ToM-prompt condition; weaker LLMs show only spurious emergence. That is, coordination emergence is prompt-dependent, model-dependent, and not free. This lesson implements a minimal ToM-aware agent, runs a cooperative task with and without ToM prompting, and measures the coordination delta against the Riedl 2025 protocol.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 16 · 07 (Society of Mind & Debate), Phase 16 · 17 (Generative Agents)
**Time:** ~75 minutes

## The Problem

Multi-agent coordination often looks magical: agents divide labor, anticipate each other, avoid duplication. Usually this "emergence" is a product of prompt engineering—someone told the agents to "coordinate." Remove the prompt, coordination vanishes.

Riedl's 2025 finding is more rigorous: under controlled conditions, coordination emerges only when agents are prompted to reason about **other agents' minds** (ToM). Without ToM prompting, even strong models' coordination patterns fail statistical tests. This matters for production: teams shipping "multi-agent coordination" features rely on prompts and are fragile.

This lesson treats ToM as a concrete capability (reasoning about "beliefs about beliefs"), builds a minimal ToM-aware agent, and measures what real coordination looks like versus what prompt decoration looks like.

## The Concept

### What ToM means

Developmental psychology: a 3-year-old assumes everyone's inner world matches their own. A 5-year-old understands others have different beliefs. A 7-year-old can reason about "beliefs about beliefs" ("she thinks I think the ball is under the cup"). These are zeroth-order, first-order, and second-order ToM respectively.

For LLM agents, ToM order maps to:

- **Zeroth-order:** No model of others. The agent acts only on its own observations.
- **First-order:** The agent has a model of each other agent's beliefs. "Alice believes X."
- **Second-order:** The agent models recursive beliefs. "Alice believes Bob believes X."

Li et al. 2023 found that first- and second-order ToM emerge in LLM agents in cooperative games, but degrade with long horizons and unreliable communication.

### Sally-Anne test, briefly

A 1985 false-belief test: Sally puts a marble in basket A, then leaves. Anne moves it to basket B. Where will Sally look when she returns? Children with first-order ToM say basket A (Sally's belief differs from reality). Those without say basket B.

GPT-4-era LLMs pass Sally-Anne-style tests when asked straightforwardly. They fail when narratives are long, scenes change multiple times, or questions are phrased indirectly. This is the practical state of ToM in production LLMs as of 2026.

### Riedl's coordination measurement

Riedl (arXiv:2510.05174) built a population-scale test: N agents, a cooperative goal, variable prompt conditions. Measures:

1. **Identity-linked differentiation.** Do agents develop stable role distinctions over time?
2. **Goal-directed complementarity.** Do agents' actions complement (different subtasks) rather than duplicate?
3. **Higher-order synergy.** A statistic measuring "the group achieves what no subset could."

Result: only under the ToM prompt condition do all three metrics produce above-baseline signal. Without ToM prompting, mid-capability models' metrics hover around random levels. Large models show some coordination without explicit ToM prompting, but the effect is smaller than with explicit prompts.

### Coordination illusion

Without statistical tests, "emergent coordination" in demos often reflects:

- Prompt engineering that bakes coordination in (system prompt says "work together").
- Observer bias (we see patterns we expect to see).
- Post-hoc cherry-picking of successful runs.

Production systems that advertise "emergent coordination" without measurable signal should be treated as marketing. Measure first, claim second.

### A minimal ToM-aware agent

Structure:

```
agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]

observation update:
  - Update own_beliefs from direct observation
  - Update other_models[agent_id] from the other's actions + prior beliefs

action selection:
  - Enumerate candidate actions
  - For each candidate, predict what each other agent will do next based on modeled beliefs
  - Pick the action that maximizes joint outcome under those predictions
```

The `other_models` attribute is the ToM state. First-order ToM keeps only one layer. Second-order adds `other_models[i][other_models_of_j]`—what I think agent i thinks agent j believes.

### Why long horizons hurt

Li et al. document: context limits cause agents to forget which belief belongs to whom. Hallucinations add spurious beliefs to other-agent models. Both produce "I thought he thought X" errors that accumulate over time.

Mitigations documented in the paper and 2024-2026 follow-up work:

- **Explicitly write ToM state in the prompt.** Structured format: `{agent_id: belief_list}`. Forced retrieval preserves identity-belief bindings.
- **Shorter reasoning chains.** Fewer ToM updates per turn reduce cumulative hallucination.
- **External ToM store.** Maintain models outside LLM context; inject only the relevant portion each turn.

### Where ToM fails in production

- **Adversarial scenarios.** Agents with good ToM are more manipulable (you can model their model of you and exploit it).
- **Heterogeneous teams.** When models differ, a ToM model that works for one counterpart won't generalize.
- **Ground-truth-dependent tasks.** ToM is about beliefs; if correctness depends on facts, ToM can be a distraction.

### Coordination you can actually measure

Three practical signals that a team's coordination is real, not prompt decoration:

1. **Complementarity over time.** In a multi-turn task, do agents' actions cover disjoint subtasks?
2. **Anticipation.** Does agent A's action at turn T+1 depend on a prediction about B's action at turn T+2 that later proves correct?
3. **Correction.** When A misreads B's belief at turn T, does A correct at turn T+2?

These are measurable in a logged multi-agent system. They are the substantive version of "coordination" narratives.

## Build It

`code/main.py` implements:

- `ToMAgent` — tracks its own beliefs and a belief model for each other agent.
- A cooperative task: three agents must each take a token from three boxes; each box holds one token. Agents cannot communicate; they infer intent from each other's actions.
- Two configurations: `zeroth_order` (no ToM) and `first_order` (ToM with one layer of belief modeling).
- Measures over 200 randomized trials: completion rate, duplication rate (two agents targeting the same box), average turns to complete.

Run:

```
python3 code/main.py
```

Expected output: zeroth-order agents duplicate effort at ~35% rate and complete ~60% of trials within 10 turns. First-order ToM agents duplicate at ~5% and complete ~95%. This delta is the measurable coordination effect.

## Use It

`outputs/skill-tom-auditor.md` is a skill that audits a multi-agent system's claims of "emergent coordination." Checks for prompt decoration, statistical significance relative to a control group, and measured complementarity.

## Ship It

Coordination claim checklist:

- **Control condition.** A version of the system with coordination prompts removed. Test both.
- **Statistical test.** Is the difference between system and control on your metric significant at `p < 0.05`?
- **Complementarity metric.** Action disjointness over time, not just final success.
- **Failure case logs.** When agents fail to coordinate, what does the ToM state look like?
- **Model capability disclosure.** If the effect vanishes on smaller models, say so.

## Exercises

1. Run `code/main.py`. Confirm first-order ToM reduces duplication rate by ~7x. Does the gap hold when you scale to 5 agents and 5 boxes?
2. Implement second-order ToM (agent A models B's view of C). Does it improve over first-order? On what task?
3. Inject a **hallucination** into ToM state: randomly flip one belief per turn. How much does this degrade first-order performance?
4. Read Li et al. (arXiv:2310.10701). Reproduce the "long-horizon degradation" finding: as turns increase from 10 to 30, how does your first-order ToM performance change?
5. Read Riedl 2025 (arXiv:2510.05174). Implement the higher-order synergy statistic on your simulation logs. Does the effect hold without the ToM prompt condition?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Theory of Mind | "Understanding others' minds" | The ability to model another agent's beliefs. Graded by order (0, 1, 2+). |
| Sally-Anne test | "False-belief test" | 1985 developmental psychology; LLMs pass straightforward versions, fail complex ones. |
| First-order ToM | "A believes X" | Modeling one other's belief about facts. |
| Second-order ToM | "A believes B believes X" | One deeper layer of recursive modeling. |
| Identity-linked differentiation | "Stable roles over time" | Riedl's metric: roles persist rather than randomize. |
| Goal-directed complementarity | "Disjoint actions" | Agents target different subtasks, not the same one. |
| Higher-order synergy | "Group exceeds any subset" | Riedl's statistic measuring genuine coordination. |
| Coordination illusion | "It looks coordinated" | Apparent coordination from prompt decoration without measurable signal. |

## Further Reading

- [Li et al. — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) — Emergent ToM in cooperative games; long-horizon failure modes
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) — Population-scale measurement; ToM prompting is the load-bearing condition
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) — The 1978 origin of the ToM concept
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/) — The Sally-Anne paper (1985)
