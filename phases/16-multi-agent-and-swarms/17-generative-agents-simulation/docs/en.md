# Generative Agents and Emergent Simulation

> Park et al. 2023 (UIST '23, arXiv:2304.03442) populated **Smallville** — a 25-agent sandbox — with a three-component architecture: **memory stream** (natural-language log), **reflection** (higher-order syntheses an agent generates from its own memory stream), and **plan** (day-level behaviors first, then sub-plans). The signature result is the emergence of a Valentine's Day party: one agent seeded with "wants to throw a Valentine's Day party," with no further scripting, caused invitations to spread through the population, times to be coordinated, and the party to actually happen — emerging from 24 agents who initially knew nothing about it. Ablation experiments show all three components are necessary for believability. Documented failures are spatial-norm violations (walking into closed stores, sharing single-occupancy bathrooms). This is the reference architecture for 2026 agent simulation and multi-agent social evaluation.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 13 (Shared Memory)
**Time:** ~75 min

## The Problem

Most multi-agent systems are tightly scripted teams: planner plans, coder codes, reviewer reviews. That works for well-defined tasks. It doesn't capture the kind of emergent, unscripted behavior that arises when agents have memory, priorities, and an open world. Research, social simulation, and increasingly game AI need the second kind.

The Smallville architecture is the benchmark for it. Before Park 2023, even the best agent simulations were shallow script-followers; after it, the pattern became the default for open-world generative agents. If you're building an agent simulation in 2026, you're either using Smallville's three components or explicitly arguing why you're not.

## The Concept

### The Three Components

**Memory stream.** An append-only log of observations, actions, reflections, and plans. Each entry has a timestamp, type, description (natural language), and derived metadata: **recency** (exponential decay by age), **importance** (agent self-rated 1-10), **relevance** (cosine similarity to current query).

```
[2026-02-14 09:12:03] observation: Isabella Rodriguez asked me if I like jazz
[2026-02-14 09:14:22] reflection:   I enjoy long conversations about music
[2026-02-14 10:05:00] plan:         Attend Isabella's Valentine's Day party tonight
```

Memory retrieval combines three scores: `score = w_recency * e^(-decay * age) + w_importance * importance + w_relevance * cos_sim`. Top-k entries enter the current prompt.

**Reflection.** Periodically (every N memories or on significant events), the agent generates higher-order syntheses from recent memories. Reflection entries flow back into the memory stream and are retrievable like any other memory. This is how the agent builds "understanding" — the closest thing to long-term beliefs in this architecture.

**Plan.** Top-down decomposition. First a coarse day-level plan ("go to work, have dinner with Klaus"). Then hour-level plans. Then action-level plans. Plans are revisable: when an observation contradicts the plan, the agent re-plans the affected segment.

### Why All Three Matter (Ablation)

Park et al. ran ablations removing observation, reflection, and planning separately. Each ablation hurts believability:

- Without **observation**, agents miss context and act on stale beliefs.
- Without **reflection**, agents can't form higher-order beliefs; interactions stay shallow.
- Without **planning**, behavior becomes reactive noise; goals dissipate.

Human raters scored believability highest with all three present; removing any one produced measurable degradation.

### The Valentine's Day Emergence

One agent, Isabella Rodriguez, was seeded with the goal "wants to throw a Valentine's Day party at Hobbs Cafe at 5pm on Feb 14." The other 24 agents received no such seeding. Over simulated days:

1. Isabella's plan includes inviting people.
2. Each invitation becomes an observation in a neighbor's memory stream.
3. That neighbor's reflection generates the belief: "Isabella is having a party."
4. The neighbor's plan incorporates "attend party on Feb 14."
5. The neighbor tells other neighbors. Invitations spread without central coordination.
6. On Feb 14 at 5pm, several agents converge at Hobbs Cafe.

This is emergence in the technical sense: a system-level behavior (a party) arises from local interactions (bilateral invitations + individual planning) without a central orchestrator.

### Documented Failure Modes

Park et al. explicitly documented:

- **Spatial-norm violations.** Agents walk into closed stores. Agents try to use the same single-occupancy bathroom. Agents eat in rooms where eating isn't appropriate. The model can't infer social-physical norms from environment alone.
- **Memory overflow.** Deep simulation runs make memory retrieval cost grow. Practical remedy: periodic memory compaction (summarize-and-prune) and decay on low-importance entries.
- **Reflection hallucination.** Reflections can fabricate relationships not present in the memory stream. Mitigation: include source memory ids in the reflection prompt and verify on retrieval.

These are production-relevant failure modes: any 2026 agent simulation inherits them.

### Three-Component Implementation Rules

1. **Memory is append-only.** Never mutate a memory entry. Corrections are new entries.
2. **Importance scores are cheap.** Call the LLM at write-time to rate importance 1-10. Cache this score.
3. **Retrieval is ranking, not filtering.** Take top-k by combined score; don't hard-filter (you lose context).
4. **Reflection runs periodically.** Trigger when unprocessed memories' importance sum exceeds a threshold (e.g., 150).
5. **Plans are revisable.** When a new observation contradicts the plan, regenerate only the affected segment, not the entire plan.

### Generative Agents Beyond Smallville

2024-2026 follow-up literature extends the architecture:

- **Multi-agent social simulation for policy/market research.** Smallville-like populations simulate user reaction behavior to features. Faster than A/B testing; accuracy is debated.
- **NPC AI for games.** RPGs with Smallville agents produce emergent storylines rather than scripted quests.
- **Generative agent evaluation benchmarks.** Metrics shift from task accuracy to believability + coherence of behavior over long runs.

The architecture is the reference. Extensions swap components (vector store for memory, retrieval-augmented reflection, neuro-symbolic planning) but preserve the three-component structure.

### Why This Matters for Multi-Agent Engineering

Smallville is a proof of concept: multi-agent emergence is cheap when the components are right. The architecture has since been reproduced with open-source models (smaller LLMs degrade believability gracefully, not catastrophically). Any production system needing **emergent social behavior** uses this shape. Any system needing **tight task execution** uses the supervisor / role / primitive patterns from earlier in this phase.

## Build It

`code/main.py` implements the three components with standard-library Python and scripted agent strategies (no real LLM). Demonstrates the Valentine's Day party emergence reproduced at micro-scale:

- `MemoryStream` — append-only log with recency/importance/relevance retrieval.
- `reflect(stream)` — scripted reflection over recent high-importance memories.
- `plan(agent_state)` — day-level and hour-level planning based on current beliefs.
- Scenario: 5 agents. Agent 1 starts with "throw a party at 5pm." Over simulated ticks, invitations spread and agents converge.

Run:

```
python3 code/main.py
```

Expected output: tick-by-tick trace. By the final tick, at least 3 of 5 agents have the party in their plan and converge at the party location. A single seed produces coordinated attendance without any orchestrator.

## Use It

`outputs/skill-simulation-designer.md` designs a generative agent simulation: agent count, memory schema, reflection cadence, planning horizon, and evaluation metrics.

## Ship It

Rules for production simulations:

- **Memory is a database.** At scale, pick a real store (vector DB, Postgres). In-memory stdlib is for prototypes.
- **Log retrieval traces.** Every action logs the top-k memories that drove it. This is your debugging capability.
- **Budget tokens per agent.** Each agent's "retrieval + reflection + planning" per tick is O(k) LLM calls. N agents × T ticks × calls-per-tick can blow your budget.
- **Compact memory periodically.** Summarize-and-prune low-importance entries. Retention policy is a design decision, not a detail.
- **Detect spatial/social norm violations explicitly.** The architecture doesn't learn them.

## Exercises

1. Run `code/main.py`. Confirm 3+ agents converge at the party. Increase agents to 10 — does emergence still happen?
2. Remove the reflection step. What does behavior look like? Map to Park 2023's ablation findings.
3. Introduce a competing seeded goal ("Klaus wants to give a research talk at 5pm"). Do agents split, or does one goal dominate? What determines this?
4. Add a spatial constraint: Hobbs Cafe holds max 4 agents. Does the simulation handle overflow gracefully, or hit the "single-occupancy bathroom" failure mode?
5. Read Park et al. (arXiv:2304.03442) Section 6 (emergent behavior experiments). Identify one behavior your micro-scale version cannot reproduce. Which component of the architecture would you need to enhance?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Memory stream | "Agent's diary" | Append-only log of observations, actions, reflections, plans. |
| Recency | "How fresh the memory is" | Exponential decay score by age. |
| Importance | "How much the agent cares" | Self-rated 1-10 at write time. Cached. |
| Relevance | "How related to current query" | Cosine similarity (embedding-based). |
| Reflection | "Higher-order beliefs" | Syntheses generated from recent memories, re-ingested as new memories. |
| Plan | "Day/hour/action decomposition" | Top-down plan tree. Revisable when observations contradict. |
| Smallville | "Park 2023's sandbox" | The 25-agent simulation that produced the Valentine's Day emergence. |
| Believability | "Quality metric" | Human raters scoring "does behavior look like a plausible agent." |

## Further Reading

- [Park et al. — Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — the reference architecture
- [UIST '23 paper page](https://dl.acm.org/doi/10.1145/3586183.3606763) — publication venue
- [Smallville code release](https://github.com/joonspk-research/generative_agents) — reference Python implementation
- [Hayes-Roth 1985 — A Blackboard Architecture for Control](https://www.sciencedirect.com/science/article/abs/pii/0004370285900639) — prior work on structured-memory agents
