# Skill Libraries and Lifelong Learning (Voyager)

> Voyager (Wang et al., TMLR 2024) treats executable code as a skill. Skills are named, retrievable, composable, and refined by environment feedback. This is the reference architecture for Claude Agent SDK skills, skillkit, and the 2026 skill-library pattern.

**Type:** Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 minutes

## Learning Objectives

- Name Voyager's three components — automatic curriculum, skill library, iterative prompting — and the role of each.
- Explain why Voyager makes the action space code rather than raw commands.
- Implement a skill library with the standard library, including registration, retrieval, composition, and failure-driven refinement.
- Map the Voyager pattern to the 2026 Claude Agent SDK skill and skillkit ecosystem.

## The Problem

An agent that rebuilds every capability from scratch each session gets three things wrong:

1. **Wasted tokens.** Every task re-elicits the same reasoning.
2. **Lost progress.** A correction learned in session A doesn't transfer to session B.
3. **Failure on long-horizon composition.** Complex tasks require capability hierarchies; a single prompt can't express them.

Voyager's answer: treat each reusable capability as a named piece of code stored in a library, retrievable by similarity, composable with other skills, and refined by execution feedback.

## The Concept

### Three Components

Voyager (arXiv:2305.16291) builds an agent around three things:

1. **Automatic curriculum.** A curiosity-driven proposer that picks the next task based on the agent's current skill set and environment state. Exploration is bottom-up.
2. **Skill library.** Every skill is executable code. New skills are added on task success. Skills are retrieved by query-to-description similarity.
3. **Iterative prompting mechanism.** On failure, the agent receives execution errors, environment feedback, and self-verification output, then refines the skill.

Minecraft evaluation (Wang et al., 2024): 3.3x more unique items, 8.5x faster to stone tools, 6.4x faster to iron tools, 2.3x longer map traversal distance compared to baselines. The numbers are Minecraft-specific, but the pattern transfers.

### Action Space = Code

Most agents emit raw commands. Voyager emits JavaScript functions. A skill is:

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

Composed from sub-skills. Stored keyed by description and embedding. Retrieved as programs, not prompts.

This is the 2026 Claude Agent SDK skill: a named, retrievable piece of code plus instructions that the agent loads on demand.

### Skill Retrieval

New task: "craft a diamond pickaxe." The agent:

1. Embeds the task description.
2. Queries the skill library for top-k similar skills.
3. Retrieves `craftIronPickaxe`, `mineDiamond`, `placeCraftingTable`, etc.
4. Composes a new skill from retrieved primitives + new logic.

This is the pattern that MCP resources (Phase 13) and Agent SDK skills implement: retrieval over a knowledge/code surface scoped to the current task.

### Iterative Refinement

Voyager's feedback loop:

1. Agent writes a skill.
2. Skill runs against the environment.
3. One of three signals returns: `success`, `error` (with stack trace), `self-verification failure`.
4. Agent rewrites the skill using that signal as context.
5. Loop until success or max iterations.

This is Self-Refine (Lesson 05) applied to code generation with environment-grounded verification. CRITIC (Lesson 05) is the same pattern but using external tools as verifiers.

### Curriculum and Exploration

Voyager's curriculum module proposes tasks like "build a shelter by the lake" based on "what the agent has" and "what it hasn't done yet." The proposer uses environment state + skill inventory to pick a task just above the agent's current capability — the exploration sweet spot.

For production agents this translates to a "what's missing" operator: given the current skill library and a domain, which skills haven't we covered? Teams usually implement this manually as curriculum review.

### Where This Pattern Breaks

- **Skill library rot.** The same skill gets added 10 times with slightly different descriptions. Deduplicate on write; retrieval returns only one.
- **Composed skill drift.** A parent skill depends on a child skill that was refined. Version skills; a parent pinned to v1 doesn't magically pick up v3.
- **Retrieval quality.** Vector retrieval over skill descriptions degrades once the library grows past a few hundred. Supplement with tag filtering and hard constraints ("only skills with `category=tooling`").

## Build It

`code/main.py` implements a skill library with the standard library:

- `Skill` — name, description, code (as string), version, tags, dependencies.
- `SkillLibrary` — register, search (token overlap), compose (dependency topological sort), and refine (version bump on update).
- A scripted agent that registers three primitive skills, composes a fourth, hits a failure, then refines.

Run it:

```
python3 code/main.py
```

The trace shows library writes, retrieval, composition, a failed execution, and a v2 refinement — the end-to-end Voyager loop.

## Use It

- **Claude Agent SDK skill** (Anthropic) — The 2026 reference: each skill has description, code, and instructions; loaded on demand during agent sessions.
- **skillkit** (npm: skillkit) — Cross-agent skill management supporting 32+ AI coding agents.
- **Custom skill libraries** — Domain-specific (SQL skills for data agents, Terraform skills for infra agents). The Voyager pattern scales down.
- **OpenAI Agents SDK `tools`** — At the low end; each tool is a lightweight skill.

## Ship It

`outputs/skill-skill-library.md` generates a Voyager-shaped skill library for any target runtime with registration, retrieval, versioning, and refinement wired in.

## Exercises

1. Add a dependency-cycle detector to `compose()`. What happens when skill A depends on B and B depends on A? Error or warning?
2. Implement per-skill version pinning. When a parent skill composes child skill `crafting@1`, refining `crafting@2` must never silently upgrade the parent.
3. Replace token-overlap retrieval with sentence-transformers embeddings (or a standard-library BM25 implementation). Measure retrieval@5 on a 50-skill toy library.
4. Add a "curriculum" agent: given the current library and a domain description, propose 5 missing skills. Run weekly.
5. Read Anthropic's Claude Agent SDK skill documentation. Port the toy library to the SDK's skill schema. What changes about discoverability?

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Skill | "reusable capability" | A named piece of code + description, retrievable by similarity |
| Skill library | "agent's memory of how-to" | Persistent store of skills, searchable and composable |
| Curriculum | "task proposer" | Bottom-up goal generator driven by current capability gaps |
| Composition | "skill DAG" | Skills calling skills; topological sort at execution |
| Iterative refinement | "self-correction loop" | Environment feedback + errors + self-verification folded into the next version |
| Action-space-as-code | "programmatic actions" | Emitting functions rather than raw commands to express temporally extended behavior |
| Dedup on write | "skill merge" | Near-duplicate descriptions merged into one canonical skill |

## Further Reading

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — Original skill library paper
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Skills as a 2026 productization
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — Skills and sub-agents in practice
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — The refinement loop underneath Voyager
