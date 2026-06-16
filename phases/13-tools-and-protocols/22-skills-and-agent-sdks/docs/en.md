# Skills and Agent SDKs — Anthropic Skills, AGENTS.md, OpenAI Apps SDK

> MCP says "what tools exist." Skills say "how to do a task." The 2026 stack layers both together. Anthropic's Agent Skills (open standard, December 2025) ship as SKILL.md with progressive disclosure. OpenAI's Apps SDK is MCP plus widget metadata. AGENTS.md (now in 60,000+ repos) sits at the repo root as project-level agent context. This lesson names what each covers and builds a minimal SKILL.md + AGENTS.md bundle that flows across agents.

**Type:** Learn
**Languages:** Python (standard library, SKILL.md parser and loader)
**Prerequisites:** Phase 13 · 07 (MCP server)
**Time:** ~45 minutes

## Learning Objectives

- Distinguish the three layers: AGENTS.md (project context), SKILL.md (reusable knowledge), MCP (tools).
- Write a SKILL.md with YAML frontmatter and progressive disclosure.
- Load skills into an agent runtime via the filesystem.
- Combine a skill with an MCP server and an AGENTS.md so a single bundle works in Claude Code, Cursor, and Codex.

## The Problem

An engineer distills a release-notes workflow into a multi-step prompt: "Read recently merged PRs. Group by area. Summarize each. Write a changelog entry in the team's style. Post to a Slack draft." They put it in a team Notion doc.

Now they want to use this workflow from Claude Code, Cursor, and Codex CLI. Each agent loads instructions differently: Claude Code's slash-command, Cursor's rule, Codex's `.codex.md`. The engineer copies the workflow three times and maintains three copies.

AGENTS.md and SKILL.md together fix this:

- **AGENTS.md** sits at the repo root. Every compatible agent reads it at session start. "How does this project work? What conventions apply? Which commands run tests?"
- **SKILL.md** is a portable bundle: YAML frontmatter (name, description) + markdown body + optional resources. Agents that support skills load them by name on demand.
- **MCP** (Phase 13 · 06–14) handles the tools the skill needs to call.

Three layers, one portable artifact.

## The Concept

### AGENTS.md (agents.md)

Launched in late 2025; adopted by 60,000+ repos by April 2026. One file at the repo root. Format:

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

The agent reads this at session start and uses it to calibrate its behavior for that project. Every coding agent in 2026 supports AGENTS.md: Claude Code, Cursor, Codex, Copilot Workspace, opencode, Windsurf, Zed.

### SKILL.md Format

Anthropic's Agent Skills (released December 2025 as an open standard):

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

The frontmatter declares the skill's identity. The body is the prompt shown to the model when the skill is loaded.

### Progressive Disclosure

Skills can reference sub-resources that the agent fetches only when needed. Example:

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

The SKILL.md says "see style-guide.md for style rules." The agent pulls style-guide.md only while the skill is running. This avoids bloating the prompt with details the model may not need.

### Filesystem Discovery

Agent runtimes scan known directories for SKILL.md files:

- `~/.anthropic/skills/*/SKILL.md`
- The project's `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

Loading is keyed by folder name and the frontmatter's `name`. Claude Code, the Anthropic Claude Agent SDK, and SkillKit (cross-agent) all follow this pattern.

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk` (TypeScript) and `claude-agent-sdk` (Python) load skills at session start and expose them as callable "agents" within the runtime. When a user triggers a skill, the agent loop dispatches to it.

### OpenAI Apps SDK

Launched October 2025; built directly on MCP. Unifies OpenAI's former Connectors and Custom GPT Actions under a single developer surface. An Apps SDK app is:

- An MCP server (tools, resources, prompts).
- Plus widget metadata for the ChatGPT UI.
- Plus an optional MCP Apps `ui://` resource for interactive surfaces.

Same protocol, richer UX.

### Cross-Agent Portability via SkillKit

Tools like SkillKit and similar cross-agent distribution layers translate a single SKILL.md into the native formats of 32+ AI agents (Claude Code, Cursor, Codex, Gemini CLI, OpenCode, etc.). One source of truth; many consumers.

### The Three-Layer Stack

| Layer | File | When loaded | Purpose |
|-------|------|-------------|---------|
| AGENTS.md | Repo root | Session start | Project-level conventions |
| SKILL.md | Skills directory | Skill triggered | Reusable workflow |
| MCP server | External process | Tool needed | Callable actions |

All three compose: the agent reads AGENTS.md at session start, the user triggers a skill, the skill's instructions contain MCP tool calls, and the agent dispatches via an MCP client.

## Use It

`code/main.py` delivers a standard-library SKILL.md parser and loader. It discovers skills under `./skills/`, parses YAML frontmatter plus markdown body, and produces a dict keyed by skill name. It then simulates an agent loop that triggers `release-notes-writer` by name.

What to look at:

- YAML frontmatter is parsed with a minimal standard-library parser (no `pyyaml` dependency).
- The skill body is stored verbatim; the agent prepends it to the system prompt on trigger.
- Progressive disclosure is demonstrated via a `read_subresource` function that pulls referenced files on demand.

## Ship It

This lesson produces `outputs/skill-agent-bundle.md`. Given a workflow, this skill produces the combined SKILL.md + AGENTS.md + MCP-server blueprint bundle, portable across agents.

## Exercises

1. Run `code/main.py`. Add a second skill under `skills/`; confirm the loader picks it up.

2. Write an AGENTS.md for this curriculum's repo. Include test commands, style conventions, and the Phase 13 mental model.

3. Port a multi-step workflow from your team's internal docs into a SKILL.md. Verify it loads in Claude Code.

4. Manually translate the skill into Cursor's and Codex's native rule formats. Count the diffs between formats — that's the translation surface SkillKit automates.

5. Read the Anthropic Agent Skills blog post. Find one feature in the Claude Agent SDK that this lesson's loader does not cover. (Hint: agent sub-invocations.)

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| SKILL.md | "Skill file" | YAML frontmatter plus markdown body, loaded by agent runtimes |
| AGENTS.md | "Repo-root agent context" | Project-level conventions file read at session start |
| Progressive disclosure | "Lazy-loaded sub-resources" | Skill body references files pulled only when needed |
| Frontmatter | "The YAML block at the top" | Metadata (name, description) inside `---` delimiters |
| Claude Agent SDK | "Anthropic's skill runtime" | `@anthropic-ai/claude-agent-sdk`, loads skills and routes |
| OpenAI Apps SDK | "MCP + widget metadata" | OpenAI's developer surface built on MCP plus ChatGPT UI hooks |
| Skill discovery | "Filesystem scan" | Walking known directories for SKILL.md, keyed by name |
| Cross-agent portability | "One skill, many agents" | Translating one SKILL.md to 32+ agents via SkillKit-style tools |
| Agent Skill | "Portable knowledge" | Reusable task template beyond the MCP tool concept |
| Apps SDK | "MCP plus ChatGPT UI" | Connectors and Custom GPTs unified on MCP |

## Further Reading

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — December 2025 release
- [Anthropic — Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md format reference
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — MCP-based developer platform for ChatGPT
- [agents.md](https://agents.md/) — AGENTS.md format and adoption list
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — Official skill examples
