# Claude Agent SDK: Subagents and Session Storage

> The Claude Agent SDK is the library form of the Claude Code harness. Built-in tools, subagents for context isolation, hooks, W3C trace propagation, and session storage aligned with TypeScript. Claude Managed Agents is the hosted alternative for long-running async work.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 01 (Agent loop), Phase 14 · 10 (Skill library)
**Time:** ~75 minutes

## Learning Objectives

- Explain the difference between the Anthropic Client SDK (raw API) and the Claude Agent SDK (harness form).
- Describe subagents — parallelization and context isolation — and when to use them.
- Name the Python SDK's session store surface (`append`, `load`, `list_sessions`, `delete`, `list_subkeys`) and what `--session-mirror` does.
- Implement a harness with the standard library featuring built-in tools, context-isolated subagent spawning, lifecycle hooks, and a session store.

## The Problem

A raw LLM API gives you a single round-trip. A production agent needs tool execution, MCP servers, lifecycle hooks, subagent spawning, session persistence, and trace propagation. The Claude Agent SDK provides this shape as a library — the same harness Claude Code uses, exposed for custom agents.

## The Concept

### Client SDK vs Agent SDK

- **Client SDK (`anthropic`).** Raw Messages API. You own the loop, tools, and state.
- **Agent SDK (`claude-agent-sdk`).** Built-in tool execution, MCP connection, hooks, subagent spawning, session storage. The Claude Code loop as a library.

### Built-in Tools

The SDK ships 10+ tools out of the box: file read/write, shell, grep, glob, web fetch, and more. Custom tools register through the standard tool schema interface.

### Subagents

Anthropic documents two use cases:

1. **Parallelization.** Run independent work concurrently. "Find test files for each of these 20 modules" becomes 20 parallel subagent tasks.
2. **Context isolation.** Subagents use their own context windows; only results return to the orchestrator. The orchestrator's budget is preserved.

Recent Python SDK additions: `list_subagents()`, `get_subagent_messages()` for reading subagent transcripts.

### Session Storage

Aligned with the TypeScript protocol:

- `append(session_id, message)` — add a turn.
- `load(session_id)` — restore a conversation.
- `list_sessions()` — enumerate.
- `delete(session_id)` — cascades to subagent sessions.
- `list_subkeys(session_id)` — list subagent keys.

`--session-mirror` (CLI flag) mirrors the transcript to an external file as it streams, for debugging.

### Hooks

Lifecycle hooks you can register:

- `PreToolUse`, `PostToolUse` — gate or audit tool calls.
- `SessionStart`, `SessionEnd` — setup and teardown.
- `UserPromptSubmit` — act on user input before the model sees it.
- `PreCompact` — run before context compaction.
- `Stop` — cleanup when the agent exits.
- `Notification` — sideband alerts.

Hooks are how pro-workflows (Phase 14 lesson reference) and similar systems add cross-cutting behavior.

### W3C Trace Context

An active OTel span on the caller propagates into the CLI subprocess via W3C trace context headers. The entire multi-process trace shows up as a single trace in your backend.

### Claude Managed Agents

The hosted alternative (beta header `managed-agents-2026-04-01`). Long-running async work, built-in prompt caching, built-in compaction. You trade control for managed infrastructure.

### Where This Pattern Breaks Down

- **Subagent over-spawning.** Spawning 100 subagents for 100 small tasks. Overhead dominates. Batch instead.
- **Hook sprawl.** Every team adds hooks; startup time balloons. Review hooks quarterly.
- **Session bloat.** Sessions accumulate; volume grows. Use `list_sessions` + expiry policies.

## Build It

`code/main.py` implements the SDK's shape using the standard library:

- `Tool`, `ToolRegistry` with built-in `read_file`, `write_file`, `list_dir`.
- `Subagent` — private context, isolated execution, returns results.
- `SessionStore` — append, load, list, delete, list_subkeys.
- `Hooks` — `pre_tool_use`, `post_tool_use`, `session_start`, `session_end`.
- A demo: the main agent spawns 3 subagents in parallel (each isolated), aggregates results, and persists the session.

Run it:

```
python3 code/main.py
```

The trace shows subagent context isolation (orchestrator context size stays bounded), hook execution, and session persistence.

## Use It

- **Claude Agent SDK** for Claude-first products that want the Claude Code harness shape.
- **Claude Managed Agents** for hosted long-running async work.
- **OpenAI Agents SDK** (Lesson 16) for the OpenAI-first counterpart.
- **LangGraph + custom tools** if you want a graph-shaped state machine.

## Ship It

`outputs/skill-claude-agent-scaffold.md` scaffolds a Claude Agent SDK application with subagents, hooks, session storage, MCP server mounts, and W3C trace propagation.

## Exercises

1. Add a subagent spawner that batches 20 tasks into groups of 5 parallel subagents. Measure orchestrator context size vs one-per-task.
2. Implement a `PreToolUse` hook that rate-limits `write_file` calls (5 per session per minute). Trace the behavior.
3. Wire up `list_subkeys` to render a subagent tree. What does deep nesting look like?
4. Port the toy to the real `claude-agent-sdk` Python package. What changes in tool registration?
5. Read the Claude Managed Agents docs. When would you switch from self-hosted to managed?

## Key Terms

| Term | Common description | What it actually is |
|------|----------------|------------------------|
| Agent SDK | "Claude Code as a library" | Harness form: tools, MCP, hooks, subagents, session storage |
| Subagent | "child agent" | Independent context, own budget; results bubble up |
| Session store | "conversation database" | Persists, loads, lists, deletes turns with subagent cascade |
| Hook | "lifecycle callback" | pre/post tool, session, prompt submit, compact, stop |
| W3C trace context | "cross-process trace" | Parent span propagates into CLI subprocesses |
| Managed Agents | "hosted harness" | Anthropic-hosted long-running async work |
| `--session-mirror` | "transcript mirror" | Writes session turns to an external file as they stream |
| MCP server | "tool surface" | External tool/resource source mounted on the agent |

## Further Reading

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — the library form of Claude Code
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — production patterns
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — the hosted alternative
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — the counterpart
