# Async Tasks (SEP-1686) — Call Now, Get Results Later for Long-Running Work

> Real agent work takes minutes to hours: CI runs, deep research synthesis, batch exports. Synchronous tool calls drop connections, time out, or block the UI. SEP-1686 merged in 2025-11-25, adding a Tasks primitive: any request can be enhanced into a task whose result can be fetched later or streamed via status notifications. Drift warning: Tasks is experimental in 2026 H1; the SDK surface is still settling around the spec.

**Type:** Build
**Languages:** Python (stdlib, async task state machine)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 09 (transports)
**Time:** ~75 minutes

## Learning Objectives

- Identify when to promote a tool from synchronous to task-enhanced (server-side work >30 seconds).
- Walk through the task lifecycle: `working` → `input_required` → `completed` / `failed` / `cancelled`.
- Persist task state so crashes don't lose in-flight work.
- Poll `tasks/status` and fetch `tasks/result` correctly.

## The Problem

A `generate_report` tool runs an extraction pipeline that takes several minutes. Options under the synchronous model:

1. Hold the connection open for three minutes. Remote transports cut it; the client times out; the UI freezes.
2. Return a placeholder immediately; require the client to poll a custom endpoint. Breaks MCP's uniformity.
3. Fire and forget; no result.

None are good. SEP-1686 adds a fourth: task enhancement. Any request (typically `tools/call`) can be marked as a task. The server returns a task id immediately. The client polls `tasks/status` and fetches `tasks/result` on completion. Server-side state survives restarts.

## The Concept

### Task enhancement

A request becomes a task by setting `params._meta.task.required: true` (or `optional: true`, server decides). The server responds immediately:

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` is the server's promise to retain state; after ttl the task result is discarded.

### Per-tool opt-in

Tool annotations can declare task support:

- `taskSupport: "forbidden"` — this tool always runs synchronously. Safe for fast tools.
- `taskSupport: "optional"` — the client can request task enhancement.
- `taskSupport: "required"` — the client must use task enhancement.

A `generate_report` tool would be `required`. A `notes_search` tool would be `forbidden`.

### States

```
working  -> input_required -> working  (via elicitation loop)
working  -> completed
working  -> failed
working  -> cancelled
```

The state machine is append-only: once `completed`, `failed`, or `cancelled`, the task is terminal.

### Methods

- `tasks/status {taskId}` — returns current state and a progress hint.
- `tasks/result {taskId}` — blocks, or returns 404 if not yet complete.
- `tasks/cancel {taskId}` — idempotent; terminal states are ignored.
- `tasks/list` — optional; enumerates active and recently completed tasks.

### Streaming state changes

When supported by the server, the client can subscribe to status notifications:

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

Clients that stream rather than poll get better UX. Polling is always supported as the minimum surface.

### Durable state

The spec requires servers that declare task support to persist state. Crashes should not lose completed results within ttl. Storage ranges from SQLite to Redis to the filesystem. The Lesson 13 scaffolding uses the filesystem.

### Cancellation semantics

`tasks/cancel` is idempotent. If the task is mid-execution, the server attempts to stop (depending on whether the executor supports cooperative cancellation). If already terminal, the request is a no-op.

### Crash recovery

When a server process restarts:

1. Load all persisted task state.
2. Mark any `working` tasks whose processes are dead as `failed` with error `CRASH_RECOVERY`.
3. Retain `completed` / `failed` / `cancelled` within ttl.

### Async tasks plus sampling

A task can itself call `sampling/createMessage`. This is how long-running research tasks work: the server's task thread samples the client's model on demand, while the client's UI shows the task as `working` with periodic progress updates.

### Why this is experimental

SEP-1686 shipped in 2025-11-25, but the broader roadmap flags three open questions: durable subscription primitives, subtasks (parent-child task relationships), and result-TTL standardization. Expect the spec to evolve through 2026. Production code should treat Tasks as stable only for common cases and add guards for future SDK changes around subtasks.

## Use It

`code/main.py` implements a durable task store (filesystem backend) and a `generate_report` tool that runs in a background thread. The client calls the tool, immediately gets a task id, polls `tasks/status` while the worker updates progress, and fetches `tasks/result` on completion. Cancellation works; crash recovery is simulated by killing the worker thread and reloading state.

What to look for:

- Task state JSON persists to `/tmp/lesson-13-tasks/<id>.json`.
- The worker thread updates the `progress` field; polling shows it advancing.
- Client-side cancellation sets an event; the worker checks and exits early.
- State reload on "crash" marks in-flight tasks as `failed` with `CRASH_RECOVERY`.

## Ship It

This lesson produces `outputs/skill-task-store-designer.md`. Given a long-running tool (research, build, export), this skill designs the task store (state shape, ttl, durability), picks the correct taskSupport flag, and sketches the progress notifications.

## Exercises

1. Run `code/main.py`. Start a `generate_report` task, poll status, then fetch the result.

2. Add a `tasks/cancel` call mid-run. Verify the worker respects it and the state becomes `cancelled`.

3. Simulate crash recovery: kill the worker thread, restart the loader, observe the `CRASH_RECOVERY` failure mode.

4. Extend the store to SQLite. Same durability benefit; query options open up (list all tasks for session X).

5. Read the 2026 MCP roadmap blog post. Identify the one Tasks-related open question most likely to affect SDK API design in the coming year.

## Key Terms

| Term | What people call it | What it actually is |
|------|---------------------|---------------------|
| Task | "long-running tool call" | Request enhanced with `_meta.task` for async execution |
| SEP-1686 | "Tasks spec" | Spec Evolution Proposal adding Tasks in 2025-11-25 |
| `_meta.task` | "task envelope" | Per-request metadata containing id, state, ttl |
| taskSupport | "tool flag" | Per-tool `forbidden` / `optional` / `required` |
| `tasks/status` | "poll method" | Fetches current state and optional progress hint |
| `tasks/result` | "fetch result" | Returns completed payload, or 404 if not yet done |
| `tasks/cancel` | "stop it" | Idempotent cancellation request |
| ttl | "retention budget" | Milliseconds the server promises to retain task state |
| `notifications/tasks/updated` | "status push" | Server-initiated state change event |
| Durable store | "crash-safe state" | Filesystem / SQLite / Redis persistence layer |

## Further Reading

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — origin proposal and full discussion
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — design walkthrough with rationale
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — mechanics and state machine
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) — SDK-level task implementation patterns
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — open questions and 2026 priorities including subtasks
