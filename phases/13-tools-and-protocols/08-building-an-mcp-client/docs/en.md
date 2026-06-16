# Building an MCP Client — Discovery, Invocation, Session Management

> Most MCP content ships server tutorials and hand-waves the client. The hard orchestration lives in client code: process spawning, capability negotiation, merging tool lists across multiple servers, sampling callbacks, reconnection, and namespace collision resolution. This lesson builds a multi-server client that lifts three different MCP servers into a flat tool namespace for model consumption.

**Type:** Build
**Languages:** Python (standard library, multi-server MCP client)
**Prerequisites:** Phase 13 · 07 (Building an MCP Server)
**Time:** ~75 minutes

## Learning Objectives

- Spawn an MCP server as a subprocess, complete `initialize`, and send `notifications/initialized`.
- Maintain per-server session state (capabilities, tool list, last-seen notification id).
- Merge tool lists across multiple servers into a single namespace and handle collisions.
- Route a tool call to the owning server and reassemble the response.

## The Problem

A real agent host (Claude Desktop, Cursor, Goose, Gemini CLI) loads multiple MCP servers at once. A user might have a filesystem server, a Postgres server, and a GitHub server running simultaneously. The client's job:

1. Spawn each server.
2. Handshake independently with each.
3. Call `tools/list` on each, flatten the results.
4. When the model emits `notes_search`, look it up in the merged namespace and route to the correct server.
5. Handle notifications from any server (`tools/list_changed`) without blocking.
6. Reconnect on transport failure.

Hand-rolling all of this is what separates "toy" from "usable." Official SDKs wrap these, but the mental model must be yours.

## The Concept

### Subprocess spawning

`subprocess.Popen` with `stdin=PIPE, stdout=PIPE, stderr=PIPE`. Set `bufsize=1` and use text mode for line-by-line reading. Each server is a process; the client holds one `Popen` handle per server.

### Per-server session state

One `Session` object per server, holding:

- `process` — the Popen handle.
- `capabilities` — what the server declared in `initialize`.
- `tools` — last `tools/list` result.
- `pending` — map of request id to a promise/future awaiting a response.

Requests are inherently async; a `tools/call` sent to server A cannot block while server B is mid-call. Use either threads with queues or asyncio.

### Merging namespaces

When the client sees the aggregated tool list, names may collide. Two servers might both expose `search`. The client has three options:

1. **Prefix by server name.** `notes/search`, `files/search`. Clear but ugly.
2. **Silent first-come.** Second server's `search` shadows the first. Risky; hides conflicts.
3. **Conflict rejection.** Refuse to load the second server; notify the user. Safest for security-sensitive hosts.

Claude Desktop uses prefix-by-server. Cursor uses conflict rejection with a clear error. VS Code MCP also adopts prefix-by-server.

### Routing

After merging, a dispatch table maps `tool_name -> session`. The model emits a call by name; the client finds the session, writes a `tools/call` message to that server's stdin, and awaits the response.

### Sampling callback

If a server declared `sampling` capability in `initialize`, it can send `sampling/createMessage` asking the client to run its LLM. The client must:

1. Block further requests to that server until the sample resolves, or pipeline if its implementation supports concurrency.
2. Call its LLM provider.
3. Send the response back to the server.

Lesson 11 covers sampling end to end. This lesson stubs it for completeness.

### Notification handling

`notifications/tools/list_changed` means re-call `tools/list`. `notifications/resources/updated` means re-read the resource if it's in use. Notifications must not produce responses — don't try to ack them.

A common client bug: blocking the read loop on a `tools/call` while a notification sits in the stream. Use a background reader thread that pushes every message onto a queue; the main thread dequeues and dispatches.

### Reconnection

Transports fail: servers crash, the OS kills the process, stdio pipes break. The client detects EOF on stdout and marks that session as dead. Options:

- Silently restart the server and re-handshake. Fine for pure read-only servers.
- Surface the failure to the user. Fine for stateful servers with user-visible sessions.

Phase 13 · 09 covers Streamable HTTP reconnection semantics; stdio is simpler.

### Keepalive and session ids

Streamable HTTP uses an `Mcp-Session-Id` header. stdio has no session id — process identity is the session. Keepalive pings are optional; stdio pipes don't timeout from inactivity.

## Use It

`code/main.py` spawns three simulated MCP servers as subprocesses, handshakes with each, merges their tool lists, and routes a tool call to the correct one. The "servers" are other Python processes running toy responders (no real LLM). Run it and watch:

- Three initializations with their own capability sets.
- Three `tools/list` results merged into a 7-tool namespace.
- A routing decision based on tool name.
- A collision prevented by namespace prefixing.

What to look for:

- The `Session` dataclass cleanly holds per-server state.
- A background reader thread dequeues every line on stdout without blocking the main thread.
- The dispatch table is a simple `dict[str, Session]`.
- Conflict handling is explicit: when two servers declare the same name, the second is prefix-renamed.

## Ship It

This lesson produces `outputs/skill-mcp-client-harness.md`. Given a declarative list of MCP servers (name, command, args), this skill produces scaffolding that spawns them, merges tool lists, and delivers a routing function with collision resolution.

## Exercises

1. Run `code/main.py` and watch the server startup logs. Kill one simulated server process with SIGTERM and observe how the client detects EOF and marks that session dead.

2. Implement namespace prefixing. When two servers both expose `search`, rename the second to `<server>/search`. Update the dispatch table and verify tool calls route correctly.

3. Add connection-pool-style backoff for server restarts: exponential backoff on consecutive failures, capped at 30 seconds, with a notification to the user after three failures.

4. Sketch a client that supports 100 concurrent MCP servers. What data structure replaces the simple dispatch dict? (Hint: a trie for prefix namespaces, plus a per-server tool-count metric.)

5. Port the client to the official MCP Python SDK. The SDK wraps `stdio_client` and `ClientSession`. Code should shrink from ~200 lines to ~40 while preserving multi-server routing.

## Key Terms

| Term | Common shorthand | What it actually is |
|------|----------------|------------------------|
| MCP client | "agent host" | The process that spawns servers and orchestrates tool calls |
| Session | "per-server state" | Capabilities, tool list, and pending-request bookkeeping |
| Merged namespace | "one tool list" | Flat set of tool names across all active servers |
| Namespace collision | "two servers same name" | Client must prefix, reject, or first-come duplicates |
| Routing | "who owns this call?" | Dispatching from tool name to the owning server |
| Background reader | "non-blocking stdout" | Thread or task that drains server stdout into a queue |
| Sampling callback | "LLM as a service" | Client's handler for `sampling/createMessage` from a server |
| `notifications/*_changed` | "primitive mutated" | Signal that the client must re-discover or re-read |
| Reconnection policy | "when a server dies" | Restart semantics on transport failure |
| Stdio session | "process = session" | No session id; subprocess lifetime is the session |

## Further Reading

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — Authoritative client behavior
- [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) — Hello-world client tutorial using the Python SDK
- [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk) — Reference `ClientSession` and `stdio_client`
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) — Parallel TS implementation
- [VS Code — MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp) — How VS Code multiplexes multiple MCP servers in a single editor host
