# MCP Fundamentals — Primitives, Lifecycle, JSON-RPC Foundation

> Every integration before MCP was one-off. The Model Context Protocol, first released by Anthropic in November 2024 and now hosted by the Linux Foundation's Agentic AI Foundation, standardizes discovery and invocation so any client can talk to any server. The 2025-11-25 specification names six primitives (three server, three client), a three-phase lifecycle, and a JSON-RPC 2.0 wire format. Learn these, and the rest of this phase's MCP chapters become a casual reference browse.

**Type:** Learn
**Languages:** Python (standard library, JSON-RPC parser)
**Prerequisites:** Phase 13 · 01–05 (Tool Interface and Function Calling)
**Time:** ~45 minutes

## Learning Objectives

- Name all six MCP primitives (server-side tools, resources, prompts; client-side roots, sampling, elicitation) with one use case each.
- Walk through the three-phase lifecycle (initialize, operation, shutdown) and identify who sends which messages at each phase.
- Parse and emit JSON-RPC 2.0 request, response, and notification envelopes.
- Explain what capability negotiation during `initialize` is, and what breaks without it.

## The Problem

Before MCP, every tool-using agent had its own protocol. Cursor had an MCP-shaped but incompatible tool system. Claude Desktop shipped with another. VS Code's Copilot extension used yet a third. A team that built a "Postgres query" tool wrote the same tool three times, each wired to a different host's API. Reusing it required copying code.

The result was a Cambrian explosion of one-off integrations and a ceiling on ecosystem velocity.

MCP fixes this by standardizing the wire format. A single MCP server works in every MCP client: Claude Desktop, ChatGPT, Cursor, VS Code, Gemini, Goose, Zed, Windsurf — 300+ clients by April 2026. 110 million SDK downloads per month. 10,000+ public servers. The Linux Foundation took over hosting in December 2025 under the newly formed Agentic AI Foundation.

The specification revision used throughout this phase is **2025-11-25**. It adds asynchronous Tasks (SEP-1686), URL-pattern elicitation (SEP-1036), sampling with tools (SEP-1577), incremental scope consent (SEP-835), and OAuth 2.1 resource-indicator semantics. Phases 13 · 09–16 cover those extensions. This lesson stays at the foundation.

## The Concept

### Three server primitives

1. **Tools.** Callable actions. Same four-step loop as Phase 13 · 01.
2. **Resources.** Exposed data. Read-only, URI-addressable content: `file:///path`, `db://query/...`, custom schemes.
3. **Prompts.** Reusable templates. Slash commands in the host UI; server provides templates, client fills parameters.

### Three client primitives

4. **Roots.** The set of URIs the server is allowed to touch. Client declares them; server respects them.
5. **Sampling.** Server asks the client's model to perform a completion. Lets server-hosted agent loops run without a server-side API key.
6. **Elicitation.** Server asks the client's user for structured input mid-flow. Form or URL (SEP-1036).

Every capability in MCP belongs to exactly one of these six. Phases 13 · 10–14 go deep on each.

### Wire format: JSON-RPC 2.0

Every message is a JSON object with these fields:

- Request: `{jsonrpc: "2.0", id, method, params}`.
- Response: `{jsonrpc: "2.0", id, result | error}`.
- Notification: `{jsonrpc: "2.0", method, params}` — no `id`, no response expected.

The base spec has ~15 methods, grouped by primitive. The important ones:

- `initialize` / `initialized` (handshake)
- `tools/list`, `tools/call`
- `resources/list`, `resources/read`, `resources/subscribe`
- `prompts/list`, `prompts/get`
- `sampling/createMessage` (server to client)
- `notifications/tools/list_changed`, `notifications/resources/updated`, `notifications/progress`

### Three-phase lifecycle

**Phase 1: initialize.**

Client sends `initialize` with its `capabilities` and `clientInfo`. Server responds with its own `capabilities`, `serverInfo`, and the spec version it speaks. After the client digests the response, it sends `notifications/initialized`. From this point on, either side can send requests per the negotiated capabilities.

**Phase 2: operation.**

Bidirectional. Client calls `tools/list` to discover, `tools/call` to invoke. Server may send `sampling/createMessage` if the server declared that capability. Server may send `notifications/tools/list_changed` when its tool set mutates. Client may send `notifications/roots/list_changed` when the user changes root scope.

**Phase 3: shutdown.**

Either side closes the transport. MCP has no structured shutdown method; the transport (stdio or Streamable HTTP, Phase 13 · 09) carries the connection-end signal.

### Capability negotiation

The `capabilities` in the `initialize` handshake are the contract. A server example:

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

The server declares it can send `tools/list_changed` notifications and supports `resources/subscribe`. The client agrees by declaring its own:

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

If the client doesn't declare `sampling`, the server must not call `sampling/createMessage`. Symmetrically: if the server doesn't declare `resources.subscribe`, the client must not attempt subscription.

This is what prevents ecosystem drift. A client that doesn't support sampling is still a valid MCP client; a server that doesn't call `sampling` is still a valid MCP server. They just don't use that feature together.

### Structured content and error shapes

`tools/call` returns a `content` array of typed blocks: `text`, `image`, `resource`. Phase 13 · 14 adds MCP Apps (`ui://` interactive UIs) to this list.

Errors use JSON-RPC error codes. Spec-defined additions: `-32002` "Resource not found", `-32603` "Internal error", plus MCP-specific error data as `error.data`.

### Client capabilities vs. tool-call details

A common confusion: `capabilities.tools` says whether the client supports tool-list-changed notifications. Whether the client will invoke specific tools is a runtime choice driven by its model, not a capability flag. Capability flags are spec-level contracts. The model's choices are orthogonal.

### Why JSON-RPC instead of REST?

JSON-RPC 2.0 (2010) is a lightweight bidirectional protocol. REST is client-initiated. MCP needs server-initiated messages (sampling, notifications), so JSON-RPC with its symmetric request/response shape is the natural fit. JSON-RPC also composes cleanly atop stdio and WebSocket/Streamable HTTP without reinventing HTTP's request shapes.

## Use It

`code/main.py` delivers a minimal JSON-RPC 2.0 parser and emitter, then manually walks through the `initialize` → `tools/list` → `tools/call` → `shutdown` sequence, printing each message. No real transport; just message shapes. Cross-reference against the spec linked in Further Reading and verify each envelope.

What to look for:

- `initialize` declares capabilities in both directions; the response contains `serverInfo` and `protocolVersion: "2025-11-25"`.
- `tools/list` returns a `tools` array; each entry has `name`, `description`, `inputSchema`.
- `tools/call` uses `params.name` and `params.arguments`.
- The response's `content` is an array of `{type, text}` blocks.

## Ship It

This lesson produces `outputs/skill-mcp-handshake-tracer.md`. Given a pcap-style log of an MCP client-server interaction, this skill annotates each message with which primitive it belongs to, which lifecycle phase, and which capability it relies on.

## Exercises

1. Run `code/main.py`. Identify the line where capability negotiation happens and describe what changes if the server doesn't declare `tools.listChanged`.

2. Extend the parser to handle `notifications/progress`. Message shape: `{method: "notifications/progress", params: {progressToken, progress, total}}`. Send it during a long-running `tools/call` and confirm the client handler displays a progress bar.

3. Read the MCP 2025-11-25 spec end to end — the full document is ~80 pages. Identify the one capability flag that most servers will never need. Hint: it's related to resource subscriptions.

4. Sketch on paper which primitive a hypothetical "cron job" feature would belong to. (Hint: the server wants the client to invoke it at a scheduled time. None of today's six primitives fit.) MCP's 2026 roadmap has a draft SEP for this.

5. Parse a session log from an open-source MCP server on GitHub. Count request vs. response vs. notification messages. Calculate what fraction of traffic is lifecycle vs. operation.

## Key Terms

| Term | Common shorthand | What it actually is |
|------|----------------|------------------------|
| MCP | "Model Context Protocol" | Open protocol for model-to-tool discovery and invocation |
| Server primitive | "what servers expose" | tools (actions), resources (data), prompts (templates) |
| Client primitive | "what clients let servers use" | roots (scope), sampling (LLM callback), elicitation (user input) |
| JSON-RPC 2.0 | "wire format" | Symmetric request/response/notification envelope |
| `initialize` handshake | "capability negotiation" | First pair of messages; server and client declare supported features |
| `tools/list` | "discovery" | Client asks server for its current tool set |
| `tools/call` | "invocation" | Client asks server to execute a tool with arguments |
| `notifications/*_changed` | "change events" | Server tells client its primitive listings changed |
| Content block | "typed result" | `{type: "text" | "image" | "resource" | "ui_resource"}` in tool results |
| SEP | "Spec Evolution Proposal" | Named draft proposals (e.g., SEP-1686 for async Tasks) |

## Further Reading

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — Authoritative specification document
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — Six-primitive mental model
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — November 2024 launch post
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — Anniversary retrospective and 2025-11-25 spec changes
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — Summary of SEP-1686, 1036, 1577, 835, 1724
