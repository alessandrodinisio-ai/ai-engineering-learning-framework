# Building an MCP Server — Python + TypeScript SDK

> Most MCP tutorials only demo a stdio hello-world. A real server exposes tools plus resources plus prompts, handles capability negotiation, emits structured errors, and behaves consistently across SDKs. This lesson builds a notes server end to end: standard-library stdio transport, JSON-RPC dispatch, all three server primitives, and a pure-function style — ready to graduate into the Python SDK's FastMCP or the TypeScript SDK once you finish.

**Type:** Build
**Languages:** Python (standard library, stdio MCP server)
**Prerequisites:** Phase 13 · 06 (MCP Fundamentals)
**Time:** ~75 minutes

## Learning Objectives

- Implement `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, and `prompts/get` methods.
- Write a dispatch loop that reads JSON-RPC messages from stdin and writes responses to stdout.
- Emit structured error responses per JSON-RPC 2.0 and MCP's additional error codes.
- Graduate a standard-library implementation to FastMCP (Python SDK) or the TypeScript SDK without rewriting tool logic.

## The Problem

Before you can use a remote transport (Phase 13 · 09) or an auth layer (Phase 13 · 16), you need a clean local server. Local means stdio: the server is spawned by the client as a subprocess, messages flow on stdin/stdout as newline-delimited JSON.

The 2025-11-25 spec mandates stdio messages encoded as JSON objects with an explicit `\n` delimiter. No SSE here; SSE is the legacy remote mode being removed mid-2026 (Atlassian's Rovo MCP server deprecates it on 2026-06-30; Keboola on 2026-04-01). For stdio, one JSON object per line is the entire wire format.

The notes server is a good shape because it exercises all three server primitives. Tools mutate (`notes_create`). Resources expose data (`notes://{id}`). Prompts deliver templates (`review_note`). This lesson's shape generalizes to any domain.

## The Concept

### Dispatch loop

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

Three rules:

- Never print anything to stdout that isn't a JSON-RPC envelope. Debug logs go to stderr.
- Every request must be paired with a response carrying the same `id`.
- Notifications must never be responded to.

### Implementing `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

Only declare what you support. The client uses this capability set to gate features.

### Implementing `tools/list` and `tools/call`

`tools/list` returns `{tools: [...]}`, each entry with `name`, `description`, `inputSchema`. `tools/call` takes `{name, arguments}`, returns `{content: [blocks], isError: bool}`.

Content blocks are typed. Most common:

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

Tool errors have two shapes. Protocol-level errors (unknown method, bad params) are JSON-RPC errors. Tool-level errors (call was valid but the tool failed) return as `{content: [...], isError: true}`. This lets the model see the failure in its context.

### Implementing resources

Resources are read-only by design. `resources/list` returns a listing; `resources/read` returns content. URIs can be `file://...`, `http://...`, or custom schemes like `notes://`.

When you expose data as a resource rather than a tool:

- The model doesn't "call" it; the client can inject it into context at user request.
- Subscriptions let the server push updates when the resource changes (Phase 13 · 10).
- Phase 13 · 14 extends it to interactive resources with `ui://`.

### Implementing prompts

Prompts are templates with named parameters. The host surfaces them as slash commands. A `review_note` prompt might take a `note_id` parameter and produce a multi-message prompt template that the client feeds to its model.

### Stdio transport subtleties

- Newline-delimited JSON. No length-prefixed framing.
- Don't buffer. `sys.stdout.flush()` after every write.
- Client controls lifecycle. When stdin closes (EOF), exit cleanly.
- Don't silently handle SIGPIPE; log and exit.

### Annotations

Each tool can carry `annotations` that describe safety properties:

- `readOnlyHint: true` — pure read, safe to retry.
- `destructiveHint: true` — irreversible side effect; client should confirm.
- `idempotentHint: true` — same input produces same output.
- `openWorldHint: true` — interacts with external systems.

Clients use these to decide UX (confirmation dialogs, status indicators) and routing (Phase 13 · 17).

### Graduation path

The standard-library server in `code/main.py` is ~180 lines. FastMCP (Python) collapses the same logic to decorator style:

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

The TypeScript SDK has an equivalent shape. The graduation path is a drop-in replacement when ready; the concepts (capabilities, dispatch, content blocks) are the same.

## Use It

`code/main.py` is a complete, stdio-based, standard-library-only notes MCP server. It handles `initialize`, `tools/list` and `tools/call` for three tools (`notes_list`, `notes_search`, `notes_create`), `resources/list` and `resources/read` for each note, and one `review_note` prompt. You can drive it by piping JSON-RPC messages:

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

What to look for:

- The dispatcher is a `dict[str, Callable]` keyed by method name.
- Each tool executor returns a content block list, not a bare string.
- `isError: true` is set when an executor throws.

## Ship It

This lesson produces `outputs/skill-mcp-server-scaffolder.md`. Given a domain (notes, tickets, files, database), this skill scaffolds an MCP server with the correct tools / resources / prompts split and SDK graduation path.

## Exercises

1. Run `code/main.py` and drive it with hand-crafted JSON-RPC messages. Exercise `notes_create`, then `resources/read` to retrieve the new note.

2. Add a `notes_delete` tool with `annotations: {destructiveHint: true}`. Verify the client surfaces a confirmation dialog (requires a real host; Claude Desktop works).

3. Implement `resources/subscribe` so the server pushes `notifications/resources/updated` when any note is modified. Add a keepalive task.

4. Port the server to FastMCP. The Python file should shrink to under 80 lines. Wire behavior must be identical; verify with the same JSON-RPC test scaffolding.

5. Read the spec's `server/tools` section and find a tool-definition field the lesson server doesn't implement. (Hint: there are several; pick one and add it.)

## Key Terms

| Term | Common shorthand | What it actually is |
|------|----------------|------------------------|
| MCP server | "thing that exposes tools" | A process speaking MCP JSON-RPC on stdio or HTTP |
| stdio transport | "subprocess model" | Server spawned by client; communicates via stdin/stdout |
| Dispatcher | "method router" | Map of JSON-RPC method names to handler functions |
| Content block | "tool result block" | Typed element in a tool response's `content` array |
| `isError` | "tool-level failure" | Flag indicating the tool failed; distinct from JSON-RPC errors |
| Annotations | "safety hints" | readOnly / destructive / idempotent / openWorld flags |
| FastMCP | "Python SDK" | Higher-level decorator-based framework atop MCP protocol |
| Resource URI | "addressable data" | `file://`, `db://`, or custom scheme identifying a resource |
| Prompt template | "slash-command brief" | Server-provided template with parameter slots for the host UI |
| Capability declaration | "feature switches" | Per-primitive flags declared in `initialize` |

## Further Reading

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — Reference Python implementation
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — Parallel TS implementation
- [FastMCP — server framework](https://gofastmcp.com/) — Decorator-style Python API for MCP servers
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) — End-to-end tutorial with either SDK
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — Full reference for tools/* messages
