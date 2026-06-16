# MCP Resources and Prompts — Context Exposure Beyond Tools

> Tools took 90% of MCP's attention. The other two server primitives solve different problems. Resources expose data for reading; prompts expose reusable templates as slash-commands. Many servers should use resources instead of wrapping reads into tools, and should use prompts instead of hardcoding workflows in client prompts. This lesson names that decision rule and walks through the `resources/*` and `prompts/*` messages.

**Type:** Build
**Languages:** Python (stdlib, resource + prompt handlers)
**Prerequisites:** Phase 13 · 07 (MCP server)
**Time:** ~45 minutes

## Learning Objectives

- Decide whether to expose a capability as a tool, resource, or prompt for a given domain.
- Implement `resources/list`, `resources/read`, `resources/subscribe`, and handle `notifications/resources/updated`.
- Implement `prompts/list` and `prompts/get` with parameterized templates.
- Recognize when the host renders prompts as slash-commands vs. auto-injected context.

## The Problem

A naive MCP server for a notes app exposes everything as tools: `notes_read`, `notes_list`, `notes_search`. This wraps every data access in a model-driven tool call. Consequences:

- For every query that might benefit from context, the model must decide whether to call `notes_read`.
- Read-only content cannot be subscribed to or streamed into the host's side panel.
- Client UI (Claude Desktop's resource attach panel, Cursor's "Include file" picker) cannot surface this data.

The correct split: expose data as a resource, expose mutations or computational actions as tools, expose reusable multi-step workflows as prompts. Each primitive has its UX affordance and its access pattern.

## The Concept

### tools vs resources vs prompts — the decision rule

| Capability | Primitive |
|------------|-----------|
| User wants to search, filter, or transform data | tool |
| User wants the host to carry this data as context | resource |
| User wants a replayable templated workflow | prompt |

Guideline: if the model would benefit from calling it on every relevant query, it's a tool. If the user would benefit from attaching it to a conversation, it's a resource. If the reusable unit is an entire multi-step workflow, it's a prompt.

### Resources

`resources/list` returns `{resources: [{uri, name, mimeType, description?}]}`. `resources/read` takes `{uri}`, returns `{contents: [{uri, mimeType, text | blob}]}`.

URIs can be anything addressable:

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14` (custom scheme)
- `memory://session-2026-04-22/recent` (server-specific)

`contents[]` supports both text and binary. Binary uses `blob` as a base64-encoded string plus a `mimeType`.

### Resource subscriptions

Declare `{resources: {subscribe: true}}` in capabilities. Client calls `resources/subscribe {uri}`. Server sends `notifications/resources/updated {uri}` when the resource changes. Client re-reads.

Use case: a notes server where a resource is a file on disk; a file watcher triggers an update notification; Claude Desktop re-pulls it into context when the file is edited outside the host.

### Resource templates (added 2025-11-25)

`resourceTemplates` let you expose a parameterized URI pattern: `notes://{id}`, with `id` as the completion target. The client can auto-complete id in the resource picker.

### Prompts

`prompts/list` returns `{prompts: [{name, description, arguments?}]}`. `prompts/get` takes `{name, arguments}`, returns `{description, messages: [{role, content}]}`.

A prompt is a template that fills into a message list which the host feeds to its model. For example, a `code_review` prompt takes a `file_path` argument and returns a three-message sequence: a system message, a user message with the file body, and an assistant preamble with a reasoning template.

### Hosts and prompts

Claude Desktop, VS Code, and Cursor render prompts as slash-commands in the chat UI. The user types `/code_review` and picks arguments from a form. The server's prompt is the contract between "user shortcut" and "full prompt sent to the model."

Not every client supports prompts yet — check capability negotiation. A server that declares prompt capability, paired with a client that doesn't support prompts, simply won't see those slash commands.

### "list changed" notification

Both resources and prompts send `notifications/list_changed` when the collection changes. A notes server that just imported 20 new notes sends `notifications/resources/list_changed`; the client re-calls `resources/list` to pick up the additions.

### Content type conventions

Text: `mimeType: "text/plain"`, `text/markdown`, `application/json`.
Binary: `image/png`, `application/pdf`, with `blob` field.
MCP Apps (Lesson 14): `text/html;profile=mcp-app` inside a `ui://` URI.

### Dynamic resources

A resource URI need not correspond to a static file. `notes://recent` can return the five most recent notes on every read. `db://query/users/active` can execute a parameterized query. Servers are free to compute content dynamically.

Rule: if the client can cache by URI, the URI must be stable. If the computation is one-shot, the URI should contain a timestamp or nonce to avoid stale caching.

### Subscription vs polling

Clients that support subscriptions get server pushes via `notifications/resources/updated`. Pre-subscription clients, or hosts that don't support it, poll by re-reading. Both are spec-compliant. The server's capability declaration tells the client which it supports.

Subscription cost: per-session state on the server (who subscribed to what). Keep subscription sets bounded; disconnected clients should time out.

### prompts vs system prompts

Prompts in MCP are not system prompts. The host's system prompt (its own operational instructions) and MCP prompts (server-provided, user-triggered templates) coexist side by side. A well-behaved client never lets a server prompt override its own system prompt; it layers them.

## Use It

`code/main.py` extends the Lesson 07 notes server with:

- Per-note resources (`notes://note-1` etc.) with `resources/subscribe` support.
- A `review_note` prompt that renders into a three-message template.
- A file-watcher simulation that sends `notifications/resources/updated` when a note is modified.
- A `notes://recent` dynamic resource that always returns the five most recent notes.

Run the demo to see the full flow.

## Ship It

This lesson produces `outputs/skill-primitive-splitter.md`. Given a proposed MCP server, this skill classifies each capability as tool / resource / prompt with a rationale.

## Exercises

1. Run `code/main.py`. Observe the initial resource list, then trigger a note edit and verify the `notifications/resources/updated` event fires.

2. Add a `resources/list_changed` emitter: when a new note is created, send that notification so the client rediscovers.

3. Design three prompts for a GitHub MCP server: `summarize_pr`, `triage_issue`, `release_notes`. Each with an argument schema. The prompt body should run without further editing.

4. Take an existing tool from the Lesson 07 server and classify whether it should stay a tool or be split into a resource + tool pair. Justify in one sentence.

5. Read the spec's `server/resources` and `server/prompts` sections. Find one rarely-populated but spec-supported field in `resources/read`. Hint: look at `_meta` on resource contents.

## Key Terms

| Term | What people call it | What it actually is |
|------|---------------------|---------------------|
| Resource | "exposed data" | URI-addressable content the host can read |
| Resource URI | "pointer to data" | Scheme-prefixed identifier (`file://`, `notes://`, etc.) |
| `resources/subscribe` | "watch for changes" | Client opt-in for server-pushed updates on a URI |
| `notifications/resources/updated` | "resource changed" | Signal telling the client a subscribed resource has new content |
| Resource template | "parameterized URI" | URI pattern with completion hints for the host picker |
| Prompt | "slash-command template" | Named multi-message template with argument slots |
| Prompt arguments | "template inputs" | Typed parameters the host collects before rendering |
| `prompts/get` | "render the template" | Server returns the filled message list |
| Content block | "typed block" | `{type: text | image | resource | ui_resource}` |
| Slash-command UX | "user shortcut" | Host rendering prompts as `/`-prefixed commands |

## Further Reading

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) — resource URIs, subscriptions, and templates
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) — prompt templates and slash-command integration
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — full `resources/*` message reference
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — full `prompts/*` message reference
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) — community guide expanding on the official docs
