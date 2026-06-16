# Roots and Elicitation — Scope Limiting and Mid-Call User Input

> Hardcoded paths break the moment a user opens another project. Pre-filled tool arguments break when the user isn't specific enough. Roots limit a server's scope to a set of user-controlled URIs; elicitation pauses mid-tool-call to ask the user for structured input via a form or URL. Two client primitives, two fixes for common MCP failure modes. SEP-1036 (URL-mode elicitation, 2025-11-25) is still experimental in 2026 H1 — check SDK version before relying on it.

**Type:** Build
**Languages:** Python (stdlib, roots + elicitation demo)
**Prerequisites:** Phase 13 · 07 (MCP server)
**Time:** ~45 minutes

## Learning Objectives

- Declare `roots` and respond to `notifications/roots/list_changed`.
- Restrict server file operations to URIs within the declared root set.
- Use `elicitation/create` to ask the user for a confirmation or structured input mid-tool-call.
- Choose between form-mode and URL-mode elicitation (the latter is experimental; drift risk noted).

## The Problem

Two concrete failures a notes MCP server hits in production.

**Broken path assumptions.** The server was written against `~/notes`. A user on a different machine whose notes live at `~/Documents/Notes` gets a silently-failing tool call (file not found) or, worse, writes to the wrong place.

**Missing parameters the user would have known.** The user says "delete that old TPS report note." The model calls `notes_delete(title: "TPS report")`, but there are three matching notes from 2023, 2024, and 2025. The tool can't guess. Failing with "ambiguous" is annoying; running on all three is catastrophic.

Roots fix the first: the client declares the URI set the server can reach at `initialize`. Elicitation fixes the second: the server pauses the tool call and sends `elicitation/create` for the user to pick one.

## The Concept

### Roots

The client declares a root list at `initialize`:

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

The server can then call `roots/list`:

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

The server must treat roots as boundaries: any file read/write outside the root set is rejected. This isn't client-enforced (the server is still user-trusted code), but a spec-compliant server honors it.

When the user adds or removes a root, the client sends `notifications/roots/list_changed`. The server re-calls `roots/list` and updates its boundaries.

### Why roots are a client primitive

Roots are declared by the client because they represent the user's consent model. The user tells Claude Desktop "give this notes server access to these two directories." The server cannot widen that scope.

### Elicitation: form mode default

`elicitation/create` takes a form schema plus a natural-language prompt:

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

The client renders a form, collects the user's answer, and returns:

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

Three possible actions: `accept` (user filled it), `decline` (user dismissed it), `cancel` (user aborted the entire tool call).

The form schema is flat — v1 does not support nested objects. SDKs typically reject anything more complex than a single level.

### Elicitation: URL mode (SEP-1036, experimental)

Added 2025-11-25. Instead of a schema, the server sends a URL:

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

The client opens the URL in a browser, waits for completion, and returns when the user comes back. Useful for OAuth flows, payment authorization, and document signing where a form isn't enough.

Drift warning: SEP-1036's response shape is still settling; some SDKs return a callback URL, others return a completion token. Read your SDK's release notes before using URL mode in production.

### When elicitation is the right tool

- User confirmation before destructive actions (destructive hint + elicitation).
- Disambiguation (pick one from N matches).
- First-run setup (API key, directory, preferences).
- OAuth-style flows (URL mode).

### When elicitation is wrong

- Filling a required parameter the tool could have asked for in prose. Use a normal re-prompt, not an elicitation dialog.
- High-frequency calls. Elicitation interrupts the conversation; don't trigger it in a loop.
- Anything the server can validate after the fact. Validate, return an error, let the model ask the user in text.

### The human-in-the-loop bridge

Elicitation plus sampling together enable MCP's "human-in-the-loop" model. A server's agent loop can pause for user input (elicitation) or model reasoning (sampling). Phase 13 · 11 covered sampling; this lesson covers elicitation. Combine them for complete mid-loop control.

## Use It

`code/main.py` extends the notes server with:

- A `roots/list` response that the server re-queries after a root-list-changed notification.
- A `notes_delete` tool that uses `elicitation/create` to disambiguate when multiple notes match.
- A `notes_setup` tool that uses URL-mode elicitation to open a first-run config page (simulated).
- A boundary check that rejects operations on URIs outside declared roots.

The demo runs three scenarios: happy path (single match), disambiguation (three matches, elicitation fires), out-of-root write (rejected).

## Ship It

This lesson produces `outputs/skill-elicitation-form-designer.md`. Given a tool that may need user confirmation or disambiguation, this skill designs the elicitation form schema and message template.

## Exercises

1. Run `code/main.py`. Trigger the disambiguation path; confirm the simulated user response is routed back to the tool.

2. Add a new tool `notes_archive` that always requires elicitation confirmation (destructive hint). Observe the UX: how does this compare to the model re-asking in text?

3. Implement URL-mode elicitation for a first-run OAuth flow. Note drift risk and add an SDK version guard.

4. Extend `roots/list` handling: when the notification arrives, the server should atomically re-read and rescan open file handles that may now be out of scope.

5. Read the SEP-1036 issue discussion thread on GitHub. Find one open question that affects how the server should handle URL-mode callbacks.

## Key Terms

| Term | What people call it | What it actually is |
|------|---------------------|---------------------|
| Root | "consent boundary" | URI the client permits the server to reach |
| `roots/list` | "server asks for scope" | Client returns the current root set |
| `notifications/roots/list_changed` | "user changed scope" | Client signals the root set changed |
| Elicitation | "ask user mid-call" | Server-initiated structured user input request |
| `elicitation/create` | "the method" | JSON-RPC method for an elicitation request |
| Form mode | "schema-driven form" | Flat JSON Schema rendered as a form in client UI |
| URL mode | "browser redirect" | SEP-1036 experimental; opens a URL and waits |
| `accept` / `decline` / `cancel` | "user response outcomes" | Three branches the server handles |
| Disambiguation | "pick one" | Common elicitation use case when the tool has N candidates |
| Flat form | "top-level properties only" | Elicitation schema cannot nest |

## Further Reading

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — authoritative roots reference
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) — authoritative elicitation reference
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — step-by-step walkthrough of 2025-11-25 additions
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL-mode elicitation proposal (experimental, drift risk)
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UX walkthrough
