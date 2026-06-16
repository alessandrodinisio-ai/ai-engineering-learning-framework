# MCP Apps — Interactive UI Resources via `ui://`

> Plain-text tool output caps what an agent can show. MCP Apps (SEP-1724, official 2026-01-26) lets a tool return sandboxed interactive HTML rendered inline in Claude Desktop, ChatGPT, Cursor, Goose, and VS Code. Dashboards, forms, maps, 3D scenes — all via one extension. This lesson walks through the `ui://` resource scheme, the `text/html;profile=mcp-app` MIME, the iframe sandbox's postMessage protocol, and the security surface that comes with letting a server render HTML.

**Type:** Build
**Languages:** Python (stdlib, UI resource emitter), HTML (example app)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources)
**Time:** ~75 minutes

## Learning Objectives

- Return a `ui://` resource from a tool call with the correct MIME and metadata.
- Declare a tool-associated UI with `_meta.ui.resourceUri`, `_meta.ui.csp`, and `_meta.ui.permissions`.
- Implement iframe-sandboxed postMessage JSON-RPC for UI-to-host communication.
- Apply CSP and permissions-policy defaults to defend against UI-initiated attacks.

## The Problem

A 2025-era `visualize_timeline` tool can return "here are the 14 notes organized chronologically: …". That's a paragraph. What the user actually wants is the interactive timeline. Before MCP Apps, options were: client-specific widget APIs (Claude artifacts, OpenAI Custom GPT HTML), or no UI at all.

MCP Apps (SEP-1724, released 2026-01-26) standardizes the contract. A tool result contains a `resource` whose URI is `ui://...` and whose MIME is `text/html;profile=mcp-app`. The host renders it in a sandboxed iframe with a restricted CSP and no network access unless explicitly granted. The UI inside the iframe communicates with the host via a tiny postMessage JSON-RPC dialect.

Every compatible client (Claude Desktop, ChatGPT, Goose, VS Code) renders the same `ui://` resource the same way. One server, one HTML bundle, universal UI.

## The Concept

### The `ui://` resource scheme

A tool returns:

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

The host then calls `resources/read` on the `ui://notes/timeline` URI and gets back:

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe sandbox

The host renders the HTML in a sandboxed `<iframe>` with:

- `sandbox="allow-scripts allow-same-origin"` (or stricter per server declaration)
- The server-declared CSP applied via response headers.
- No cookies, no localStorage from the host origin.
- Network access limited to `connectSrc` in the CSP.

### postMessage protocol

The iframe communicates with the host via `window.postMessage`. A tiny JSON-RPC 2.0 dialect:

Always pin `targetOrigin` to the exact origin of the peer, and validate `event.origin` against a whitelist on the receiving side before processing any payload. Never use `"*"` on either side of this channel — the body carries tool calls and resource reads.

```js
// iframe to host  (pinned to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pinned to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// Receiving side on both ends
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

Host-side methods the UI can call:

- `host.callTool(name, arguments)` — invoke a server tool.
- `host.readResource(uri)` — read an MCP resource.
- `host.getPrompt(name, arguments)` — fetch a prompt template.
- `host.close()` — close this UI.

Each call still goes through the MCP protocol and inherits the server's permissions.

### Permissions

The `_meta.ui.permissions` list requests additional capabilities:

- `camera` — access the user's camera (for document-scanning UIs).
- `microphone` — voice input.
- `geolocation` — location.
- `network:*` — broader network access than `connectSrc` alone.

Each permission is a prompt the user sees before the UI renders.

### Security risks

HTML inside an iframe is still HTML. New attack surfaces:

- **Prompt injection via UI.** A malicious server UI can display text that looks like system messages to fool the user. Host rendering should visually distinguish server UI from host UI.
- **Exfiltration via `connectSrc`.** If the CSP allows `connect-src: *`, the UI can send data anywhere. Defaults should be strict.
- **Clickjacking.** UI overlays on host chrome. The host must prevent z-index manipulation and enforce opacity rules.
- **Focus stealing.** UI grabs keyboard focus and captures the next message. The host must intercept.

Phase 13 · 15 covers these in depth as part of MCP security; this lesson only introduces them.

### The `ui/initialize` handshake

After the iframe loads, it sends `ui/initialize` via postMessage:

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

The host responds with capabilities and a session token. The UI uses that session token on every subsequent host call.

### AppRenderer / AppFrame SDK primitives

The ext-apps SDK exposes two convenience primitives:

- `AppRenderer` (server side) — wraps a React / Vue / Solid component and emits a `ui://` resource with correct MIME and metadata.
- `AppFrame` (client side) — receives the resource, mounts the iframe, and mediates postMessage.

You can use these or hand-roll HTML and JSON-RPC.

### Ecosystem status

MCP Apps shipped 2026-01-26. Client support as of 2026 April:

- **Claude Desktop.** Full support since 2026 January.
- **ChatGPT.** Full support via Apps SDK (same underlying MCP Apps protocol).
- **Cursor.** Beta; enabled via settings.
- **VS Code.** Insider builds only.
- **Goose.** Full support.
- **Zed, Windsurf.** Roadmapped.

Servers in production: dashboards, map visualizations, data tables, chart builders, sandboxed IDE previews.

## Use It

`code/main.py` extends the notes server with a `visualize_timeline` tool that returns a `ui://notes/timeline` resource, plus a `resources/read` handler for that URI that returns a small complete HTML bundle with an SVG timeline. The HTML is templated with stdlib — no build system. The postMessage is sketched in JS comments since stdlib can't drive a browser.

What to look for:

- `_meta.ui` on the tool response carries resourceUri, CSP, permissions.
- The HTML renders without network access; all data is inlined.
- JS calls `window.parent.postMessage` to invoke `host.callTool` (documented but inert in this stdlib demo).

## Ship It

This lesson produces `outputs/skill-mcp-apps-spec.md`. Given a tool that would benefit from interactive UI, this skill produces the complete MCP Apps contract: `ui://` URI, CSP, permissions, postMessage entry points, and a security checklist.

## Exercises

1. Run `code/main.py` and inspect the emitted HTML. Open the HTML directly in a browser; verify the SVG renders. Then sketch the postMessage contract the UI would use to call `host.callTool("notes_update", ...)`.

2. Tighten the CSP: remove `'unsafe-inline'` and use a nonce-based script policy. What changes in the HTML generation code?

3. Add a second UI resource `ui://notes/editor` with a form that edits a note in place. On user submit, the iframe calls `host.callTool("notes_update", ...)`.

4. Audit the attack surface of this UI. Where could a malicious server inject content? What does the iframe sandbox prevent vs. not prevent?

5. Read the SEP-1724 spec and find one capability in the MCP Apps SDK that this toy implementation doesn't use. (Hint: component-level state sync.)

## Key Terms

| Term | What people call it | What it actually is |
|------|---------------------|---------------------|
| MCP Apps | "interactive UI resources" | SEP-1724 extension released 2026-01-26 |
| `ui://` | "App URI scheme" | Resource scheme for UI bundles |
| `text/html;profile=mcp-app` | "the MIME" | Content-type for MCP App HTML |
| Iframe sandbox | "render container" | Browser sandboxing for the UI with CSP and permissions |
| postMessage JSON-RPC | "UI-to-host wire" | Tiny JSON-RPC-over-postMessage dialect for host calls |
| `_meta.ui` | "tool-UI binding" | Metadata linking a tool result to a UI resource |
| CSP | "Content-Security-Policy" | Declares allowed sources for scripts, network, styles |
| AppRenderer | "server SDK primitive" | Turns a framework component into a `ui://` resource |
| AppFrame | "client SDK primitive" | Iframe mounting helper that mediates postMessage |
| `ui/initialize` | "handshake" | First postMessage from UI to host |

## Further Reading

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — reference implementation and SDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — formal spec document
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) — high-level documentation
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026 January launch blog post
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc-style SDK reference
