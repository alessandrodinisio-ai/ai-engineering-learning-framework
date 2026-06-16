# MCP Transports — stdio vs Streamable HTTP vs SSE Migration

> stdio works locally, nowhere else. Streamable HTTP (2025-03-26) is the remote standard. The legacy HTTP+SSE transport is deprecated and being removed mid-2026. Picking the wrong transport costs a migration; picking right buys you a remotely hostable MCP server with session continuity and DNS-rebinding protection.

**Type:** Learn
**Languages:** Python (standard library, Streamable HTTP endpoint skeleton)
**Prerequisites:** Phase 13 · 07, 08 (MCP server and client)
**Time:** ~45 minutes

## Learning Objectives

- Choose between stdio and Streamable HTTP based on deployment shape (local vs. remote, single-process vs. cluster).
- Implement Streamable HTTP's single-endpoint pattern: POST for requests, GET to open a session stream.
- Enforce `Origin` validation and session-id semantics to defeat DNS-rebinding.
- Migrate a legacy HTTP+SSE server to Streamable HTTP before the mid-2026 removal deadline.

## The Problem

The first MCP remote transport (2024-11) was HTTP+SSE: two endpoints, one receiving client POSTs, one Server-Sent-Events channel for server-to-client streaming. It worked. It was also clunky: two endpoints per session, caching issues in front of some CDNs, and a hard dependency on long-lived SSE connections — which some WAFs aggressively terminate.

The 2025-03-26 spec replaced it with Streamable HTTP: one endpoint, POST for client requests, GET to establish a session stream, both sharing an `Mcp-Session-Id` header. Every server built or migrated since uses Streamable HTTP. The old SSE mode is being deprecated — Atlassian Rovo removes it June 30, 2026; Keboola on April 1, 2026; most remaining enterprise servers by end of 2026.

stdio remains important for local servers. Claude Desktop, VS Code, and every IDE-shaped client spawn servers over stdio. The correct mental model: stdio for "this machine," Streamable HTTP for "across a network." No crossover.

## The Concept

### stdio

- Subprocess transport. Client spawns server, communicates via stdin/stdout.
- One JSON object per line. Newline-delimited.
- No session id; process identity is the session.
- No auth needed (subprocess inherits parent's trust boundary).
- Never use for remote servers — you'd need SSH or socat tunneling, and at that point just use Streamable HTTP.

### Streamable HTTP

Single endpoint `/mcp` (or any path). Supports three HTTP methods:

- **POST /mcp.** Client sends a JSON-RPC message. Server responds with a single JSON response, or an SSE stream containing one or more responses (useful for batched responses and notifications related to that request).
- **GET /mcp.** Client opens a long-lived SSE channel. Server uses it to send server-to-client requests (sampling, notifications, elicitation).
- **DELETE /mcp.** Client explicitly terminates the session.

Sessions are identified by the `Mcp-Session-Id` header: server sets it on the first response, client echoes it on every subsequent request. Session ids must be cryptographically random (128+ bits); client-chosen ids are rejected for security.

### Single endpoint vs. two endpoints

The old spec's two-endpoint pattern is still callable in 2026 — the spec calls it "legacy compatible." But all new servers should be single-endpoint. Official SDKs emit single-endpoint; use legacy mode only when talking to an unmigrated remote.

### `Origin` validation and DNS-rebinding

Browsers (today) are not MCP clients, but an attacker can craft a web page that tricks the browser into POSTing to `localhost:1234/mcp` — where the user's local MCP server listens. If the server doesn't check `Origin`, the browser's same-origin policy won't save it because `Origin: http://evil.com` is a valid cross-origin. 

The 2025-11-25 spec requires servers to reject requests whose `Origin` is not on an allowlist. The allowlist typically contains MCP client hosts (`https://claude.ai`, `vscode-webview://*`) and localhost variants for local UIs.

### Session-id lifecycle

1. Client sends first request without `Mcp-Session-Id`.
2. Server allocates a random id, sets `Mcp-Session-Id` on the response header.
3. Client echoes that header on all subsequent requests and on the streaming `GET /mcp`.
4. Session can be revoked by the server; client sees 404 on next request and must re-initialize.
5. Client can explicitly DELETE the session for a clean shutdown.

### Keepalive and reconnection

SSE connections drop. Client re-establishes by re-GETting with the same `Mcp-Session-Id`. Server must queue events missed during the interruption (up to a reasonable window) and replay them via the `last-event-id` header echoed by the client.

Phase 13 · 13 covers Tasks, which let long-running work survive even a full session reconnect.

### Backwards-compatibility probing

A client wanting to support both old and new servers:

1. POST to `/mcp`.
2. If the response is a `200 OK` with JSON or SSE, this is Streamable HTTP.
3. If the response is a `200 OK` with `Content-Type: text/event-stream` and a `Location` header pointing to a secondary endpoint, this is legacy HTTP+SSE; follow the `Location`.

### Cloudflare, ngrok, and hosting

Production remote MCP servers in 2026 run on Cloudflare Workers (with their MCP Agents SDK), Vercel Functions, or containerized Node/Python. Key: your host must support long-lived HTTP connections to sustain the SSE GET. Vercel's free tier caps at 10 seconds — unsuitable. Cloudflare Workers support indefinite streaming.

### Gateway composition

When you front multiple MCP servers with a gateway (Phase 13 · 17), the gateway is a single Streamable HTTP endpoint that rewrites session ids and multiplexes to upstreams. Tools merge at the gateway layer; the client sees one logical server.

### Transport failure modes

- **stdio SIGPIPE.** A subprocess dying mid-write triggers SIGPIPE; the server should exit cleanly. Client should detect EOF and mark the session dead.
- **HTTP 502 / 504.** Cloudflare, nginx, and other proxies emit these on upstream failure. Streamable HTTP clients should retry once with short backoff.
- **SSE connection drop.** TCP RST, proxy timeout, or client network switch closes the stream. Client reconnects with `Mcp-Session-Id` and optional `last-event-id` to resume.
- **Session revocation.** Server invalidates a session id; client sees 404 on next request. Client must re-handshake.
- **Clock skew.** Resource-TTL calculations on the client diverge from the server. Client should treat server timestamps as authoritative.

### When to bypass Streamable HTTP

Some enterprises deploy MCP servers behind gRPC or message-queue transports within their own networks. This is non-standard — the MCP spec doesn't formally define these. A gateway can expose a Streamable HTTP surface to MCP clients while using gRPC internally. The external surface stays spec-compliant; the gateway owns the translation.

## Use It

`code/main.py` implements a minimal Streamable HTTP endpoint using `http.server` (standard library). It handles POST, GET, and DELETE on `/mcp`, sets `Mcp-Session-Id` on the first response, validates `Origin`, and rejects requests from non-allowlisted origins. The handlers reuse the dispatch logic from the Lesson 07 notes server.

What to look for:

- The POST handler reads the JSON-RPC body, dispatches, and writes a JSON response (single-response variant; SSE variant is structurally similar).
- The `Origin` check rejects a default `http://evil.example` probe but accepts `http://localhost`.
- Session id is a random 128-bit hex string; the server keeps per-session state in memory.

## Ship It

This lesson produces `outputs/skill-mcp-transport-migrator.md`. Given a legacy HTTP+SSE MCP server, this skill produces a migration plan to Streamable HTTP with session-id continuity, Origin checks, and backwards-compatibility probe support.

## Exercises

1. Run `code/main.py`. POST an `initialize` using `curl` and observe the `Mcp-Session-Id` response header. POST a second request echoing that header and verify session continuity.

2. Add a GET handler that opens an SSE stream. Send a `notifications/progress` event every five seconds. Reconnect by re-GETting with the same session id and confirm the server accepts it.

3. Implement `last-event-id` replay logic. On reconnect, replay any events produced after that id.

4. Extend `Origin` validation to support wildcard patterns (`https://*.example.com`) and confirm it accepts `https://app.example.com` but rejects `https://evil.example.com.attacker.net`.

5. Take a legacy HTTP+SSE server from the official registry (several exist) and sketch the migration: what changes in endpoint handling, session-id generation, and header semantics.

## Key Terms

| Term | Common shorthand | What it actually is |
|------|----------------|------------------------|
| stdio transport | "local subprocess" | JSON-RPC on stdin/stdout, newline-delimited |
| Streamable HTTP | "remote transport" | Single-endpoint POST + GET + optional SSE, 2025-03-26 spec |
| HTTP+SSE | "legacy" | Two-endpoint model being removed mid-2026 |
| `Mcp-Session-Id` | "session header" | Server-allocated random id echoed on every subsequent request |
| `Origin` allowlist | "DNS-rebinding defense" | Rejects requests whose Origin is not approved |
| Single endpoint | "one URL" | `/mcp` handles POST / GET / DELETE for all session operations |
| `last-event-id` | "SSE replay" | Header for resuming a dropped stream without missing events |
| Backwards-compat probe | "old-vs-new detection" | Client auto-selects transport by checking response shape |
| Long-lived HTTP | "SSE streaming" | Server pushes events over a single TCP connection for minutes or hours |
| Session revocation | "forced re-init" | Server invalidates a session id; client must handshake again |

## Further Reading

- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — Authoritative reference for stdio and Streamable HTTP
- [MCP — Basic transports spec 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — The revision that introduced Streamable HTTP
- [Cloudflare — MCP transport](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Streamable HTTP patterns for Workers hosting
- [AWS — MCP transport mechanisms](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — Comparison across deployment shapes
- [Atlassian — HTTP+SSE deprecation notice](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — Concrete migration deadline example
