# MCP Security II — OAuth 2.1, Resource Indicators, Incremental Scopes

> Remote MCP servers need authorization, not just authentication. The 2025-11-25 spec aligns with OAuth 2.1 + PKCE + resource indicators (RFC 8707) + protected resource metadata (RFC 9728). SEP-835 adds incremental scope consent, doing step-up authorization on a 403 WWW-Authenticate. This lesson implements the step-up flow as a state machine so you can see every hop.

**Type:** Build
**Languages:** Python (stdlib, OAuth state-machine simulator)
**Prerequisites:** Phase 13 · 09 (Transports), Phase 13 · 15 (Security I)
**Time:** ~75 minutes

## Learning Objectives

- Distinguish the responsibilities of resource server vs authorization server.
- Walk through a PKCE-protected OAuth 2.1 authorization code flow.
- Use `resource` (RFC 8707) and protected resource metadata (RFC 9728) to prevent confused-deputy attacks.
- Implement step-up authorization: server responds 403 with WWW-Authenticate requesting a higher scope; client re-prompts for user consent and retries.

## The Problem

Early MCP (pre-2025) shipped remote servers with ad-hoc API keys or no auth at all. The 2025-11-25 spec closes that gap with a full OAuth 2.1 profile.

Three real needs:

- **Typical remote server.** A user installs a remote MCP server that accesses their Notion / GitHub / Gmail. OAuth 2.1 with PKCE is the right shape.
- **Scope escalation.** A notes server granted `notes:read` may later need `notes:write` for a specific action. Step-up (SEP-835) requests that extra scope without redoing the whole flow.
- **Confused deputy prevention.** A client holds a token audience-scoped to Server A. Server A is malicious and tries to present that token to Server B. Resource indicators (RFC 8707) pin the token to its intended audience.

OAuth 2.1 isn't new. What's new is MCP's profile: specific required flows (authorization code + PKCE only; no implicit, no client credentials by default), mandatory resource indicators on every token request, and published protected resource metadata so clients know where to go.

## The Concept

### Roles

- **Client.** The MCP client (Claude Desktop, Cursor, etc.).
- **Resource server.** The MCP server (notes, GitHub, Postgres, whatever).
- **Authorization server.** Issues tokens. Can be the same service as the resource server or a separate IdP (Auth0, Keycloak, Cognito).

In MCP's profile, resource and authorization servers can be the same host but should be URL-distinct.

### Authorization Code + PKCE

Flow:

1. Client generates `code_verifier` (random) and `code_challenge` (SHA256).
2. Client redirects user to `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`.
3. User consents. Authorization server redirects to `redirect_uri?code=...`.
4. Client POSTs to `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`.
5. Authorization server verifies the verifier's hash matches the stored challenge, issues an access token.
6. Client uses the token: `Authorization: Bearer ...` on every request to the resource server.

PKCE prevents authorization code interception attacks. Resource indicators prevent the token from being valid elsewhere.

### Protected Resource Metadata (RFC 9728)

The resource server publishes a `.well-known/oauth-protected-resource` document:

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

The client discovers the authorization server from the resource server. Reduces configuration — the client only needs the resource URL.

### Resource Indicators (RFC 8707)

The `resource` parameter on the token request pins the token's intended audience. The issued token contains `aud: "https://notes.example.com"`. Another MCP server receiving this token checks `aud` and rejects it.

### Scope Model

Scopes are space-separated strings. Common MCP conventions:

- `notes:read`, `notes:write`, `notes:delete`
- `admin:*` for administrative capabilities (use sparingly)
- `profile:read` for identity

Scope selection should be least-privilege: request what's needed now, step up when more is needed.

### Step-Up Authorization (SEP-835)

The user granted `notes:read`. They later ask the agent to delete a note. Server responds:

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

The client sees the insufficient_scope error, prompts the user with a consent dialog for that extra scope, runs a mini OAuth flow for it, and retries the request with the new token.

### Token Audience Validation

On every request: server checks `token.aud == self.resource_url`. Mismatch = 401. This prevents cross-server token reuse.

### Short-Lived Tokens & Rotation

Access tokens should be short-lived (1 hour default). Refresh tokens rotate on each refresh. The client handles silent refresh in the background.

### No Token Pass-Through

A sampling server (Phase 13 · 11) must never pass the client's token through to other services. The sampling request is the boundary.

### Confused Deputy Prevention

Token binds to `aud`. Client binds to `client_id`. Every request validates both. The spec explicitly forbids the legacy "pass the token" pattern — common in pre-MCP remote tool ecosystems.

### Client ID Discovery

Each MCP client publishes its metadata at a fixed URL. The authorization server can fetch the client's metadata document to discover redirect URIs and contact info. This eliminates manual client registration.

### Gateways & OAuth

Phase 13 · 17 shows how an enterprise gateway handles OAuth: the gateway holds upstream server credentials, the token issued to the client is gateway-issued, upstream tokens never leave the gateway. This inverts the trust model — users authenticate once with the gateway; the gateway handles N server authorizations.

## Use It

`code/main.py` simulates the full OAuth 2.1 step-up flow as a state machine. It implements:

- PKCE code-verifier / challenge generation.
- Authorization code flow with resource indicators.
- Protected resource metadata endpoint.
- Token validation with audience checks.
- Step-up on `insufficient_scope`.

No HTTP server in this lesson; the state machine runs in-memory so you can trace every hop. Phase 13 · 17's gateway lesson wires it to a real transport.

## Ship It

This lesson produces `outputs/skill-oauth-scope-planner.md`. Given a remote MCP server with tools, this skill designs the scope set, pinning rules, and step-up strategy.

## Exercises

1. Run `code/main.py`. Trace the two-scope step-up flow. Note which hops repeat during step-up.

2. Add refresh-token rotation: issue a new refresh token on each refresh and revoke the old one. Simulate a stolen refresh token being used after rotation, confirm it fails.

3. Implement the protected resource metadata endpoint as a real HTTP response using stdlib http.server. Mirror lesson 09's /mcp endpoint.

4. Design a scope hierarchy for a GitHub MCP server: read repo, write PR, approve PR, merge PR, admin. Step-up between each level.

5. Read RFC 8707 and RFC 9728. Find one field in 9728 where MCP's usage differs from the RFC's example. (Hint: it's about `scopes_supported`.)

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| OAuth 2.1 | "modern OAuth" | Consolidated RFC mandating PKCE, banning implicit flow |
| PKCE | "proof of possession" | Code verifier + challenge that defeats authorization code interception |
| Resource indicator | "token audience" | RFC 8707 `resource` parameter pinning a token to one server |
| Protected-resource metadata | "discovery document" | RFC 9728's `.well-known/oauth-protected-resource` |
| Step-up authorization | "incremental consent" | SEP-835 flow that adds scopes on demand |
| `insufficient_scope` | "403 with WWW-Authenticate" | Server signal to re-consent for a bigger scope |
| Confused deputy | "cross-service token reuse" | Attack where a trusted holder improperly forwards a token |
| Short-lived token | "access token TTL" | Bearer that expires quickly; refresh token renews it |
| Scope hierarchy | "least-privilege stack" | Graduated scope sets with step-up between levels |
| Client ID metadata | "client discovery document" | URL where a client publishes its own OAuth metadata |

## Further Reading

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — The authoritative MCP OAuth profile
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) — Step-by-step walkthrough of the 2025-11-25 changes
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — The audience-pinning RFC
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — The discovery document RFC
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — Practical step-up flow walkthrough
