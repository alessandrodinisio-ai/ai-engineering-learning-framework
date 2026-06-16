# Production MCP Auth — Registration, JWKS Refresh, Audience-Pinned Tokens

> Lesson 16 stood up the OAuth 2.1 state machine in memory. By 2026, every MCP server you ship to a real organization sits behind production auth: client registration supporting an unbounded growth of clients (Client ID Metadata Documents preferred, dynamic client registration as a backward-compatible fallback), authorization-server metadata discovery (RFC 8414 *or* OpenID Connect Discovery), JWKS cache refresh that won't break token validation at 3 AM, and audience-pinned tokens that reject cross-resource replay. This lesson models the entire surface with three roles — an authorization server, a resource server (i.e., MCP server), and a client — so you can trace every hop from discovery to a validated tool call.
>
> **Spec note (2025-11-25):** The November 2025 MCP authorization spec downgraded Dynamic Client Registration from `SHOULD` to `MAY` and established **Client ID Metadata Documents (CIMD)** as the recommended default registration mechanism. This lesson covers both, in the priority order specified by the spec; the code still retains DCR for demonstration since it is fully self-contained in a single process.

**Type:** Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 13 · 16 (OAuth 2.1 state machine), Phase 13 · 17 (gateways)
**Time:** ~90 minutes

## Learning Objectives

- Discover an authorization server via RFC 8414 metadata and verify the contract.
- Implement RFC 7591 dynamic client registration so MCP clients can register without admin intervention.
- Cache and refresh JWKS keys on schedule so signature validation survives key rotation.
- Pin tokens to a single MCP resource using RFC 8707 resource indicators, rejecting confused-deputy reuse.
- Cleanly separate three roles — authorization server, resource server, client — so each role performs only the checks that belong to it.
- Read an IdP capability matrix and refuse deployment when the IdP cannot satisfy the MCP auth profile.

## The Problem

Lesson 16's simulator runs OAuth 2.1 in memory. Production has three operational gaps that a pure in-memory simulator cannot see.

The first gap is registration. A real organization runs hundreds of MCP servers and thousands of MCP clients. Ops will not manually register every Cursor user as an OAuth client. The 2025-11-25 spec gives clients a priority order to solve this: use a pre-registered `client_id` if you have one, otherwise use a **Client ID Metadata Document** (the client identifies itself with an HTTPS URL it controls, and the authorization server *pulls* metadata), otherwise fall back to **RFC 7591 dynamic client registration** (the client *pushes* a `POST /register` and receives a `client_id` on the spot), and finally fall back to prompting the user. CIMD is the recommended default because it eliminates per-server registration while preserving a DNS-rooted trust model; DCR is retained for backward compatibility. Both discover their entry point from the authorization server's metadata: CIMD checks `client_id_metadata_document_supported`, DCR checks `registration_endpoint`.

The second gap is key rotation. JWT validation depends on the authorization server's signing keys, published as a JSON Web Key Set (JWKS). The authorization server rotates them on schedule (often hourly, sometimes faster during incident response). An MCP server that fetches JWKS once at startup validates correctly until the rotation window — then every request fails until restart. Production threads JWKS into a cached value with a refresh job that overwrites the cache before the previous keys expire, plus a cache-miss fallback fetch for when a token arrives signed by a key newer than the cache.

The third gap is audience binding. Lesson 16 introduced the RFC 8707 resource indicator. In production, that indicator becomes a hard claim check on every request. The MCP server compares `token.aud` against its own canonical resource URL and rejects with HTTP 401 on mismatch. This is the only defense that stops an upstream MCP server (or a malicious client holding a token intended for one server) from replaying that token against another server in the same trust mesh.

This lesson maps each of these gaps to a concrete piece of the surface. The metadata document is an HTTP endpoint. The JWKS cache refresh is a scheduled job plus a key-value cache. JWT validation is a routine the resource server runs before handing out any tool. Separating three roles means each role performs only the checks that belong to it: the authorization server issues and rotates keys, the resource server caches and validates, the client performs discovery and registration.

## The Concept

### RFC 8414 — OAuth Authorization Server Metadata

A document at `/.well-known/oauth-authorization-server` describes everything a client needs:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

A client given an MCP resource URL chains discovery: RFC 9728's `oauth-protected-resource` (the resource server's document) gives the issuer, then `oauth-authorization-server` (this RFC) gives every endpoint. The client never hard-codes authorization URLs.

The contract to verify before trusting an IdP to run MCP:

- `code_challenge_methods_supported` includes `S256` (i.e., RFC 7636 PKCE). The spec is explicit: if this field is **absent**, the authorization server does not support PKCE and the client **MUST** refuse to proceed.
- `grant_types_supported` includes `authorization_code` and rejects `password` and `implicit`.
- At least one registration path is advertised: `client_id_metadata_document_supported: true` (CIMD, preferred) **or** `registration_endpoint` (RFC 7591 DCR, fallback). Either satisfies the contract; you no longer hard-require DCR.
- For OAuth 2.1, `response_types_supported` is exactly `["code"]`.

If `S256` is absent, the MCP server refuses to deploy on that IdP — there is no degraded mode for PKCE. If *neither* registration path is advertised and you have no pre-registered `client_id`, you cannot register; at that point it is a deployment checklist error, not a code error.

### RFC 9728 (Review) — Protected Resource Metadata

Lesson 16 covered RFC 9728. The production difference is: this document is the only place a client looks to find which authorization servers *this* MCP server trusts. An MCP server may accept tokens from multiple IdPs (one for employees, one for partners). RFC 9728 declares that set; RFC 8414 documents what each IdP supports.

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Client ID Metadata Documents (Recommended Default)

CIMD inverts registration from *push* to *pull*. Instead of asking the authorization server to mint a `client_id`, the client uses an HTTPS URL it controls **as** its `client_id`. That URL resolves to a JSON metadata document; the authorization server fetches it on demand during the OAuth flow. Trust is DNS-rooted: if the server operator trusts `app.example.com`, it trusts the client served from `https://app.example.com/client.json`. No registration round-trip, no `client_id` namespace that can be exhausted, and no state to keep in sync per-server.

The client-hosted metadata document:

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

The `client_id` value in the document **MUST** equal the URL from which it is served (the authorization server verifies this; mismatch results in rejection). The authorization server advertises support in its RFC 8414 metadata with `client_id_metadata_document_supported: true`.

The spec is direct about two security facts:

- **SSRF.** The authorization server will fetch an attacker-supplied URL. It must defend against server-side request forgery (do not fetch internal/admin endpoints).
- **localhost impersonation.** CIMD alone cannot stop a local attacker from claiming a legitimate client's metadata URL and binding an arbitrary `localhost` redirect. The authorization server **MUST** clearly display the redirect URI's hostname during consent and **SHOULD** warn on redirects using only `localhost`.

Because CIMD requires no server-side state, there is no registrar to stand up like with DCR. The client side is read-only: serve your metadata document from a static HTTPS endpoint and let the authorization server fetch it.

### RFC 7591 — Dynamic Client Registration (Fallback / Backward Compatibility)

DCR is now a `MAY`, retained for backward compatibility with pre-2025-11-25 deployments and IdPs that don't yet support CIMD. Without it (and without CIMD or pre-registration), every MCP client (Cursor, Claude Desktop, a custom agent) requires an out-of-band exchange with the IdP admin. With DCR, the client posts:

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

The server responds with a `client_id` and a `registration_access_token` for subsequent updates:

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

For MCP clients running on user devices, `token_endpoint_auth_method: none` is the correct default. They receive only a `client_id` — no `client_secret` to steal. PKCE provides the proof-of-possession that public clients need.

Three production pitfalls:

- The registration endpoint must be rate-limited by source IP. Without this, an adversary can script millions of fake registrations and exhaust the `client_id` namespace. Run a rate-limit check before the registrar processes the request.
- Some enterprise IdPs require a `software_statement` (a signed JWT vouching for the client). This lesson's mock skips it; production must wire a validation step that rejects any unsigned registration with a non-localhost redirect URI.
- The `registration_access_token` must be stored as a hash, not in plaintext. A stolen token lets an attacker rewrite the client's redirect URIs.

### RFC 8707 (Review) — Resource Indicators

Lesson 16 established the shape. Production rule: every token request carries `resource=<canonical-mcp-url>`, and the MCP server validates on every call that `token.aud` matches its own resource URL. The canonical URI is the most specific identifier for this server: lowercase scheme and host, no fragment, and by convention no trailing slash. The path component is **not** stripped by rule — the spec preserves it when needed to identify a specific MCP server. `https://mcp.example.com`, `https://mcp.example.com/mcp`, `https://mcp.example.com:8443`, `https://mcp.example.com/server/mcp` are all valid canonical URIs. Each server picks one and pins `aud` exactly to that value. (This lesson's mock uses bare-host audiences like `https://notes.example.com` for brevity; a deployment co-locating multiple MCP servers under the same origin would use paths to distinguish them.)

### RFC 7636 (Review) — PKCE

PKCE is mandatory in OAuth 2.1. This lesson's authorization-code flow always carries `code_challenge` and `code_verifier`. The server rejects any token request that lacks a verifier or whose verifier doesn't hash to the stored challenge.

### MCP Spec 2025-11-25 Auth Profile

The MCP spec (2025-11-25) is precise about what an MCP server's authorization layer must do:

- Implement RFC 9728 protected-resource metadata and provide its location via the `WWW-Authenticate: Bearer resource_metadata="..."` header on 401 **or** the well-known URI `/.well-known/oauth-protected-resource` (SEP-985 makes the header optional with a well-known fallback). The metadata's `authorization_servers` field **MUST** list at least one server.
- Accept tokens only via `Authorization: Bearer ...` on **every** request — never in a query string, never validated only at session start.
- Validate `aud`, `iss`, `exp`, and required scopes on every request. The server **MUST** verify that the token was issued specifically for it (audience); missing or mismatched `aud` is always rejected, never treated as a wildcard.
- On 401/403, return `WWW-Authenticate: Bearer` with `error=...`, a `resource_metadata="<PRM-URL>"` parameter (the URL of the metadata document, *not* the bare resource), and `scope="..."` on `insufficient_scope` (403). Note: the parameter is `resource_metadata`, a discovery pointer — there is no `resource` parameter in the challenge.
- Authorization-server discovery accepts **both** RFC 8414 OAuth metadata **and** OpenID Connect Discovery 1.0; clients must try both well-known suffixes in priority order.
- Defense against **mix-up attacks** is the client's job (not the server's): it records the expected `issuer` before redirect and verifies the `iss` authorization response parameter (RFC 9207) before exchanging the code. PKCE alone cannot stop mix-up because the client will send its own `code_verifier` to whichever token endpoint it is directed to.

The OAuth 2.1 draft is the base; RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD are the surface; the MCP spec is the profile.

### IdP Capability Matrix

Not every IdP supports the full MCP profile. The matrix below documents objective capability statements as of the 2025-11-25 spec. It is a *deployment gate*, not a recommendation.

CIMD only landed in the 2025-11-25 spec and the underlying OAuth draft wasn't adopted until October 2025, so vendor support is still rolling out — treat "CIMD" below as "where it stands today; verify in your own tenant" rather than a permanent statement.

| IdP Category | AS Metadata (8414/OIDC) | CIMD | RFC 7591 DCR | RFC 8707 Resource | RFC 7636 S256 PKCE | Notes |
|---|---|---|---|---|---|---|
| Self-hosted (Keycloak) | Supported | Emerging | Supported | Supported (24.x+) | Supported | Reference IdP for the MCP profile in this lesson; DCR path works end-to-end, CIMD following the new spec. |
| Enterprise SSO (Microsoft Entra ID) | Supported | Emerging | Supported (premium tier) | Supported | Supported | DCR availability varies by tenant tier; verify in the target tenant before deployment. |
| Enterprise SSO (Okta) | Supported | Emerging | Supported (Okta CIC / Auth0) | Supported | Supported | DCR available on Auth0 (now Okta CIC); classic Okta orgs require admin pre-registration. |
| Social login IdPs (general) | Varies | Not supported | Rare | Rare | Supported | Most social IdPs treat clients as static partners; no self-service registration. Use only as an identity source with your own MCP-aware authorization server layered on top. |
| Custom / self-built | Implementation-dependent | Implementation-dependent | Implementation-dependent | Implementation-dependent | Implementation-dependent | If you ship it yourself, ship the full profile and prefer CIMD. Skipping PKCE or audience binding breaks the MCP auth contract. |

Deployment checklist rejection rules: if the chosen IdP does not list `S256` in `code_challenge_methods_supported`, the MCP server refuses to start — there is no degraded mode for PKCE. Registration is a softer gate: you need *one* workable path (a pre-registered `client_id`, `client_id_metadata_document_supported: true`, or a `registration_endpoint`). Lacking DCR alone no longer triggers rejection since CIMD or pre-registration can cover it.

### JWKS Refresh Pattern (Rotate at the AS, Refresh at the Resource Server)

Separate the two verbs clearly, because conflating them is a real production bug:

- **Rotate** is what the *authorization server* does: mint a new signing key, publish it to the JWKS, then retire the old one later. The resource server has no part in this and cannot do it — it does not have the IdP's private keys.
- **Refresh** is what the *resource server* does: re-`GET` the published JWKS into its own cache. This is the only JWKS action the resource server ever takes.

The failure mode in production is a stale cache. Solve it with a scheduled refresh job and a key-value cache. The resource server runs a job (cron, timer, whatever your runtime offers) that fetches `<issuer>/.well-known/jwks.json` at a fixed interval and overwrites `cache[issuer] = {keys, fetched_at}`. The validator reads from that cache. A token whose `kid` is not in the cache triggers **one** synchronous refresh as a fallback, then rechecks. This handles both cases at once: scheduled refresh, and the key-overlap window — a token signed by a brand-new key arrives before the next scheduled refresh.

The fallback **must be a re-fetch, never a rotation**. If you wire the cache-miss path to a "rotate and mint new," two things break: (1) minting a new key produces a `kid` that *still* doesn't match the token, so the lookup fails anyway; and (2) an attacker spraying tokens with random `kid` values forces an unbounded stream of key creations — a self-inflicted DoS. A re-fetch is idempotent, so a forged `kid` wastes at most one fetch.

Cache shape:

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

Two keys existing simultaneously is the steady state. The authorization server rotates by introducing the next key (`k_2026_04`) first, then retiring the previous one (`k_2026_03`), so tokens signed under the old key remain valid until they expire. The cache holds this union; the validator picks by `kid`.

### Validation Routine

The MCP server runs validation before handing out any tool. The shape used by `code/main.py`:

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` decodes the JWT, resolves the signing key from the JWKS cache (refreshing once on miss), verifies the signature, then checks `iss` against the allow-list, `aud` against this server's canonical resource, `exp`, and required scopes — returning a `WWW-Authenticate` challenge at the first failure. Making this a single routine on the resource server means every entry point (every tool call, every transport) runs the same checks; no path can reach a tool without validating first.

### Audience Replay Walkthrough (Access-Token Privilege Restriction)

Server A (`notes.example.com`) and Server B (`tasks.example.com`) are both registered with the same authorization server. Server A is compromised. The attacker takes a user's notes token and replays it against Server B.

Server B's validator:

1. Decodes the JWT, fetches JWKS by `kid`, verifies the signature.
2. Checks `iss` against its protected-resource metadata's `authorization_servers`. (Passes — same IdP.)
3. Checks `aud == "https://tasks.example.com"`. (Fails — the token's `aud` is `https://notes.example.com`.)
4. Returns 401 with `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`.

The audience claim is the protocol-level defense against this attack. Skipping it for performance is the most common production mistake; the validator must run on every request, not just at session start. The spec calls this **access-token privilege restriction**: an MCP server `MUST` reject any token that does not name it in the audience.

> **Naming note.** The spec reserves the term *confused deputy* for a related but different problem: an MCP server acting as an OAuth **proxy** to a third-party API, using a static client ID, forwarding a token without obtaining per-client user consent. Audience binding fixes the replay above; the confused-deputy fix is per-client consent **plus** never passing an inbound token through to an upstream API (the MCP server `MUST` obtain its own separate upstream token).

### Mix-up Attacks (A Client-Side Defense the Server Cannot Provide)

A client talks to many authorization servers over its lifetime. A malicious AS can attempt to get the client to exchange an honest AS's authorization code at the attacker's token endpoint. Audience binding does not help here — the attack happens before any token exists. The defense lives in the client (RFC 9207):

1. Before redirect, the client records the expected `issuer` from validated AS metadata.
2. On the authorization response, the client compares the returned `iss` parameter against that recorded issuer (simple string comparison, no normalization) before sending the code anywhere.
3. Mismatch (or a missing `iss` when the AS advertised `authorization_response_iss_parameter_supported`) → reject, without even displaying the `error` field.

PKCE alone cannot stop mix-up because the client will send its own `code_verifier` to whichever token endpoint it is directed to. This is precisely why the spec requires recording the issuer alongside the PKCE verifier and `state` on a per-request basis.

### Failure Modes

- **Stale JWKS.** The AS rotates a key; the validator rejects valid tokens. The fix is the cron refresh + cache-miss re-fetch pattern above. Never cache JWKS without a refresh job.
- **Rotation as fallback.** Wiring the cache-miss path to a "rotate and mint new" instead of a re-fetch is a real bug: it never produces the missing `kid`, and it turns attacker-controlled `kid` values into a key-creation DoS. The fallback must be an idempotent `refresh-jwks`.
- **Missing `aud` claim.** Some IdPs omit `aud` by default unless the token request includes `resource`. The validator must reject tokens with missing `aud`, not treat absence as a wildcard.
- **Mix-up from missing `iss` check.** A client that does not verify the RFC 9207 `iss` authorization response parameter (against the issuer it recorded before redirect) can be directed to exchange an honest AS's code at the attacker's token endpoint. This is a client-side failure; the resource server cannot compensate.
- **Scope escalation race.** Two concurrent step-up flows for the same user may both succeed, producing two access tokens with different scopes. The validator must use the token presented on the request, not look up "the user's current scope" — that creates a TOCTOU window.
- **Registration token theft.** A leaked `registration_access_token` lets an attacker rewrite redirect URIs. Store them hashed at rest; require the client to present the plaintext on each update; rotate on suspicion.
- **Unpinned `iss`.** A validator that accepts any `iss` lets attackers stand up their own authorization server, register a client for the target audience, and issue tokens. The `authorization_servers` list in the protected-resource metadata is the allow-list; enforce it.

## Use It

`code/main.py` walks the full production flow using standard-library Python and three roles — `AuthorizationServer`, `ResourceServer`, `Client`. The flow:

1. The authorization server publishes RFC 8414 metadata at `/.well-known/oauth-authorization-server`.
2. The MCP client calls the metadata endpoint, checks its registration options (`client_id_metadata_document_supported` for CIMD, `registration_endpoint` for DCR) and `S256` PKCE support.
3. The demo walks the DCR fallback path: the client posts to `/register` (RFC 7591) and receives a `client_id`. (A CIMD client would instead present its own HTTPS `client_id` URL and skip this step.)
4. The MCP client runs a PKCE-protected authorization code flow (RFC 7636) with a `resource` indicator (RFC 8707).
5. The MCP client calls a tool on the MCP server with `Authorization: Bearer ...`.
6. The MCP server runs `validate`, resolving the signing key from the JWKS cache.
7. The IdP rotates a key; the scheduled refresh re-fetches JWKS into the cache.
8. The next call validates against the refreshed key without restart, and the previous token still validates during the overlap window.
9. An audience replay attempt against a different MCP resource receives a 401 with `audience mismatch` and a `resource_metadata` pointer.

The JWTs here use HS256 with a shared secret (so the lesson runs on the standard library alone). Production uses RS256 or EdDSA with the JWKS pattern above; the validation logic is otherwise identical. Because the IdP and resource server live in the same process, `refresh_jwks` reads the authorization server's key list directly; in production it is an HTTP `GET` to `jwks_uri`.

## Ship It

This lesson produces `outputs/skill-mcp-auth.md`. Given an MCP server configuration and a set of IdP capabilities, this skill outputs the auth surface to stand up — protected-resource metadata, the registration path to use (CIMD, pre-registration, or DCR fallback), JWKS refresh schedule, scope mapping, and rejection rules to impose when the IdP doesn't support the full RFC profile.

## Exercises

1. Run `code/main.py`. Trace the flow. Note how the IdP rotates a key at step 6, the scheduled `refresh_jwks` re-fetches the published key set, and then both the old token (overlap window) and a new token validate without restart.

2. Add a new IdP to the protected-resource metadata's `authorization_servers` list. Issue a token signed by the new IdP; confirm the validator accepts it. Then issue a token signed by an unlisted IdP; confirm the validator rejects with `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`.

3. Add a rate-limit check to `register_client` that runs before the registrar accepts the request. Use a per-source-IP token bucket stored in a small dict keyed by IP.

4. Read RFC 7591 and find two fields that this lesson's `/register` handler does not validate. Add the validation. (Hint: `software_statement` and the URI scheme of `redirect_uris`.)

5. Add a Client ID Metadata Document path. Serve a `client.json` whose `client_id` equals its own URL and have the authorization server fetch and validate it (rejecting if `client_id` ≠ URL). Confirm a CIMD client registers without calling `register_client`.

6. Prove the DoS fix. Send the validator a token with a random `kid`; confirm `refresh_jwks` runs at most once and that the authorization server's key count does not grow. Then deliberately re-wire the fallback to a "rotate and mint new," watch the key count climb with each forged token — then restore the re-fetch.

7. Implement the client-side RFC 9207 `iss` check from the mix-up section: record the expected issuer before the authorization request, then reject an authorization response whose `iss` doesn't match.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| ASM | "OAuth metadata document" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | "Client metadata URL" | Client ID Metadata Document — an HTTPS URL used as `client_id`; the AS fetches that JSON. Recommended default since 2025-11-25 |
| DCR | "Self-service client registration" | RFC 7591 `POST /register` flow; downgraded to a `MAY` fallback in 2025-11-25 |
| JWKS | "Public keys for JWT validation" | JSON Web Key Set fetched from `jwks_uri`, indexed by `kid` |
| Rotate vs refresh | "Updating keys" | *Rotate* = AS mints/retires signing keys; *refresh* = resource server re-fetches the published key set. The resource server only ever refreshes |
| Resource indicator | "Audience parameter" | RFC 8707 `resource` parameter pinning a token to one server |
| `aud` claim | "Audience" | JWT claim the validator compares against the canonical resource URL |
| Audience replay | "Token replay" | A token issued for Server A is presented to Server B; defended by audience validation (spec: access-token privilege restriction) |
| Confused deputy | "Proxy token misuse" | An MCP proxy using a static client ID forwards a token without per-client consent; separate from audience replay |
| Mix-up attack | "Wrong token endpoint" | Client is directed to exchange an honest AS's code at an attacker's endpoint; defended by client-side RFC 9207 `iss` |
| `iss` allow-list | "Trusted authorization servers" | The set given by `authorization_servers` in the protected-resource metadata |
| `resource_metadata` | "Where to find the PRM document" | `WWW-Authenticate` parameter on 401/403 pointing to the RFC 9728 metadata URL |
| Public client | "Native or browser client" | An OAuth client with no `client_secret`; compensated by PKCE |
| `WWW-Authenticate` | "401/403 response header" | Carries `Bearer error=...` directives that drive client recovery |

## Further Reading

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — The MCP auth profile this lesson implements
- [MCP blog — One Year of MCP: November 2025 Spec Release](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — What changed in 2025-11-25 (CIMD, XAA, DCR downgrade)
- [Aaron Parecki — Client Registration in the November 2025 MCP Authorization Spec](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update) — The rationale for CIMD over DCR
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — The discovery contract
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR (fallback path)
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — Public-client proof-of-possession
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — Audience pinning
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — Resource server discovery
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — The `iss` parameter defending against mix-up attacks
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — The consolidated OAuth base
