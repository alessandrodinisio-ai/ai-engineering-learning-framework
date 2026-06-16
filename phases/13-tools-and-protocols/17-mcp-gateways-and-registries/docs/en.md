# MCP Gateways and Registries — The Enterprise Control Plane

> Enterprises cannot let every developer install arbitrary MCP servers. A gateway centralizes authentication, RBAC, auditing, rate limiting, caching, and tool-poisoning detection, then exposes the merged tool surface as a single MCP endpoint. The official MCP Registry (Anthropic + GitHub + PulseMCP + Microsoft, with verified namespaces) is the authoritative upstream. This lesson names where the gateway sits, walks through a minimal implementation, and surveys the 2026 vendor landscape.

**Type:** Learn
**Languages:** Python (standard library, minimal gateway)
**Prerequisites:** Phase 13 · 15 (tool poisoning), Phase 13 · 16 (OAuth 2.1)
**Time:** ~45 minutes

## Learning Objectives

- Explain where an MCP gateway sits (between the MCP client and multiple backend MCP servers).
- Implement the gateway's five responsibilities: authentication, RBAC, auditing, rate limiting, and policy.
- Enforce a pinned tool-hash manifest at the gateway layer.
- Distinguish the official MCP Registry from meta-registries (Glama, MCPMarket, MCP.so, Smithery, LobeHub).

## The Problem

A Fortune 500 company has 30 approved MCP servers, 5,000 developers, compliance and audit requirements, and a security team that wants centralized policy. Letting every developer install arbitrary servers in their IDE simply does not work.

The gateway pattern:

1. The gateway runs as a single Streamable HTTP endpoint; developers connect to it.
2. The gateway holds credentials for each backend MCP server.
3. Every developer request is authenticated and scoped via the gateway's own OAuth.
4. The gateway routes calls to backend servers and applies policy.
5. All calls are logged for audit.

Cloudflare MCP Portals, Kong AI Gateway, IBM ContextForge, MintMCP, TrueFoundry, Envoy AI Gateway — all shipped gateways or gateway features in 2025–2026.

Meanwhile, the official MCP Registry went live as the authoritative upstream: curated, namespace-verified, reverse-DNS-named servers that gateways can pull from. Meta-registries (Glama, MCPMarket, MCP.so, Smithery, LobeHub) aggregate servers across multiple sources.

## The Concept

### Five Gateway Responsibilities

1. **Authentication.** OAuth 2.1 identifies the developer; maps to user roles.
2. **RBAC.** Per-user policy: which servers, which tools, which scopes.
3. **Auditing.** Every call records who, did what, when, and with what result.
4. **Rate limiting.** Per-user / per-tool / per-server caps to prevent abuse.
5. **Policy.** Reject poisoned descriptions, enforce Rule of Two, redact PII.

### The Gateway as a Single Endpoint

To developers, the gateway looks like a single MCP server. Internally it routes to N backends. Session IDs (Phase 13 · 09) are rewritten at the boundary.

### Credential Vault

Developers never see backend tokens. The gateway holds them (or proxies to an identity provider that holds them). A developer with `notes:read` on the gateway can transitively access the notes MCP server using the gateway's own backend credentials — but only under the policy that binds that transitive access.

### Tool-Hash Pinning at the Gateway

The gateway holds a manifest of approved tool descriptions (SHA256 hashes). At discovery time, it fetches each backend's `tools/list`, compares hashes against the manifest, and removes any tool whose description has mutated. This is the rug-pull defense from Phase 13 · 15, applied centrally.

### Policy as Code

Advanced gateways express policy in OPA/Rego, Kyverno, or Styra. Rules like "user `alice` can only call `github.open_pr` on repos in the `acme` org" are encoded declaratively. Simple gateways use handwritten Python. Both shapes are valid.

### Session-Aware Routing

When a user's session involves mixed servers, the gateway multiplexes: the developer's single MCP session holds N backend sessions, one per server. Notifications from any backend are routed through the gateway to the developer's session.

### Namespace Merging

The gateway merges tool namespaces across all backends, typically prefixing on conflict. `github.open_pr`, `notes.search`. This makes routing unambiguous.

### Registries

- **Official MCP Registry (`registry.modelcontextprotocol.io`).** Live under Anthropic, GitHub, PulseMCP, and Microsoft stewardship. Namespaces are verified (reverse DNS: `io.github.user/server`). Basic quality pre-screening.
- **Glama.** Search-centric meta-registry aggregating multiple sources.
- **MCPMarket.** Commercially-oriented directory with vendor listings.
- **MCP.so.** Community directory; open submission.
- **Smithery.** Package-manager-style installation flow.
- **LobeHub.** UI registry integrated into their LobeChat app.

Enterprise gateways pull from the official Registry by default, allow admin-curated additions from meta-registries, and reject anything unpinned.

### Reverse-DNS Naming

The official Registry enforces reverse-DNS names for public servers: `io.github.alice/notes`. Namespaces prevent squatting and make trust delegation clearer.

### Vendor Survey, April 2026

| Vendor | Strength |
|--------|----------|
| Cloudflare MCP Portals | Edge hosting; integrated OAuth; free tier |
| Kong AI Gateway | K8s-native; fine-grained policy; logs to OpenTelemetry |
| IBM ContextForge | Enterprise IAM; compliance; audit export |
| TrueFoundry | DevOps-oriented; metrics-first |
| MintMCP | Developer-platform-focused |
| Envoy AI Gateway | Open source; customizable filters |

Phase 17 (production infrastructure) goes deeper into gateway operations.

## Use It

`code/main.py` delivers a minimal gateway in ~150 lines: authenticates users with a fake Bearer token, holds per-user RBAC policy, routes requests to two backend MCP servers, writes every call to an audit log, enforces a rate limit, and rejects any backend tool whose description hash doesn't match the pinned manifest.

What to look at:

- The `RBAC` dict keyed by `user_id` with allowed `server_tool` entries.
- `AUDIT_LOG` is an append-only event list.
- Rate limiting uses a per-user token bucket.
- The pinned manifest is a `server::tool -> hash` dict.

## Ship It

This lesson produces `outputs/skill-gateway-bootstrap.md`. Given an enterprise MCP plan (users, backends, compliance), this skill produces a gateway configuration spec.

## Exercises

1. Run `code/main.py`. Make a call as an allowed user; then as a forbidden user; then a rate-limit-busting burst. Verify the three flows.

2. Add a policy that redacts PII from results before returning them to the client. Use a simple regex scanning for SSN-shaped strings; note the gaps (emails, phone numbers).

3. Extend the audit log to emit OpenTelemetry GenAI spans. Phase 13 · 20 covers the exact attributes.

4. Design an RBAC policy for a 50-developer team with five backends (notes, github, postgres, jira, slack). Who gets read-only on each? Who gets write?

5. Read Cloudflare's enterprise MCP blog post end to end. Identify one feature Cloudflare delivers that this standard-library gateway does not.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Gateway | "MCP proxy" | A centralized server sitting between clients and backends |
| Credential vaulting | "Backend tokens stay server-side" | Developers never see upstream tokens |
| Session-aware routing | "Multi-backend sessions" | The gateway multiplexes N backend sessions per developer session |
| Tool-hash pinning | "Approved manifest" | SHA256 of each approved tool description; centrally blocks rug-pulls |
| RBAC | "Per-user policy" | Role-based access control over tools and servers |
| Policy-as-code | "Declarative rules" | OPA/Rego, Kyverno, or Styra policies enforced at the gateway |
| Audit log | "Who, what, when" | Append-only event log for compliance |
| Rate limit | "Per-user token bucket" | Per-minute caps to prevent abuse |
| Official MCP Registry | "Authoritative upstream" | `registry.modelcontextprotocol.io`, namespace-verified |
| Reverse-DNS naming | "Registry namespace" | `io.github.user/server` convention |

## Further Reading

- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — Authoritative upstream, namespace-verified
- [Cloudflare — Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/) — Gateway pattern with OAuth and policy
- [agentic-community — MCP gateway registry](https://github.com/agentic-community/mcp-gateway-registry) — Open-source reference gateway
- [TrueFoundry — What is an MCP gateway?](https://www.truefoundry.com/blog/what-is-mcp-gateway) — Feature comparison article
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) — Enterprise gateway from IBM
