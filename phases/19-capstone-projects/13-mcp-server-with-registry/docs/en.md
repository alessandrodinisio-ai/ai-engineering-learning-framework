# Capstone Project 13 — MCP Server with Registry and Governance

> Model Context Protocol is no longer the future — it became the default tool-use spec in 2026. Anthropic, OpenAI, Google, and every major IDE shipped MCP clients. Pinterest publicly shared its internal MCP server ecosystem. The AAIF Registry standardized capability metadata at `.well-known`. AWS ECS published a reference-grade stateless deployment. Block's goose-agent stuffed the same protocol into a hosted assistant. The 2026 production shape is: StreamableHTTP transport, OAuth 2.1 scopes, OPA policy gating, and a registry that lets platform teams discover, validate, and enable servers. Build it end to end.

**Type:** Capstone
**Languages:** Python (server, via FastMCP) or TypeScript (@modelcontextprotocol/sdk), Go (registry service)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools & MCP), Phase 14 (Agents), Phase 17 (Infrastructure), Phase 18 (Safety)
**Phases Involved:** P11 · P13 · P14 · P17 · P18
**Time:** 25 hours

## The Problem

MCP became the lingua franca of tool use. Claude Code, Cursor 3, Amp, OpenCode, Gemini CLI, and every hosted agent now consume MCP servers. The production challenge is not writing a server (FastMCP makes that trivial) but deploying them at scale with enterprise requirements: per-tenant OAuth scopes, OPA policies on destructive tools, StreamableHTTP stateless scaling, a registry for discovery, and per-tool-call audit logs. Pinterest's internal MCP ecosystem and the AAIF Registry spec set the 2026 standard.

You will build an MCP server exposing 10 internal tools (Postgres read-only, S3 listing, Jira, Linear, Datadog, etc.), a registry UI for platform discovery, and a human-approval gate for destructive tools. Load testing demonstrates StreamableHTTP horizontal scaling. The audit trail satisfies an enterprise security review.

## The Concept

The MCP 2026 revision mandates StreamableHTTP as the default transport. Unlike the earlier stdio-plus-SSE shape, StreamableHTTP is stateless by default: a single HTTP endpoint accepts JSON-RPC requests, streams responses back, and supports long-lived connections for notifications. Statelessness means horizontal scaling behind a load balancer.

Authorization is OAuth 2.1 with per-tool scopes. A token carries scopes like `jira:read`, `s3:list`, `postgres:query:readonly`. The MCP server checks scopes at tool-call time, not just at session start. For high-risk tools, the server rejects any call whose scope has not been elevated to `approved:by:human` within the last N minutes — that elevation comes from a Slack review card.

The registry is a separate service. Each MCP server exposes a document at `.well-known/mcp-capabilities` with its tool list, transport URL, and auth requirements. The registry polls, validates, and indexes. Platform teams use the registry UI to see what tools are available, what scopes they require, and which teams own them.

## Architecture

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## Tech Stack

- Server framework: FastMCP (Python) or `@modelcontextprotocol/sdk` (TypeScript)
- Transport: StreamableHTTP over HTTPS (stateless)
- Auth: OAuth 2.1, workload identity via SPIFFE / SPIRE
- Policy: Per-tool OPA / Rego rules; one policy decision service per request
- Registry: Self-hosted, consuming `.well-known/mcp-capabilities` manifests
- Human approval: Slack interactive messages for destructive tools
- Deployment: AWS ECS Fargate or Fly.io, one server per tenant or shared with tenant scoping
- Audit: Structured JSONL in per-tenant buckets with per-call lineage

## Build It

1. **Tool surface.** Expose 10 internal tools: Postgres read-only query, S3 list objects, Jira search/get, Linear search/get, Datadog metrics query, PagerDuty on-call query, GitHub read-only, Notion search, Slack search, Salesforce read. Each tool has a typed schema and a scope tag.

2. **FastMCP server.** Wire up the tools. Configure StreamableHTTP transport. Add a middleware that performs OAuth token introspection and scope enforcement.

3. **OPA policies.** Per-tool Rego policies: what scopes allow the call, what PII redaction applies, what payload size limits apply. A policy decision service is called on every tool call.

4. **Registry service.** A separate Go or TS service that polls `.well-known/mcp-capabilities` from registered servers, validates against JSON Schema, and exposes a list / search / validate / enable-disable UI.

5. **Capability manifest.** Each server exposes `.well-known/mcp-capabilities` with: tool list, auth requirements, transport URL, owner team, SLO.

6. **Destructive tool separation.** State-changing tools (Jira create, Linear create, Postgres write) live on a second MCP server with a stricter auth flow: the token must carry an `approved:by:human` scope that was elevated via Slack card within the last 15 minutes.

7. **Audit log.** Per-tenant append-only JSONL: `{timestamp, user, tool, args_redacted, response_redacted, outcome}`. PII redaction via Presidio before writing.

8. **Load test.** 100 concurrent clients over StreamableHTTP. Demonstrate horizontal scaling by adding a second replica; show the load balancer redistributing without session stickiness.

9. **Conformance test.** Run the official MCP conformance suite against both servers. Pass all mandatory sections.

## Use It

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## Ship It

`outputs/skill-mcp-server.md` describes the deliverable. A production-grade MCP server + registry + audit layer for internal tools with OAuth 2.1 scopes and OPA gating.

| Weight | Criterion | How to Measure |
|:-:|---|---|
| 25 | Spec conformance | StreamableHTTP + capability manifest passing MCP conformance tests |
| 20 | Security | Scope enforcement, OPA covering every tool, secret hygiene |
| 20 | Observability | Per-tool-call audit log with PII redaction |
| 20 | Scale | Horizontal scaling demo with 100-client load test |
| 15 | Registry experience | Discover / validate / enable-disable workflow |
| **100** | | |

## Exercises

1. Add a new tool (Confluence search). Bring it live through the registry validation flow without touching the core server.

2. Write an OPA policy that redacts columns named `email`, `ssn`, `phone` from Postgres query results. Exercise with a probe query.

3. Benchmark StreamableHTTP vs stdio on local latency. Report per-call p50/p95.

4. Implement per-tenant quotas: at most N calls per tool per tenant per minute. Enforce with a second OPA rule.

5. Run the MCP conformance suite from [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) and fix every failure.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| StreamableHTTP | "2026 MCP transport" | Stateless HTTP + streaming; replaces SSE + stdio for networked servers |
| Capability manifest | "Well-known document" | `.well-known/mcp-capabilities` with tool list, auth, transport URL |
| OPA / Rego | "Policy engine" | Open Policy Agent, authorizing tool calls against external rules |
| Scope elevation | "Human-approved" | A short-lived scope granted via Slack approval, required for destructive tools |
| Registry | "Tool discovery" | A service indexing MCP servers from their capability manifests |
| Workload identity | "SPIFFE / SPIRE" | Cryptographic service identity for OAuth token issuance |
| Conformance suite | "Spec tests" | The official MCP test suite checking StreamableHTTP + tool manifest correctness |

## Further Reading

- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP, capability metadata, registry
- [AAIF MCP Registry spec](https://github.com/modelcontextprotocol/registry) — 2026 registry specification
- [AWS ECS reference deployment](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — reference production deployment
- [Pinterest internal MCP ecosystem](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — reference internal deployment
- [Block `goose` MCP usage](https://block.github.io/goose/) — reference agent consumption pattern
- [FastMCP](https://github.com/jlowin/fastmcp) — Python server framework
- [Open Policy Agent](https://www.openpolicyagent.org/) — policy engine reference
- [SPIFFE / SPIRE](https://spiffe.io) — workload identity reference
