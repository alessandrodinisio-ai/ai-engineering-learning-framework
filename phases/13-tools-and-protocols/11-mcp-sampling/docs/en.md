# MCP Sampling — Server-Requested LLM Completions and Agent Loops

> Most MCP servers are dumb executors: take arguments, run code, return content. Sampling lets the server reverse direction: it requests the client's LLM to make a decision. This enables server-hosted agent loops without the server owning any model credentials. SEP-1577 merged in 2025-11-25, adding tools inside sampling requests so the loop can incorporate deeper reasoning. Drift warning: SEP-1577's in-sampling tool shape is still experimental through 2026 Q1 and is still settling in SDK APIs.

**Type:** Build
**Languages:** Python (stdlib, sampling scaffolding)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources and prompts)
**Time:** ~75 minutes

## Learning Objectives

- Explain what `sampling/createMessage` solves (server-hosted loops without server-side API keys).
- Implement a server that requests the client to sample on a multi-turn prompt and returns the completion.
- Use `modelPreferences` (cost / speed / intelligence priorities) to guide the client's model selection.
- Build a `summarize_repo` tool that internally iterates via sampling rather than hardcoded behavior.

## The Problem

An MCP server useful for code summarization workflows needs to: traverse a file tree, pick which files to read, synthesize a summary, return. Where does the LLM reasoning happen?

Option A: the server calls its own LLM. Requires an API key, bills on the server side, expensive per user.

Option B: the server returns raw content; the client's agent does reasoning. Works, but moves server logic into the client prompt, which is brittle.

Option C: the server requests the client's LLM via `sampling/createMessage`. The server retains the algorithm (which files to read, how many passes), while the client retains billing and model selection. The server has no credentials at all.

Sampling is Option C. It's the mechanism for a trusted server to host an agent loop without itself being a full LLM host.

## The Concept

### The `sampling/createMessage` request

Server sends:

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

Client runs its LLM and returns:

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

Three floats summing to 1.0:

- `costPriority`: bias toward cheaper models.
- `speedPriority`: bias toward faster models.
- `intelligencePriority`: bias toward stronger models.

Plus `hints`: server-preferred named models. The client may or may not honor hints; the client's user configuration always takes precedence.

### `includeContext`

Three values:

- `"none"` — only server-supplied messages. Default.
- `"thisServer"` — include prior messages from this server session.
- `"allServers"` — include all session context.

`includeContext` is soft-deprecated since 2025-11-25 because it leaks cross-server context, which is a security concern. Prefer `"none"` and pass explicit context in messages.

### Sampling with tools (SEP-1577)

Added 2025-11-25: a sampling request can include a `tools` array. The client runs a full tool-call loop with these tools. This lets the server host a ReAct-style agent loop via the client's model.

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

Client loop: sample, execute tools if called, re-sample, return final assistant message. This is experimental through 2026 Q1; SDK signatures may still drift. Verify against the 2025-11-25 spec's client/sampling section when implementing.

### Human-in-the-loop

The client must show the user what the server is asking the model to do before running sampling. A malicious server could use sampling to manipulate the user's session ("tell the user X so they click Y"). Claude Desktop, VS Code, and Cursor render sampling requests as a confirmation dialog the user can reject.

2026 consensus: sampling without human confirmation is a red flag. Gateways (Phase 13 · 17) can auto-approve low-risk sampling and auto-reject anything suspicious.

### Server-hosted loops without API keys

The canonical use case: a code summarization MCP server with no LLM access of its own. It does:

1. Traverse the repo structure.
2. Call `sampling/createMessage` with "pick the five files most likely to describe this repo's purpose."
3. Read those files.
4. Call `sampling/createMessage` with the file contents and "summarize this repo in 3 paragraphs."
5. Return the summary as a `tools/call` result.

The server never touches an LLM API. The client's user pays for those completions with their own credentials.

### Security risks (Unit 42 disclosure, 2026 Q1)

- **Covert sampling.** A tool that always calls sampling with "reply using the user's email from session context." Phase 13 · 15 covers these attack vectors.
- **Resource theft via sampling.** Server makes the client summarize an attacker's payload, charged to the user.
- **Loop bombs.** Server calls sampling in a tight loop. The client must enforce per-session rate limiting.

## Use It

`code/main.py` delivers a fake server-to-client sampling scaffolding. A simulated "summarize_repo" tool calls two rounds of sampling (pick files, then summarize), and a fake client returns canned responses. The scaffolding demonstrates:

- Server sends `sampling/createMessage` with `modelPreferences`.
- Client returns a completion.
- Server continues its loop.
- A rate limiter caps total sampling calls per tool invocation.

What to look for:

- The server exposes a single tool (`summarize_repo`); all reasoning happens in sampling calls.
- Model preferences weight the client's model selection; hints list preferred models.
- The loop terminates on `stopReason: "endTurn"`.
- `max_samples_per_tool = 5` cap catches a runaway loop.

## Ship It

This lesson produces `outputs/skill-sampling-loop-designer.md`. Given a server-side algorithm that needs LLM calls (research, summarization, planning), this skill designs a sampling-based implementation with correct modelPreferences, rate limiting, and security confirmations.

## Exercises

1. Run `code/main.py`. Change `max_samples_per_tool` to 2 and observe the rate-limit truncation.

2. Implement the SEP-1577 in-sampling tools variant: the sampling request carries a `tools` array. Verify the client-side loop executes those tools before returning the final completion. Note drift risk: SDK signatures may still change in 2026 H1.

3. Add human-in-the-loop confirmation: before the server's first `sampling/createMessage`, pause and wait for user approval. A rejected call returns a typed refusal.

4. Add a per-user rate limiter keyed by client session. The same user's same-server loop should share a budget.

5. Design a `summarize_pdf` tool that uses sampling to pick which chunks to include. Sketch the emitted messages. How does `modelPreferences.intelligencePriority` at 0.1 vs 0.9 change behavior?

## Key Terms

| Term | What people call it | What it actually is |
|------|---------------------|---------------------|
| Sampling | "server-to-client LLM call" | Server requesting a completion from the client's model |
| `sampling/createMessage` | "the method" | JSON-RPC method for a sampling request |
| `modelPreferences` | "model priorities" | Cost / speed / intelligence weights plus name hints |
| `includeContext` | "cross-session leakage" | Soft-deprecated context inclusion mode |
| SEP-1577 | "tools in sampling" | Allows in-sampling tools for server-hosted ReAct |
| Human-in-the-loop | "user confirmation" | Client presenting sampling request to user before running |
| Loop bomb | "runaway sampling" | Server-side infinite sampling loop; client must rate-limit |
| Covert sampling | "hidden reasoning" | Malicious server hiding intent in sampling prompts |
| Resource theft | "spending user's LLM budget" | Server forcing client to pay for unwanted sampling |
| `stopReason` | "why generation stopped" | `endTurn`, `stopSequence`, or `maxTokens` |

## Further Reading

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — high-level sampling overview
- [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — authoritative `sampling/createMessage` shape
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — Spec Evolution Proposal for tools in sampling (experimental)
- [Unit 42 — MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — covert sampling and resource theft patterns
- [Speakeasy — MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) — step-by-step walkthrough with client-side code examples
