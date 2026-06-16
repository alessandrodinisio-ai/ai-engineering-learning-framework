# The Tool Interface — Why Agents Need Structured I/O

> Language models produce tokens. Programs execute actions. The gulf between the two is the tool interface: a contract that lets the model request actions and the host execute them. Every 2026 stack — function calling on OpenAI, Anthropic, Gemini; MCP's `tools/call`; A2A's task part — is a different encoding of the same four-step loop. This lesson names that loop and shows the minimum machinery needed to run it.

**Type:** Learn
**Languages:** Python (standard library, no LLM calls)
**Prerequisites:** Phase 11 (LLM Completion APIs)
**Time:** ~45 minutes

## Learning Objectives

- Explain why a text-only LLM cannot take real-world actions on its own.
- Draw the four-step tool-call loop (describe → decide → execute → observe) and identify who owns each step.
- Write a tool description with three parts: name, JSON Schema input, and a deterministic executor function.
- Distinguish pure tools from consequential tools and explain why the distinction matters for safety.

## The Problem

An LLM outputs a probability distribution over the next token. That is its entire output surface. If you ask a chat model "What's the weather in Bengaluru right now?" it can write a plausible-sounding sentence, but it cannot connect to a weather API. That sentence might happen to be correct, or it might be stale data from three days ago.

Bridging this gap is what the tool interface does. The host program — your agent runtime, Claude Desktop, ChatGPT, Cursor, or a custom script — advertises a list of callable tools to the model. When the model decides an action is needed, it emits a structured payload specifying the tool name and arguments. The host parses the payload, actually executes the tool, and feeds the result back. The loop continues until the model decides no further calls are needed.

The first version of this contract shipped in June 2023 with OpenAI's "functions" parameter. Anthropic followed with `tool_use` blocks in Claude 2.1. Gemini added `functionDeclarations` months later. Today every provider exposes the same shape: a tool list typed with JSON Schema as input, a tool call as JSON payload output. The Model Context Protocol (November 2024) generalized this contract so a single tool registry can serve all models. A2A (April 2026, v1.0) stacks another layer on the same primitive for agent-to-agent delegation.

The four-step loop is the invariant underneath all of this. Everything else in Phase 13 is a refinement of it.

## The Concept

### Step 1: describe

The host declares each tool with three fields.

- **Name.** A stable, machine-readable identifier. `get_weather`, not "that weather thing."
- **Description.** A natural-language brief. "Use when the user asks for current weather in a specific city. Do not use for historical data."
- **Input schema.** A JSON Schema object (draft 2020-12) describing the tool's parameters.

The model receives this list. Modern providers serialize the declarations into the system prompt using their own specific templates, so you as the caller only deal with the structured form.

### Step 2: decide

Given the user message and available tools, the model picks one of three behaviors.

1. **Answer directly with text.** No tool call.
2. **Call one or more tools.** Emit structured call objects. Under `parallel_tool_calls: true` (default on OpenAI and Gemini, opt-in on Anthropic), the model can emit multiple calls in a single turn.
3. **Refuse.** Strict-mode structured outputs can produce a typed `refusal` block instead of a call.

A tool call payload has three stable fields: a call `id`, the tool `name`, and a JSON `arguments` object. The id exists so the host can match a later result to the specific call — critical when parallel calls return out of order.

### Step 3: execute

The host receives the call, validates arguments against the declared schema, and runs the executor. Invalid arguments mean the model hallucinated a field or used the wrong type — a very common failure mode on weaker models. Production hosts do one of three things on invalid arguments: fail fast and throw the error back to the model, fix the JSON with a constrained parser, or retry the model with a validation error.

The executor itself is ordinary code. Python, TypeScript, a shell command, a database query. It produces a result — usually a string, but potentially any JSON value or a structured content block (text, image, or resource reference in MCP). The result must be serializable.

### Step 4: observe

The host appends the tool result to the conversation (as a `tool`-role message with the matching `id`) and calls the model again. The model now has the tool output in context and can produce a final answer or request more calls. This continues until the model stops emitting calls or the host hits a safety cap on iteration count.

### The trust split

Tools come in two kinds, and the distinction matters for safety.

- **Pure.** Read-only, deterministic, no side effects. `get_weather`, `search_docs`, `get_current_time`. Safe to speculatively invoke.
- **Consequential.** Mutates state, costs money, touches user data. `send_email`, `delete_file`, `execute_trade`. Must be gated.

Meta's 2026 agent-safety "Rule of Two" says: in a single turn, you may combine at most two of these three — untrusted input, sensitive data, consequential action. The tool interface is where you enforce that rule — by refusing calls, requiring user confirmation, or escalating permission scope. Full security treatment in Phase 13 · 15; agent-level permission policies in Phase 14 · 09.

### Where the loop lives

| Scenario | Who describes | Who decides | Who executes |
|---------|---------------|-------------|--------------|
| Single-turn function calling (OpenAI/Anthropic/Gemini) | App developer | LLM | App developer |
| MCP | MCP server | LLM via MCP client | MCP server |
| A2A | Agent Card publisher | Caller agent | Callee agent |
| Web browser (function-calling agent) | Browser extension / WebMCP | LLM | Browser runtime |

Same four steps everywhere. The column names change, not the structure.

### Why not just prompt the model to emit JSON?

"Make the model reply in JSON" was the pre-function-calling approach. It fails about 5–15% of the time on frontier models, much more on smaller ones. Failure modes include missing braces, trailing commas, hallucinated fields, type errors. Then you add a JSON repair pass, a retry, or a constrained decoder.

Native function calling is better for three reasons. First, providers train the model end-to-end on the exact call shape, so valid-JSON rates rise to 98–99% in strict mode. Second, the call payload lives in its own protocol slot rather than in free text — so tool calls never leak into user-visible replies. Third, providers enforce schema compliance with constrained decoding (OpenAI's strict mode, Anthropic's `tool_use`, Gemini's `responseSchema`). The output is guaranteed to pass validation.

Phase 13 · 02 walks through all three provider APIs side by side. Phase 13 · 04 goes deep on structured outputs.

### Circuit breakers

The loop terminates when the model stops emitting calls or the host hits a max-turns cap. Production hosts set this between 5 and 20 turns. Beyond that, you are almost certainly stuck in a loop the model cannot exit. Claude Code defaults to 20; OpenAI Assistants to 10; Cursor's agent mode to 25.

The alternative — unbounded loops — surfaces every six months as a post-mortem titled "agent burned $400 in API calls overnight." No cap, no deploy.

Phase 14 · 12 goes deeper on error recovery and self-healing; Phase 17 covers production rate limiting.

### Where Phase 13 goes from here

- Lessons 02–05 refine the provider-level tool-call surface.
- Lessons 06–14 generalize the loop into MCP.
- Lessons 15–18 harden the loop against malicious servers, adversarial users, and unauthenticated remote auth surfaces.
- Lessons 19–22 extend the pattern to agent-to-agent collaboration, observability, routing, and packaging.
- Lesson 23 uses every primitive to ship a complete ecosystem.

Every remaining lesson is a refinement of this four-step loop. Carry it as the invariant.

## Use It

`code/main.py` runs the four-step loop without calling an LLM. A fake "decider" function simulates the model by pattern-matching the user message; the executor, schema validator, and observe-step scaffolding are real. Run it, watch the full request/response choreography with printable intermediate states; swap the fake decider for any real provider in a later lesson.

What to look for:

- The tool registry stores four fields per tool: name, description, schema, and an executor reference.
- The validator is a minimal JSON Schema subset (types, required, enum, min/max) written in pure standard library. Phase 13 · 04 ships a more complete one.
- The loop caps iterations at 5. Production agents need exactly this kind of circuit breaker.

## Ship It

This lesson produces `outputs/skill-tool-interface-reviewer.md`. Given a draft tool definition (name + description + schema + executor outline), this skill audits it for loop fitness: whether the name is machine-stable, the description is a complete usage brief, the schema correctly uses JSON Schema 2020-12, and the pure vs. consequential classification is explicit.

## Exercises

1. Add a fourth tool to `code/main.py` called `get_stock_price(ticker)`. Write its description as "Use when the user asks for the current price of a stock by ticker symbol. Do not use for historical prices or market summaries." Run the scaffolding and confirm the fake decider routes ticker-related queries to the new tool.

2. Break the schema validator. Pass a call with an `arguments` object missing a required field and confirm the host rejects it before execution. Then pass a call with extra unknown fields. Decide: should the host reject or ignore? Justify your choice with a safety argument.

3. Classify every tool in the scaffolding as pure or consequential. Add a `consequential: true` flag to the registry entries that need it, and modify the loop to print a "would confirm with user" line every time a consequential tool is selected. This is the shape of the confirmation gate every production host needs.

4. Draw the four-step loop on paper and fill in the provider table above for your favorite client (Claude Desktop, Cursor, ChatGPT, or a custom stack). Cross-reference with the MCP-specific variant in Phase 13 · 06.

5. Read OpenAI's function calling guide end to end. Identify the one field that lives in the request but is not in this lesson's four-step loop. Explain what it adds and why it is convenience rather than necessity.

## Key Terms

| Term | Common shorthand | What it actually is |
|------|----------------|------------------------|
| Tool | "a thing the model can call" | A triple of name + JSON Schema–typed input + executor function |
| Function calling | "native tool calls" | Provider-level API support for emitting structured tool calls instead of prose |
| Tool call | "the model's action request" | A JSON payload with `id`, `name`, `arguments` emitted by the model |
| Tool result | "what the tool returned" | The executor's output, wrapped in a `tool`-role message with matching id |
| Parallel tool calls | "multiple calls at once" | Multiple call objects in one model turn, independent, keyed by id |
| Strict mode | "guaranteed JSON" | Constrained decoding that forces model output to pass the declared schema |
| Pure tool | "read-only tool" | No side effects; safe to re-run |
| Consequential tool | "action tool" | Mutates external state; needs gating, audit, or user confirmation |
| Four-step loop | "tool-call cycle" | describe → decide → execute → observe |
| Host | "agent runtime" | The program that holds the tool registry, calls the model, and runs executors |

## Further Reading

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — Authoritative reference for OpenAI-style tool declarations and call shapes
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — Claude's `tool_use` / `tool_result` block format
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — `functionDeclarations` and parallel call semantics in Gemini
- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — Provider-agnostic generalization of the tool interface
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — The schema dialect every modern tool API speaks
