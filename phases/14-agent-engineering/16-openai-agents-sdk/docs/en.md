# OpenAI Agents SDK: Handoffs, Guardrails, Tracing

> The OpenAI Agents SDK is a lightweight multi-agent framework built on the Responses API. Five primitives: Agent, Handoff, Guardrail, Session, Tracing. A Handoff is a tool named `transfer_to_<agent>`. Guardrails fire on input or output. Tracing is on by default.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 01 (Agent loop), Phase 14 · 06 (Tool use)
**Time:** ~75 minutes

## Learning Objectives

- Name the five primitives of the OpenAI Agents SDK.
- Explain handoffs: why they are modeled as tools, what name shape the model sees, and how context is transferred.
- Distinguish input guardrails, output guardrails, and tool guardrails; explain `run_in_parallel` vs blocking mode.
- Implement a runtime with the standard library featuring handoffs + guardrails + span-style tracing.

## The Problem

Agents that cannot cleanly delegate end up stuffing everything into a single prompt. Agents without guardrails leak PII, produce policy-violating content, or loop forever. OpenAI's SDK codifies the three primitives that make multi-agent work controllable.

## The Concept

### Five Primitives

1. **Agent.** LLM + instructions + tools + handoffs.
2. **Handoff.** Delegation to another agent. Appears to the model as a tool named `transfer_to_<agent_name>`.
3. **Guardrail.** Validation on input (first agent only), output (last agent only), or tool calls (every function tool).
4. **Session.** Automatic conversation history across turns.
5. **Tracing.** Built-in spans for LLM generations, tool calls, handoffs, and guardrails.

### Handoffs as Tools

The model sees `transfer_to_billing_agent` in its tool list. Calling it signals the runtime to:

1. Copy the conversation context (or collapse it via the `nest_handoff_history` beta).
2. Initialize the target agent with its instructions.
3. Continue the run with the target agent.

This is the productionization of the supervisor pattern (Lesson 13 / Lesson 28).

### Guardrails

Three flavors:

- **Input guardrail.** Runs on the first agent's input. Rejects unsafe or out-of-scope requests before any LLM call.
- **Output guardrail.** Runs on the last agent's output. Catches PII leaks, policy violations, malformed responses.
- **Tool guardrail.** Runs per function tool. Validates arguments, checks permissions, audits execution.

Modes:

- **Parallel** (default). The guardrail LLM runs alongside the main LLM. Lower tail latency. If triggered, the main LLM's work is discarded (wasted tokens).
- **Blocking** (`run_in_parallel=False`). The guardrail LLM runs first. If triggered, no tokens are wasted on the main call.

A triggered tripwire raises `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`.

### Tracing

On by default. Every LLM generation, tool call, handoff, and guardrail emits a span. Opt out with `OPENAI_AGENTS_DISABLE_TRACING=1`. `add_trace_processor(processor)` fans spans out to your own backend in addition to OpenAI.

### Session

`Session` stores conversation history in a backend (SQLite, Redis, custom). `Runner.run(agent, input, session=session)` automatically loads and appends.

### Where This Pattern Breaks Down

- **Handoff drift.** Agent A hands off to Agent B, B hands back to A. Add a hop counter.
- **Guardrail bypass.** Tool guardrails only fire on function tools; built-in tools (file reader, web scraper) need separate policies.
- **Over-tracing.** Spans contain sensitive content. Configure OTel GenAI content-capture rules (Lesson 23) — store externally, reference by ID.

## Build It

`code/main.py` implements the SDK's shape using the standard library:

- `Agent`, `FunctionTool`, `Handoff` (as a function tool with transfer semantics).
- `Runner` with input/output/tool guardrails, handoff dispatch, and a hop counter.
- A simple span emitter to demonstrate trace shape.
- A triage agent that hands off to billing or support based on the user query; a guardrail trips on one input.

Run it:

```
python3 code/main.py
```

The trace shows two successful handoffs, one input guardrail trigger, and a span tree mirroring what the real SDK emits.

## Use It

- **OpenAI Agents SDK** for OpenAI-first products.
- **Claude Agent SDK** (Lesson 17) for Claude-first products.
- **LangGraph** (Lesson 13) when you want explicit state and durable recovery.
- **Custom** when you need precise control (voice, multi-vendor, federated deployments).

## Ship It

`outputs/skill-agents-sdk-scaffold.md` scaffolds an Agents SDK application with a triage agent, handoffs, input/output/tool guardrails, session storage, and a trace processor.

## Exercises

1. Add a handoff hop counter: reject after N transfers. Trace the behavior.
2. Implement `nest_handoff_history` as an option — collapse prior messages into a summary before transfer.
3. Write a blocking output guardrail. Compare latency on a prompt that triggers it vs one that passes.
4. Wire `add_trace_processor` to a JSON logger. What shape does it emit per span?
5. Read the SDK docs. Port your standard-library toy to `openai-agents-python`. Where did you model things incorrectly?

## Key Terms

| Term | Common description | What it actually is |
|------|----------------|------------------------|
| Agent | "LLM + instructions" | The Agent type in the SDK; owns tools and handoffs |
| Handoff | "transfer" | A tool the model calls to delegate to another agent |
| Guardrail | "policy check" | Validation on input / output / tool calls |
| Tripwire | "guardrail trigger" | Exception thrown when a guardrail rejects |
| Session | "history store" | Conversation memory persisted across runs |
| Tracing | "spans" | Built-in observability covering LLM + tools + handoffs + guardrails |
| Blocking guardrail | "sequential check" | Guardrail runs first; no wasted tokens on trigger |
| Parallel guardrail | "concurrent check" | Guardrail runs alongside; lower latency, wasted tokens on trigger |

## Further Reading

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — primitives, handoffs, guardrails, tracing
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude-flavored counterpart
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — when to actually use handoffs
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the standard Agents SDK spans map to
