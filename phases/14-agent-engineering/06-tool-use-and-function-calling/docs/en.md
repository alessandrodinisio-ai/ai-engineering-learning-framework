# Tool Use and Function Calling

> Toolformer (Schick et al., 2023) pioneered self-supervised tool annotation. The Berkeley Function Calling Leaderboard V4 (Patil et al., 2025) set the 2026 benchmark: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination. Single-turn is solved. Memory, dynamic decision-making, and long-horizon tool chains are not.

**Type:** Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 minutes

## Learning Objectives

- Explain Toolformer's self-supervised training signal: retain a tool annotation only when execution reduces the next-token loss.
- Name the five evaluation categories in BFCL V4 and what each measures.
- Implement a tool registry with schema validation, argument coercion, and execution sandboxing using the standard library.
- Diagnose the three open problems of 2026: long-horizon tool chains, dynamic decision-making, and memory.

## The Problem

Early tool use asked: can a model predict a single correct function call? Modern tool use asks: can a model chain tools across 40 steps — with memory, partial observability, recovery from tool failures, and without hallucinating nonexistent tools?

Toolformer established the baseline: models can learn when to invoke tools through self-supervision. BFCL V4 defines the 2026 evaluation target. The gap between the two is exactly the space where production agents live.

## The Concept

### Toolformer (Schick et al., NeurIPS 2023)

The idea: let the model annotate its own pretraining corpus with candidate API calls. For each candidate, execute it. Retain the annotation only when "including the tool result" reduces the next-token loss. Fine-tune on the filtered corpus.

Tools covered: calculator, QA system, search engine, translator, calendar. The self-supervised signal is purely about whether the tool helps predict text — no human annotation.

Scaling results: tool-use ability emerges with scale. Small models are hurt by tool annotations; large models benefit. This is why frontier models in 2026 ship with strong tool-use capabilities, while most 7B models need explicit tool-use fine-tuning to be reliable.

### Berkeley Function Calling Leaderboard V4 (Patil et al., ICML 2025)

BFCL is the de facto evaluation standard for 2026. V4 composition:

- **Agentic (40%)** — Full agent trajectories: memory, multi-turn, dynamic decision-making.
- **Multi-Turn (30%)** — Interactive conversations with tool chains.
- **Live (10%)** — User-submitted real prompts (harder distribution).
- **Non-Live (10%)** — Synthetic test cases.
- **Hallucination (10%)** — Detecting when no tool should be called.

V3 introduced state-based evaluation: after a tool sequence, check the actual API state (e.g., "was the file created?") rather than matching tool-call ASTs. V4 added web search, memory, and format-sensitivity categories.

Key 2026 findings: single-turn function calling is nearly solved. Failures concentrate in memory (carrying context across turns), dynamic decision-making (choosing tools based on prior results), long-horizon chains (drift after 20+ steps), and hallucination detection (refusing to call when no suitable tool exists).

### Tool Schema

Every vendor has a schema format. They differ in details but share the same shape:

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic uses `input_schema` directly. OpenAI uses `function.parameters`. Both accept JSON Schema. The description is load-bearing — the model reads it to pick the right tool. Poor tool descriptions are the number-one root cause of "picked the wrong tool" failures.

### Argument Validation

Trust no tool call. Validate:

1. **Type coercion.** Schema says int, model might return string "5". Convert when unambiguous; reject when ambiguous.
2. **Enum validation.** If the schema says `status in {"open", "closed"}` and the model emits `"in_progress"`, reject with a descriptive error.
3. **Required fields.** Missing required field -> immediately feed an error observation back to the model instead of crashing.
4. **Format validation.** Dates, emails, URLs — validate with specific parsers, not regex.

Every validation failure should return a structured observation so the model can retry with the correct shape.

### Parallel Tool Calls

Modern vendors support parallel tool calls in a single assistant turn. The loop:

1. Model emits 3 tool calls, each with a different `tool_use_id`.
2. Runtime executes them (in parallel if mutually independent).
3. Each result is returned as a `tool_result` block, correlated by `tool_use_id`.

Engineering rule: treat correlation IDs as load-bearing. Mix them up and you get "wrong tool matched to wrong result" routing bugs.

### Sandboxing

Tool execution is the sandbox boundary. See Lesson 09 for details. In short: every tool should declare its read/write surface, network access, timeout, and memory cap. A generic `run_shell(cmd)` is a red flag; a specific `git_status()` is safer.

## Build It

`code/main.py` implements a production-shaped tool registry:

- JSON Schema subset validator (standard library only).
- Tool registration with description, input schema, timeout, and executor.
- Argument coercion and enum validation.
- Parallel tool dispatch with correlation IDs.
- Error observations as structured strings.

Run it:

```
python3 code/main.py
```

The trace shows a mini agent calling three tools in one turn, with one deliberately malformed call rejected by a descriptive error the model can act on.

## Use It

Every vendor has its own tool schema — Anthropic, OpenAI, Gemini, Bedrock. If you need multi-vendor support, use a conversion layer (OpenAI Agents SDK, Vercel AI SDK, LangChain tool adapters). BFCL is the reference benchmark — if tool use is core to your product, run your agent against it before shipping.

## Ship It

`outputs/skill-tool-registry.md` generates a tool catalog, schema, and registry for a given task domain. Includes a description quality check (does each tool's description tell the model when to use it?).

## Exercises

1. Add a "no-op" tool that lets the model explicitly refuse to use any other tool. Measure on a BFCL-style hallucination test.
2. Implement argument coercion for "int-as-string" and "float-as-string." At what point does coercion start masking real bugs?
3. Add a per-tool timeout and a circuit breaker (refuse the tool for 60 seconds after 3 consecutive failures). How does this change model recovery behavior?
4. Read the BFCL V4 description. Pick one category (e.g., "multi-turn") and run 10 example prompts through your agent. Report the pass rate.
5. Port the standard-library validator to Pydantic or Zod. What does Pydantic/Zod catch that this toy misses?

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Function calling | "tool use" | Structured-output tool invocation with validated schema |
| Toolformer | "self-supervised tool annotation" | Schick 2023 — retain tool calls whose results reduce next-token loss |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 benchmark: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination |
| Tool schema | "function signature for the model" | name, description, JSON Schema for parameters |
| tool_use_id | "correlation ID" | Binds a tool call to its result; essential for parallel dispatch |
| Hallucination detection | "knowing when not to call" | V4 category: refusing to call when no suitable tool exists |
| Argument coercion | "string-to-int fix" | Narrow fix for predictable schema mismatches; reject when ambiguous |
| Sandboxing | "tool execution boundary" | Per-tool read/write surface, network, timeout, memory cap |

## Further Reading

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — Self-supervised tool annotation
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — 2026 evaluation benchmark
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) — Production tool schema in the Claude Agent SDK
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — Function tool types and Guardrails
