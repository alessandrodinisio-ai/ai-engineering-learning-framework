# Structured Output — JSON Schema, Pydantic, Zod, Constrained Decoding

> "Politely asking the model to return JSON" fails 5–15% of the time, even on frontier models. Structured output bridges the gap with constrained decoding: the model is literally prevented from emitting any token that would violate the schema. OpenAI's strict mode, Anthropic's schema-typed tool calls, Gemini's `responseSchema`, Pydantic AI's `output_type`, and Zod's `.parse` are five surface forms of the same idea. This lesson builds the schema validator and strict-mode contracts that every subsequent production extraction pipeline will use.

**Type:** Build
**Languages:** Python (standard library, JSON Schema 2020-12 subset)
**Prerequisites:** Phase 13 · 02 (Function Calling Deep Dive)
**Time:** ~75 minutes

## Learning Objectives

- Write a JSON Schema 2020-12 for an extraction target using the right constraints (enum, min/max, required, pattern).
- Explain why strict mode and constrained decoding give a different guarantee than "generate then validate."
- Distinguish three failure modes: parse error, schema violation, model refusal.
- Ship an extraction pipeline with typed repair and typed refusal handling.

## The Problem

An agent reading purchase-order emails needs to turn free text into `{customer, line_items, total_usd}`. Three approaches.

**Approach 1: prompt for JSON.** "Reply in JSON with fields customer, line_items, total_usd." Works 85–95% of the time on frontier models. Fails in six ways: missing braces, trailing commas, type errors, hallucinated fields, truncation at token limit, prose leaking ("Here is your JSON:").

**Approach 2: generate then validate.** Generate freely, parse, validate against schema, retry on failure. Reliable but expensive — each retry costs a call, and truncation bugs cost an extra round every time they hit.

**Approach 3: constrained decoding.** The provider enforces the schema during decoding. Invalid tokens are masked out of the sampling distribution. Output is guaranteed to parse and guaranteed to pass validation. Failure collapses to a single mode: refusal (model judges the input cannot fit the schema).

Every frontier provider ships some form of Approach 3 as of 2026.

- **OpenAI.** `response_format: {type: "json_schema", strict: true}`, plus `refusal` in the response if the model declines.
- **Anthropic.** Schema enforcement on `tool_use` inputs; `stop_reason: "refusal"` isn't a thing, but `end_turn` with no tool call is the signal.
- **Gemini.** Request-level `responseSchema`; in 2026 Gemini ships token-level grammar constraints for selected types.
- **Pydantic AI.** `output_type=InvoiceModel` emits a structured `RunResult` typed to `InvoiceModel`.
- **Zod (TypeScript).** Runtime parser that validates provider output against a Zod schema; pairs with OpenAI's `beta.chat.completions.parse`.

Common thread: declare the schema once, enforce it end to end.

## The Concept

### JSON Schema 2020-12 — the lingua franca

Every provider accepts JSON Schema 2020-12. The constructs you'll use most:

- `type`: one of `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`.
- `properties`: field-name to sub-schema map.
- `required`: list of field names that must be present.
- `enum`: closed set of allowed values.
- `minimum` / `maximum` (numbers), `minLength` / `maxLength` / `pattern` (strings).
- `items`: sub-schema applied to every array element.
- `additionalProperties`: `false` forbids extra fields (default varies by mode).

OpenAI strict mode adds three requirements: every property must be in `required`, `additionalProperties: false` everywhere, and no unresolved `$ref`. Break these and the API returns 400 at request time.

### Pydantic, the Python binding

Pydantic v2 generates JSON Schema from dataclass-shaped models via `model_json_schema()`. Pydantic AI wraps it so you write:

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

and the agent framework translates the schema at the edges to OpenAI strict mode, Anthropic `input_schema`, or Gemini `responseSchema`. The model's output comes back as a typed `Invoice` instance. Validation errors raise `ValidationError` with a typed error path.

### Zod, the TypeScript binding

Zod (`z.object({customer: z.string(), ...})`) is the TS counterpart. OpenAI's Node SDK exposes `zodResponseFormat(Invoice)` which translates to the API's JSON Schema payload.

### Refusal

Strict mode cannot force the model to answer. If the input can't fit the schema ("this email is a poem, not an invoice"), the model emits a `refusal` field with a reason. Your code must handle it as a first-class result, not a failure. Refusal is also useful as a safety signal: a model asked to extract credit-card numbers from a protected-content email returns a refusal with a safety reason.

### Open-weight constrained decoding

Open-source-weight implementations use three techniques.

1. **Grammar-based decoding** (`outlines`, `guidance`, `lm-format-enforcer`): builds a deterministic finite automaton from the schema; masks logits of tokens that would violate the FSM at each step.
2. **Logit masking with JSON parser**: runs a streaming JSON parser in lockstep with the model; computes valid next-token set at each step.
3. **Speculative decoding with validator**: cheap draft model proposes tokens, validator enforces schema.

Commercial providers pick one under the hood. 2026 state-of-the-art is faster than unconstrained generation for short structured outputs and roughly on par for long ones.

### Three failure modes

1. **Parse error.** Output is not valid JSON. Impossible under strict mode. Still happens on non-strict providers.
2. **Schema violation.** Output parses but violates the schema. Impossible under strict mode. Common outside it.
3. **Refusal.** Model declines. Must be handled as a typed result.

### Retry strategy

When you are outside strict mode (Anthropic tool calls, non-strict OpenAI, older Gemini), the recovery pattern is:

```
generate -> parse -> validate -> if failed, inject error and retry up to 3 times
```

One retry usually suffices. Three retries cover occasional fumbles from weaker models. Beyond three is a signal that the schema is bad: the model cannot satisfy it for some inputs, and the prompt or schema needs revision.

### Small-model support

Constrained decoding works on small models too. A 3B-parameter open-weight model with grammar enforcement beats a 70B model with bare prompting on structured tasks. This is the primary reason structured output matters for production: it decouples reliability from model size.

## Use It

`code/main.py` delivers a minimal JSON Schema 2020-12 validator (types, required, enum, min/max, pattern, items, additionalProperties) using the standard library. It wraps an `Invoice` schema, runs a fake LLM output through the validator, and demonstrates the parse-error, schema-violation, and refusal paths. In production, swap the fake output for a real provider response.

What to look for:

- The validator returns a typed `[ValidationError]` list with path and message. This is exactly the shape you want to throw into a retry prompt.
- The refusal branch does not retry. It logs and returns a typed refusal. Phase 14 · 09 uses refusal as a safety signal.
- The `additionalProperties: false` check fires on an adversarial test input, showing why strict mode closes the door on hallucinated fields.

## Ship It

This lesson produces `outputs/skill-structured-output-designer.md`. Given a free-text extraction target (invoice, ticket, resume, etc.), this skill produces a strict-mode-compatible JSON Schema 2020-12 plus a mirroring Pydantic model, with typed refusal and retry handling stubbed out.

## Exercises

1. Run `code/main.py`. Add a fourth test case whose `total_usd` is negative. Confirm the validator rejects it with a `minimum` constraint path.

2. Extend the validator to support `oneOf` with a discriminator field. Common case: a `line_item` is either a product or a service, discriminated by `kind`. Strict mode has subtle rules here; consult OpenAI's structured outputs guide.

3. Write the same Invoice schema as a Pydantic BaseModel and compare `model_json_schema()` output against your hand-rolled schema. Spot the one field Pydantic sets by default that the hand-rolled version misses.

4. Measure refusal rate. Construct ten inputs that should not be extractable (a song lyric, a math proof, an empty email) and run them through a real provider with strict mode. Count refusals vs. hallucinated outputs. This is your ground truth for refusal-aware retry.

5. Read OpenAI's structured outputs guide end to end. Identify one construct it explicitly forbids in strict mode that plain JSON Schema allows. Then design a schema that non-necessarily uses that forbidden construct and refactor it to be strict-mode-compatible.

## Key Terms

| Term | Common shorthand | What it actually is |
|------|----------------|------------------------|
| JSON Schema 2020-12 | "the schema spec" | IETF draft schema dialect spoken by every modern provider |
| Strict mode | "guaranteed schema" | OpenAI's flag that enforces schema via constrained decoding |
| Constrained decoding | "logit masking" | Decode-time enforcement that masks invalid next-tokens |
| Refusal | "model declines" | Typed result when input cannot fit the schema |
| Parse error | "invalid JSON" | Output didn't parse as JSON; impossible under strict mode |
| Schema violation | "wrong shape" | Parsed but violates types / required / enum / range |
| `additionalProperties: false` | "no extras allowed" | Forbids unknown fields; required in OpenAI strict mode |
| Pydantic BaseModel | "typed output" | Python class that emits and validates JSON Schema |
| Zod schema | "TypeScript output type" | TS runtime schema for provider output validation |
| Grammar enforcement | "open-weight constrained decoding" | FSM-based logit masking as in outlines / guidance |

## Further Reading

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — Strict mode, refusal, and schema requirements
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — August 2024 launch post explaining decoding guarantees
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — Typed output_type binding serialized to each provider
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — Authoritative specification
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — Enterprise deployment notes and strict-mode caveats
