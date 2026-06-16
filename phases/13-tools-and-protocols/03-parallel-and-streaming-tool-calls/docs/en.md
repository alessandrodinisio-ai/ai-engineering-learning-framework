# Parallel Tool Calls & Streaming with Tools

> Three independent weather queries run serially cost three round-trips. Run them in parallel and total latency collapses to the slowest single call. Every frontier provider now emits multiple tool calls in one turn. The gains are real; the plumbing is subtle. This lesson covers two halves: parallel fan-out, and streaming argument reassembly — with the focus on the id-correlation pitfall.

**Type:** Build
**Languages:** Python (standard library, thread pool + streaming scaffolding)
**Prerequisites:** Phase 13 · 02 (Function Calling Deep Dive)
**Time:** ~75 minutes

## Learning Objectives

- Explain why `parallel_tool_calls: true` exists and when to turn it off.
- Correlate streaming argument chunks to the correct tool call id during parallel fan-out.
- Reassemble partial `arguments` strings into complete JSON without parsing early.
- Run a three-city weather benchmark demonstrating serial vs. parallel latency.

## The Problem

Without parallel calls, an agent answering "What's the weather in Bengaluru, Tokyo, and Zurich?" does this:

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

Three LLM round-trips, each paying executor latency. Roughly 4× the ideal wall-clock time.

With parallel calls:

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

One LLM round-trip. Executor time is the max of the three, not the sum. Production benchmarks on OpenAI, Anthropic, and Gemini show 60–70% wall-clock reduction for fan-out workloads.

The cost is correlation complexity. When three calls complete out of order, your results must carry matching `tool_call_id` values for the model to pair them. When results stream back, you must assemble partial argument fragments into complete JSON before executing. Gemini 3 added unique ids partly to solve a real problem: two parallel calls to the same tool were indistinguishable.

## The Concept

### Enabling parallelism

- **OpenAI.** `parallel_tool_calls: true` is the default. Set to `false` to force serial.
- **Anthropic.** Parallel via `disable_parallel_tool_use: false` (default since Claude 3.5). Set to `true` for serial.
- **Gemini.** Always parallel-capable; `tool_config.function_calling_config.mode = "AUTO"` lets the model decide.

Turn off parallelism when tools have ordering dependencies (`create_file` before `write_file`), when one call's output determines another's input, or when rate limiters can't absorb the fan-out.

### Id correlation

Every call the model emits has an `id`. Every result the host returns must carry the same id. Without it, results are ambiguous.

- **OpenAI.** `tool_call_id` on each tool-role message.
- **Anthropic.** `tool_use_id` on each `tool_result` block.
- **Gemini.** `id` on each `functionResponse` (Gemini 3+; Gemini 2 matched by name, which broke for same-name parallel calls).

### Running calls concurrently

The host runs each call's executor in its own thread, coroutine, or remote worker. The simplest scaffolding uses a thread pool; production uses asyncio with `asyncio.gather` or structured concurrency. Completion order is unpredictable — ids are the identifiers.

A common bug: replying results in call-list order rather than completion order. This usually works because the model only cares about `tool_call_id`, but if a result is dropped or duplicated, out-of-order submission makes debugging harder. Prefer completion-order replies with explicit ids.

### Streaming tool calls

When the model streams, `arguments` arrive in fragments. Three parallel calls produce three interleaved streams of chunks on the wire. You need one accumulator per id.

Per-provider shapes:

- **OpenAI.** Each chunk is `choices[0].delta.tool_calls[i].function.arguments` (partial string). Chunks carry `index` (position in call list). You accumulate by index, read the `id` when it first appears, and parse JSON on `finish_reason = "tool_calls"`.
- **Anthropic.** Stream events are `message_start`, then one `content_block_start` of type `tool_use` per block (with id, name, empty input). `content_block_delta` events carry `input_json_delta` chunks. `content_block_stop` closes each block.
- **Gemini.** `streamFunctionCallArguments` (Gemini 3+) emits chunks with `functionCallId`, enabling clean interleaving. Before Gemini 3, streaming returned one complete call at a time.

### Partial JSON and the parse-early trap

You cannot parse `arguments` until they are complete. Partial JSON like `{"city": "Beng` is invalid and will throw. The correct threshold is the provider's call-end signal: OpenAI's `finish_reason = "tool_calls"`, Anthropic's `content_block_stop`, or Gemini's stream-end event. Only then attempt `json.loads`. A more robust approach uses an incremental JSON parser that yields events as structure completes; OpenAI's streaming guide recommends this for UX that shows real-time "thinking" indicators. Counting braces as a completeness test is unreliable (braces inside quoted strings or escaped content cause false positives) and should only be used as an informal debugging heuristic.

### Out-of-order completion

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: medium API, returns third
```

Host reply still must reference the ids:

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

On OpenAI or Anthropic, order in the reply doesn't matter for correctness. Gemini accepts any order as long as ids match.

### Benchmark: serial vs. parallel

The scaffolding in `code/main.py` simulates three executors with latencies of 400, 600, and 800 ms. Serial totals 1800 ms. Parallel runs in max(400, 600, 800) = 800 ms. The gap is constant, not proportional, so time saved grows with tool count.

Real-world caveat: parallel calls pressure downstream APIs. A 10-way fan-out to a rate-limited service will fail. Phase 13 · 17 covers gateway-level backpressure; retry semantics are planned for a later phase.

### Wall-clock time for streaming fan-out

If the model itself streams, you can begin executing a call as soon as its arguments are complete, without waiting for all calls to be finalized. This is a documented optimization from OpenAI but not all SDKs expose it. This lesson's scaffolding does: the simulated stream starts a call the moment it yields a complete argument object.

## Use It

`code/main.py` has two halves. The first runs three simulated weather calls serial and parallel using `concurrent.futures.ThreadPoolExecutor` and prints wall-clock times. The second replays a fake streaming response — three parallel calls with interleaved `arguments` chunks on a single stream — and reassembles them with a `StreamAccumulator` keyed by id. No LLM, no network, just reassembly logic.

What to look for:

- Serial timer hits 1.8 s. Parallel timer hits 0.8 s on the same fake latencies.
- The accumulator handles out-of-order chunks by buffering per id and parsing only when each call's JSON is complete.
- Executors start as soon as an id's arguments are finalized, rather than waiting for the entire stream to end.

## Ship It

This lesson produces `outputs/skill-parallel-call-safety-check.md`. Given a tool registry, this skill audits which tools can safely run in parallel, which have ordering dependencies, and which would overwhelm downstream rate limits — returning a revised registry with a per-tool `parallel_safe` flag.

## Exercises

1. Run `code/main.py` and vary the simulated latencies. Confirm the parallel-to-serial ratio is approximately `max/sum` (real runs deviate slightly from the ideal due to thread scheduling, serialization, and scaffolding overhead). At what latency distribution does parallelism stop mattering?

2. Extend the accumulator to handle a "call cancelled mid-stream" case: drop its buffer and emit a `cancelled` event. Which provider explicitly documents this scenario? Check Anthropic's `content_block_stop` semantics and OpenAI's `finish_reason: "length"` behavior.

3. Replace the thread pool with `asyncio.gather`. Benchmark both. You should see async pull slightly ahead due to lower context-switch cost, but only when executors do real I/O.

4. Pick two tools that must not run in parallel (e.g., `create_file` then `write_file`). Add an `ordering_dependency` graph to the registry and gate parallel fan-out on that graph. This is the minimal mechanism for dependency-aware scheduling, formalized in a later agent-engineering phase.

5. Read OpenAI's parallel-function-calling section and Anthropic's `disable_parallel_tool_use` docs. Identify the one real-world tool type where Anthropic recommends disabling parallelism. (Hint: consequential mutations to the same resource.)

## Key Terms

| Term | Common shorthand | What it actually is |
|------|----------------|------------------------|
| Parallel tool calls | "fan-out in one turn" | Model emits multiple tool calls in a single assistant message |
| `parallel_tool_calls` | "OpenAI's flag" | Enables or disables multi-call emission |
| `disable_parallel_tool_use` | "Anthropic's inverse flag" | Opt-out flag; default is parallel enabled |
| Tool call id | "correlation handle" | Per-call identifier that result messages must echo back |
| Accumulator | "stream buffer" | Per-id string buffer for partial `arguments` chunks |
| Out-of-order completion | "fastest first" | Parallel calls complete in unpredictable order; ids are the glue |
| Dependency graph | "ordering constraint" | Tools whose output feeds other tool inputs; cannot parallelize |
| Parse-early trap | "JSON.parse blew up" | Attempting to parse an incomplete `arguments` string |
| `streamFunctionCallArguments` | "Gemini 3 feature" | Streaming argument chunks with unique id per call |
| Completion-order reply | "don't wait for all" | Reply with results as they arrive, keyed by id |

## Further Reading

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — Default behavior and opt-out flag
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` and result batching
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — Id-correlated parallel calls from Gemini 3
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — Chunk argument reassembly for OpenAI streams
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — `content_block_delta` with `input_json_delta`
