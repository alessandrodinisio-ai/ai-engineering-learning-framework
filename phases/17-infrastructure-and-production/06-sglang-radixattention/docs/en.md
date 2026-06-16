# SGLang and RadixAttention for Prefix-Heavy Workloads

> SGLang treats the KV cache as a first-class, reusable resource stored in a radix tree. vLLM schedules requests FCFS (first-come-first-served), while SGLang's cache-aware scheduler prioritizes requests that share longer prefixes — essentially a depth-first radix traversal that keeps hot branches resident in HBM. On Llama 3.1 8B with ShareGPT-style 1K prompts, SGLang achieves ~16,200 tok/s versus vLLM's ~12,500, roughly a 29% advantage. On prefix-heavy RAG workloads, the advantage reaches 6.4x. On voice-cloning-style workloads, cache hit rates exceed 86%. Deployed in 2026 across 400,000+ GPUs at xAI, LinkedIn, Cursor, Oracle, GCP, Azure, and AWS. The pitfall: when prefix ordering is inconsistent, that 6.4x number evaporates — ordering is the lever in the engineer's hand.

**Type:** Learn
**Languages:** Python (standard library, a toy radix tree cache + cache-aware scheduler)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 14 (Agentic RAG)
**Time:** ~75 minutes

## Learning Objectives

- Draw RadixAttention: how prefixes are stored in a radix tree, and how KV blocks are shared among sequences hanging off the same branch.
- Explain cache-aware scheduling, and why FCFS is wrong for prefix-heavy traffic.
- Given a prefix cache hit rate and prompt length distribution, calculate the expected speedup for a workload.
- State the prompt ordering discipline that makes the 6.4x number real rather than wasted.

## The Problem

Classic serving treats each request's prompt as opaque. Even if 5,000 RAG requests all start with the same 2,000-token system prompt plus the same retrieval preamble, vLLM will prefill that 2,000-token prefix 5,000 times. The GPU does the same work over and over.

The observation: prompts in agentic and RAG workloads almost always share long prefixes. System prompts, tool schemas, few-shot examples, retrieval headers, conversation history — all repeat across requests. If you store that prefix's KV cache once and reuse it, you never prefill it again.

RadixAttention does exactly this. Tokens are indexed into a radix tree; each node owns the KV blocks for the token sequence from root to that node. A new request walks the tree: any node whose tokens match reuses that node's KV blocks. Prefill cost becomes proportional to the "new" suffix, not the entire prompt.

The challenge is scheduling. If two requests share a 2,000-token prefix and a third shares only 200 tokens of that prefix, you want to serve the two long-sharing requests together so the long prefix stays in HBM. FCFS does the opposite — it serves whichever arrived first, potentially evicting the hot branch before the next long-prefix request arrives.

## The Concept

### Radix tree as KV index

A radix tree (compact trie) stores token sequences. Each node owns a token span and the KV blocks computed for that span. Child nodes extend the sequence by one or more tokens.

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

A new request arrives with system prompt + "Context: <doc A>" + "Question: Carol". The scheduler walks the tree: system prefix matches (reuse 124 blocks), doc-A branch matches (reuse 31 blocks), then allocates new blocks only for "Question: Carol" (4 blocks). Prefill cost: 4 blocks of new tokens. Without the tree: 160 blocks. Roughly a 40x saving on prefill.

### Cache-aware scheduling

Radix-tree-backed reuse is pointless if the cache thrashes. Two key strategies:

1. **Depth-first dispatch.** When picking the next request from the queue, prefer requests hanging off the same branch as the currently running set. This pins hot branches.
2. **Branch-level LRU, not block-level.** Evict entire branches (starting from the least recently used leaf) rather than individual blocks, keeping cache shape aligned with radix shape.

FCFS violates both. A request sharing 2,000 tokens sits behind one sharing only 50 tokens, and then the 2,000-token branch gets evicted to admit the 50-token one.

### Benchmark numbers to memorize

- Llama 3.1 8B, H100, ShareGPT 1K prompts: SGLang ~16,200 tok/s vs vLLM ~12,500 (~29% advantage).
- Prefix-heavy RAG (same system + same docs, different questions): up to 6.4x on SGLang.
- Voice-cloning workload: 86.4% prefix cache hit rate.
- Production hit rates across SGLang customers: 50-99%, depending on prompt discipline.
- Deployed on 400,000+ GPUs in 2026.

### The ordering pitfall

That 6.4x number depends on consistent prompt template ordering. If your client assembles prompts as `[system, tools, context, history, question]` in some requests and `[system, context, tools, history, question]` in others, the tree cannot find a shared prefix. What looks like a shared prefix to a human is two different sequences to the radix tree.

The lever in the engineer's hand: your prompt template is a cache key. Fix the order. Put all invariant content (system, tools, schema) first. Put retrieval context next. Put the user question last. Do not interleave dynamic content into the prefix.

A real case from research: moving dynamic content out of the cacheable prefix took one deployment from 7% to 74% cache hit rate in a single change.

### Where RadixAttention wins and loses

Wins:
- RAG (same retrieval preamble, different questions).
- Agents (same tool schema, different queries).
- Chat with long system prompts.
- Voice/vision workloads with repeated preambles.

Loses (falls back to vLLM-level throughput):
- One-shot generation with unique prompts (code completion, open-ended chat without system prompts).
- Dynamic prompts that interleave unique content into the prefix on every request.

### Why this is a scheduling problem, not just a kernel problem

You could implement KV reuse as a kernel trick. SGLang's insight is that reuse only pays off when the scheduler keeps hot branches resident. A naive "reuse if available" strategy thrashes the cache under mixed load. The radix-tree-indexed scheduler is what turns the kernel trick into a 29% production advantage.

### Interaction with vLLM

These two systems are not strict competitors. In 2026 vLLM added prefix caching (`--enable-prefix-caching`) and a cache-aware router (vLLM Router, written in Rust). The gap has narrowed but not disappeared — SGLang's entire stack is radix-first; vLLM's is grafted on. For workloads dominated by prefix reuse, SGLang remains the default. For general-purpose serving without strong prefix patterns, vLLM remains on par or better.

## Use It

`code/main.py` implements a toy radix tree KV cache plus a scheduler with two strategies: FCFS and cache-aware. It runs the same workload twice, reports prefix cache hit rates and throughput differences. Then it runs a "shuffled order" workload to show how 6.4x collapses.

## Ship It

This lesson produces `outputs/skill-radix-scheduler-advisor.md`. Given a workload description (prompt template shape, retrieval patterns, concurrent tenant count), it produces a prompt ordering prescription and a go/no-go on adopting SGLang.

## Exercises

1. Run `code/main.py`. Compare FCFS vs cache-aware on the same workload. Where does the gap come from — prefill savings, decode savings, or queue latency?
2. Modify the workload so prompts randomly shuffle `[system, tools, context]`. Re-run. What happened to the hit rate? Why?
3. Calculate the HBM cost of pinning a 2,000-token system prompt as a single radix branch on Llama 3.1 8B. Compare against the cost of a 16-sequence batch without prefix reuse.
4. Read the SGLang RadixAttention paper. Explain in three sentences why tree-level LRU eviction beats block-level LRU under prefix-heavy load.
5. A customer reports only 8% cache hit rate. Name three possible causes and the diagnostic you would run for each.

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| RadixAttention | "the SGLang thing" | Indexes KV cache into a radix tree so shared prefixes reuse blocks |
| Radix tree | "compact trie" | A tree where each node owns a token span and its KV blocks |
| Cache-aware scheduler | "hot branch first" | Scheduler that prefers requests sharing resident branches |
| Prefix cache hit rate | "how much of your prompt is free" | Fraction of prompt tokens served by reused KV blocks |
| FCFS | "first-come-first-served" | Default scheduling that destroys prefix locality |
| Branch-level LRU | "evict leaves" | Eviction strategy that matches radix shape |
| Prompt template ordering | "cache key" | The order of prompt sections determines what the tree can share |
| System prompt pinning | "resident prefix" | Pinning invariant system sections to avoid eviction churn |

## Further Reading

- [SGLang GitHub](https://github.com/sgl-project/sglang) — source code and documentation.
- [SGLang documentation](https://sgl-project.github.io/) — RadixAttention and scheduling details.
- [SGLang paper — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — design reference.
- [LMSYS blog — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — benchmark numbers and scheduler rationale.
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) — vLLM's own radix-like implementation, for comparison.
