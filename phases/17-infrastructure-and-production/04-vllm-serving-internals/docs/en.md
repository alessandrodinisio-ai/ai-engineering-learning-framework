# vLLM Serving Internals: PagedAttention, Continuous Batching, Chunked Prefill

> vLLM's dominance in 2026 rests on three defaults that compound, not any single trick. PagedAttention is always on. Continuous batching injects new requests into the active batch between decode iterations. Chunked prefill slices long prompts so decode tokens never starve. With all three enabled, Llama 3.3 70B FP8 on a single H100 SXM5 at 128 concurrency reaches 2,200-2,400 tok/s — approximately 25% above vLLM's own defaults and 3-4x a naive PyTorch loop. This lesson reads the scheduler and attention kernel to the point where you can draw the diagram, and closes with a toy continuous batcher in `code/main.py` that schedules prefill and decode the way vLLM does.

**Type:** Learn
**Languages:** Python (standard library, a toy-level continuous batching scheduler)
**Prerequisites:** Phase 17 · 01 (Model Serving), Phase 11 (LLM Engineering)
**Time:** ~75 minutes

## Learning Objectives

- Explain PagedAttention as a KV cache allocator: blocks, block tables, and why fragmentation stays below 4% under production load.
- Diagram continuous batching at the iteration level: how finished sequences leave the batch and new ones join without draining.
- Describe chunked prefill in one sentence and name which latency metric it protects (hint: P99 TTFT tail, not average throughput).
- Name the 2026 vLLM v0.18.0 gotcha that bites teams who enable all optimizations at once.

## The Problem

A naive PyTorch serving loop processes one request at a time: tokenize, prefill, decode to EOS, return. Fine for one user. For a hundred users, it's a polite queue. The obvious fix — static batching — pads every request to the longest prompt in the window, pads every decode to the longest expected output, and the entire batch is held hostage by the slowest sequence. You pay for padding that's never used, and fast requests wait for slow ones.

vLLM solves three problems at once. PagedAttention prevents KV cache fragmentation from eating 60-80% of GPU memory the way classic contiguous allocation does. Continuous batching lets requests join and leave the batch between every decode iteration, so the batch is always full of real work. Chunked prefill splits a 32k-token prompt into ~512-token slices interleaved with decode, so one long prompt doesn't freeze every decode token on the GPU.

The 2026 production default is all three enabled. You need to understand each because the failure modes are all in the scheduler, not the model.

## The Concept

### PagedAttention as a Virtual Memory System

Each sequence's KV cache is `num_layers x 2 x num_heads x head_dim x seq_len x bytes_per_element`. For Llama 3.3 70B at 8192 tokens, that's approximately 1.25 GB per sequence in BF16. If you reserve 8192 slots per request but the average request uses 1500 tokens, you waste approximately 82% of reserved HBM. Classic batching pays for this waste.

PagedAttention borrows the OS virtual memory idea. The KV cache is no longer contiguous per sequence. It's allocated in fixed-size blocks (default 16 tokens). Each sequence has a block table that maps its logical token positions to physical block IDs. When a sequence grows beyond its allocated blocks, another block is added. When it finishes, blocks are returned to the pool.

Fragmentation drops from 60-80% (classic) to below 4% (PagedAttention). You don't enable PagedAttention with a flag — it's the only allocator vLLM ships. The knob is `--gpu-memory-utilization` (default 0.9), which tells vLLM how much HBM to reserve for KV blocks after loading weights and activations.

### Continuous Batching at the Iteration Level

Old "dynamic batching" waits a window (say 10 ms) to fill a batch, then runs prefill + decode + decode + decode until every sequence finishes. Fast sequences that finish early sit idle while the GPU completes the slow ones.

Continuous batching operates between every decode step. Call the set of running sequences the `RUNNING` list. At each iteration:

1. Any sequence in `RUNNING` that just hit EOS or max_tokens is removed.
2. The scheduler checks the waiting queue. If free KV blocks are available, it admits new sequences (prefill or resume).
3. The forward pass runs on whatever is in `RUNNING` at that moment, producing one new token per sequence.

Batch size is never padded to a fixed number. Sequences at different points in their respective outputs share a single fused forward. In 2026 vLLM this is called the `V1 scheduler`. The key invariant: the scheduler runs once per decode iteration, not once per request.

### Chunked Prefill Protects the TTFT Tail

Prefill is compute-bound. A 32k-token prompt on Llama 3.3 70B takes approximately 800 ms of pure prefill on one H100. While prefill is running, every other sequence in the batch has its decode token waiting. In a serving loop, one long prompt's time-to-first-token (TTFT) becomes inter-token latency (ITL) jitter for dozens of other users.

Chunked prefill splits the prefill into fixed-size chunks (default 512 tokens) and schedules each chunk as a unit. Between chunks, the scheduler can advance decode sequences by one token. You trade a small amount of absolute prefill latency (a few ms per chunk boundary) for dramatically lower decode jitter. In published benchmarks, P99 ITL drops from approximately 50 ms to approximately 15 ms under mixed workloads.

### The Three Defaults Interact

These three features are preconditions for each other. PagedAttention gives the scheduler a fine-grained KV budget to reason about. Continuous batching needs that fine-grained budget so admitting a new sequence doesn't force a global repack. Chunked prefill is a decision the scheduler makes on the same `RUNNING` list — it's one more scheduling policy, not a separate system.

You don't need to know every flag. You need to know what the scheduler optimizes: goodput subject to KV block budget, constrained by chunked prefill slicing.

### The 2026 v0.18.0 Gotcha

In vLLM v0.18.0, you cannot combine `--enable-chunked-prefill` with draft-model speculative decoding (`--speculative-model`). The documented exception is N-gram GPU speculative decoding in the V1 scheduler. Teams that flip on all flags without reading release notes get a runtime error at startup, not a silent fallback. If your speculative gain is worth enabling chunked prefill alongside it, reconsider the choice — the correct answer in 2026 is often EAGLE-3 without chunked prefill rather than a "draft model + chunked prefill" combination that won't compile.

### Numbers You Should Remember

- Llama 3.3 70B FP8, H100 SXM5, 128 concurrency, all three enabled: 2,200-2,400 tok/s.
- Same model, default vLLM (no chunked prefill): ~1,800 tok/s.
- Same model, naive PyTorch forward loop: ~600 tok/s.
- PagedAttention KV fragmentation waste under production load: <4%.
- P99 ITL under mixed workloads: ~15 ms with chunked prefill, ~50 ms without.

### What the Scheduler Looks Like

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # Schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` is this loop in standard-library Python with fake token counts and fake forward latencies. Run it to see how chunked prefill keeps decode sequences alive during a long prefill.

## Use It

`code/main.py` simulates a vLLM-style scheduler with toggleable features. Run it to see:

- `NAIVE` mode: one request at a time, no batching.
- `STATIC` mode: pad and wait, classic batching.
- `CONTINUOUS` mode: iteration-level admission and release.
- `CONTINUOUS + CHUNKED` mode: prefill slicing interleaved with decode.

Output shows total throughput (tokens per virtual second), mean TTFT, and P99 ITL. The `CONTINUOUS + CHUNKED` line should dominate on mixed traffic.

## Ship It

This lesson produces `outputs/skill-vllm-scheduler-reader.md`. Given a serving configuration (batch size, KV memory utilization, chunked prefill size, speculative config), it produces a scheduler diagnostic — identifying which of the three defaults is the bottleneck and what to tune.

## Exercises

1. Run `code/main.py`. Compare `STATIC` and `CONTINUOUS` on a mixed workload of long and short requests. Where does the throughput gap come from — prefill efficiency, decode efficiency, or tail latency?
2. Modify the toy scheduler to add `--max-num-batched-tokens`. For an H100 running Llama 3.3 70B FP8, what's the correct value? (Hint: it's a function of KV block size and free block count, not raw HBM.)
3. Re-read the vLLM v0.18.0 release notes. Which flag combinations are mutually exclusive? List them.
4. For a trace of 1,000 requests with average 1,500 output tokens and standard deviation 600 tokens, calculate KV cache fragmentation waste under (a) per-request contiguous allocation with max 8192, and (b) PagedAttention with 16-token blocks.
5. Explain in one paragraph why chunked prefill in isolation helps P99 ITL rather than throughput. Where does the practical throughput gain come from?

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| PagedAttention | "The KV trick" | Fixed-size block allocator for KV cache; fragmentation <4% |
| Block table | "Page table" | Per-sequence mapping of logical token positions to physical KV blocks |
| Continuous batching | "Dynamic batching, done right" | Admission/release decisions made every decode iteration |
| Chunked prefill | "Prefill slicing" | Splits long prefills into 512-token slices interleaved with decode |
| TTFT | "Time to first token" | Prefill + queue + network; dominated by prefill for long prompts |
| ITL | "Inter-token latency" | Time between consecutive decode tokens; dominated by batch size |
| Goodput | "SLO-meeting throughput" | Tokens per second where every request still hits TTFT and ITL targets |
| V1 scheduler | "The new scheduler" | vLLM's 2026 scheduler; N-gram spec decode is the chunked-prefill-compatible path |
| `--gpu-memory-utilization` | "The memory knob" | Fraction of HBM reserved for KV blocks after loading weights and activations |

## Further Reading

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — Authoritative source on chunked prefill and speculative decoding compatibility in V1.
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 release cadence and version-specific behavior.
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — The original post that still defines how to think about the allocator.
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — Fragmentation analysis and scheduler design.
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) — V1 scheduler walkthrough with flame graphs.
