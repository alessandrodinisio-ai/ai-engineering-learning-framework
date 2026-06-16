# Long Video Understanding at Million-Token Context

> A 1-hour, 24 FPS, 4K video, once patched and embedded, produces on the order of 60 million tokens. A 2-hour podcast transcript is 30k tokens. A full Blu-ray movie, even with aggressive pooling, is hundreds of thousands of tokens. Google's Gemini 1.5 (March 2024) opened this era with a 10-million-token context, enabling reliable needle-in-a-haystack recall over hour-long video. LWM (Liu et al., February 2024) demonstrated the ring attention scaling path. LongVILA and Video-XL further scaled ingestion. VideoAgent swapped raw context for agentic retrieval. Each path trades off compute, recall, and engineering complexity differently. This lesson reads them side by side.

**Type:** Build
**Languages:** Python (stdlib, needle-in-a-haystack simulator + agentic retrieval router)
**Prerequisites:** Phase 12 · 17 (video temporal tokens)
**Time:** ~180 minutes

## Learning Objectives

- Calculate total visual token count for long videos under different FPS and pooling settings.
- Explain three scaling paths: brute-force context (Gemini 1.5), ring attention (LWM), token compression (LongVILA / Video-XL).
- Compare raw-context video VLMs against agentic-retrieval video VLMs (VideoAgent) on accuracy and latency.
- Design a needle-in-a-haystack test for a 30-minute video and measure recall at a specific minute.

## The Problem

A Qwen2.5-VL-sized patch at 384 native resolution produces ~729 tokens per frame. With 3x3 pooling, 81 tokens per frame. A 30-minute clip at 1 FPS = 1800 frames = 145,800 tokens. Doable for 2025 open VLMs, but tight. At 2 FPS, 291,600 tokens — only the largest contexts fit.

A 2-hour movie at 1 FPS is 583k tokens. Beyond most 2026 open models; requires Gemini 2.5 Pro or more aggressive pooling.

Three scaling paths have emerged.

## The Concept

### Path 1: Brute-Force Context (Gemini 1.5, Claude Opus)

Throw hardware at the problem. Scale context to millions of tokens, process everything in a single forward pass.

Gemini 1.5 Pro shipped with 1M tokens; Gemini 1.5 Ultra reached 10M; Gemini 2.5 Pro in 2026 reliably handles hours of video. The paper (arXiv:2403.05530) documents 99.7% needle-in-a-haystack recall at ~9.5 million token range.

Engineering: a custom attention implementation with memory hierarchy (local + global + sparse), plus MoE expert routing tuned for long-context efficiency. Not fully published. Not open-source.

### Path 2: Ring Attention (LWM, LongVILA)

Ring attention distributes a long sequence across a "ring" of devices, each holding one chunk. Attention across the full sequence is computed by each device sending its chunk to the next in a ring pattern, computing partial attention, then aggregating.

LWM (Liu et al., 2024) trained a 1M-token context model this way. Training compute scales linearly with context rather than quadratically — attention's quadratic cost is amortized across devices on the ring.

LongVILA (arXiv:2408.10188) adapted this pattern to VLMs. 1400-frame video at 192 tokens per frame = 268k context, trained with 8-way ring attention.

### Path 3: Token Compression (Video-XL, LongVA)

Cheaper than brute-force context: aggressively compress before the LLM sees the sequence.

Video-XL (arXiv:2409.14485) uses a visual summary token: each segment of N frames produces a single "summary" token that attends to those N frames. At inference, the LLM sees only one summary token per segment, dramatically shrinking context.

LongVA uses a "long-context transfer" technique to extend the LLM's context from 200k to 2M. Train on long-context text, transfer to long-context video via shared representations.

Token compression trades recall at specific timestamps for scalability. The model broadly knows what happened but sometimes misses the exact frame.

### Path 4: Agentic Retrieval (VideoAgent)

Don't feed the entire video to the LLM. Instead, treat the video as a database and use the LLM to query it.

VideoAgent (arXiv:2403.10517):

1. LLM reads the question.
2. LLM asks a retrieval tool for relevant segments ("get me segments with a cat").
3. Tool returns matching segment timestamps.
4. LLM reads those segments via a VLM.
5. LLM assembles the answer, or issues follow-up queries.

This applies the LLM-as-agent pattern to long video. Inference is cheaper (only encode relevant segments), engineering is harder (retrieval quality becomes the bottleneck).

### Needle-in-a-Haystack Benchmark

The standard long-context test: insert a unique visual or textual marker at a random point in the video, then ask a query that requires recalling it.

Metric: Recall@k across video lengths and marker positions.

Gemini 2.5 Pro achieves >99% recall on videos up to 90 minutes. Open 72B models (Qwen2.5-VL-72B, InternVL3-78B) get ~85-90% at 30 minutes, degrading past 60 minutes.

VideoAgent can match or beat raw-context models on 2+ hour content, because as long as the tool is good, retrieval hits the needle.

### Which Path to Pick

15-minute clips, frontier accuracy: open 72B + native context usually works. Pick Qwen2.5-VL-72B.

30 minutes to 1 hour: LongVILA or Video-XL for open; Gemini 2.5 Pro for closed. Quality bar matters — frontier goes closed.

2+ hour content: VideoAgent or similar retrieval pattern. Or summarize into smaller chunks and feed hierarchical summaries.

### 2026 Production Pattern

In practice, production long-video pipelines are hybrid:

1. Run dynamic FPS sampling + aggressive pooling on the full video (get a ~100k-token global representation).
2. Feed to a 72B VLM for global summarization.
3. If the user asks a detail question, use the summary as an index for agentic retrieval.

This combines brute-force context for global understanding with retrieval for local detail.

## Use It

`code/main.py`:

- Calculates token budgets for videos from 1 minute to 3 hours under different FPS + pooling settings.
- Simulates a needle-in-a-haystack: injects a marker at a random timestamp, asks a question, scores recall.
- Includes an agentic retrieval router simulator that picks specific segments to feed to a downstream VLM.

Run the budget table and feel the scale gap.

## Ship It

This lesson produces `outputs/skill-long-video-strategy-planner.md`. Given a video duration and query complexity, it picks between brute-force context, compression, and agentic retrieval, and estimates latency + quality expectations.

## Exercises

1. A 45-minute lecture at 1 FPS, 81 tokens per frame. Total tokens? Which models' context windows can fit it?

2. Design a needle-in-a-haystack test: at which minute do you inject the marker, and what's the exact query format?

3. Compare brute-force Qwen2.5-VL-72B (80k context) against VideoAgent (Claude 3.5 + retrieval) on a 1-hour video. Who wins on recall? Who wins on latency?

4. Ring attention's memory cost scales linearly with sequence length, linearly with number of devices. Explain why, and what breaks if you remove the ring rotation stage.

5. Read Gemini 1.5 Section 5 on needle-in-a-haystack. What does the paper find about recall at the 1M vs 10M token boundary?

## Key Terms

| Term | How people say it | What it actually means |
|------|-----------------|------------------------|
| Brute-force context | "just more tokens" | Scaling LLM context to millions of tokens; process everything in one forward pass |
| Ring attention | "LWM-style parallelism" | Distributed attention pattern where each device holds one chunk and rotates |
| Token compression | "summary tokens" | A learned compressor reduces per-segment tokens before entering the LLM |
| Needle-in-a-haystack | "NIH test" | Insert a unique marker at a random point, test recall at query time |
| Agentic retrieval | "LLM as query planner" | LLM asks a retrieval tool for relevant segments, reads via VLM, assembles answer |
| VideoAgent | "retrieval pattern for video" | Canonical agentic retrieval design: question -> tool -> segments -> answer |

## Further Reading

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu et al. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue et al. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu et al. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang et al. — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)
