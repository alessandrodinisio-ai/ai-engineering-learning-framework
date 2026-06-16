# DualPipe Parallelism

> DeepSeek-V3 trains on 2,048 H800 GPUs with MoE experts scattered across nodes. Cross-node expert all-to-all communication costs 1 GPU-hour of communication for every 1 GPU-hour of compute. GPUs sit idle half the time. DualPipe (DeepSeek, December 2024) is a bidirectional pipeline that overlaps forward and backward compute with the all-to-all communication they trigger. Bubbles drop, throughput climbs, and keeping two copies of model parameters (the "dual" that gives it its name) is cheap after expert parallelism already spreads experts thin across ranks. This lesson is a Learn-type walkthrough of what DualPipe actually does, and why Sea AI Lab's DualPipeV improvement removes the 2x parameter cost at the price of slightly tighter bubbles.

**Type:** Learn
**Languages:** Python (stdlib, schedule simulator)
**Prerequisites:** Phase 10 · 05 (distributed training, FSDP, DeepSpeed), Phase 10 · 14 (open model architectures and MoE)
**Time:** ~60 minutes

## Learning Objectives

- Name the four components of a DualPipe forward-backward chunk and why each has its own overlap window.
- Explain the pipeline bubble problem at scale and what "bubble-free" means in practice vs. marketing.
- Hand-trace a DualPipe schedule for 8 PP ranks and 16 micro-batches, confirming that forward and backward streams fill each other's idle slots.
- State the tradeoff DualPipeV (Sea AI Lab, 2025) makes: removing the 2x parameter duplication at the cost of slightly larger bubbles when expert parallelism is inactive.

## The Problem

Training a 671B MoE model on 2k H800 GPUs hits three compounding bottlenecks:

1. **Memory pressure.** Each GPU holds a slice of the model. Activation memory at sequence 8k, 61 layers, 128 heads is enormous.
2. **Pipeline bubbles.** Traditional pipeline parallelism (GPipe, 1F1B) leaves GPUs idle while waiting for their stage's inputs or gradients. At 8 stages, even with 1F1B scheduling, roughly 12% of GPU time can be bubble.
3. **Cross-node all-to-all.** MoE with expert parallelism scatters experts across nodes. Each forward triggers an all-to-all to dispatch tokens to their experts, and another to combine. At 2k GPUs this easily becomes a 1:1 compute-to-communication ratio.

Each has separate solutions: gradient checkpointing for memory, Zero Bubble (Sea AI Lab, 2023) for pipeline bubbles, expert parallelism communication kernels for all-to-all. What DualPipe does is make them work together. The schedule overlaps compute and communication within a single forward-backward chunk, injects micro-batches from both ends of the pipeline simultaneously, and uses the resulting schedule to hide all-to-all inside compute windows.

Reported result: pipeline bubbles nearly eliminated, GPU utilization above 95% throughout DeepSeek-V3's 14.8T-token training run.

## The Concept

### Pipeline parallelism recap

Slice an N-layer model across P devices. Device `i` holds layers `i * N/P .. (i+1) * N/P - 1`. A micro-batch flows forward through devices 0 to P-1, then backward from P-1 to 0. Each device can only start its forward stage when the previous device sends its output, and can only start backward when the downstream device sends upstream gradients.

GPipe (Huang et al., 2019) schedules one micro-batch at a time, wasting most GPU time. 1F1B (Narayanan et al., 2021) interleaves forward and backward passes for multiple micro-batches. Zero Bubble (Qi et al., 2023) splits the backward pass into two parts—backward for inputs (B) and backward for weights (W)—and schedules them to fill bubbles. After Zero Bubble, pipelines are nearly tight.

DualPipe is the next step. It adds two ideas on top:

### Idea 1: Chunk decomposition

Each forward chunk is split into four components:

- **Attention.** Q/K/V projections, attention, output projection.
- **All-to-all dispatch.** Cross-node communication sending tokens to their experts.
- **MLP.** MoE expert computation.
- **All-to-all combine.** Cross-node communication bringing expert outputs back.

A backward chunk adds gradient versions of each. DualPipe schedules them so that all-to-all dispatch happens in parallel with the next chunk's attention computation, and all-to-all combine happens in parallel with the next chunk's MLP computation.

### Idea 2: Bidirectional schedule

Most pipeline schedules inject micro-batches from stage 0, flowing toward stage P-1. DualPipe injects micro-batches from *both ends*. Stage 0 sees forward micro-batches originating from it; stage P-1 also sees forward micro-batches originating from it. The two streams meet in the middle.

To make this work, device `i` must hold *both* early pipeline layer `i` *and* late pipeline layer `P - 1 - i`. That's the "dual" in DualPipe: each device keeps two copies of the model layers it needs to serve (one for each direction). At DeepSeek-V3's scale this is 2x parameter duplication cost. It's affordable because expert parallelism already spreads MoE experts thin enough that duplicating non-expert layers is minor.

Critically, the forward stream of one direction overlaps with the backward stream of the other direction precisely where bubbles would appear in a unidirectional schedule. Bubbles vanish.

### Hand-traced schedule

Consider P = 4 ranks, 8 micro-batches split 4 forward / 4 backward. Time moves left to right; rows are device ranks.

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

Read "F4/F5R" as: rank 1 is running micro-batch 4's forward (traveling left-to-right in the pipeline) *and* micro-batch 5's forward (traveling right-to-left) in the same time slot. That's what "bidirectional" means operationally.

At rank 2 the cross-streams overlap earlier; at rank 0 and P-1 they overlap latest. In the steady middle of the schedule, every rank runs some direction's forward overlapping with another direction's backward. Compute stays busy. The forward's all-to-all dispatch hides inside backward compute. The all-to-all combine hides inside forward compute. Bubbles are squeezed out.

### Bubble accounting

Standard 1F1B pipeline bubble (wasted time per rank):

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble improves this but doesn't reach zero. DualPipe achieves zero bubble in the steady stage when the number of micro-batches divides evenly by 2x pipeline depth. Outside the steady stage (warmup and cooldown), some bubble exists, but it doesn't grow with micro-batch count—a key property the paper emphasizes.

In marketing: "bubble-free." In technical terms: bubble doesn't grow with micro-batch count. Sea AI Lab's subsequent analysis (DualPipeV / Cut-in-half) shows that full zero-bubble holds only when expert parallelism isn't the bottleneck; with EP-driven all-to-all there's always some scheduling compromise.

### DualPipeV — the improvement

Sea AI Lab (2025) observed that the 2x parameter duplication is wasteful when EP communication overlap is not the focus. Their DualPipeV schedule folds the bidirectional injection into a "V-shaped" schedule running on a single parameter copy. Bubbles are slightly larger than DualPipe's, but memory savings are substantial. DeepSeek adopted DualPipeV as an EP-off mode in their open-source DualPipe implementation.

Tradeoff:

| Feature | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| Parameter copies per device | 2 | 1 | 1 | 1 |
| Bubble vs micro-batches | Constant | Grows slightly | Grows | Grows |
| Compute-communication overlap | Full | Partial | Minimal | Partial |
| When to use | EP-heavy MoE | Dense or EP-light | Baseline | Any pipeline |

### What it means for a 14.8T-token run

DeepSeek-V3's pretraining consumed 14.8T tokens on 2,048 H800 GPUs, roughly 2.8 million GPU-hours. With naive 1F1B they'd lose 12-15% of that to pipeline bubbles—340k to 420k GPU-hours, enough to train a complete 70B model. DualPipe recovers most of it. Hard to quantify the contribution directly without internal logs, but the paper's claim is >95% GPU utilization averaged across the training run.

For smaller runs (under 1k GPUs), DualPipe is overkill—pipeline bubbles are smaller relative to total cost, and dense model training rarely hits the all-to-all bottleneck. For frontier MoE training at multi-thousand GPU scale, it's effectively required.

### Where it sits in the stack

- Complements **FSDP** (Phase 10 · 05). FSDP shards model parameters across ranks; DualPipe schedules compute across ranks. They combine.
- Compatible with **ZeRO-3** gradient sharding. The two-copy duplication bookkeeping needs to cooperate with ZeRO's sharded gradients.
- Requires **custom all-to-all kernels** tuned for the specific cluster topology. DeepSeek's open-source kernels are the reference implementation.

## Use It

`code/main.py` is a pipeline schedule simulator. It takes `(P, n_micro_batches, schedule)` and prints steady-state utilization for 1F1B, Zero Bubble, DualPipe, and DualPipeV. It's a teaching tool—numbers match the paper's qualitative claims, not a statement about production measured speedups.

The simulator's value: run it with different P and micro-batch counts and watch bubble fraction grow for 1F1B but not for DualPipe.

Integration considerations for real training runs:

- Pick a pipeline parallelism depth that cleanly divides your micro-batch count.
- Ensure your expert parallelism mesh supports bidirectional all-to-all. DeepSeek's kernels are the reference.
- Expect to burn a week debugging the schedule itself the first time. The bookkeeping is hairy.
- Monitor per-rank GPU utilization, not just aggregate. DualPipe's gains come from tightening the stragglers.

## Ship It

This lesson produces `outputs/skill-dualpipe-planner.md`. Given a training cluster spec (GPU count, topology, interconnect, model shape), it recommends a pipeline parallelism strategy, the schedule algorithm to use, and expected bubble fraction at the target scale.

## Exercises

1. Run `code/main.py` on `(P=8, micro_batches=16, schedule=dualpipe)` and `(P=8, micro_batches=16, schedule=1f1b)`. Compute the GPU utilization difference and express it as GPU-hours recovered per million tokens trained.

2. Hand-draw the schedule for `(P=4, micro_batches=8, schedule=dualpipe)`. Label each time slot with micro-batch ID and direction. Identify the first time slot where the bubble vanishes.

3. Read DeepSeek-V3 technical report (arXiv:2412.19437) Figure 5. Identify the overlap window for all-to-all dispatch within a single DualPipe forward chunk. Explain how the compute schedule hides it.

4. Compute the 2x parameter overhead of DualPipe for a 70B dense model with P=8 pipeline stages and a 671B MoE model with P=16 pipeline stages. Show why the overhead is proportionally smaller for the MoE case (most parameters are experts, sharded across a large EP group).

5. Compare DualPipe with Chimera (a 2021 competing bidirectional scheduler). Using the paper's Section 3.4 as reference, identify two specific properties DualPipe adds that Chimera lacks.

## Key Terms

| Term | How people say it | What it actually is |
|------|----------------|------------------------|
| Pipeline bubble | "idle time per rank" | GPU cycles wasted because a pipeline stage is waiting for its input or gradient |
| 1F1B | "default pipeline schedule" | One-forward / one-backward interleaved schedule; the baseline DualPipe beats |
| Zero Bubble | "Sea AI Lab 2023" | Splits backward into B (input gradient) and W (weight gradient); nearly fully tightens the pipeline |
| DualPipe | "DeepSeek-V3 schedule" | Bidirectional pipeline + compute-communication overlap; bubble doesn't grow with micro-batch count |
| DualPipeV | "Cut-in-half" | V-shaped improvement removing the 2x parameter duplication at the cost of slightly larger bubbles |
| Chunk | "pipeline work unit" | One forward or backward pass of a micro-batch through one pipeline stage |
| All-to-all dispatch | "send tokens to experts" | Cross-node communication routing tokens to their assigned MoE experts |
| All-to-all combine | "bring expert outputs back" | Cross-node communication gathering expert outputs after MLP |
| Expert parallelism (EP) | "experts across GPUs" | Sharding MoE experts across ranks so different GPUs hold different experts |
| Pipeline parallelism (PP) | "layers across GPUs" | Sharding model layers across ranks; the dimension DualPipe schedules |
| Bubble fraction | "wasted GPU time" | (bubble_time / total_time); the fraction DualPipe drives toward zero |

## Further Reading

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) — Primary DualPipe reference
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) — Open-source reference implementation with DualPipeV (Cut-in-half) mode
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — The Zero Bubble predecessor
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — DualPipeV analysis that inspired DeepSeek's EP-off mode
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — The 1F1B schedule DualPipe compares against
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — The original pipeline parallelism paper and bubble problem
