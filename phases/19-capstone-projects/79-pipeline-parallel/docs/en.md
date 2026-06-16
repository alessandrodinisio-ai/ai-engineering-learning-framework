# Pipeline Parallel and Bubble Analysis

> Tensor parallelism splits matrix multiplications across ranks. Pipeline parallelism splits the model across ranks — one rank per stage. Microbatches flow through the pipeline. The idle time at the beginning and end is the bubble; minimizing it is the entire craft.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19 Track C Lessons 42-49
**Time:** ~90 minutes

## Learning Objectives

- Split a sequential model into N stages and simulate a forward pipeline across N ranks.
- Schedule M microbatches through the pipeline using the GPipe schedule (fill forward, then drain backward) and compute the bubble fraction.
- Compare the bubble against the interleaved 1F1B schedule used by Megatron-LM and PipeDream.
- Argue that equal compute per stage matters more than equal parameters per stage for stage assignment.

## The Problem

A 70-billion-parameter model in fp16 needs 140 GB just for parameters. No consumer GPU fits it. ZeRO-3 shards parameters across ranks, but each forward still requires every rank to allgather the full layer — paying log(N) hops per layer. Pipeline parallel takes a different path: split the model into N stages, one stage per rank. Layer 1's forward runs on rank 0, which passes the activation tensor to rank 1; rank 1 runs layer 2 and passes to rank 2; and so on. Backward flows in reverse. Memory drops linearly because each rank only holds one stage; computation is sequential — that is the bubble problem.

The bubble is the idle time at the start of the pipeline (waiting for the first microbatch to reach the last stage) and at the end (waiting for the last microbatch to drain backward). With M microbatches and N stages, the per-stage bubble fraction is (N-1)/(M+N-1). At M=8, N=4 this is 27%. At M=64, N=4 this is 4.5%. More microbatches per step means a smaller bubble, which means a smaller per-microbatch batch size — and this constraint is what drives microbatch design.

## The Concept

```mermaid
flowchart LR
  R0[rank 0: stage 0 / layer 0] --> R1[rank 1: stage 1 / layer 1]
  R1 --> R2[rank 2: stage 2 / layer 2]
  R2 --> R3[rank 3: stage 3 / loss]
  R3 -.backward.-> R2
  R2 -.backward.-> R1
  R1 -.backward.-> R0
```

### GPipe Schedule

Before starting any backward, fill the pipeline forward with all M microbatches; then drain backward. Activations for each microbatch must be kept until its backward, so memory grows linearly with M. Forward occupies M+N-1 clock cycles, backward another M+N-1. Useful work per stage is 2M cycles; bubble per stage is 2(N-1) cycles. When each forward and backward occupies one time unit, the bubble fraction is (N-1)/(M+N-1). Choosing M much larger than N hides the bubble.

### 1F1B Schedule

Interleaved: as soon as one microbatch's forward reaches the last stage, its backward is launched and flows back. This schedule alternates one forward and one backward on each stage. The bubble is still N-1, but activation memory is bounded by pipeline depth rather than microbatch count. Production pipelines use 1F1B (Megatron, PipeDream). This lesson implements GPipe first because it is simpler; 1F1B is left as an exercise.

### Why Equal Compute Per Stage Matters

If stage 0 takes 50 ms and stage 1 takes 100 ms, every cycle is bottlenecked by stage 1. Other stages idle 50 ms per cycle waiting for stage 1 to release. Equal parameters is the wrong axis: a transformer's compute is dominated by attention plus MLP per layer, while embedding layers have many parameters but little compute. Stage assignment should equalize FLOPs per stage, not weights per stage.

### Microbatch vs. Batch Relationship

A pipeline runs M microbatches of size B. The effective batch size is M*B. The gradient at the end of one pipeline step equals the gradient over the combined M*B samples. Bubble fraction depends on M; the optimizer sees M*B. Tuning M trades bubble (lower with higher M) against per-microbatch memory (higher activation memory under GPipe with higher M).

## Build It

`code/main.py` implements:

- `PipelineStage`: a small `nn.Module` holding one stage's parameters and exposing `forward(activation)`.
- `Pipeline(stages, num_microbatches)`: orchestrates the GPipe schedule across simulated stages using per-stage simulated wall-clock time.
- `bubble_fraction(num_stages, num_microbatches)`: closed-form (N-1)/(M+N-1).
- A 4-stage demo printing a per-microbatch trace and the measured bubble fraction.

Run:

```bash
python3 code/main.py
```

Output: a stage-vs-microbatch Gantt chart plus bubble percentage compared against the closed-form prediction.

## Ship It

Three patterns harden pipeline parallel for production.

**Activation checkpointing pairs with pipelines.** With M microbatches in flight under GPipe, activation memory is M times that of a single microbatch. Activation checkpointing recomputes forward during backward, trading compute for memory; this combination is what makes pipelines viable for long sequences.

**Stage balance is measured, not assumed.** Production teams run a profiling pass to measure actual per-layer compute (FLOPs and wall-clock) on target hardware, then partition based on that measurement. Megatron-LM's `--num-layers-per-stage` flag accepts a list, allowing unequal layer counts when per-layer costs differ across stages.

**Send-recv scheduling must avoid deadlock.** A pipeline where every stage sends before receiving will deadlock on the wire. The standard fix is interleaving: even-ranked stages send then receive; odd-ranked stages receive then send. This lesson explicitly orders ranks to make the pattern visible.

## Use It

Production patterns:

- **Megatron-LM.** The benchmark for large-scale pipeline parallel. Uses 1F1B and supports tensor + pipeline + data parallel combined.
- **DeepSpeed Pipeline.** Integrates with ZeRO; ZeRO-1 + pipeline is a common combination for the largest open-source models.
- **PyTorch Pipe.** PyTorch-native pipeline wrapper built on `torch.distributed.pipeline.sync.Pipe`.

## Connections

Lesson 80 stores each stage's parameter shard into a sharded checkpoint. Lesson 81 assembles DDP + ZeRO + pipeline in an end-to-end demo (spiritually; for runtime the demo keeps pipeline as simulation).

## Exercises

1. Implement 1F1B and verify the bubble fraction matches GPipe but activation memory is bounded.
2. Profile real per-stage times on a deeper model and rebalance stages by measured wall-clock.
3. Add gradient accumulation across pipeline microbatches and verify the gradient equals an equivalent full-batch forward.
4. Pair pipeline with activation checkpointing and measure the memory reduction vs. compute cost.
5. Combine pipeline with DDP (each pipeline rank replicated in a data-parallel group) and reason about this 2D schedule.

## Key Terms

| Term | Common usage | Precise meaning |
|------|----------------|------------------------|
| Pipeline | "model parallel along depth" | One rank per stage; activations flow between stages |
| Bubble | "pipeline idle time" | (N-1) steps at the start + end where some stages have no work |
| Microbatch | "a slice of the batch" | One forward/backward unit; larger M means smaller bubble |
| GPipe | "fill then drain" | All M forwards before any backward; high activation memory |
| 1F1B | "interleaved schedule" | One forward then one backward per stage; bounded activation memory |

## Further Reading

- [Huang et al., GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [Narayanan et al., PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377)
- [Megatron-LM pipeline parallel documentation](https://github.com/NVIDIA/Megatron-LM)
- Phase 19 Lesson 76 — The send/recv primitives used by this schedule
- Phase 19 Lesson 78 — ZeRO is orthogonal to pipeline and commonly combined
