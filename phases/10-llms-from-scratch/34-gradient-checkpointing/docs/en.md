# Gradient Checkpointing and Activation Recomputation

> Backpropagation retains every intermediate activation. At 70B parameters and 128K context, that's 3 TB of activations per rank. Checkpointing trades FLOPs for memory: recompute instead of save. The question is which segments to drop, and the answer isn't "all of them."

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## The Problem

Training a transformer stores, for each layer, the input to every operation that gets differentiated during backward: attention inputs, Q/K/V projections, softmax outputs, FFN inputs, norm outputs, and residual streams. For a layer with hidden size `d`, sequence length `L`, batch `B`, that's on the order of `12 * B * L * d` floats per layer.

For `d=8192, L=8192, B=1`, that's 800 MB per layer in BF16. A 64-layer model is 51 GB of activations—and that's before multiplying by micro-batch size, before attention softmax intermediates (`L^2` per head), before partial copies from tensor parallelism.

The bill squeezes from both sides: BF16 weights plus optimizer states might fit in 80GB, but activations push you over. Gradient checkpointing (a.k.a. activation recomputation) is the standard fix. Drop most activations; rerun the forward pass to retrieve them during backward. Cost: extra FLOPs. Benefit: memory drops proportionally to the ratio of checkpointed segments to total layers.

Done naively, checkpointing costs roughly 33% extra forward FLOPs per step. Done well—selective checkpointing per Korthikanti et al.—you save 5x memory at under 5% FLOP overhead. And with FP8 matmuls, FSDP offloading, and expert-parallel MoE, it truly matters: you can afford neither the memory nor the wasted compute.

## The Concept

### What backward actually needs

`output = layer(input)`. Backward wants `grad_input` and `grad_params`. To compute them it needs:

- `input` (to compute `grad_params = input.T @ grad_output` for linear layers)
- Some activation derivative intermediates (derivatives of ReLU/GELU/softmax depend on the activation values)

Forward propagation automatically stashes these into the autograd graph. Every `tensor.retain_grad()` and every op that needs its input keeps a reference.

### Naive full checkpointing

Partition the network into `N` segments. During forward, store only each segment's *input*. When backward needs intermediates, rerun the segment's forward to materialize them, then differentiate.

Example: 32-layer transformer partitioned into 32 segments, each being 1 layer.

- Memory: 32 layer inputs (small) vs 32 * (per-layer activation volume) (huge).
- Extra compute: one additional forward per segment, i.e., roughly 33% more forward FLOPs total (since backward is 2x forward, the full step becomes 1 + 1 + 2 = 4 units instead of 1 + 2 = 3).

This is the original Chen et al. 2016 recipe: one checkpoint every `sqrt(L)` layers to balance memory and compute. For L=64, that's 8 checkpoints.

### Selective checkpointing (Korthikanti 2022)

Not all activations cost the same. Attention softmax output is `B*L*L*heads`, growing *quadratically* with sequence length. FFN hidden activations are `B*L*4d`, growing linearly. For long sequences, the softmax dominates.

Selective checkpointing keeps the cheap-to-store activations (linear projections, residuals) and recomputes only the expensive ones (attention). You pay minimal FLOPs for recomputation but save O(L^2) memory.

Megatron-Core implements this as "selective" activation recomputation. Used in most 2024+ frontier training runs.

### Offloading

An alternative to recomputation: ship activations to CPU memory between forward and backward. Requires PCIe bandwidth; beneficial when idle bandwidth exceeds rematerialization cost. Hybrid strategies are common: some layers checkpoint, others offload.

FSDP2 exposes offloading as a first-class option. Offloading shines when the GPU is memory-bound but CPU-GPU transfer has spare capacity.

### Recomputation cost model

FLOPs per step with naive checkpointing every `k` layers out of `L` layers:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in segments
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

With selective checkpointing you recompute only the attention kernel, not the full layer:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Memory savings model

Per-layer activation volume: `A`. For `L` layers, total activation memory: `L * A`.

Full checkpointing (segment size 1): store only `L * input_volume` (approximately `L * 1/10 A` for a standard transformer). Save roughly `9 * L * A * 1/10`.

Checkpoint every `k` layers: store `L/k * A` plus the `k-1` layers within the active segment.

At `k = sqrt(L)`, both memory and recomputation cost scale as `sqrt(L)`—the optimal trade-off for equal-cost layers.

### When not to checkpoint

- The innermost layer of an already-in-flight pipeline stage. It has to complete anyway.
- If the first and last layers dominate the stage's compute (rare in transformers).
- Attention kernels already using FlashAttention—Flash already recomputes softmax efficiently, so additional layer-level checkpointing on top of it adds little.

### Implementation patterns

1. **Function wrapping:** Wrap a segment in `torch.utils.checkpoint.checkpoint(fn, input)`. PyTorch stores only `input`, recomputing everything else during backward.

2. **Decorator-based:** Mark layers as checkpointable; the trainer decides which segments get wrapped at config time.

3. **Manual explicit recompute:** Write your own backward pass, calling a custom `recompute_forward` that replays the forward using stored inputs.

All three produce the same functional result. Wrapping is the standard idiom.

### Interaction with TP / PP / FP8

- **Tensor parallelism:** Checkpointed inputs must be gathered or re-scattered during recomputation; handle communication costs.
- **Pipeline parallelism:** Typical pattern is to checkpoint the forward of each pipeline stage so that reverse-order micro-batches can reuse activation memory.
- **FP8 recomputation:** The amax history updated during recomputation must match the original forward's, or FP8 scaling drifts. Most frameworks snapshot this scaling.

## Build It

### Step 1: A toy model with segments

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Step 2: Naive backward requiring all activations

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Step 3: Checkpointed memory with segments of k layers

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Step 4: Cost model

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Step 5: Memory estimator

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Step 6: Optimal segment size

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Step 7: Selective checkpointing decision

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## Use It

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint`—the classic wrapper in PyTorch. Wraps a function; stores only inputs, recomputes during backward.
- **Megatron-Core activation recomputation**: Supports `selective`, `full`, and `block` modes. Standard for 2024+ frontier training.
- **FSDP2 offloading**: Use `module.to_empty(device="cpu")` with `offload_policy` in FSDP2 to shard activations to CPU instead of recomputing.
- **DeepSpeed ZeRO-Offload**: CPU offloading for optimizer states and activations, complementary to checkpointing.

## Ship It

This lesson produces `outputs/prompt-activation-recompute-policy.md`—a prompt that takes your model configuration (layers, hidden, seq, batch) and available GPU memory, and outputs a per-layer recomputation policy (none / selective / full / offload).

## Exercises

1. Verify correctness. Run `model_forward` + `model_backward` (full activations) vs `model_forward_checkpointed` + `model_backward_checkpointed` (segmented). Parameter gradients must match to machine precision.

2. Sweep segment size `k` from 1 to `L`. Plot FLOP overhead and memory. Find the knee of the curve.

3. Implement selective checkpointing: store the attention module's input but not its intermediates. For a 32-layer model with seq=8192, measure the FLOP overhead relative to full-layer checkpointing.

4. Add offloading. Store segment inputs into a simulated "CPU buffer" (a separate list). Measure "PCIe bandwidth" as bytes/time and find the break-even point between offloading and recomputation.

5. Benchmark a real PyTorch transformer with and without `torch.utils.checkpoint`. Measure memory (via `torch.cuda.max_memory_allocated`) and step time.

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| Gradient checkpointing | "Redo the forward to save memory" | Store only segment inputs; recompute intermediates during backward to produce the tensors that support gradients |
| Activation recomputation | "Same as checkpointing" | HPC-flavored name for the same technique |
| Segment size (k) | "How many layers per checkpoint" | The number of layers whose intermediates are discarded together and rematerialized |
| Selective checkpointing | "Korthikanti's trick" | Recompute only the storage-expensive activations (attention softmax); keep the cheap ones |
| Full checkpointing | "The naive version" | Recompute every layer's intermediates within each segment |
| Block checkpointing | "Coarse-grained" | Checkpoint entire transformer blocks; maximum granularity |
| FLOP overhead | "The compute tax" | Extra FLOPs per step = (recompute FLOPs) / (forward + backward FLOPs); 33% naive, 5% selective |
| Activation offloading | "Ship to CPU" | Move activations to CPU memory between forward and backward; alternative to recomputation |
| sqrt-L rule | "Classic optimum" | For equal-cost layers, optimal checkpoint spacing is sqrt(L) layers |
| Attention-softmax volume | "The O(L^2) problem" | L^2 * heads * batch floats; dominates activation memory at long contexts |

## Further Reading

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- The original paper formalizing gradient checkpointing
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- Selective activation recomputation with formal cost analysis
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- Alternative constant-memory approach via reverse-mode rematerialization
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- Activation offloading at scale
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- Standard API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- Selective, full, and block modes
