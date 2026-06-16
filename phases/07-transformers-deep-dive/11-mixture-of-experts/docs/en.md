# Mixture of Experts (MoE)

> A dense 70B transformer activates all parameters for every token. A 671B MoE activates only 37B per token, yet beats it on every benchmark. Sparsity is the most important scaling idea of the decade.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 min

## The Problem

A dense transformer's inference FLOPs equal its parameter count (×2 for forward pass). Scale it up, every token pays full price. By 2024, the frontier hit a compute wall: getting meaningfully smarter required exponentially more FLOPs per token.

Mixture of Experts breaks this coupling. Replace each FFN with `E` independent experts + a router that picks `k` experts per token. Total params = `E × FFN_size`. Active params per token = `k × FFN_size`. Typical 2026 config: `E=256`, `k=8`. Storage scales with `E`, compute scales with `k`.

The 2026 frontier is almost entirely MoE: DeepSeek-V3 (671B total / 37B active), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2, gpt-oss. On Artificial Analysis's independent leaderboard, the top 10 open-source models are all MoE.

## The Concept

![MoE layer: router selects k of E experts per token](../assets/moe.svg)

### FFN Replacement

Dense transformer block:

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE block:

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

Each expert is an independent FFN (usually SwiGLU). The router is a single linear layer. Each token picks its own `k` experts and gets a gated mixture of their outputs.

### The Load Balancing Problem

If the router sends 90% of tokens to expert 3, the others starve. Three fixes have been tried:

1. **Auxiliary load-balancing loss** (Switch Transformer, Mixtral). Add a penalty proportional to expert usage variance. Works, but adds a hyperparameter and a second gradient signal.
2. **Expert capacity + token dropping** (early Switch). Each expert processes at most `C × N/E` tokens; overflow tokens skip the layer. Hurts quality.
3. **Auxiliary-loss-free balancing** (DeepSeek-V3). Add a learned bias per expert that shifts the router's top-k selection. Bias updates outside training loss. No penalty on the main objective. The big unlock of 2024.

DeepSeek-V3's approach: after each training step, for each expert, check whether its usage is above or below target. Push the bias `±γ`. Selection uses `scores + bias`. Gating uses the raw `scores`, unchanged. Decouples routing from representation.

### Shared Experts

DeepSeek-V2/V3 also split experts into *shared* and *routed*. Every token passes through all shared experts. Routed experts are selected via top-k. Shared experts capture general knowledge; routed experts specialize. V3 runs 1 shared expert plus top-8 from 256 routed experts.

### Fine-Grained Experts

Classic MoE (GShard, Switch): each expert is as wide as a full FFN. `E` is small (8–64), `k` is small (1–2).

Modern fine-grained MoE (DeepSeek-V3, Qwen-MoE): each expert is narrower (1/8 FFN size). `E` is large (256+), `k` is larger (8+). Same total params, but combinatorial space expands massively. Each token has `C(256, 8) = 400 trillion` possible "experts." Quality goes up, latency stays flat.

### Cost Profile

Per token, per layer:

| Configuration | Active params per token | Total params |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 beats Llama 3 70B (dense) on nearly every benchmark while doing **fewer active FLOPs per token**. More params = more knowledge. More active FLOPs = more compute per token. MoE decouples the two.

### The Cost: Memory

Regardless of which experts fire, all experts live on GPU. A 671B model in fp16 needs ~1.3 TB of memory. Frontier MoE deployment requires expert parallelism — sharding experts across GPUs and routing tokens over the network. Latency is dominated by all-to-all communication, not matmul.

## Build It

See `code/main.py`. A compact, stdlib-only MoE layer with:

- `n_experts=8` SwiGLU-style experts (single linear each, for demonstration)
- top-k=2 routing
- softmax-normalized gating weights
- auxiliary-loss-free balancing via per-expert bias

### Step 1: Router

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over raw scores for selected experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Bias affects selection, not gating weights. This is the DeepSeek-V3 trick — bias corrects load imbalance without steering the model's predictions.

### Step 2: Route 100 Tokens Through the Router

Track which experts fire and how often. Without bias, usage is skewed. Add a bias update loop (overused experts get `-γ`, underused get `+γ`), and usage converges to uniform within a few iterations.

### Step 3: Parameter Count Comparison

Print the "dense equivalent" of an MoE configuration. DeepSeek-V3 shape: 256 routed + 1 shared, 8 active, d_model=7168. Total param count is staggering. Active count is one-seventh of dense Llama 3 70B.

## Use It

HuggingFace loading:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 production inference: vLLM natively supports MoE routing. SGLang has the fastest expert-parallel path. Both handle top-k selection and expert parallelism automatically.

**When to pick MoE:**
- You want frontier quality with lower per-token inference cost.
- You have the memory / expert-parallel infrastructure.
- Your workload is token-dense (chat, code) rather than context-dense (long docs).

**When not to pick MoE:**
- Edge deployment — you pay full storage for any active FLOP.
- Latency-critical single-user serving — expert routing adds overhead.
- Small models (<7B) — MoE quality advantage only appears above a compute threshold (~6B active params).

## Ship It

See `outputs/skill-moe-configurator.md`. This skill picks E, k, and shared-expert layout for a new MoE given a parameter budget, training token count, and deployment target.

## Exercises

1. **Easy.** Run `code/main.py`. Observe how auxiliary-loss-free bias updates flatten expert usage within 50 iterations.
2. **Medium.** Replace the learned router with a hash-based router (deterministic, no learning). Compare quality and balance. Why does the learned router win?
3. **Hard.** Implement GRPO-style "rollout-matched routing" (DeepSeek-V3.2 trick): log which experts fire at inference time, force the same routing during gradient computation. Test the effect on a toy policy-gradient setup.

## Key Terms

| Term | How people talk about it | What it actually means |
|------|-----------------|-----------------------|
| Expert | "one of many FFNs" | An independent feed-forward network; parameters dedicated to a sparse slice of FFN computation. |
| Router | "the gate" | A tiny linear layer that scores each token against each expert; top-k selection. |
| Top-k routing | "k active experts per token" | Each token's FFN computation goes through exactly k experts, weighted by gating. |
| Auxiliary loss | "load-balancing penalty" | An extra loss term penalizing skewed expert usage. |
| Auxiliary-loss-free | "DeepSeek-V3 trick" | Balancing via per-expert bias on router selection only; no extra gradient. |
| Shared expert | "always on" | Extra experts every token passes through; captures general knowledge. |
| Expert parallelism | "shard by expert" | Place different experts on different GPUs; route tokens over the network. |
| Sparsity | "active params < total params" | The ratio `k × expert_size / (E × expert_size)`; DeepSeek-V3 is 37/671 ≈ 5.5%. |

## Further Reading

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — The idea.
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch, classic MoE.
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B.
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + auxiliary-loss-free MoE + MTP.
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — The bias-based balancing paper.
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — Fine-grained + shared expert split used in this lesson's router.
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — The original shared-expert paper.
