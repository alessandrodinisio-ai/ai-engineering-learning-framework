# Speculative Decoding and EAGLE

> A frontier LLM generates one token at the cost of a full forward pass through billions of parameters. That forward pass is massively over-provisioned: most of the time a much smaller model can correctly guess the next 3-5 tokens, and the large model only needs to *verify* the guess. Guess right and you get 5 tokens for the cost of one. Speculative decoding (Leviathan et al. 2023) makes this exact, and EAGLE-3 (2025) pushes the acceptance rate to ~4.5 tokens per verification — achieving 4-5x speedups while matching the output distribution.

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10 Lesson 12 (Inference Optimization), Phase 10 Lesson 04 (Pretraining Mini-GPT)
**Time:** ~75 minutes

## The Problem

A 70B-class model on an H100 typically decodes at 40-80 tokens/second. Each token requires a full forward pass that reads all model weights from HBM. You can't make the model smaller without changing the output. You can't increase batch size beyond memory. You're stuck — unless you can make the model output more than one token per forward pass.

Autoregressive generation looks inherently serial: `x_{t+1} = sample(p(· | x_{1:t}))`. But there's a concurrency opportunity. If you have a cheap predictor that says "the next 4 tokens are probably [a, b, c, d]", you can verify all 5 positions in **a single forward pass** of the large model and accept the longest matching prefix.

Leviathan, Kalai, Matias (2023, "Fast Inference from Transformers via Speculative Decoding") made this exact via a clever accept/reject rule that preserves the target model's sampling distribution. Same output distribution, 2-4x faster.

## The Concept

### The two-model setup

- **Target model** `M_p`: the large, slow, high-quality model you actually want to sample from. Distribution: `p(x)`.
- **Draft model** `M_q`: a small, fast, lower-quality model. Distribution: `q(x)`. 5-30x smaller.

Each step:

1. Draft model proposes `K` tokens autoregressively: `x_1, x_2, ..., x_K ~ q`.
2. Target model runs *one* forward pass on all `K+1` positions in parallel, producing `p(x_k)` for each proposed token.
3. Accept/reject each token left-to-right using the modified rejection sampling rule below. Accept the longest matching prefix.
4. If any token is rejected, sample a replacement from a corrected distribution and stop. Otherwise sample a bonus token from `p(· | x_1...x_K)`.

If the draft perfectly matches the target, you get K+1 tokens per target forward. If the draft is wrong at position 1, you get just 1 token.

### The exactness rule

Speculative decoding is **provably distribution-equivalent to sampling from p**. The rejection rule:

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

where `(p - q)+` denotes the pointwise positive part of the difference. When draft and target agree (`p ≈ q`), acceptance is near 1. When they disagree, the residual distribution is constructed so the overall sample is still exactly `p`.

**Greedy case.** For temperature=0 sampling, just check `argmax(p) == x_t`. Yes means accept; no means emit `argmax(p)` and stop.

### Expected speedup

If the draft model's per-token acceptance rate is `α`, the expected tokens produced per target forward is:

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

At `α = 0.8, K = 4`: `(1 - 0.8^5)/(1 - 0.8) = 3.36` tokens/forward. One target forward costs roughly `cost_q * K + cost_p` (K draft steps plus one target verification). If `cost_p >> cost_q * K`, the speedup in throughput is `3.36× / 1 = 3.36×`.

The only real knob is `α`, and it depends entirely on draft-target alignment. A good draft is everything.

### Training the draft: distillation

A random small model makes a poor draft. The standard recipe is to distill from the target:

1. Pick a small architecture (~1B for a 70B target, ~500M for a 7B target).
2. Run the target model on a large text corpus; store its next-token distributions.
3. Train the draft with KL divergence against the target's distributions (not against ground-truth tokens).

Result: `α` is typically 0.6-0.8 on code, 0.7-0.85 on natural language chat. Production speedups of 2-3x.

### EAGLE: tree drafting + feature reuse

Li, Wei, Zhang, Zhang (2024, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") observed two inefficiencies in standard speculative decoding:

1. The draft makes K serial steps, each full-stack. But the draft could reuse the target's most recent verification features (hidden states) — the target already computed the rich representation that the draft is re-deriving from scratch.
2. The draft outputs a single linear chain. If the draft could output a *tree* of candidates (multiple guesses per node), the target's single forward could verify multiple candidate paths in parallel via a tree attention mask, and pick the longest accepted branch.

EAGLE-1's changes:
- Draft input = target's final hidden state at position t, not raw tokens.
- Draft architecture = 1 transformer decoder layer (not a separate small model).
- Output = a tree of K = 4-8 candidates per depth, depth 4-6.

EAGLE-2 (2024) added dynamic tree topology: the tree grows wider where the draft is uncertain and stays narrow where it's confident. Improves `α_effective` without increasing verification cost.

EAGLE-3 (Li et al. 2025, "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test") removes the fixed top-level feature dependency and trains the draft with a new "test-time simulation" loss — the draft is trained on outputs matching the target's test-time distribution rather than teacher-forced training distribution. Acceptance rate rises from 0.75 (EAGLE-2) to 0.82 (EAGLE-3), average tokens per verification from 3.0 to 4.5.

### Tree attention verification

When the draft outputs a tree, the target model verifies it in a single forward pass using a **tree attention mask** — a causal mask that encodes the tree topology rather than a flat line. Each token attends only to its ancestors in the tree. The verification pass is still one forward, one matmul; the topology mask costs just a few extra KV entries.

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

If `a, b` are competing first-token candidates and `c, d, e, f` are second-token candidates, all six positions are verified in one forward pass. The output is the longest prefix on any accepted path.

### When it wins, when it doesn't

**Wins:**
- Predictable text in chat/completion (code, common English, structured output). `α` is high.
- Scenarios with spare GPU compute during decode (the memory-bound phase). Tree drafting uses the available FLOPs.

**Loses / no gain:**
- Highly stochastic output (high-temperature creative writing). `α` drops toward `1/|vocab|`.
- Very high-concurrency batched serving — batching already fills FLOPs, leaving little room for tree verification.
- Very small target models where the draft isn't much smaller.

Production teams typically report 2-3x wall-clock speedup on chat, 3-5x on code generation, near zero on creative writing.

## Build It

`code/main.py`:

- A reference implementation of `speculative_decode(target, draft, prompt, K, temperature)` that implements the exact rejection rule and verifies it preserves the target's distribution (empirical KL < 0.01 vs plain target sampling).
- An EAGLE-style tree drafter that builds a tree of depth K using top-p branching.
- A tree attention mask builder that produces the correct causal pattern for the verifier.
- An acceptance-rate test harness that runs both on a tiny LM (a GPT-2-small distilled from a GPT-2-medium target).

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at each drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left to right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## Use It

- **vLLM** and **SGLang** have first-class built-in speculative decoding. Flags: `--speculative_model`, `--num_speculative_tokens`. EAGLE-2/3 supported via `--spec_decoding_algorithm eagle` flag.
- **NVIDIA TensorRT-LLM** natively supports Medusa and EAGLE trees.
- **Reference draft models**: `Qwen/Qwen3-0.6B-spec` (drafts for Qwen3-32B), `meta-llama/Llama-3.2-1B-Instruct-spec` (drafts for 70B).
- **Medusa heads** (Cai et al. 2024, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"): instead of a draft model, add K parallel prediction heads to the target itself. Simpler deployment, slightly lower acceptance rate than EAGLE.

## Ship It

This lesson produces `outputs/skill-speculative-tuning.md` — a skill that profiles a target model's workload and selects: draft model, K (draft length), tree width, temperature, and when to fall back to plain decoding.

## Exercises

1. Implement the exact rejection rule and empirically verify it. Run 10,000 samples via `speculative_decode` and via plain target sampling; compute TV distance between the two output distributions. Should be < 0.01.

2. Derive the speedup formula. Given fixed `α` and `K`, plot expected tokens per target forward. Find the optimal K for α ∈ {0.5, 0.7, 0.9}.

3. Train a tiny draft. Take a 124M GPT-2 target, distill a 30M GPT-2 draft on 100M tokens using KL loss. Measure `α` on held-out text. Expected: 0.6-0.7.

4. Implement EAGLE-style tree drafting. Instead of a chain, have the draft output top-3 branches at each depth. Build the tree attention mask. Verify the target accepts the longest correct branch.

5. Measure the failure mode. Run speculative decoding at temperature=1.5 (high randomness). Show α collapses and the algorithm is slower than plain decoding due to draft overhead.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| Target model | "The big model" | The slow, high-quality model you want to sample from (p distribution) |
| Draft model | "The speculator" | The small, fast predictor (q distribution); 5-30x smaller |
| K / draft length | "Lookahead" | Number of tokens speculated per verification pass |
| α / acceptance rate | "Hit rate" | Per-token probability that the draft's proposal is accepted |
| Exact rejection rule | "The acceptance test" | The r < p/q comparison that preserves the target distribution |
| Residual distribution | "The corrected p-q" | (p - q)+ / ||(p - q)+||_1, the distribution to sample from on rejection |
| Tree drafting | "Branching speculation" | Draft outputs a tree of candidates, verified at once with a tree-structured attention mask |
| Tree attention mask | "Topology mask" | Causal mask encoding tree topology, letting each node attend only to its ancestors |
| Medusa heads | "Parallel heads" | K extra prediction heads added to the target itself; no separate draft model |
| EAGLE feature reuse | "Hidden-state drafting" | Draft input is the target's last hidden state rather than raw tokens, shrinking the draft |
| Test-time simulation loss | "EAGLE-3 training" | Training the draft on outputs matching the target's test-time distribution rather than teacher forcing |

## Further Reading

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — The exact rejection rule and theoretical speedup analysis
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind's parallel speculative sampling paper
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — Parallel-head alternative to draft models
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — Feature reuse and tree drafting
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — Dynamic tree topology
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — Training-time to test-time alignment
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/lookahead decoding, a speculator-free alternative
