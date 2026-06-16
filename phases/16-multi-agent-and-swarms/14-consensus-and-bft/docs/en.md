# Agent Consensus and Byzantine Fault Tolerance

> Classical distributed-systems BFT meets stochastic LLMs. Between 2025-2026 three research directions appeared: **CP-WBFT** (arXiv:2511.10400) attaches a confidence probe to each vote and weights accordingly; **DecentLLMs** (arXiv:2507.14928) goes leaderless, with workers proposing in parallel and aggregating via geometric median; **WBFT** (arXiv:2505.05103) combines weighted voting with Hierarchical Structure Clustering, splitting nodes into Core and Edge. "Can AI Agents Agree?" (arXiv:2603.01213) gives an honest empirical result: even scalar agreement is fragile today — a single deceptive agent can wreck a Mixture-of-Agents. BFT is necessary but not sufficient. This lesson builds a minimal BFT protocol, injects three agent-specific attacks (Byzantine lie, sycophantic conformity, correlated-error monoculture), and measures how each consensus variant copes.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 16 · 07 (Society of Mind & Debate), Phase 16 · 13 (Shared Memory)
**Time:** ~75 min

## The Problem

You have N LLM agents, each producing an answer. They disagree. Majority vote picks the wrong one because two agents are correlated (same base model, same training data, same failure mode). The third agent happens to be wrong in a novel way — so the majority is a false majority.

Now add a deceptive agent: it lies deliberately. Or add a sycophantic agent: it echoes whoever spoke last. In classical BFT, the assumption is Byzantine nodes are a fraction `f < n/3` and behave arbitrarily. The 2026 reality: LLM nodes are stochastic even when honest, correlated across models, and influenced by each other's outputs. You cannot treat them as independent Bernoulli voters.

Classical BFT (PBFT, 1999) isn't wrong — it's just incomplete. It handles arbitrary bit-flips. It doesn't handle "three honest agents share a hallucination because they share training data." This lesson starts from PBFT's foundation and layers on three 2025-2026 adaptations.

## The Concept

### What Classical BFT Gives You

Practical Byzantine Fault Tolerance (Castro & Liskov, OSDI 1999) tolerates `f < n/3` Byzantine nodes. The protocol has three phases (pre-prepare, prepare, commit) and two primitives (signed messages, quorum certificates). It reaches agreement on a single value among `n >= 3f + 1` honest or malicious nodes.

These guarantees are strong, but assume:

1. **Independent failures.** Byzantine nodes don't coordinate.
2. **Honest nodes are truly honest.** Correctness of honest outputs is out of scope; the protocol only aligns disagreements.
3. **The problem has a ground-truth answer.** Consensus on a wrong fact is still consensus.

LLM agents violate all three. Two agents running the same base model share failures. An "honest" LLM still hallucinates. And on ambiguous problems, "truth" is whatever the agents settle on — there's no external oracle.

### Three LLM-Specific Attacks

**Byzantine lie.** One agent outputs a deliberately wrong answer. Classical BFT handles this if `f < n/3`.

**Sycophantic conformity.** One agent reads others' answers before voting, then aligns with whoever spoke last. Not malicious, but correlated with the loudest voice. Classical BFT can't block it because this agent passes every signature check.

**Correlated-error monoculture.** Three agents share a base model. They hallucinate the same wrong answer. The majority is wrong. Classical BFT can't help because all three are "honestly" in agreement.

### 2025-2026 Responses

**CP-WBFT** (arXiv:2511.10400) — Confidence-Probe Weighted BFT. Each voter attaches a confidence probe to its answer (a self-reported probability, or a prediction from an independent calibration model). Vote weight scales with confidence. Reports +85.71% BFT improvement on complete graphs. Mitigates: sycophantic conformity (conforming agents tend to have low confidence in positions they didn't originate).

**DecentLLMs** (arXiv:2507.14928) — Leaderless. Worker agents propose in parallel, evaluator agents score proposals, and the final answer is the geometric median of scored positions. Robust at `f < n/2`. Mitigates: Byzantine lies and correlated errors (geometric median is robust to outliers, pulled toward the dense cluster rather than the model-biased mean).

**WBFT** (arXiv:2505.05103) — Weighted BFT with Hierarchical Structure Clustering. Vote weight is assigned by response quality plus a trust score learned from history. Clusters agents into Core and Edge; Core agents must reach consensus first, Edge agents follow. Mitigates: scalability (Core consensus is small and fast) and partially monoculture (Core can be selected for diversity).

### Empirical Study: "Can AI Agents Agree?" (arXiv:2603.01213)

This paper measures scalar agreement (LLM agents agreeing on a single numerical value) across multiple frontier models. The findings are uncomfortable:

- Even without adversaries, LLM agents disagree on scalar problems at rates exceeding 30% on many benchmarks.
- A single agent with a deceptive persona can shift Mixture-of-Agents consensus away from the honest baseline by 40+ percentage points.
- Disagreement rate correlates with model diversity — heterogeneous ensembles disagree more (benefit: uncorrelated errors) but also drift slower (cost: takes longer to converge).

Takeaway: BFT gives you the machinery to align outputs, but it doesn't tell you whether the aligned output is correct. Combine with verification (Phase 16 · 08 Role Specialization), diversity (Phase 16 · 15 Debate Variants), and evaluator agents (Phase 16 · 24 Benchmarks).

### The Stripped-Down Core Protocol

One-round minimal BFT for LLM agents:

```
1. Task arrives; each agent i produces answer a_i
2. Each agent attaches confidence probe c_i in [0, 1]
3. Aggregator collects (a_i, c_i) from all n agents
4. Aggregator groups by semantic cluster (equivalent answers in one cluster)
5. Aggregator computes weight for each cluster C:
     w(C) = sum_{i in C} c_i
6. winner = cluster with max weight, if max > threshold * sum(c_i)
   else: retry or escalate
7. Minority clusters are logged with provenance for post-hoc audit
```

The semantic clustering step is the LLM-specific twist. Two answers "the study reports 4.2%" and "4.2% improvement" belong to the same cluster. Naive string equality misses this. In production, use a cheap embedding model or explicit normalization.

### Threshold Tuning

The `threshold` parameter determines when to accept vs retry. Too low: you accept weak majorities. Too high: you never accept anything. Empirical range: 0.5-0.67 for `n=5-7` agents, higher for smaller `n`. Below threshold, escalate to a human or swap in a different ensemble.

### Where Consensus Doesn't Help

- **Ambiguous problems.** If the problem has no ground truth, consensus is an opinion. Call it that.
- **Compound problems.** "Write code and explain it" — two answers. Vote on each independently.
- **Adversarial multi-round.** If agents can observe previous rounds and imitate (Du 2023 debate), they start agreeing with each other regardless of truth. Bound rounds (usually 2-3).

## Build It

`code/main.py` implements:

- `AgentVoter` — a scripted strategy with (answer, confidence).
- `MajorityVote` — classic plurality.
- `CPWBFT` — confidence-weighted voting with semantic clustering.
- `DecentLLMs` — geometric median aggregation over scored proposals.
- `Scenario` — runs each aggregator under three attack modes.

Attack modes implemented:

1. `byzantine`: one agent lies with high confidence.
2. `sycophancy`: one agent copies the first answer it sees, matching confidence.
3. `monoculture`: three agents share a wrong answer at medium confidence (correlated error).

Run:

```
python3 code/main.py
```

Expected output: a table of (attack, aggregator) -> final answer, with correct answer highlighted. Plurality fails on monoculture. CPWBFT's confidence weighting mitigates sycophancy. DecentLLMs' geometric median pulls toward the honest cluster when monoculture is less than half the total.

## Use It

`outputs/skill-consensus-designer.md` designs a consensus protocol for a multi-agent ensemble: clustering method, weighting, threshold, and escalation strategy for below-threshold rounds.

## Ship It

Before shipping any consensus mechanism:

- **Attack-test with at least the three modes above.** Your protocol should fail predictably, not silently.
- **Log every minority cluster with provenance.** Minority clusters are your early-warning system for correlated errors.
- **Enforce bounded rounds.** No "debate until agreement" — that rewards sycophancy.
- **Separate agreement from correctness.** Consensus output goes to a verifier; verifier is independent of the ensemble.
- **Monitor agreement rate.** Sudden rise means conformity bias; sudden drop means model drift.

## Exercises

1. Run `code/main.py`. Confirm plurality fails on monoculture attack, but CPWBFT partially mitigates when monoculture confidence is below 0.7.
2. Add a fourth attack mode: **silent abstention** — one agent refuses to answer ("I don't know"). How should each aggregator treat abstention? Implement your choice.
3. Replace semantic clustering with embedding similarity (use any open-source embedding model). What happens to the sycophancy attack?
4. Read CP-WBFT (arXiv:2511.10400). Implement the confidence probe calibration step (an independent calibration model verifies each agent's self-reported confidence). Measure accuracy gain on the monoculture scenario.
5. Read "Can AI Agents Agree?" (arXiv:2603.01213). Reproduce a simplified scalar agreement experiment: three agents, one scalar question, deceptive persona prompt. Can CPWBFT or DecentLLMs catch it?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| BFT | "Byzantine fault tolerance" | Castro-Liskov 1999 protocol that reaches consensus under `f < n/3` arbitrary failures. |
| Byzantine | "Any bad behavior" | A node that can lie, drop messages, fail silently — anything except crash-safe. |
| Confidence probe | "How sure are you?" | A self-reported or calibrator-predicted probability attached to a vote. |
| Semantic clustering | "Same answer, different wording" | Grouping equivalent answers before counting votes. |
| Geometric median | "Robust center" | The point minimizing sum of distances to all samples. Robust to outliers, unlike the mean. |
| Monoculture | "Same model, same bugs" | Correlated errors when agents share training data or base model. |
| Sycophantic conformity | "Echoing the loudest voice" | An agent's vote biased toward whoever spoke first/loudest. |
| Core/Edge | "Hierarchical BFT" | WBFT's split: small Core reaches consensus first, Edge nodes follow. Bounds latency. |

## Further Reading

- [Castro & Liskov — Practical Byzantine Fault Tolerance (OSDI 1999)](https://pmg.csail.mit.edu/papers/osdi99.pdf) — the foundation
- [CP-WBFT — Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400) — weighting votes by confidence
- [DecentLLMs — leaderless multi-agent consensus](https://arxiv.org/abs/2507.14928) — geometric median aggregation
- [WBFT — Weighted BFT with Hierarchical Structure Clustering](https://arxiv.org/abs/2505.05103) — Core/Edge split to bound latency
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213) — fragility of scalar agreement and deceptive persona attacks
