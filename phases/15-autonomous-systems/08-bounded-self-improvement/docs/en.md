# Bounded Self-Improvement Design

> Research has converged on four primitives for constraining a self-improvement loop. Formal invariants that must hold across every edit. Alignment anchors that cannot be modified. Multi-objective constraints—every dimension (safety, fairness, robustness) must hold, not just performance. Regression detection—pausing the loop when historical metrics suggest capability loss. None are proofs of safety—information-theoretic results (Kolmogorov complexity, Lob's theorem) limit what any system can prove about its own successor. They are mitigations that make silent failure more expensive.

**Type:** Learn
**Languages:** Python (standard library, bounded loop with invariant checks)
**Prerequisites:** Phase 15 · 07 (RSI), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## The Problem

Lesson 7's race simulator showed that small differences in rates compound into large gaps. Lesson 4's DGM case study showed that loops actively exploit their own evaluators. Both results point to the same engineering problem: what constraints can you impose on a self-improvement loop such that the constraints cannot be silently undermined by the loop itself?

The ICLR 2026 RSI Workshop summary (openreview.net/pdf?id=OsPQ6zTQXV) identifies four such primitives. Anthropic's RSP v3.0 (Lesson 19) and DeepMind's FSF v3 (Lesson 20) both reference them in their capability thresholds. Meta's HyperAgents work and community frameworks like SAHOO (March 2026) implement subsets in production.

The honest framing: these are mitigations. Information-theoretic results limit what any system can prove about its own successor, and no current design formally closes the problem. A well-constrained loop is safer than an unconstrained one, but not safe in an absolute sense.

## The Concept

### Primitive 1: Formal Invariants

An invariant is a property that must hold before and after every self-modification. Examples:

- Output distributions are conditioned on a fixed constitutional header (Lesson 17).
- No tool calls hit unauthorized endpoints.
- Memory writes go through a logged, signed path.
- Module hashes of the evaluator match approved versions.

Invariants are checked by external code the loop cannot edit. If a proposed modification violates an invariant, it is rejected. This catches the DGM-style evaluator-tampering pattern (Lesson 4).

The hard part is choosing invariants that are both necessary for safety and computable in reasonable time. Over-specification stalls progress; under-specification lets the loop step over important properties.

### Primitive 2: Alignment Anchors

An alignment anchor is an immutable representation of the loop's core objective, pinned outside the loop's edit surface. Examples:

- A constitutional text (Lesson 17) loaded from a read-only location on every invocation.
- An evaluator in an independent repository with separate access control.
- A signed tool-permission manifest re-verified every cycle.

Anchors prevent goal drift. The loop can improve how it pursues the goal, but cannot edit what the goal is.

The subtle failure mode: an anchor the loop cannot edit can still be reinterpreted by a loop that drifts in how it reads the anchor. Constitutional AI (Lesson 17) is explicitly reasoning-based for novel situations; that reasoning layer is where interpretation drift lives. Anchors are necessary, not sufficient.

### Primitive 3: Multi-Objective Constraints

A loop optimizing a single scalar score finds shortcuts. A loop required to satisfy multiple hard constraints simultaneously has fewer shortcuts available. Typical dimensions:

- Performance (task-level benchmarks)
- Safety (red-team evaluations, refusal rate on known-bad inputs)
- Fairness (disparate impact bounds on sensitive subgroups)
- Robustness (OOD test sets, adversarial input handling)

A modification is accepted only when every constraint holds. Lesson 13's cost governors layer financial constraints on top. Lesson 18's Llama Guard plugs in as a safety dimension.

### Primitive 4: Regression Detection

Each cycle's scores are compared against the historical distribution. Drops beyond a configured tolerance pause the loop. This catches silent capability loss—which would otherwise be absorbed into the running mean as though the loop "improved past" it.

A practical implementation: store per-task scores for the last N cycles. Each new cycle computes per-task deltas. If any delta drops below threshold, the cycle is rejected and flagged for human review.

### Information-Theoretic Limits

Kolmogorov complexity and Lob's theorem set upper bounds on what a system can prove about itself. Schmidhuber's formal Godel Machine (Lesson 4) aimed at the highest bound; no one has completed a non-trivial proof. Lob's result says: if a system provably believes "if I prove I should do X, I will do X," it will do X without having proved it should, a famous self-reference failure.

The implication for our primitives: they don't close the safety problem. They make silent failure more expensive. A malicious or drifting loop that would have silently bypassed a missing check now must actively subvert an explicit one, and that's a more detectable signature.

### A Worked Example

Suppose an agent proposes an edit. The gate stack:

1. Invariant check: module hashes, tool permission manifest, constitutional header.
2. Anchor check: objective statement matches approved version (byte-for-byte or semantically).
3. Multi-objective evaluation: performance, safety, fairness, robustness dimensions.
4. Regression detection: no dimension dropped beyond tolerance.

All four pass, the edit lands. Any single failure pauses the loop.

## Use It

`code/main.py` runs a bounded self-improvement loop on Lesson 4's DGM-style toy, but layers the four primitives on top. Each primitive can be independently enabled or disabled. The demonstration point: each primitive catches a specific class of failure, and removing any one lets that class through.

## Ship It

`outputs/skill-bounded-loop-review.md` audits a proposed bounded loop and scores how many of the four primitives it actually implements (vs. claims to).

## Exercises

1. Run `code/main.py` with all primitives enabled. Confirm the loop still improves on the primary metric without letting hacking through.

2. Disable regression detection. Construct an input that causes a silent capability loss to be accepted.

3. Disable multi-objective constraints. Show the loop converging on the performance dimension while a safety dimension drops.

4. Design an alignment anchor for a coding agent. What text, stored where, checked how?

5. Read the ICLR 2026 RSI Workshop summary. Pick one of the four primitives and propose a concrete improvement over the current state of the art.

## Key Terms

| Term | Colloquial usage | Actual meaning |
|---|---|---|
| Invariant | "Property that must always be true" | A property checked by external code before and after every edit |
| Alignment anchor | "Pinned objective" | Immutable core-goal representation outside the loop's edit surface |
| Multi-objective constraint | "All dimensions must hold" | Performance, safety, fairness, robustness—all required |
| Regression detection | "Drop means pause" | Pauses loop when historical metric deltas suggest capability loss |
| Kolmogorov bound | "Information-theoretic limit" | Limits what a system can prove about its own successor |
| Lob's theorem | "Self-reference trap" | A system can act on "I should do X" without having proven it should |
| Gate stack | "Layered checks" | Multiple primitives composed; any failure rejects the edit |
| Bounded improvement | "Mitigation, not proof" | Raises cost of silent failure; doesn't close the safety problem |

## Further Reading

- [ICLR 2026 RSI Workshop summary (OpenReview)](https://openreview.net/pdf?id=OsPQ6zTQXV) — Convergence on the four primitives.
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — Multi-objective capability thresholds.
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — Deceptive alignment monitoring as an invariant primitive.
- [Schmidhuber (2003). Godel Machines](https://people.idsia.ch/~juergen/goedelmachine.html) — Formal-proof ancestor of these primitives.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — A reasoning-based alignment anchor.
