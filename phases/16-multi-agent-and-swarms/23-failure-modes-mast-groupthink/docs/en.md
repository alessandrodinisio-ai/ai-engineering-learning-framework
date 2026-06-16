# Failure Modes — MAST, Groupthink, Monoculture, Cascading Errors

> The 2026 reference taxonomy is **MAST** (Cemri et al., NeurIPS 2025, arXiv:2503.13657), distilled from 1642 execution traces across 7 state-of-the-art open-source MAS, showing **41–86.7% failure rates**. Three root categories: **Specification Problems** (41.77%) — ambiguous roles, underspecified tasks; **Coordination Failures** (36.94%) — communication breakdowns, state desynchronization; **Verification Gaps** (21.30%) — missing checks, no quality gates. The **Groupthink** family (arXiv:2508.05687) adds: monoculture collapse (same base model → correlated failures), conformity bias (agents reinforce each other's errors), theory-of-mind deficits, mixed-motive dynamics, cascading reliability failures. Cascade example: retry storms — one payment failure triggers order retries, which trigger inventory retries, crushing the inventory service (10× load in seconds — needs circuit breakers). Memory poisoning: one agent's hallucination enters shared memory, downstream agents treat it as fact; gradual accuracy decay makes root-cause diagnosis painful. **STRATUS** (NeurIPS 2025) reports 1.5× mitigation success when deploying specialized detection / diagnosis / validation agents. This lesson treats failure modes as a first-class engineering concern.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 13 (Shared Memory), Phase 16 · 14 (Consensus & BFT), Phase 16 · 15 (Voting & Debate Topologies)
**Time:** ~75 minutes

## The Problem

Multi-agent systems fail 41-86.7% of the time on real tasks (Cemri et al. 2025 measured this across 7 open-source MAS). This isn't something you can tune away by "just adding more agents." These failures have structural causes. The MAST taxonomy gives you the categories. This lesson maps each category to a concrete detection, diagnosis, and mitigation pattern so the numbers stop looking random.

The 2026 production practice is to treat failure modes as design inputs. Your architecture isn't "good enough" until you can point at each MAST category and name the mitigation you deployed.

## The Concept

### MAST Categories

**Specification Problems (41.77% of failures).** Agent tasks aren't defined tightly enough. Examples:

- Ambiguous roles: two agents both think they're the reviewer.
- Underspecified tasks: "summarize this" when the user wants a specific angle.
- Implicit success criteria: agents can't tell if they succeeded.

Mitigations:
- Write explicit role contracts. Each agent's prompt states what it does *and what it doesn't do*.
- Acceptance tests per task. Before an agent starts, define "what done looks like."
- Pre-flight spec checks: a separate agent reviews the task definition before dispatch.

**Coordination Failures (36.94%).** Communication or state breakdown.

Examples:
- Two agents update shared state without synchronization.
- Messages lost between agents (queue failures, timeouts).
- State drift: agent A thinks the task is done; agent B is still executing.

Mitigations:
- Versioned shared state with optimistic concurrency.
- Explicit acknowledgment for critical messages (retry until acked).
- Periodic state-sync checkpoints; detect drift early.

**Verification Gaps (21.30%).** No independent check on outputs.

Examples:
- An agent claims success; nobody verifies.
- A chain of agents each trust the previous one's output.
- No test coverage for emergent combined behavior.

Mitigations:
- Independent verifier agents (Lesson 13). Read-only, with independent source access.
- Explicit handoff contracts: "A's output must pass checker C before B can start."
- Result logging for post-hoc analysis.

### The Groupthink Family (arXiv:2508.05687)

Five related failures when agents converge or imitate each other:

**Monoculture collapse.** Same base model or training data → correlated errors. When three agents share one LLM, they share its hallucinations.

**Conformity bias.** Agents adjust toward the loudest or most confident peer, even when wrong.

**ToM deficits.** Agents can't model each other's beliefs; coordination falls apart (Lesson 18).

**Mixed-motive dynamics.** Partially aligned incentives cause agents to drift toward a compromise middle ground that satisfies nobody.

**Cascading reliability failures.** One component's error mode triggers error modes in dependent components.

### Cascade Example — Retry Storms

A classic 2026 incident pattern:

```
Payment service has 10% request failures
   ↓
Order agent retries payment (exponential backoff, but naive)
   ↓
Each retry is a new "order-inventory" check
   ↓
Inventory service sees 2× normal load
   ↓
Inventory service starts timing out
   ↓
Every order retries inventory checks
   ↓
Inventory service sees 10× normal load
   ↓
Cluster goes down
```

The fix is classical: **circuit breakers**. When downstream error rates exceed a threshold, short-circuit with cached or default results. Plus a bounded retry budget per request.

Circuit breakers are one of the few multi-agent failure mitigations you can borrow directly from distributed systems without modification.

### Memory Poisoning (Revisited)

From Lesson 13: one agent's hallucination becomes shared-memory fact; downstream agents reason over poisoned facts. In MAST terms, this is a verification gap at the shared-memory layer.

Gradual accuracy decay is the symptom. You don't get a crash; you get slow drift that's hard to root-cause.

Mitigation: append-only logs, provenance, read-only verifiers. Covered in Lesson 13.

### STRATUS — Specialized Agents for Failure Detection

STRATUS (NeurIPS 2025) reports 1.5× mitigation success when you deploy three roles:

- **Detection agent.** Watches for symptom patterns (high divergence, retry spikes, accuracy drift).
- **Diagnosis agent.** Given symptoms, infers likely root causes from the MAST taxonomy.
- **Validation agent.** After mitigation is applied, checks whether symptoms are resolved.

This is SRE-style incident response applied to agent systems. All three roles can be LLM agents with specialized prompts.

### Failure Mode Audits

The 2026 best practice is an annual (or per-major-release) failure mode audit:

1. **Sample traces.** Collect ~1000 real execution traces.
2. **Classify.** For each trace's failure, map to a MAST + groupthink category.
3. **Compute category failure rates.** Which categories dominate in your system?
4. **Prioritize mitigations.** Which fix eliminates the most failures?
5. **Pick 2-3 mitigations.** Implement; re-audit next quarter.

The discipline matters more than the specific choices. Without audits, failures blend into noise and never get addressed systematically.

### When Systems Fail Silently

The most dangerous failure category is silent correctness failures. A system that fails loudly (crashes, exceptions, alerts) can be monitored. A system that produces "plausible but wrong" output can't be detected via exception logs. This is why verification gaps, while only 21.30% by count, are the most expensive failure category per incident.

Invest in:
- Sampling-based human review.
- Gold dataset regression tests.
- Cross-agent spot-checks on critical outputs.

### Fast Failures vs Slow Failures

Some failures are instant; some are slow. Instant failures (timeouts, schema mismatches, auth errors) are cheap to detect. Slow failures (memory poisoning, monoculture drift, role ambiguity) are expensive to detect and prevent.

The 2026 engineering action: instrument proxy metrics for slow failures so you can catch drift before it becomes a visible error. Agreement rate, retry rate, output length distributions, and edit distance between adjacent agent versions are all useful proxies.

## Build It

`code/main.py` implements:

- `FailureTaxonomy` — classifies simulated incidents into MAST + groupthink categories.
- `CircuitBreaker` — the classic pattern; opens when error rate exceeds threshold.
- `RetryStormSimulator` — demonstrates cascading failure; toggles circuit breaker on/off.
- `DetectionAgent` — scripted STRATUS-style symptom matcher.

Run:

```
python3 code/main.py
```

Expected output:
- Retry storm without circuit breaker: inventory errors explode (simulated).
- With circuit breaker: capped at threshold; degraded-mode responses.
- Detection agent flags the pattern and names the MAST category.

## Use It

`outputs/skill-mast-auditor.md` runs a MAST-style failure mode audit on a multi-agent system. Traces → classification → mitigation prioritization.

## Ship It

Failure mode discipline in production:

- **Do a MAST audit quarterly.** Not annually. Categories shift as your system grows.
- **Put circuit breakers everywhere.** Every outbound call to any dependent service. Default open threshold at 5-10% error rate.
- **Gold datasets.** Small, high-quality, human-reviewed. Run regression against them weekly.
- **STRATUS triad.** Detection + diagnosis + validation agents monitoring production. Start with just the detection agent; add diagnosis when symptoms are noisy.
- **Failure budgets.** Set explicit SLOs for failure rates by category. Exceeding the budget triggers a "stop the release" conversation.

## Exercises

1. Run `code/main.py`. Confirm the circuit breaker caps the retry storm. Change the failure threshold and observe the tradeoffs.
2. Implement a **slow-failure proxy**: agreement rate among 3 parallel agents. Trigger an alert when it drops sharply. Simulate monoculture drift by gradually correlating agent outputs.
3. Read Cemri et al. (arXiv:2503.13657). Pick one of their 7 MAS systems and map its top 3 failure categories. How do these compare to what MAST predicts?
4. Read the groupthink paper (arXiv:2508.05687). Identify which of the five patterns is hardest to detect in production. Propose a proxy metric.
5. Design a STRATUS-style detection-diagnosis-validation triad for a specific multi-agent system you know. What symptoms does detection watch? What mitigations does diagnosis recommend? How does validation confirm they worked?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MAST | "The 2026 taxonomy" | Cemri 2025; 3 root categories + 14 failure subtypes. |
| Specification Problem | "Role ambiguity" | Insufficient task or role definition; agents don't know what to do. |
| Coordination Failure | "State drift" | Communication or synchronization breakdown between agents. |
| Verification Gap | "Nobody checked" | Outputs accepted without independent validation. |
| Groupthink family | "Homogenization failures" | Monoculture, conformity, ToM deficits, mixed-motive, cascades. |
| Monoculture collapse | "Same model, same hallucinations" | Correlated errors from shared base model or training data. |
| Retry storm | "Cascading error amplification" | One failure triggers retries that amplify load downstream. |
| Circuit breaker | "Fast-fail by error rate" | Opens when error rate exceeds threshold; short-circuits with defaults. |
| STRATUS | "Incident response triad" | Detection + diagnosis + validation agents. 1.5× mitigation success. |
| Memory poisoning | "Hallucination propagation" | Shared memory facts contaminated; downstream agents reason over poison. |

## Further Reading

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — The MAST taxonomy, NeurIPS 2025
- [Groupthink failures in multi-agent LLMs](https://arxiv.org/abs/2508.05687) — Monoculture, conformity, and the five-family taxonomy
- [STRATUS — specialized agents for MAS incident response](https://neurips.cc/) — NeurIPS 2025 proceedings entry (detection + diagnosis + validation)
- [Release It! — stability patterns (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/) — The standard reference for circuit breakers
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — Production failure mode notes
