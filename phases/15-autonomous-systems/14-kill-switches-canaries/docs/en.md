# Kill Switches, Circuit Breakers & Canary Tokens

> A kill switch is a boolean held outside the agent's edit surface—a Redis key, a feature flag, a signed config—that disables the agent entirely. A circuit breaker is more granular: it trips on a specific pattern (five identical tool calls in a row), pauses the offending path, and escalates to a human. Canary tokens inherit from classic deception: a fake credential or honeypot record the agent has no legitimate reason to touch, whose access triggers an alert. eBPF-based datapaths (like Cilium) can rewrite a quarantined pod's egress to a forensics honeypot at the kernel level; published Cilium benchmarks report sub-millisecond P99 datapath latency under load (your propagation budget depends on how a policy update reaches nodes, not the datapath itself). Statistical detectors that adapt to a moving baseline (EWMA, CUSUM) will silently accept drift—layer them with hard constitutional limits that don't bend.

**Type:** Learn
**Languages:** Python (standard library, three-detector simulator: kill switch, circuit breaker, canary)
**Prerequisites:** Phase 15 · 13 (Cost governors), Phase 15 · 10 (Permission modes)
**Time:** ~60 minutes

## The Problem

Cost governors (Lesson 13) constrain what an agent can spend. They don't constrain what an agent can do within budget. An agent with a $50 velocity limit can still exfiltrate a secret, post an incorrect message, or delete a resource—expensive actions are often cheap in tokens.

This lesson covers three detectors that sit alongside the cost layer:

1. **Kill switch**: a boolean off-button held where the agent can't reach.
2. **Circuit breaker**: an action-pattern detector that pauses a specific path.
3. **Canary token**: a decoy that exposes an agent who touches something it should never touch.

All three are pre-LLM engineering. Classic deception, rate-limiting circuit breakers, and feature-flag kill are all older than autonomous agents. What's new is the attack surface: agents read untrusted content (Lesson 11), edit their own memory, and can compose many safe-looking actions into an unsafe one. The detectors named here work because they don't trust the agent's self-reporting.

## The Concept

### Kill Switch

A kill switch is a boolean the agent can read but cannot write. Implementation choices:

- **Feature flag in a managed service.** LaunchDarkly, Statsig, Unleash. Updates propagate in seconds.
- **Redis key polled by the agent.** Simple; requires the agent process to check every turn.
- **Signed config in object storage.** Agent validates signature on startup; refuses unsigned state.
- **OS-level signal or container lifecycle kill.** Docker `kill`, Kubernetes `kubectl delete pod`, systemd stop.

Properties of a correct kill switch:

- The agent cannot set it to `off`. (Lives in a system the agent's credentials can't write to.)
- It's checked on every consequential action, not just startup.
- When off, the agent does nothing externally observable, including writing logs to systems the agent can reach.
- Re-enabling is an explicit human action, not an automatic timeout.

### Circuit Breaker

A circuit breaker pauses a specific pattern, not the whole agent. Classic shape (from Nygard's 2007 book, still applicable):

- **Closed**: actions allowed.
- **Open**: actions blocked.
- **Half-open**: after cooldown, allow 1-3 probe attempts (default 1); success closes the breaker, any remaining failure re-opens it.

Agent-relevant triggers:

- Five identical tool calls in a row (repetitive loop).
- Five consecutive failures of the same tool with different inputs (systemic failure).
- Tool call rate above threshold (Lesson 13 velocity).
- A specific tool (e.g., `file.delete`) called after an out-of-trust content read (Lesson 11).

### Canary Tokens

A canary token (also called a honeytoken) is an input the agent should never touch. Access equals alert.

Examples for agents:

- A fake `AWS_SECRET_ACCESS_KEY` with no real permissions in the environment. Exfiltration attempts are instantly attributable.
- A fake database record marked with a known sentinel value. Any read or update to this record triggers an alert.
- A fake file in the workspace. Read or edit of this file is suspicious.
- A fake memory injected at session start. If it appears in a later output, the memory has been exfiltrated.

Canary design is workflow-specific. A canary the agent has legitimate reason to touch is noise.

### Why Layer Statistical and Hard Limits

Statistical detectors (EWMA, CUSUM, z-score on tool call rate) adapt to a moving baseline. This is a feature for legitimate drift and a bug for adversarial drift. A patient attacker can move the baseline slowly enough that no alert ever fires.

Hard constitutional limits (Lesson 17) don't bend. "Never email `secrets@`" as a constitutional rule is simpler, auditable, and immune to slow-drift gaming. The layered pattern: statistical detectors catch most noise, hard limits catch the attack that slips past the detector.

### Quarantine via eBPF Datapath Rewrite

Cilium's eBPF-based network policies can rewrite a quarantined pod's egress to a forensics honeypot at the kernel level. Cilium's published benchmarks report sub-millisecond P99 datapath latency on TCP_RR; the wall-clock budget for *triggering* a quarantine in your cluster is dominated by how a policy update propagates from control plane to node (on Kubernetes with operator pre-loading, hundreds of milliseconds is realistic; tail latency depends on cluster size and reconcile cadence). This gives you two things: the agent exfiltrates nothing real, and the agent's behavior inside the honeypot is fully observed. Same pattern as canary tokens, different layer: the agent reveals intent by what it tries to do while contained.

### What No Detector Catches

- **Semantic attacks composed from permitted primitives.** Every individual action passes every detector; the problem is the composition. This is Lesson 10's Auto Mode compound failure.
- **Attacks that include disabling the detector as part of the payload.** If an agent reads a page saying "before you act, disable canaries" and the agent has that capability, the detector is defeated. Canary tokens should live in systems the agent cannot modify.

## Use It

`code/main.py` simulates a short agent trace passing through three detectors. A kill switch held in an external dict; a circuit breaker that trips on five identical tool calls; a canary file whose read triggers an alert. Feed in a synthetic trace: legitimate actions, a repetitive loop, a canary probe, and a kill-switch-triggered scenario where the agent's actions are halted.

## Ship It

`outputs/skill-tripwire-design.md` reviews a proposed agent deployment's detector stack and flags gaps (missing kill switch, missing canary, circuit breaker thresholds too loose).

## Exercises

1. Run `code/main.py`. Confirm the circuit breaker fires on turn 5 (fifth identical call) and the canary fires on turn 9 (reading the fake secret).

2. Add a statistical detector: EWMA z-score on tool call rate. Feed a slowly drifting trace and show the detector never fires. Now add a hard limit (no more than 50 tool calls in 10 minutes) and show the hard limit fires on the same trace.

3. Design a set of canary tokens for a browser agent (Lesson 11). List at least three canaries and what each would detect.

4. Read Cilium network policy documentation. Describe concretely an egress-rewrite quarantine flow: which policy selector, which pod, which egress rewrite, which alert. What determines the wall-clock latency from "decide to quarantine" to "first redirected packet"?

5. Define a re-enablement process for a kill-switched agent. Who can re-enable? What must be logged? What must the agent change before re-enablement?

## Key Terms

| Term | Colloquial usage | Actual meaning |
|---|---|---|
| Kill switch | "Off button" | Boolean outside agent's edit surface; checked on every consequential action |
| Circuit breaker | "Pattern pause" | Trips on repetition, failure rate, or velocity for specific actions |
| Canary token | "Honeytoken" | Decoy the agent has no legitimate reason to touch; access triggers alert |
| Honeypot | "Forensics sandbox" | Redirected traffic/workspace where quarantined agent is observed |
| EWMA | "Moving average" | Exponentially weighted; adapts to drift (both feature and bug) |
| CUSUM | "Cumulative sum" | Detects sustained shift from baseline |
| Hard limit | "Constitutional rule" | Non-adaptive; constant regardless of history |
| Constitutional limit | "Always-true rule" | Tied to Lesson 17's constitution; agent cannot edit |

## Further Reading

- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Kill switch and circuit breaker framing for autonomous agents.
- [Microsoft Agent Framework — HITL and oversight](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — Production governance patterns.
- [OWASP LLM / Agentic Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — Detection and response requirements.
