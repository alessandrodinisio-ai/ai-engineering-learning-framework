# Chaos Engineering for LLM Production

> In 2026, chaos engineering for LLMs is its own discipline. Prerequisites before running experiments in production: defined SLIs/SLOs, observability with traces+metrics+logs, automatic rollback, runbooks, on-call. The architecture has four facets: control (experiment scheduler), target (services, infrastructure, data stores), safety (guards + abort + traffic filtering), observability (metrics + traces + logs), and feedback (into SLO adjustments). Guardrails are mandatory: burn-rate alerts pause experiments when daily error-budget burn > 2x expected; suppression windows + trace-ID correlation deduplicate alert noise. Cadence: weekly small canary + SLO review; monthly game day + post-mortem; quarterly cross-team resilience audit + dependency mapping. LLM-specific experiments: memory overload, network fault, provider outage, malformed prompt, KV cache eviction storm. Tools: Harness Chaos Engineering (LLM-derived suggestions, blast-radius reduction, MCP tool integration); LitmusChaos (CNCF); Chaos Mesh (CNCF Kubernetes-native).

**Type:** Learn
**Languages:** Python (standard library, a toy chaos-experiment executor)
**Prerequisites:** Phase 17 · 23 (SRE for AI), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## Learning Objectives

- Name the five prerequisites for chaos engineering (SLI/SLO, observability, rollback, runbooks, on-call) and explain why skipping any one breaks the practice.
- Draw the four facets (control, target, safety, observability) and the feedback loop into SLOs.
- List five LLM-specific experiments (memory overload, network fault, provider outage, malformed prompt, KV eviction storm).
- Pick the right tool for a given stack — Harness, LitmusChaos, Chaos Mesh.

## The Problem

Chaos testing in traditional stacks is mature. LLM stacks add new failure modes. A 4K-token prompt with a poison character stalls the tokenizer for 12 seconds. An upstream provider returns 429; your gateway retries; your service OOMs under the retry-amplified concurrency. A KV cache eviction storm under burst load triggers heavy re-prefill cascades that saturate compute.

None of these surface in unit tests. Chaos engineering is how you find them before your users do.

## The Concept

### Prerequisites

Don't run chaos in production without the following:

1. **SLIs/SLOs** — defined service-level indicators and objectives.
2. **Observability** — traces, metrics, logs wired to dashboards.
3. **Automatic rollback** — Phase 17 · 20's policy-flag rollback.
4. **Runbooks** — structured, per Phase 17 · 23.
5. **On-call** — someone to respond.

Missing any one and chaos becomes a real incident.

### Four Facets + Feedback

**Control plane** — experiment scheduler (Litmus workflow, Chaos Mesh schedule, Harness UI).

**Target plane** — services, pods, nodes, load balancers, data stores.

**Safety plane** — kill switch, suppression windows, blast-radius limits, error-budget gates.

**Observability plane** — normal metrics + trace-ID correlation to distinguish chaos-induced from natural failures.

**Feedback loop** — findings feed into SLO adjustments, runbook updates, code fixes.

### Guardrails Are Mandatory

- **Burn-rate alert**: if daily error-budget burn exceeds 2x expected, pause the experiment.
- **Suppression windows**: during an experiment, silence non-experiment alerts in the blast radius.
- **Trace-ID correlation**: all experiment-induced errors carry a tag so on-call can deduplicate.

### Five LLM-Specific Experiments

1. **Memory overload** — send long-context requests at high concurrency, forcing a KV cache preemption storm. Observe: does the service degrade gracefully or crash?

2. **Network fault** — sever the connection between inference gateway and provider. Observe: does fallback activate within SLA? (Phase 17 · 19)

3. **Provider outage simulation** — OpenAI returns 100% 429s. Observe: does routing fail over to Anthropic? (Phase 17 · 16, 19)

4. **Malformed prompt** — inject payloads that stall the tokenizer (e.g., deeply nested unicode, enormous UTF-8 code points). Observe: can a single request lock a worker?

5. **KV eviction storm** — force eviction by saturating vLLM's block budget. Observe: does LMCache recover, or does the service degrade?

### Cadence

- **Weekly** — small canary experiments in staging, maybe 5% of production.
- **Monthly** — planned game day targeting a specific scenario; cross-team participation; post-mortem.
- **Quarterly** — cross-team resilience audit; dependency graph update.

### Tools

- **Harness Chaos Engineering** — commercial; AI-derived experiment suggestions; blast-radius reduction; MCP tool integration.
- **LitmusChaos** — CNCF graduated; Kubernetes workflow-based.
- **Chaos Mesh** — CNCF sandbox; Kubernetes-native CRD-style.
- **Gremlin** — commercial; broad support.
- **AWS FIS** / **Azure Chaos Studio** — managed cloud offerings.

### Start Small

First experiment: pod-kill a single decode replica under steady-state traffic. Observe rerouting and recovery. If that runs safely, escalate to network chaos.

First LLM-specific experiment: inject a provider 429 for 5 minutes. Observe fallback. Most teams discover their fallback hasn't been fully tested.

### Numbers You Should Remember

- Four facets: control, target, safety, observability.
- Burn-rate pause: 2x expected daily budget burn.
- Cadence: weekly canary, monthly game day, quarterly audit.
- Five LLM experiments: memory, network, provider, malformed prompt, KV storm.

## Use It

`code/main.py` simulates three chaos experiments with safety-plane gates. Reports which experiments would trigger burn-rate abort.

## Ship It

This lesson produces `outputs/skill-chaos-plan.md`. Given a stack and maturity level, it picks the top three experiments and tools.

## Exercises

1. Run `code/main.py`. Which experiment triggers the burn-rate gate, and why?
2. Design the first five chaos experiments for a vLLM-based RAG service. Include success criteria.
3. Your burn-rate alert paused an experiment. How do you determine root cause — chaos or natural?
4. Argue whether chaos should run in production or only in staging. When is production the right answer?
5. Name three LLM-specific failure modes that generic network chaos cannot reproduce.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| SLI / SLO | "Service objectives" | Indicators + targets; required prerequisite |
| Blast radius | "Scope" | Set of services / users affected by an experiment |
| Burn-rate alert | "Budget gate" | Fires when error-budget burn rate > 2x expected |
| Game day | "Monthly drill" | Planned cross-team chaos exercise |
| LitmusChaos | "CNCF workflow" | CNCF-graduated Kubernetes chaos tool |
| Chaos Mesh | "CNCF CRD" | CNCF sandbox Kubernetes-native chaos |
| Harness CE | "Commercial AI-assisted" | Harness chaos with AI suggestions |
| Malformed prompt | "Tokenizer bomb" | Input that stalls tokenization |
| KV eviction storm | "Preemption cascade" | Mass eviction triggering heavy re-prefill |

## Further Reading

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
