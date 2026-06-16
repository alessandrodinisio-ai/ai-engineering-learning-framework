# SRE for AI — Multi-Agent Incident Response, Runbooks, Predictive Detection

> AI SRE uses LLMs, grounded via RAG on infrastructure data (logs, runbooks, service topology), to automate the investigation, documentation, and coordination phases. The 2026 architectural pattern is multi-agent orchestration — specialized agents (logs, metrics, runbook) coordinated by a supervisor; the AI proposes hypotheses and queries, humans approve judgmental decisions. Datadog Bits AI and Azure SRE Agent ship this as managed products. Runbooks are evolving: NeuBird Hawkeye uses adversarial evaluation (two models analyze the same incident; agreement = confidence, disagreement = uncertainty); operational memory persists across team changes. Auto-remediation stays narrow: AI suggests, humans approve. Fully autonomous actions are limited (restart pod, roll back a specific deployment) with tight guardrails — anyone selling "set and forget" is overstating. Emerging frontier: pre-incident prediction. MIT research reports an LLM trained on historical logs + GPU temperatures + API error patterns predicted 89% of outages 10-15 minutes in advance. Prediction: by end of 2026, 95% of enterprise LLMs will have automated failover.

**Type:** Learn
**Languages:** Python (standard library, a toy multi-agent incident-triage simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 24 (Chaos Engineering)
**Time:** ~60 minutes

## Learning Objectives

- Draw the multi-agent AI SRE architecture: supervisor + specialized agents (logs, metrics, runbook) + human approval gate.
- Explain why auto-remediation is narrow (restart pod, revert deployment) rather than broad (refactor service).
- Describe the adversarial evaluation pattern (NeuBird Hawkeye): two models agree = confidence; disagree = escalate.
- Cite MIT's 89% early-detection result and the operational constraint: a prediction without execution is just a dashboard.

## The Problem

An on-call engineer gets paged at 3 AM. "High error rate on checkout." They check Datadog, Loki, three runbooks, deployment logs. 30 minutes later they realize the root cause is a vLLM OOM triggered by a KV cache spike. They restart the pod; errors disappear.

In 2026, the first 20 minutes of that investigation are automatable. Group logs by service, correlate to recent deployments, match against runbooks — all RAG + tool use. A supervised agent can do first-pass triage, presenting a hypothesis before the human opens Datadog.

Fully autonomous remediation is a different matter. Restart a pod: safe. Scale a GPU pool: safe if policy allows. Refactor a service: absolutely not. This discipline is about drawing that narrow line.

## The Concept

### Multi-Agent Architecture

```
          Incident
             |
             v
        Supervisor
        /    |    \
       v     v     v
  Log Agent  Metrics Agent  Runbook Agent
       |     |     |
       +-----+-----+
             |
             v
        Hypothesis + Evidence
             |
             v
        Human Approval
             |
             v
        Action (narrow set)
```

The supervisor decomposes the incident into sub-queries. Specialized agents have tool access (log search, PromQL, document retrieval). The supervisor synthesizes and presents hypothesis + evidence to the human. The human approves or redirects.

### Scope of Auto-Remediation

**Safe (narrow)**: restart pod, revert a specific deployment, scale pool within pre-approved bounds, enable a pre-approved feature flag.

**Unsafe (broad)**: change service topology, alter resource limits, deploy new code, modify IAM, alter databases.

Anyone selling "set and forget" is overstating. The safe set expands as AI SRE matures, but the boundary is real.

### Adversarial Evaluation (NeuBird Hawkeye)

Two models independently analyze the same incident. If they agree on root cause, confidence is high. If they disagree, escalate to a human with both hypotheses. Simple pattern, effective filter against hallucinated root causes.

### Operational Memory

Personnel turnover is the silent killer in traditional SRE — tribal knowledge leaves with people. AI SRE stores runbooks + post-mortems in a vector database; agents retrieve on every new incident. When a new engineer joins, the AI has the full history.

### Pre-Incident Prediction

MIT 2025 research: an LLM trained on historical logs, GPU temperatures, and API error patterns predicted 89% of outages 10-15 minutes ahead on a test set.

Reality check: a prediction without execution is just a dashboard. The operational question is "what do we do when we predict?" Pre-emptive drain? Page? Autoscale? The answer is policy-specific.

### 2026 Products

- **Datadog Bits AI** — managed SRE copilot within Datadog.
- **Azure SRE Agent** — Azure-native.
- **NeuBird Hawkeye** — adversarial evaluation + operational memory.
- **PagerDuty AIOps** — triage + deduplication.
- **Incident.io Autopilot** — incident commander + coordination.

### Runbooks as Code

Runbooks evolved from Confluence pages to versioned markdown with structured sections (symptoms, hypotheses, verification, actions). Structured runbooks feed better RAG retrieval. Any AI-SRE rollout starts with converting unstructured runbooks to structured ones.

### Numbers You Should Remember

- MIT early detection: 89% of outages, 10-15 minutes lead time.
- Multi-agent triage: supervisor + (logs, metrics, runbook) + human.
- Safe auto-remediation set: restart pod, revert deployment, scale within bounds.
- Adversarial evaluation: two models independently; agreement = confidence.

## Use It

`code/main.py` simulates a multi-agent triage: log agent finds errors, metrics agent finds CPU spike, runbook agent matches a known issue. Supervisor ranks hypotheses.

## Ship It

This lesson produces `outputs/skill-ai-sre-plan.md`. Given current on-call setup, incident volume, and team maturity, it designs an AI SRE rollout.

## Exercises

1. Run `code/main.py`. What if the log and metrics agents disagree? How does the supervisor arbitrate?
2. Define three "safe" auto-remediation actions for your service. Justify each.
3. Write a structured runbook template: sections, required fields, verification commands.
4. Predictive detection fires with 12 minutes lead time. What's your policy — page, pre-drain, or both?
5. Argue whether a 3-person team in 2026 should adopt AI SRE or wait. Consider maturity, volume, and risk.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| AI SRE | "Agent on-call" | LLM-powered incident investigation + coordination |
| Supervisor agent | "Orchestrator" | Top-level agent that decomposes incidents into sub-queries |
| Specialized agent | "Domain agent" | Sub-agent with tool access (logs, metrics, runbook) |
| Auto-remediation | "AI fixes it" | Narrow pre-approved actions; not broad refactoring |
| Operational memory | "Vector runbooks" | Post-mortems + runbooks stored in vector DB for RAG |
| Adversarial evaluation | "Dual-model check" | Independent analysis; agreement = confidence |
| NeuBird Hawkeye | "The adversarial one" | Product with adversarial evaluation + memory pattern |
| Bits AI | "Datadog's SRE agent" | Datadog-managed AI SRE |
| Pre-incident prediction | "Early detection" | Outage prediction with 10-15 minute lead time |

## Further Reading

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)
