# Capstone Project 06 — Kubernetes DevOps Troubleshooting Agent

> AWS's DevOps Agent is now GA, Resolve AI published its K8s troubleshooting playbook, NeuBird demonstrated semantic monitoring, and Metoro tied its AI SRE to per-service SLOs. The production pattern has solidified: an alert webhook fires, an agent reads telemetry, walks a graph of K8s objects, ranks root-cause hypotheses, and sends a Slack brief with approval buttons. Read-only by default. Every repair goes through a human gate. This capstone project builds exactly such an agent, evaluates it on 20 synthetic incidents, and benchmarks against AWS's Agent on three shared cases.

**Type:** Capstone
**Languages:** Python (agent), TypeScript (Slack integration)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools & MCP), Phase 14 (Agents), Phase 15 (Autonomous Systems), Phase 17 (Infrastructure), Phase 18 (Safety)
**Phases involved:** P11 · P13 · P14 · P15 · P17 · P18
**Time:** 30 hours

## The Problem

The 2025-2026 SRE narrative became: "AI agents triage incidents, humans approve fixes." AWS DevOps Agent, Resolve AI, NeuBird, Metoro, and PagerDuty AIOps all ship this pattern in production. The agent reads Prometheus metrics, Loki logs, Tempo traces, kube-state-metrics, and a knowledge graph of K8s objects. It produces a ranked, citation-backed set of root-cause hypotheses within five minutes. It never executes a destructive command without obtaining explicit human approval via Slack.

Most of the hard work is scoping and safety, not reasoning. The agent needs a read-only-by-default RBAC surface, a hardened MCP tool server, and an audit log of every command "considered vs executed." It needs to know when it is out of its depth and escalate. And it must run cheaply enough that an OOM-kill cascade doesn't generate a $5,000 agent bill.

## The Concept

The agent operates over a knowledge graph. Nodes are K8s objects (Pod, Deployment, Service, Node, HPA, PVC) plus telemetry sources (Prometheus time series, Loki streams, Tempo traces). Edges encode ownership (Pod -> ReplicaSet -> Deployment), scheduling (Pod -> Node), and observation (Pod -> Prometheus time series). The graph is kept fresh by a kube-state-metrics sync and resampled on each alert.

When an alert fires, the agent starts from the affected object to locate the root cause. It walks edges, pulls relevant telemetry slices (last 15 minutes), and drafts a hypothesis. Hypotheses are ranked by evidence: how many telemetry citations support it, how recent they are, and how specific they are. The top three hypotheses are sent to Slack with graph-path visualizations and approval buttons for repair actions.

Repairs are gated. Allowed actions by default are read-only. Destructive actions (scale down, rollback, delete Pod) require Slack approval; ArgoCD rollback hooks require an auth token the agent never holds. The audit log records every command the agent *considered*—not just those it executed—so the review process can catch near-misses.

## Architecture

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## Tech Stack

- Observability sources: Prometheus, Loki, Tempo, kube-state-metrics
- Knowledge graph: Neo4j (managed) or kuzu (embedded) for K8s objects + telemetry edges
- Agent: LangGraph with per-tool allowlisting, read-only by default
- Tool transport: FastMCP over StreamableHTTP; destructive tools on a separate server behind an approval gate
- Models: Claude Sonnet 4.7 for root-cause reasoning, Gemini 2.5 Flash for log summarization
- Repair: ArgoCD rollback webhook, PagerDuty escalation, Slack approval cards
- Audit: Append-only structured log (considered, executed, approved, outcome)
- Deployment: K8s deployment with its own narrow RBAC role; dedicated namespace

## Build It

1. **Graph ingestion.** Sync kube-state-metrics into Neo4j/kuzu every 30s. Nodes: Pod, Deployment, Node, Service, PVC, HPA. Edges: OWNED_BY, SCHEDULED_ON, EXPOSES, MOUNTS, SCALES. Telemetry overlay edges: OBSERVED_BY (a Pod is observed by a Prometheus time series).

2. **Alert receiver.** A FastAPI endpoint that accepts PagerDuty or Alertmanager webhooks. Extracts affected objects and SLO violations.

3. **Read-only tool surface.** Wrap kubectl, Prometheus query, Loki logql, and Tempo traceql via FastMCP. Each tool has only a narrow RBAC verb ("get", "list", "describe"). No "delete", "exec", or "scale" in the default server.

4. **Root-cause agent.** LangGraph with three nodes: `sample` pulls the last 15 minutes of telemetry slices, `walk` queries the graph for adjacent objects, `hypothesize` drafts ranked root-cause candidates with telemetry citations.

5. **Evidence scoring.** Each hypothesis gets a score = recency * specificity * inverse graph-path length * citation count. Return top three.

6. **Slack brief.** Send an attachment with hypotheses, graph-path visualizations (server-rendered subgraph images), and up to one repair action approval button.

7. **Repair gate.** Destructive tools (scale down, rollback, delete) live on a second MCP server behind an approval token. Only after the Slack card is approved by a human can the agent invoke them.

8. **Audit log.** Append-only JSONL: for each candidate command, record whether it was considered, executed, who approved it. Ship to S3 daily.

9. **Synthetic incident suite.** Build 20 scenarios: OOMKill cascade, DNS flap, HPA thrash, PVC full, noisy neighbor, failing sidecar, bad ConfigMap rollout, certificate rotation, image-pull backoff, etc. Score the agent on root-cause accuracy and time-to-hypothesis.

## Use It

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## Ship It

`outputs/skill-devops-agent.md` is the deliverable. Given a K8s cluster and alert sources, the agent produces ranked root-cause hypotheses and a Slack-gated repair flow.

| Weight | Criterion | How to Measure |
|:-:|---|---|
| 25 | RCA accuracy on scenario suite | Root cause correct on >=80% of 20 synthetic incidents |
| 20 | Safety | Destructive-action guard in audit log never fires without Slack approval |
| 20 | Time-to-hypothesis | Alert-to-Slack-brief p50 under 5 minutes |
| 20 | Explainability | Every hypothesis has a graph path and telemetry citations |
| 15 | Integration completeness | PagerDuty, Slack, ArgoCD, Prometheus end-to-end |
| **100** | | |

## Exercises

1. Run your agent on the same three incidents used for the AWS DevOps Agent demo. Publish a side-by-side comparison. Report where the agents diverge.

2. Add a "near-miss" audit that flags any command the agent *considered* that would have been destructive without approval. Measure the near-miss rate over one week.

3. Swap the hypothesis model from Claude Sonnet 4.7 to self-hosted Llama 3.3 70B. Measure the RCA accuracy delta and cost per incident in dollars.

4. Build a causal filter: distinguish correlated telemetry spikes from actual root causes. Train a small classifier on the labels of the 20 scenarios.

5. Add a rollback dry-run: perform the ArgoCD rollback against a staging cluster with the same manifests. Validate the rollback plan in a real cluster before the Slack approval button.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| K8s knowledge graph | "cluster graph" | Nodes = K8s objects + telemetry time series; edges = ownership, scheduling, observation |
| Read-only-by-default | "scoped RBAC" | The agent's service account has only get/list/describe verbs; destructive verbs live on a separate server behind approval |
| Audit log | "considered vs executed" | An append-only record of every candidate command, whether it ran, and who approved it |
| Hypothesis ranking | "evidence score" | Recency x specificity x inverse graph-path length x citation count |
| Slack approval card | "HITL gate" | An interactive Slack message with repair buttons; the agent cannot proceed until a human clicks |
| Telemetry citation | "evidence pointer" | A Prometheus query, Loki selector, or Tempo trace URL backing a given assertion |
| MTTR | "time to resolve" | Wall-clock time from alert fire to SLO recovery |

## Further Reading

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — the 2026 reference standard
- [Resolve AI K8s troubleshooting](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — competitor reference
- [NeuBird semantic monitoring](https://www.neubird.ai) — semantic graph approach
- [Metoro AI SRE](https://metoro.io) — SLO-first production perspective
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — cluster state source
- [LangGraph](https://langchain-ai.github.io/langgraph/) — reference agent orchestrator
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP server framework
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — gated repair target
