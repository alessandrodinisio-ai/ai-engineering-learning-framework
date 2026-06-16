# Checkpoints & Rollback

> Every graph-state transition is persisted. When a worker crashes, its lease expires and another worker picks up at the latest checkpoint. Cloudflare Durable Objects hold state across hours or weeks. Propose-then-commit (Lesson 15) defines a rollback plan for every action. Post-action verification closes the loop. EU AI Act Article 14 mandates effective human oversight for high-risk systems — in practice this means checkpoints must be queryable, rollbacks must be rehearsed, and audit trails must survive a deployment. The sharp failure mode: without idempotency keys and precondition checks, a retry after a transient failure can execute an already-approved action twice. Post-action verification is what catches it.

**Type:** Learn
**Languages:** Python (stdlib, checkpoint-and-rollback state machine)
**Prerequisites:** Phase 15 · 12 (Durable Execution), Phase 15 · 15 (Propose-then-Commit)
**Time:** ~60 min

## The Problem

Durable execution (Lesson 12) makes a crashed agent recoverable. Propose-then-commit (Lesson 15) makes an approved action auditable. This lesson connects them: what happens when an approved action partially executes, crashes, then recovers? When does rollback run, and against what state?

Real systems wire this differently:

- **LangGraph** checkpoints every graph-state transition to PostgreSQL. When a worker crashes, its lease releases and another worker resumes at the latest checkpoint. Workflows pause at `interrupt()`, which itself persists.
- **Cloudflare Durable Objects** hold keyed state across hours or weeks. Co-locate compute with storage of approved actions.
- **Microsoft Agent Framework** exposes a `Checkpoint` primitive in its workflow API; replay plus idempotency covers retries.

In every case the combination that works is: idempotency keys (prevent duplicate execution) + precondition checks (state is still what we approved against) + post-action verification (side effect actually happened) + rollback when verification fails.

## The Concept

### Every Transition Persists

A graph-state transition is any step that moves a workflow from one named state to another. Naive implementations persist only at specific commit points; production implementations persist every transition. The cost (a few extra writes) is small relative to the reliability gain (replay can land anywhere, lease recovery is precise).

### Lease Recovery

When a worker crashes, the workflow is not lost; the lease (a short-lived claim stating this worker is executing this run) simply expires. Another worker picks up the latest checkpoint and resumes. This lease mechanism is what lets production systems survive rolling deployments without losing in-flight work.

### Idempotency Plus Preconditions

Idempotency alone is not enough. Consider: a workflow is approved to "transfer $100 from A to B when balance > $1000." The workflow is committed, crashes mid-execution, then recovers. If only the idempotency key is checked, execution resumes and the transfer fires once (correct). But consider that between crash and recovery, A's balance dropped to $500 via another workflow. The idempotency check still passes; the precondition does not. Without precondition checks we deliver an overdraft.

Every consequential action needs both:

- **Idempotency key**: prevents duplicate execution.
- **Precondition check**: confirms state still matches what was approved.

### Post-Action Verification

"The tool returned 200" is not verification. Real verification re-reads target state and confirms the side effect actually happened. Pattern:

- Database update: `UPDATE ... RETURNING *`, then assert the returned row matches intended state.
- Email send: check message ID in sent folder after submission.
- File write: read the file back and hash it.
- API call: follow-up `GET` on the target resource.

If verification fails, the workflow is in a known bad state. Rollback fires.

### Rollback Plans

Every consequential action in propose-then-commit (Lesson 15) carries a rollback plan. Types:

- **In-band rollback**: directly reverse the side effect (`DELETE` after `INSERT`, send a correction email after send).
- **Compensating transaction**: a new action that cancels the original (standard SAGA pattern).
- **Out-of-band rollback**: alert a human, pause the workflow, leave the bad state for investigation.

No-op rollback ("we can't undo this") must be named in the proposal. Actions without rollback require stronger HITL at commit time (Lesson 15's challenge-response).

### EU AI Act Article 14 — Operational Reading

Article 14 requires "effective human oversight" for high-risk systems. In operational terms, implementers read it as:

- Checkpoints are queryable by an auditor.
- Rollback is rehearsed (tested end-to-end at least once).
- Audit trails survive a deployment (checkpoint backend is not ephemeral).
- Failed verifications are alerted, not silently logged.

A workflow that crashes mid-commit, recovers, and then completes its side effects without a verify + rollback path does not survive an Article 14 test.

### The Sharp Failure Mode: Duplicate Execution

The most common production incident in this space:

1. Action approved, idempotency key k.
2. Commit begins, executes, returns 200.
3. Workflow crashes before persisting "committed" state.
4. Workflow recovers; sees "approved but not committed"; re-executes.
5. Side effect fires twice.

Mitigation: persist an "in-flight" intent before execution, execute with an idempotency key, then mark "committed" only after post-action verification succeeds. If the action fires but the state write fails, you know to verify and (if needed) re-fire. If the state write succeeds but the action fails, you verify through the recovery path and fire exactly once.

## Use It

`code/main.py` implements a checkpointed workflow with idempotency, preconditions, verification, and rollback. The driver simulates four scenarios: clean run, retry after crash (idempotency catches it), precondition failure (workflow aborts without firing), verification failure (rollback fires).

## Ship It

`outputs/skill-rollback-rehearsal.md` designs a rollback rehearsal test for a proposed workflow and audits checkpoint backend audit-trail durability.

## Exercises

1. Run `code/main.py`. Verify all four scenarios. For the "crash during commit" case, confirm the action fires exactly once across retries.

2. Flip the "mark-as-done first" pattern so the state write fires after the action. Re-run the crash scenario. Measure how many duplicate actions fire.

3. Design a rollback plan for a specific production action (e.g., "post to a Slack channel"). Classify as in-band, compensating, or out-of-band. Justify the choice.

4. Take a workflow you know. Identify every state transition. Label each with its durability requirement (persist / don't persist). Count the ones you're currently not persisting.

5. Rehearsal rollback test: design an end-to-end test that runs a real workflow, crashes it, and confirms the rollback path fires. What does this test assert?

## Key Terms

| Term | Common shorthand | Actual meaning |
|---|---|---|
| Checkpoint | "save point" | Every graph-state transition persisted to a durable store |
| Lease | "worker claim" | Short-lived claim that a worker is executing a given run; expires on crash |
| Precondition | "state gate" | Assertion that state still matches the approved action |
| Post-action verify | "re-read check" | Confirms the side effect actually happened in the target system |
| In-band rollback | "direct undo" | Reverse a side effect with an inverse operation |
| Compensating transaction | "SAGA undo" | A new action that cancels the original |
| Mark-as-done-first | "state write order" | Persist committed state before returning from commit |
| Article 14 | "EU AI Act human oversight" | Operationally: queryable checkpoints, rehearsed rollback, auditable trails |

## Further Reading

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — Checkpoint primitive and lease recovery.
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — Durable Objects as state substrate.
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — Regulatory baseline.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Reliability framework for long-running workflows.
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — Workflow shape of Claude Code Routines.
