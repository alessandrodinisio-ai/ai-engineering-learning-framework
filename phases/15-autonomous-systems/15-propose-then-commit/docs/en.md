# Human in the Loop: Propose-Then-Commit

> The 2026 consensus on HITL is specific. It's not "agent asks, user clicks Approve." It's propose-then-commit: proposed actions are persisted to a durable store with an idempotency key; surfaced to a reviewer with intent, data lineage, touched permissions, blast radius, and rollback plan; committed only on affirmative confirmation; and verified post-execution to confirm the side effect actually happened. LangGraph's `interrupt()` with PostgreSQL checkpoint, Microsoft Agent Framework's `RequestInfoEvent`, Cloudflare's `waitForApproval()` all implement the same shape. The canonical failure mode is rubber-stamp approval: clicking "Approve?" without reading. The documented mitigation is challenge-and-response with explicit checklists.

**Type:** Learn
**Languages:** Python (standard library, propose-then-commit state machine with idempotency)
**Prerequisites:** Phase 15 · 12 (Durable execution), Phase 15 · 14 (Tripwires)
**Time:** ~60 minutes

## The Problem

An agent takes an action. A user must decide: approve or not. If the decision is instantaneous, it's probably not a review. If it's structured, it's slow but trustworthy. The engineering question is how to make structured review the path of least resistance.

The 2023-era HITL pattern was a synchronous prompt: "Agent wants to send email to X with body Y—Approve?" User clicks Approve. Everyone feels the system is safe. In practice this interface was rubber-stamped heavily: users approved rapidly, approval predicted little, and when the agent erred, the audit trail showed a long list of approvals the user couldn't recall.

The 2026 pattern—propose-then-commit—moves HITL onto a persistent substrate with structured metadata and requires affirmative commitment. Every managed agent SDK has a version: LangGraph's `interrupt()`, Microsoft Agent Framework's `RequestInfoEvent`, Cloudflare's `waitForApproval()`. API names differ; the shape is the same.

## The Concept

### The Propose-Then-Commit State Machine

1. **Propose.** Agent produces a proposed action. Persisted to a durable store (PostgreSQL, Redis, Durable Object). Contains:
   - Intent (why the agent wants to do this)
   - Data lineage (what sources led to this proposal)
   - Touched permissions (which scopes / files / endpoints)
   - Blast radius (what's the worst case)
   - Rollback plan (if committed, how do we undo it)
   - Idempotency key (unique per proposal; re-submit returns the same record)
2. **Surface.** A reviewer sees the proposal with all metadata. The reviewer is a human (not the agent reviewing itself).
3. **Commit.** Affirmative confirmation. Action executes.
4. **Verify.** Post-execution, side effects are read back to confirm. If the verify step fails, the system is in a known-bad state and alerts fire.

### Idempotency Key

Without an idempotency key, a retry after a transient failure might execute an already-approved action twice. Concrete example: user approves "transfer $100 from A to B." Network blip. Workflow retries. User approved once, but the transfer executes twice. The idempotency key binds the approval to a single unique side effect; the second execution is a no-op.

This is the same idempotency pattern Stripe and AWS APIs use. Reusing it for agent approvals is explicit in Microsoft Agent Framework documentation.

### Persistence: Why Approvals Outlive Processes

The approval waiting room is state the agent doesn't own. The workflow is suspended (Lesson 12). When the approval arrives, the workflow resumes from that exact point. This is why LangGraph pairs `interrupt()` with PostgreSQL checkpoints rather than in-memory state—a two-day-later approval still finds the workflow intact.

### Rubber-Stamp Approval and Challenge-and-Response Mitigation

The default HITL UI ("Approve" / "Reject" button) produces rapid approvals with no real review. The documented mitigation: a challenge-and-response checklist that requires affirmative answers to specific questions before the Approve button is enabled. Concrete shape:

- "Do you understand which resource this touches? [ ]"
- "Have you verified the blast radius is acceptable? [ ]"
- "Do you have a rollback plan if this fails? [ ]"

Not bureaucracy for its own sake—a forcing function. Reviewers who can't check these boxes either ask for clarification (escalation) or reject (safe default). Anthropic's agent safety research explicitly cites checklist-driven HITL as a mitigation for the rubber-stamp approval pattern.

### What Counts as Consequential

Not every action needs propose-then-commit. 2026 guidance:

- **Consequential actions** (always HITL): irreversible writes, financial transactions, external communications, production database changes, destructive filesystem operations.
- **Reversible actions** (sometimes HITL): edits to local files, staging-environment changes, reversible writes with clear rollback.
- **Reads and inspections** (never HITL): reading a file, listing resources, calling a read-only API.

### Post-Action Verification

"Commit ran" does not equal "side effect happened." Network partitions and race conditions can produce a workflow that thinks it succeeded while the backend didn't persist. The verify step re-reads the target resource after commit to confirm. This is the same pattern as database transactions with `RETURNING` clauses, or AWS `GetObject` after a `PutObject`.

### EU AI Act Article 14

Article 14 mandates effective human oversight for EU high-risk AI systems. "Effective" is not decorative. The regulatory language explicitly excludes rubber-stamp patterns. Propose-then-commit with challenge-and-response is the shape that survives Article 14 scrutiny in Microsoft's Agent Governance Toolkit compliance documentation.

## Use It

`code/main.py` implements a propose-then-commit state machine in standard-library Python. The durable store is a JSON file. The idempotency key is a hash of (thread_id, action_signature). The driver simulates three cases: a clean approval flow, a retry after transient failure (must not re-execute), and a rubber-stamp default vs. a challenge-and-response flow.

## Ship It

`outputs/skill-hitl-design.md` reviews a proposed HITL workflow for conformance to the propose-then-commit shape and flags missing metadata, idempotency, verification, or challenge-and-response layers.

## Exercises

1. Run `code/main.py`. Confirm that a retry of an already-approved proposal uses the persisted record and does not re-execute. Now change the idempotency key to include a timestamp and show the retry re-executes.

2. Extend the proposal record with a `rollback` field. Simulate an execution whose verify step fails. Show that rollback fires automatically.

3. Read Microsoft Agent Framework's `RequestInfoEvent` documentation. Identify one metadata field the API includes that the toy engine lacks. Add it and explain what it guards against.

4. Design a challenge-and-response checklist for a specific action (e.g., "post to a public Twitter account"). What three questions must the reviewer answer? Why those three?

5. Pick a case where a synchronous "Approve?" prompt is sufficient (no persistence needed). Explain why, and name the risk class you're accepting.

## Key Terms

| Term | Colloquial usage | Actual meaning |
|---|---|---|
| Propose-then-commit | "Two-phase approval" | Persisted proposal + affirmative commit + verification |
| Idempotency key | "Retry-safe token" | Unique per proposal; second execution is a no-op |
| Data lineage | "Where it came from" | The specific source content that led to this proposal |
| Blast radius | "Worst case" | Scope of impact if the action goes wrong |
| Rubber-stamp | "Fast approval" | Clicking "Approve" without real review |
| Challenge-and-response | "Forced checklist" | Reviewer must answer specific questions affirmatively |
| RequestInfoEvent | "MS Agent Framework primitive" | Persistent HITL request with structured metadata |
| `interrupt()` / `waitForApproval()` | "Framework primitives" | LangGraph / Cloudflare equivalents of the same shape |

## Further Reading

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`, persistent approvals.
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` and Durable Objects.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL as a mitigation for long-horizon risk.
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — Regulatory baseline for high-risk systems.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — Constitutional framing around oversight.
