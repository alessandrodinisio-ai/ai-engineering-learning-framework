# Claude Code as Autonomous Agent: Permission Modes & Auto Mode

> Claude Code exposes seven permission modes. "plan" asks before every action, "default" asks only for risky actions, "acceptEdits" auto-approves file writes but still confirms shell execution, "bypassPermissions" approves everything. Auto Mode (March 24, 2026) replaces per-action approval with a two-stage parallel safety classifier: a single-token fast check runs on every action; flagged actions trigger a chain-of-thought deep review. Action budgets are enforced via `max_turns` and `max_budget_usd`. Auto Mode shipped as a research preview—Anthropic has stated explicitly that the classifier alone is not sufficient.

**Type:** Learn
**Languages:** Python (standard library, two-stage classifier simulator)
**Prerequisites:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 09 (Coding agent landscape)
**Time:** ~45 minutes

## The Problem

An autonomous coding agent on your machine is a standalone security category. The attack surface is everything the agent can reach—filesystem, network, credentials, clipboard, any browser tab, any open terminal. Bruce Schneier and others have publicly noted this: computer-use agents are not a "feature update" to chatbots; they are a new tool with a new risk profile.

Claude Code's permission system is Anthropic's answer. Not a single "autonomous / not autonomous" switch, but seven modes across a capability ladder: plan → default → acceptEdits → … → bypassPermissions. Each mode is a different trade-off between speed and per-action review. Auto Mode (March 2026) adds a two-stage classifier that moves approval of classifier-deemed-safe actions out of the user's critical path while retaining a review layer for flagged actions.

The engineering question: what does this system catch, what does it miss, and which mode should a given task actually use.

## The Concept

### Seven Permission Modes

| Mode | Behavior | When to use |
|---|---|---|
| `plan` | Agent proposes a plan; user approves the whole plan; each action reviewed before execution | Unfamiliar tasks; production-adjacent code; first time using the agent on a repo |
| `default` | Agent runs actions; any "risky" action (shell execution, destructive ops, network calls) prompts user | Most interactive coding sessions |
| `acceptEdits` | File writes auto-approved; shell execution and network calls still prompt | Refactoring across many files |
| `acceptExec` | Curated whitelist of shell commands auto-approved; writes auto-approved | Tight inner loop where every shell command is something like `npm test` |
| `autoMode` | Two-stage safety classifier; flagged actions escalate to review | Long unattended runs in constrained workspaces |
| `yolo` | Skips most prompts; still runs tool whitelist/blacklist | Throwaway sandboxes, CI tasks, research scripts |
| `bypassPermissions` | Approves everything | Documentation states "only inside throwaway containers you're willing to lose" |

(Names above match public Claude Code documentation; "yolo" is the informal shorthand for the permissive middle ground.)

### Auto Mode in One Page

Auto Mode (shipped March 24, 2026) is the first permission mode that delegates per-action approval to a model. Structure:

1. **Stage 1 — Single-token classifier.** Runs on every proposed action. Cheap. Returns "safe" or "flagged." Designed to run in parallel with the main agent loop so approved actions don't wait in queue.
2. **Stage 2 — Secondary policy/safety review.** Runs only on flagged actions. Does a focused policy and safety evaluation against the action, the session's current state, and the declared task. If confidence is low, escalates to user HITL.

Budget controls sit alongside the classifier:

- `max_turns` — total iterations in a session.
- `max_budget_usd` — dollar cap that aborts the session.
- Per-tool action count caps (no more than N `WebFetch` calls, etc.).

### What the System Catches

- Overt prompt injections stuffed into tool inputs whose injection instruction maps to a known-risky action shape.
- Repetitive tool loops—the classifier can see that action N+1 is nearly identical to action N, five times running.
- Obviously out-of-scope shell commands in a session that was doing only file edits.

### What the System Misses

- **Subtle prompt injections** that modulate behavior without producing any single flagged action. Indirect prompt injection is not a fully patchable bug (OpenAI preparedness lead, 2025, on browser agents—see Lesson 11).
- **Semantic out-of-scope.** Each individual action can look safe while the combined trajectory is harmful. The classifier judges actions; it does not re-derive user intent.
- **Exfiltration via legitimate channels.** Writing data to a file you own and then `git push`-ing to a public repo is a sequence of allowed actions; the problem is their composition.

### The Research Preview Positioning

Anthropic shipped Auto Mode as a research preview. Documentation states explicitly that the classifier is a layer, not a solution: users are expected to combine Auto Mode with budgets, whitelists, isolated workspaces, and trace auditing (Lessons 12-16). The preview positioning also reflects the documented eval-vs-deployment gap (Lesson 1)—a classifier that passes offline evaluation may behave differently in real sessions where user context is ambiguous.

### Where the Ladder Fits in Your Workflow

- Unfamiliar tasks: start with `plan`. Reading a plan is cheaper than rolling back a bad run.
- Known refactoring: `acceptEdits` saves many confirmation clicks.
- Unattended background runs: `autoMode`, only inside a workspace where you've tested the blast radius (no credentials, no production mounts, no outbound traffic you haven't opted into).
- Throwaway containers: `yolo` / `bypassPermissions` is acceptable—if and only if the container and its credentials are disposable.

## Use It

`code/main.py` simulates the two-stage classifier. Stage 1 is a cheap keyword rule on proposed actions; stage 2 is a slower multi-rule reviewer. The driver feeds a small synthetic trace (safe actions, one prompt-injection attempt, one repetitive loop) and shows where the classifier catches and where it misses.

## Ship It

`outputs/skill-permission-mode-picker.md` matches a task description to the right permission mode, budget cap, and required isolation.

## Exercises

1. Run `code/main.py`. Which synthetic action type is never flagged by stage 1 but always caught by stage 2? Which type is missed by both?

2. Extend stage 1's rule set to catch a specific known-bad shape (e.g., `curl $ATTACKER/exfil`). Measure false-positive rate on a sample of benign actions.

3. Read Anthropic's "How the agent loop works" documentation. List every piece of external state the agent touches by default in `default` mode. Which ones would you need to gate separately before running `autoMode` unattended?

4. Design a budget for a 24-hour unattended run: `max_turns`, `max_budget_usd`, per-tool caps, whitelist. Justify each number.

5. Describe a trajectory where every individual action is approved by both stage 1 and stage 2, yet the combined behavior is misaligned. (Lesson 14 covers how kill switches and canary tokens address this.)

## Key Terms

| Term | Colloquial usage | Actual meaning |
|---|---|---|
| Permission mode | "How much the agent can do" | One of seven named policies controlling per-action approval |
| plan mode | "Ask before anything" | Agent writes a plan; user approves before execution |
| acceptEdits | "Let it write files" | File writes auto-approved; shell execution still prompts |
| autoMode | "Auto-approve" | Two-stage safety classifier; flagged actions escalate |
| bypassPermissions | "Full YOLO" | Approves everything; intended for throwaway containers |
| Stage 1 classifier | "Fast token check" | Single-token rule on proposed action; runs in parallel |
| Stage 2 classifier | "Deep review" | Chain-of-thought reasoning on flagged actions |
| Research preview | "Not GA" | Anthropic's positioning for features whose failure modes are still being mapped |

## Further Reading

- [Anthropic — How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) — Permission modes, budgets, action format.
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — Managed service execution model.
- [Anthropic — Claude Code product page](https://www.anthropic.com/product/claude-code) — Feature surface and Auto Mode release announcement.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — The reasoning-based layer that shapes classifier judgments.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Internal perspective on long-horizon permission design.
