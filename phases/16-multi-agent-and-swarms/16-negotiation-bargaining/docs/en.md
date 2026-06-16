# Negotiation and Bargaining

> Agents negotiate resources, prices, task assignments, and terms. The 2026 benchmark landscape is clear: NegotiationArena (arXiv:2402.05863) shows LLMs can boost payoff ~20% via persona manipulation ("desperation"); "Measuring Bargaining Abilities" (arXiv:2402.15813) shows buying is harder than selling, and scale doesn't help — their **OG-Narrator** (deterministic offer generator + LLM narrator) pushes deal rate from 26.67% to 88.88%; a large-scale autonomous negotiation competition (arXiv:2503.06416) ran ~180k negotiations, finding agents with **hidden chain-of-thought** win by concealing reasoning from opponents; Bhattacharya et al. 2025, using Harvard Negotiation Project metrics, ranked Llama-3 as most effective, Claude-3 as most aggressive, GPT-4 as most fair. This lesson implements the Contract Net Protocol (FIPA's ancestor, Lesson 02), plugs in an LLM-style buyer/seller, runs an OG-Narrator-style decomposition, and measures how each structural choice changes deal rate.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 16 · 02 (FIPA-ACL Legacy), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 min

## The Problem

Two agents need to agree on a price. If you just give them pure language prompts and let them go, 2024-2026 LLMs have surprisingly low deal rates (~27% on tightly parameterized bargaining in arXiv:2402.15813). Scale doesn't fix it: GPT-4 isn't better than GPT-3.5 at the *structure* of bargaining; it's better at the *language* of bargaining.

The root cause is that LLMs conflate two jobs — deciding the offer and narrating the offer. OG-Narrator separates them: a deterministic offer generator computes the numerical action; the LLM only narrates. Deal rate jumps to ~89%.

This echoes a classic multi-agent finding: decoupling mechanism from communication layer wins. The Contract Net Protocol (FIPA, 1996; Smith, 1980) is the reference task-market mechanism. Plug an LLM into the narration slot and you get a modern LLM-powered task market.

## The Concept

### Contract Net in One Paragraph

Smith 1980's Contract Net Protocol: a **manager** broadcasts a **call for proposals (cfp)**; **bidders** respond with **propose** messages containing offers; the manager picks a winner, sending **accept-proposal** to the winner and **reject-proposal** to losers. The winner executes the work. Optional messages: **refuse** (bidder declines). FIPA codified it as the `fipa-contract-net` interaction protocol.

### Why OG-Narrator Wins

"Measuring Bargaining Abilities of Language Models" (arXiv:2402.15813) observed:

- LLMs frequently violate bargaining rules (offer at absurd prices, ignore the counterpart's ZOPA).
- They anchor poorly (accept bad first offers; counter-offer by symbolic rather than strategic amounts).
- Scale alone doesn't fix these. Larger models produce more fluent language, but strategic errors are similar.

OG-Narrator decomposition:

```
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic) │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘
```

The offer generator is a classic negotiation strategy: Rubinstein bargaining model, Zeuthen strategy, or simple tit-for-tat on price. The LLM narrates. The message contains a deterministic price and a natural-language framing.

Deal rate jumps because:
- Prices stay within the bargaining zone.
- Anchoring is strategic, not emotional.
- The LLM does what it's good at: writing.

### NegotiationArena Findings

arXiv:2402.05863 provides the standard benchmark. Headline findings:

- LLMs can boost payoff ~20% by adopting a persona ("I'm desperate to sell this by Friday") — persona manipulation is a real tactic.
- Fair/cooperative agents get exploited by adversarial agents; defense requires explicit counter-posturing.
- In ~40% of benchmark scenarios, symmetric pairings converge to unfair outcomes.

This isn't "LLMs are bad negotiators." It's "LLMs negotiate too much like humans, including the exploitable parts."

### Chain-of-Thought Concealment

The large-scale autonomous negotiation competition (arXiv:2503.06416) ran ~180k negotiations across multiple LLM strategies. Winners concealed their reasoning from opponents:

- If an agent writes "My max is $75; my reservation price is $70" into a publicly visible scratchpad, the opponent reads it.
- Winners computed strategy privately; only offers and minimal necessary narration went on the output channel.

This is classical game theory (Aumann 1976 on rationality and information) echoing in 2026: exposing your private valuation loses payoff. LLMs don't know this instinctively, happily writing reservation prices into reasoning traces visible to the opponent.

Engineering takeaway: separate private scratchpad context from public message context. Not optional.

### Bhattacharya et al. 2025 — Model Rankings

On Harvard Negotiation Project metrics (principled negotiation, respect for BATNA, interest reciprocity):

- **Llama-3** is most effective at reaching deals (deal rate + payoff).
- **Claude-3** is the most aggressive negotiator (high anchoring, late concessions).
- **GPT-4** is most fair (lowest payoff variance across pairings).

This is a 2025 snapshot. The point isn't which model wins in April 2026 — it's that different base models have persistent negotiation styles. Heterogeneous ensembles (Lesson 15) use this as a diversity source.

### Task Allocation via Contract Net + LLM

The modern reuse of Contract Net in LLM multi-agent:

1. Manager agent decomposes a task into units.
2. Broadcasts a `cfp` with task description to worker agents.
3. Each worker returns a bid: `(price, eta, confidence)`, where price can be tokens, compute units, or dollars.
4. Manager picks winners (single or multiple, depending on task) and awards.
5. Rejected workers are free to bid on other tasks.

This scales easily past 100 workers because coordination is broadcast-respond, not synchronous chat. In production use: Microsoft Agent Framework's orchestration pattern, some LangGraph implementations.

### LLM-Stakeholder Interactive Negotiation

NeurIPS 2024 (https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) introduces multi-party scoreable games with **secret scores** and **minimum acceptance thresholds**. Each stakeholder has private utility; the LLM must infer them from messages. This is the generalization from two-party bargaining to N-party coalition formation. Relevant for production task markets with heterogeneous worker capabilities.

### Narration vs Mechanism Rules

Across all 2024-2026 negotiation benchmarks, the consistent engineering rule is:

> Let the LLM narrate. Do not let the LLM compute offers.

If the offer needs to be a number (price, ETA, quantity), generate it deterministically from negotiation state, and let the LLM produce framing. If the offer needs to be a proposal structure (task decomposition, role assignment), let the LLM draft it, but validate against schema and constraint-check before sending.

## Build It

`code/main.py` implements:

- `ContractNetManager`, `ContractNetTask`, `Bid` — manager + bidders, broadcasting cfp, collecting proposals, awarding.
- `og_narrator_bargain(state, rng)` — OG-Narrator buyer: deterministic Zeuthen-style concessions toward midpoint.
- `seller_response(state, rng)` — deterministic seller counter-offer strategy (structural ground truth shared by both styles).
- `naive_llm_bargain(state, rng)` — simulates a full-LLM bargainer: picks prices with high variance, often outside ZOPA.
- Measurement: deal rate over 1000 trials, each trial sampling new reservation prices.

Run:

```
python3 code/main.py
```

Expected output: naive LLM deal rate ~65-75%; OG-Narrator deal rate ~85-95%; the 15-25 point gap is the structural advantage of separating offer generation from narration. Plus a three-bidder, one-task Contract Net task-market allocation example.

## Use It

`outputs/skill-bargainer-designer.md` designs a bargaining protocol: who generates offers (deterministic or LLM), who narrates, how private scratchpad separates from public messages, and how deal rate is monitored.

## Ship It

Production bargaining checklist:

- **Separate scratchpad.** Private state never enters the opponent's context. Non-negotiable.
- **Deterministic offer generation.** Prices, quantities, ETAs: compute, don't prompt.
- **Validate all incoming offers** against schema. Reject out-of-ZOPA offers at the protocol boundary.
- **Bound rounds.** 3-5 max; on deadlock, escalate to a mediator.
- **Measure deal rate and payoff variance continuously.** Deal rate dropping is a symptom — often prompt drift or opponent-side attacks.
- **Log all rejected proposals** with deterministic reasons. For Contract Net managers, losing bidders need to understand why.

## Exercises

1. Run `code/main.py`. Confirm OG-Narrator beats naive LLM on deal rate. By how much?
2. Implement **persona-based payoff boost** (arXiv:2402.05863) — the buyer adopts a "need to buy this week urgently" persona in narration only, offer generator unchanged. Does deal rate or payoff change?
3. Implement CoT **concealment**: maintain a private scratchpad string not transmitted to the opponent. What happens if you accidentally leak it (simulate via the exchange channel)?
4. Extend Contract Net to an N-bidder auction with a reserve price. When all bids exceed reserve, how does the manager choose between lowest price and highest quality? Which award rule do you choose, and why?
5. Read Bhattacharya et al. 2025 on Harvard Negotiation Project metrics. Implement two bargainers with different styles (aggressive vs fair). Measure payoff variance across symmetric and asymmetric pairings.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Contract Net | "Task market" | Smith 1980, FIPA 1996. cfp + propose + accept/reject. The standard task market. |
| ZOPA | "Zone of possible agreement" | Overlap between buyer's ceiling and seller's floor. Offers outside it can't close. |
| BATNA | "Best alternative to a negotiated agreement" | Your fallback if this deal fails. It sets your reservation price. |
| OG-Narrator | "Offer generator + narrator" | Decomposition: deterministic offers, LLM narration. |
| Zeuthen strategy | "Risk-minimizing concession" | Classic offer generator that concedes based on risk limits. |
| Rubinstein bargaining | "Alternating-offer equilibrium" | Game-theoretic model of infinite-horizon bargaining with discounting. |
| CoT concealment | "Hide your reasoning" | Winners in arXiv:2503.06416 keep private scratchpads; only offers on the public channel. |
| Persona manipulation | "Emotional posturing" | arXiv:2402.05863: ~20% payoff boost from desperation/urgency persona. |

## Further Reading

- [NegotiationArena](https://arxiv.org/abs/2402.05863) — benchmark; persona manipulation and exploitation findings
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813) — OG-Narrator and "buying is harder than selling" results
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416) — ~180k negotiations; CoT concealment wins
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) — multi-party scoreable games with secret utilities
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) — classic mechanism, IEEE Transactions on Computers
