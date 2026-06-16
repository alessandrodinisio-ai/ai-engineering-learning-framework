# Agent Economies, Token Incentives, Reputation

> Long-horizon autonomous agents (METR's "1-hour to 8-hour work curves") need economic agency. The emerging **five-layer stack** is: **DePIN** (physical compute) → **Identity** (W3C DID + reputation capital) → **Cognition** (RAG + MCP) → **Settlement** (account abstraction) → **Governance** (Agentic DAO). Production-grade agent incentive networks include **Bittensor** (TAO subnets rewarding task-specific models), **Fetch.ai / ASI Alliance** (ASI-1 Mini LLM + FET token), and **Gonka** (transformer-based PoW that redirects compute to productive AI tasks). Academic work: AAMAS 2025's decentralized LaMAS uses **Shapley value credit assignment** to fairly reward contributing agents; Google Research's "Mechanism design for large language models" proposes **token auctions** with second-price payment under monotone aggregation. This lesson builds a minimal agent marketplace, applies Shapley value credit assignment to a multi-agent pipeline, and runs a second-price token auction to make the game-theory machinery concrete.

**Type:** Learn
**Languages:** Python (standard library)
**Prerequisites:** Phase 16 · 16 (Negotiation & Bargaining), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## The Problem

Multi-agent systems get complicated when agents jointly produce value but need to be individually rewarded. Classic mechanisms—equal split, last-contributor-takes-all—are either unfair or gameable. Coalition-based rewards via Shapley values are fair by construction but computationally expensive. The 2025-2026 literature ships useful approximations: Shapley sampling, monotone-aggregation auctions, and on-chain reputation that accumulates from confirmed contributions.

Beyond credit assignment, the field is moving toward genuinely economic agents: Bittensor TAO rewards mining compute for fine-tuning subnet-specific models, Fetch.ai/ASI rewards ASI-1 Mini LLM usage with FET tokens, and Gonka redirects transformer proof-of-work to productive AI tasks. Agents that transact autonomously exist today; the question is how to align incentives.

This lesson treats agent economics as a concrete problem family—credit assignment, mechanism design, reputation—and builds each with minimal math so the ideas stick.

## The Concept

### The five-layer agent economy stack

1. **DePIN (physical compute).** Decentralized infrastructure renting GPUs, storage, bandwidth. Bittensor subnets, Render Network, Akash. Not agent-specific; agents consume it.
2. **Identity.** W3C Decentralized Identifiers (DIDs) give each agent a persistent ID independent of any platform. Reputation accrues to the DID. Agent Network Protocol (ANP) uses DIDs as the discovery layer.
3. **Cognition.** The agent's reasoning loop: LLM + RAG + MCP. This is what other phases build.
4. **Settlement.** Account abstraction (ERC-4337) lets agents pay gas from their own balance without holding ETH. Agents can pay for services, for each other, or for compute.
5. **Governance.** Agentic DAOs: governance structures where humans *and* agents vote on protocol changes, with voting power tied to reputation.

Not every production system uses all five layers. Bittensor uses 1, 2, partly 3, partly 4, none of 5. OpenAI agents use none except 3. The stack is a reference map, not a requirement.

### Bittensor, Fetch.ai, Gonka — what's running

**Bittensor (TAO).** Subnets are specialized tasks (language modeling, image generation, prediction). Miners submit model outputs. Validators rank them; stake-weighted scoring distributes TAO rewards. Each subnet has its own evaluation. Economic lesson: pay for task-specific output quality, not for compute consumed.

**Fetch.ai / ASI Alliance.** ASI-1 Mini LLM runs on Fetch.ai's network; users pay FET tokens for inference. The "agent-as-peer" narrative is stronger here: an agent on Fetch can call another agent for a task and pay with FET.

**Gonka.** Transformer proof-of-work: the "work" is a transformer forward pass. Miners earn by running inference tasks with known-correct outputs (from training data). Replaces hash-based PoW with productive resource use.

As of April 2026, all three are production-grade. Revenue distribution varies. Bittensor rewards quality relative to subnet validators; Fetch rewards utility as measured by paying users; Gonka rewards verifiable inference work.

### Shapley value credit assignment

Three agents cooperate on a task. The output scores 0.8. Who contributed what?

Shapley value: the unique credit allocation satisfying four axioms (efficiency, symmetry, linearity, null player). For agent `i`:

```
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

where `S_i_O` is the set of agents before `i` in ordering `O`. In practice: enumerate all permutations, record each agent's marginal contribution in each permutation, and average.

N=3 agents have 6 permutations. N=10 have 3.6 million—so in practice you sample orderings rather than enumerate.

### Second-price auctions for aggregation

Google Research ("Mechanism design for large language models") proposes second-price token auctions for aggregating LLM outputs. Setup: N agents each submit a completion; each has a private value for "being selected." The auctioneer picks the highest-value proposal and pays the *second-highest* value. Under monotone aggregation (value depends on which proposal is selected, not how many bids), this is truthful—agents bid their real values.

Why this matters for LLM systems: you can outsource a completion task to multiple differently-priced agents; the auction picks the best + pays fairly, and agents have no incentive to misreport.

### Reputation capital

A DID-bound reputation score that accrues from confirmed contributions. A simple update rule:

```
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

Decay factor `alpha` close to 1. Reputation:

- Cheap to read and usable for routing decisions ("send hard tasks to high-rep agents").
- Expensive to forge (accrues over time, bound to DID).
- Slashable: contributions that fail verification subtract.

### AAMAS 2025 decentralized LaMAS

The LaMAS proposal (AAMAS 2025) combines: DID identity, Shapley value credit assignment, and a simple auction mechanism. Key claim: decentralizing the credit assignment step makes the system auditable and immune to single-point manipulation.

### Where economics breaks

- **Price oracle manipulation.** If the credit function can be gamed, agents will game it. Every mechanism needs an adversarial test.
- **Sybil attacks.** One operator spins up N fake agents to inflate their own contribution. DIDs slow this down but don't stop it; "cost to forge reputation" is the mitigation.
- **Verification cost.** Credit assignment is only as fair as the verifier. If verification is cheap (small LLM), it can be gamed; if expensive (human jury), the system can't scale.
- **Regulatory overhang.** Agent economies intersect financial regulation. As of 2026, Bittensor, Fetch, and Gonka all operate in legal gray areas in some jurisdictions.

### When agent economics makes sense

- **Open networks with heterogeneous operators.** No single team controls all agents.
- **Verifiable outputs.** Without verification, credit assignment is guesswork.
- **Long-horizon workflows.** One-shot tasks don't benefit from reputation accumulation.
- **Tokenized payment is legally viable** in your jurisdiction.

In closed enterprise systems, economics gives way to simpler allocation (managers assign work, metrics are internal). The economics literature primarily applies to open networks.

## Build It

`code/main.py` implements:

- `shapley(value_fn, agents)` — exact Shapley computation by enumeration for small N.
- `second_price_auction(bids)` — truthful mechanism; winner pays second-highest price.
- `Reputation` — DID-bound reputation with exponential decay and slashing.
- Demo 1: three agents cooperate, exact Shapley assigns credit.
- Demo 2: five agents bid for a task slot; second-price auction picks winner + payment.
- Demo 3: 100 rounds routing tasks to reputation-heterogeneous agents; reputation-weighted routing beats random.

Run:

```
python3 code/main.py
```

Expected output: Shapley values per agent; auction results showing truthful bidding equilibrium; reputation-weighted routing shows 10-20% quality improvement over random after warmup.

## Use It

`outputs/skill-economy-designer.md` designs a minimal agent economy: identity layer choice, credit assignment mechanism, payment mechanism, reputation rules.

## Ship It

Running an agent economy in 2026:

- **Ship reputation before tokens.** Reputation is cheap to implement and valuable on its own; tokens add legal and economic complexity.
- **Verify before rewarding.** Never distribute credit without an independent verification step. Self-reported quality invites Sybil games.
- **Use Shapley sampling, not exact Shapley.** Sample 100-1000 orderings; exact enumeration doesn't scale.
- **Cap decay and floor reputation.** Unbounded decay erases legitimate contributors; decay too slow rewards stale high-rep agents.
- **Adversarially audit the mechanism.** Run red-team scenarios before opening the network. Every mechanism has a game theory; you're looking for exploits, not attackers.

## Exercises

1. Run `code/main.py`. Confirm Shapley values sum to total value (efficiency axiom). Change the value function; does the Shapley allocation shift in the expected direction?
2. Implement Shapley *sampling* (Monte Carlo over K orderings). How does K affect approximation accuracy? Compare to exact for N=4.
3. Implement a coalition-formation step before the auction: agents can merge into teams and bid as a unit. Which coalitions form? Is the result Pareto-better than individual bidding?
4. Read the Google Research mechanism design blog post. Identify one assumption that, if violated, breaks truthfulness. What does that failure mode look like in an LLM setting?
5. Read the AAMAS 2025 decentralized LaMAS paper. Implement their Shapley step on a synthetic task for 10 agents. How long does exact computation take? How close can sampling 100 orderings get?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| DePIN | "Decentralized physical infrastructure" | Token-incentivized compute/storage/bandwidth. Bittensor, Akash, Render. |
| DID | "Decentralized Identifier" | W3C spec for portable IDs. Agent reputation binds to DID, not platform. |
| ERC-4337 | "Account abstraction" | Contract accounts that can sponsor gas, enabling agent payments. |
| Shapley value | "Fair credit assignment" | The unique allocation satisfying efficiency, symmetry, linearity, null player. |
| Second-price auction | "Vickrey auction" | Truthful mechanism: winner pays second-highest bid. Compatible with monotone aggregation. |
| Reputation capital | "Accumulated quality score" | DID-bound score from confirmed contributions; decays over time. |
| Agentic DAO | "Agent + human co-governance" | DAOs with agent voters as first-class citizens, voting power tied to reputation. |
| TAO / FET / GPU credits | "Token denominations" | Bittensor TAO, Fetch.ai FET, various DePIN tokens. |

## Further Reading

- [The Agent Economy](https://arxiv.org/abs/2602.14219) — 2026 survey of the five-layer agent economy stack
- [Google Research — Mechanism design for large language models](https://research.google/blog/mechanism-design-for-large-language-models/) — Token auctions with monotone aggregation
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — Shapley value credit assignment
- [Bittensor TAO documentation](https://docs.bittensor.com/) — Subnet structure and reward distribution
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — ASI-1 Mini LLM and FET token
- [W3C Decentralized Identifiers (DIDs) spec](https://www.w3.org/TR/did-core/) — Identity foundation
