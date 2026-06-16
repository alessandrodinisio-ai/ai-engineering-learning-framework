# The FIPA-ACL and Speech Act Theory Heritage

> Before MCP, before A2A, there was FIPA-ACL. In 2000, IEEE's Foundation for Intelligent Physical Agents ratified an agent communication language with twenty performatives, two content languages, and a suite of interaction protocols — contract net, subscribe/notify, request-when. It faded from industry because ontology overhead was too heavy for the web of that era, but the LLM-driven multi-agent revival is quietly reimplementing the same ideas with the formalism stripped out: JSON contracts replacing performatives, natural language replacing ontologies. This lesson reads FIPA-ACL carefully so you can see which 2026 protocol decisions are old wine in new bottles, which are genuine innovations, and where the current wave will re-step on pitfalls the 2000s had already solved.

**Type:** Learn
**Languages:** Python (standard library)
**Prerequisites:** Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 min

## The Problem

The 2026 agent protocol landscape is crowded: MCP for tools, A2A for agents, ACP for enterprise auditing, ANP for decentralized trust, NLIP for natural-language content, plus CA-MCP and twenty-plus research proposals. Each spec claims to be foundational.

Honestly, most of them are rediscovering a very specific decision tree drawn two decades ago. Austin (1962) and Searle (1969)'s speech act theory gave us "utterances as actions." KQML (1993) turned it into a wire protocol. FIPA-ACL (ratified 2000) delivered the reference standardization: twenty performatives, content languages SL0/SL1, interaction protocols for contract-net and subscribe-notify. JADE and JACK were the Java reference platforms. The whole thing faded around 2010 because ontology overhead was too heavy and the web was winning.

When you look at MCP's `tools/call`, A2A's task lifecycle, or CA-MCP's shared context store, you're looking at a softer, JSON-native rewrite of the same FIPA decisions. Understanding this heritage tells you two things: which new "innovations" are reinventions, and which old failure modes the new specs will re-encounter.

## The Concept

### Speech Acts in One Paragraph

Austin noticed that some sentences don't describe the world — they change it. "I promise." "I request." "I declare." He called these performative utterances. Searle formalized them into five categories: assertives, directives, commissives, expressives, declaratives. KQML (Finin et al., 1993) grounded this into software agents: a message = a performative (action) plus content (what the action is about). FIPA-ACL filled KQML's gaps and standardized around twenty performatives.

### FIPA's Twenty Performatives (Partial List)

| Performative | Intent |
|---|---|
| `inform` | "I tell you P is true" |
| `request` | "I ask you to do X" |
| `query-if` | "Is P true?" |
| `query-ref` | "What is the value of X?" |
| `propose` | "I propose we do X" |
| `accept-proposal` | "I accept this proposal" |
| `reject-proposal` | "I reject this proposal" |
| `agree` | "I agree to do X" |
| `refuse` | "I refuse to do X" |
| `confirm` | "I confirm P is true" |
| `disconfirm` | "I deny P" |
| `not-understood` | "Your message didn't parse" |
| `cfp` | "Call for proposals on X" |
| `subscribe` | "Notify me when X changes" |
| `cancel` | "Cancel the ongoing X" |
| `failure` | "I tried X and failed" |

The full list lives in `fipa00037.pdf` (FIPA ACL Message Structure). The point is not to memorize it — it's that every single one maps to a primitive that some LLM protocol will eventually re-add.

### The Standard FIPA-ACL Message

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

Seven fields carry the protocol envelope, one field (`content`) carries the payload. The rest are exactly what you reinvent every time you bolt retry logic, message threading, and ontology onto a JSON protocol.

### Two Legacy Platforms

**JADE** (Java Agent DEvelopment framework, 1999–2020s) was the most widely used FIPA-compliant runtime. Agents extended a base class, exchanged ACL messages, ran in containers, and used "behaviors" for coordination. Its interaction protocol library shipped with contract-net, subscribe-notify, request-when, and propose-accept.

**JACK** (Agent Oriented Software, commercial) emphasized BDI (Belief-Desire-Intention) reasoning on top of FIPA messages. More formal, less adopted.

Both declined as web technology stacks absorbed multi-agent use cases. MCP and A2A are the 2026 runtime "containers."

### Why FIPA Faded

- **Ontology overhead.** FIPA required a shared ontology to parse `content`. Agreeing on an ontology was a multi-year standardization process. The web just used HTTP + JSON.
- **Unused formal semantics.** SL (Semantic Language) provided strict truth conditions, but most production systems used free-form content and ignored the formalism.
- **Tooling lock-in.** JADE was Java-only, JACK was commercial. Polyglot teams routed around both.
- **The internet won the stack.** REST, then JSON-RPC, then gRPC replaced the ACL transport layer.

### The LLM Revival Is FIPA Simplified

Compare a FIPA `request` with an MCP `tools/call`:

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

Same envelope, different syntax. Both carry: who, to whom, intent, payload, correlation id. Neither is more revolutionary than the other — they're different tradeoffs on the same design.

Liu et al.'s 2025 survey ("A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP," arXiv:2505.02279) makes the lineage explicit: MCP maps to tool-call speech acts, A2A to agent-peer speech acts, ACP to audit-trail speech acts, ANP to decentralized identity extensions. The new specs are all ACL descendants with JSON syntax and looser semantics.

### The Tradeoff, Made Explicit

**What FIPA gave you that modern specs drop:**

- Formal semantics — you could prove that `inform` implies "the sender believes the content."
- A standard performative catalog — you didn't repeatedly argue "should we add a `cancel`?"
- Decades of interaction protocol patterns — contract-net, subscribe-notify, propose-accept — with known correctness properties.

**What modern specs give you that FIPA didn't:**

- JSON-native payloads compatible with every modern tool.
- Natural-language content that LLMs can interpret without hand-written ontologies.
- Web-stack transports (HTTP, SSE, WebSocket).
- Capability discovery via self-describing documents (MCP `listTools`, A2A Agent Card).

Trade tighter intent semantics for easier implementation. That's the deal.

### Interaction Protocols Worth Porting

FIPA shipped ~15 interaction protocols. Three are worth bringing into LLM multi-agent systems:

1. **Contract Net Protocol (CNP).** A manager issues a `cfp` (call for proposals); bidders respond with `propose`; the manager accepts/rejects. This is the standard task marketplace pattern (Phase 16 · 16 Negotiation).
2. **Subscribe/Notify.** A subscriber sends `subscribe`; a publisher sends `inform` when a topic changes. This is every event bus in 2026.
3. **Request-When.** "Do X when condition Y holds." A deferred action with a precondition. Its 2026 analog is deferred tasks in durable workflow engines (Phase 16 · 22 Production Scaling).

Each maps cleanly onto modern message queues, HTTP + polling, or SSE streaming.

### What Breaks Without Ontology

Without a shared ontology, agents infer meaning from natural-language content. The documented 2026 failure mode is **semantic drift**: two agents use the same word (`"customer"`) to mean slightly different concepts, the receiving agent acts on the wrong interpretation, and no schema validator can catch it. FIPA's ontology requirement would have rejected the message at parse time.

Mitigations short of full ontology:

- JSON Schema on `content` — reject structural errors at the wire layer.
- Typed artifacts (A2A) — reject wrong modalities.
- Explicit performatives in the envelope — even if content is natural language, intent is unambiguous.

### Mapping 2026 Specs to the Speech Act Heritage

| Modern spec | FIPA analog | What it retains | What it drops |
|---|---|---|---|
| MCP `tools/call` | `request` | Explicit intent, correlation id | Formal semantics, ontology |
| MCP `resources/read` | `query-ref` | Explicit intent, correlation id | Formal semantics |
| A2A task lifecycle | contract-net + request-when | Async lifecycle, state transitions | Formal completeness guarantees |
| A2A streaming events | subscribe/notify | Async push | Typed-predicate subscription |
| CA-MCP shared context | Blackboard (Hayes-Roth 1985) | Multi-writer shared memory | Logical consistency model |
| NLIP | Natural-language content | LLM-native | Schema |

Read the table top to bottom and the pattern is: retain structural primitives, drop formalism, let LLMs muddle through ambiguity.

## Build It

`code/main.py` implements a FIPA-ACL translator in pure standard library. It encodes and decodes standard ACL envelopes, and demonstrates how each MCP / A2A message shape reduces to the same seven fields. The demo:

- Encodes five MCP-style and A2A-style messages as FIPA-ACL.
- Decodes FIPA-ACL back to modern equivalents.
- Runs a toy contract-net negotiation between a manager and three bidders using `cfp`, `propose`, `accept-proposal`, `reject-proposal`.

Run:

```
python3 code/main.py
```

Output is a side-by-side trace showing each modern message in both its 2026 JSON form and its FIPA-ACL form, then a contract-net bidding round-trip. The same protocol primitives survive the round trip; only syntax changes.

## Use It

`outputs/skill-fipa-mapper.md` is a skill that reads any agent protocol spec and produces a FIPA-ACL mapping. Use it before adopting a new protocol to answer: "Is this genuinely new, or is it `inform` with JSON syntax?"

## Ship It

Don't bring FIPA-ACL back. Bring its checklist back:

- What is the intent primitive (performative) of each message?
- Is there a correlation id for request-response and cancel?
- Is there a clear content language (JSON-RPC, plain text, typed artifact)?
- Are interaction protocols first-class, or are you reimplementing contract-net from scratch?
- What happens when two agents disagree on content meaning (semantic drift)?

Write these five questions into documentation before pushing any new protocol to production.

## Exercises

1. Run `code/main.py`. Observe the round-trip encoding. Identify which FIPA performative corresponds to `tools/call`, `resources/read`, and A2A task creation.
2. Add a `cancel` performative to the contract-net demo so the manager can retract a task mid-bid. What failure scenario does `cancel` solve that retries alone cannot?
3. Read FIPA ACL Message Structure (http://www.fipa.org/specs/fipa00037/) sections 4.1–4.3. Pick one performative not covered in this lesson and describe its modern JSON-RPC equivalent.
4. Read Liu et al., arXiv:2505.02279. For MCP, A2A, ACP, ANP respectively, list which FIPA performative families they retain and which they drop.
5. Design a minimal JSON-Schema for the `content` field of a `request` performative in your own system. What does this schema give you that pure natural language doesn't, and what does it cost?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Speech act | "A sentence that does something" | Austin/Searle: utterances as actions. The theoretical ancestor of ACL. |
| FIPA | "That old XML thing" | IEEE Foundation for Intelligent Physical Agents. Standardized ACL in 2000. |
| ACL | "Agent Communication Language" | FIPA's envelope format: performative + content + metadata. |
| Performative | "The verb" | The intent category of a message: `inform`, `request`, `propose`, `cfp`, etc. |
| KQML | "FIPA's predecessor" | Knowledge Query and Manipulation Language (1993). Simpler, narrower. |
| Ontology | "Shared vocabulary" | A formal definition of the concepts discussed in a content language. |
| SL0 / SL1 | "FIPA content language" | Semantic Language levels 0 and 1 — the formal content language family. |
| Contract Net | "Task marketplace" | Manager issues cfp; bidders propose; manager accepts. Standard interaction protocol. |
| Interaction protocol | "The pattern of messages" | A known-correct sequence of performatives: request-when, subscribe-notify, etc. |

## Further Reading

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — the landmark 2025 survey connecting modern specs back to the FIPA heritage
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — the 2000-ratified envelope format
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — the full performative catalog
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — `request`/`query-ref` modern tool-call equivalents
- [A2A specification](https://a2a-protocol.org/latest/specification/) — contract-net and subscribe-notify modern agent-peer equivalents
