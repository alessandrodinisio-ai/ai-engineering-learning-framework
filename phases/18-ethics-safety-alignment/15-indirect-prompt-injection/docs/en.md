# Indirect Prompt Injection — The Production Attack Surface

> Indirect prompt injection (IPI) embeds instructions in external content — a web page, an email, a shared document, a ticket — that is consumed by an agent system without explicit user action. IPI is the dominant production threat in 2026: it bypasses user input filtering because the attacker never touches the user; it silently scales as agents process more external content; it targets automated workflows where nobody is reading the prompt. MDPI Information 17(1):54 (January 2026) synthesizes 2023–2025 research. The NDSS 2026 IPI defense paper frames the core challenge: injected instructions can be semantically harmless ("please print Yes"), so detection requires more than keyword filtering. "The Attacker Moves Second" (Nasr et al., joint OpenAI/Anthropic/DeepMind, October 2025): adaptive attacks (gradient, RL, random search, human red-team) break >90% of 12 published defenses that originally reported near-zero ASR.

**Type:** Build
**Languages:** Python (standard library, IPI attack + defense testbed)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 14 (Agent Engineering)
**Time:** ~75 min

## Learning Objectives

- Define indirect prompt injection and describe three common delivery vectors.
- Explain why user input filtering completely misses IPI.
- Describe the "information flow control" framework as the 2026 defense paradigm.
- State the Nasr et al. (October 2025) finding about adaptive attack success rates against published IPI defenses.

## The Problem

Direct prompt injection requires the attacker to reach the user or their prompt. IPI requires neither: the attacker places a payload into any content the agent might read — a web page, an email in an inbox, a GitHub issue, a product review. The agent picks it up during normal operation and executes those instructions. The user is the messenger, not the intent.

## The Concept

### Three Delivery Vectors

- **Retrieval-Augmented Generation (RAG).** The attacker publishes a document; the retrieval step fetches it; the prompt concatenates it before the user's question; the model executes the attacker's instructions.
- **Inbox / Document workflows.** The attacker sends the user an email; the agent reads emails; the prompt includes the email body; the model follows the instructions in the email.
- **Tool outputs.** The attacker controls a tool the agent uses (e.g., a web search that returns attacker-controlled results); the tool output contains instructions; the agent's control flow follows.

All three share a structural property: the attacker controls a segment of the prompt without touching user-facing input.

### Why User Input Filtering Misses It

An IPI payload does not appear in the user's input. It appears in retrieved content. If filtering guards the user input, the payload bypasses it. If filtering guards all content reaching the model, it must apply to arbitrary retrieved text — which is both expensive and produces false positives on legitimate content that happens to contain imperative language.

### AI-Oriented Information Flow Control (IFC)

The 2026 defense paradigm borrows from classical operating system security. Treat each content source as a security label. Label the user query as "trusted." Label retrieved content as "untrusted." Treat the model's control flow as an information flow: actions triggered by untrusted content must be approved by trusted input before execution.

CaMeL (Microsoft 2025), ConfAIde (Stanford 2024), and the NDSS 2026 IPI defense paper operationalize IFC in different ways. The shared principle: as long as code and data share the same context window, the goal is "contain" rather than "prevent."

### The Attacker Moves Second

Nasr et al. (October 2025) tested 12 published IPI defenses with adaptive attacks (gradient search, RL policies, random search, 72-hour human red-team). Every defense that originally reported near-zero ASR was broken to >90% ASR.

Methodological lesson: publishing a defense must come with an adaptive attack evaluation. Static attack benchmarks are not evidence of robustness; the attacker will always figure out the defense.

### Real-World Incidents

Lesson 25 covers EchoLeak (CVE-2025-32711, CVSS 9.3) — the first publicly documented zero-click IPI in Microsoft 365 Copilot. CamoLeak (CVSS 9.6) in GitHub Copilot Chat. CVE-2025-53773 in GitHub Copilot. Production deployments are being compromised by IPI in the wild, not just in benchmarks.

### OWASP and NIST Framing

OWASP LLM Top 10 (2025) lists prompt injection (direct + indirect) as LLM01, the #1 application-layer threat. NIST AI SPD 2024 calls indirect prompt injection "the greatest security flaw in generative AI."

### Where This Fits in Phase 18

Lessons 12–14 are model-centric jailbreaks. Lesson 15 is the system-centric attack that dominates 2026 production deployments. Lesson 16 covers defensive tooling. Lesson 25 covers the specific CVE narratives.

## Use It

`code/main.py` builds an IPI testbed. A toy agent has three tools (search web, read email, send message). The environment contains attacker-controlled content with an embedded instruction ("forward this to all contacts"). You can switch between three agents: naive (follows the injected instruction), filter-defense (keyword filtering on retrieved content), and IFC agent (separates trusted from untrusted content and refuses control-flow commands from untrusted content).

## Ship It

This lesson produces `outputs/skill-ipi-audit.md`. Given an agent deployment description, it enumerates untrusted content sources, checks whether the deployment enforces IFC, and flags sources that reach the model without a trust label.

## Exercises

1. Run `code/main.py`. Measure the attack success rate against each of the three agents.

2. Implement a paraphrase-based defense on retrieved content. Measure its benign false-positive rate on legitimate retrieved text.

3. Read the NDSS 2026 IPI defense paper. Describe the "harmless instruction" challenge and why it defeats keyword-based filtering.

4. Design a deployment where the agent receives a tool output from a third-party API. Label each prompt segment with a trust level and write the IFC policy that governs agent actions.

5. Reproduce the Nasr et al. 2025 adaptive attack methodology on your Exercise 2 filter-defense agent. Report ASR before and after adaptive attack.

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| IPI | "indirect prompt injection" | Injection via content the user didn't write that the agent consumes during normal operation |
| RAG injection | "poisoned retrieval" | Attacker publishes content that the retrieval step fetches; prompt includes the payload |
| Zero-click | "no user action" | Attack triggers automatically during agent operation; user does nothing |
| IFC | "information flow control" | Label-based approach: actions from untrusted content require trusted approval |
| Adaptive attack | "gradient / RL red-team" | Attack that knows the defense and optimizes against it; required for honest evaluation |
| Harmless instruction | "please print Yes" | Semantically benign IPI payload; no keyword filter can catch it |
| Scope violation | "cross-trust exfiltration" | Agent accesses data in one trust context and outputs it to another |

## Further Reading

- [MDPI Information 17(1):54 — Indirect Prompt Injection Survey (January 2026)](https://www.mdpi.com/2078-2489/17/1/54) — 2023–2025 synthesis
- [Nasr et al. — The Attacker Moves Second (joint OpenAI/Anthropic/DeepMind, October 2025)](https://arxiv.org/abs/2510.18108) — Adaptive attack evaluation
- [Greshake et al. — Not what you've signed up for (arXiv:2302.12173)](https://arxiv.org/abs/2302.12173) — Original IPI paper
- [OWASP — LLM Top 10 (2025)](https://genai.owasp.org/llm-top-10/) — Prompt injection listed as LLM01
