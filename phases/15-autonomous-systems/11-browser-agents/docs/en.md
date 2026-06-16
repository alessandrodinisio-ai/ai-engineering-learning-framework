# Browser Agents and Long-Horizon Web Tasks

> ChatGPT agent (July 2025) merged Operator and deep research into one browser/terminal agent, pulling BrowseComp SOTA to 68.9%. OpenAI shut down standalone Operator on August 31, 2025—product-layer consolidation. Anthropic's Vercept acquisition pushed Claude Sonnet on OSWorld from under 15% to 72.5%. WebArena-Verified (ServiceNow, ICLR 2026) fixed the original WebArena's 11.3 percentage-point false-negative rate and delivered a 258-task Hard subset. These numbers are real. The attack surface is also real: OpenAI's preparedness lead stated publicly that indirect prompt injection into browser agents "isn't a fully patchable bug." Documented 2025-2026 attacks: Tainted Memories (Atlas CSRF), HashJack (Cato Networks), and one-click hijack in Perplexity Comet.

**Type:** Learn
**Languages:** Python (standard library, indirect prompt injection attack surface model)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45 minutes

## The Problem

A browser agent is a long-horizon agent that reads untrusted content and takes consequential actions. Every page the agent visits is input the user didn't write. Every form on every page is a potential command channel. The 2025-2026 attack corpus shows this isn't hypothetical: Tainted Memories lets attackers bind malicious instructions to an agent's memory via a crafted page; HashJack hides commands in URL fragments the agent visits; Perplexity Comet's hijack fires in one click.

The defensive picture is uncomfortable. OpenAI's preparedness lead said the quiet part loud: indirect prompt injection "isn't a fully patchable bug." This is because the attack lives at the agent's read-vs-act boundary, and that boundary is architecturally fuzzy—in principle, every token the model reads could be read as an instruction.

This lesson names the attack surface, names the benchmark landscape (BrowseComp, OSWorld, WebArena-Verified), and models a minimal indirect prompt injection scenario so you can reason about real defenses in Lessons 14 and 18.

## The Concept

### The 2026 Landscape, One Paragraph Per System

**ChatGPT agent (OpenAI).** Released July 2025. Unifies Operator (browsing) and Deep Research (multi-hour research). Shut down standalone Operator August 31, 2025. SOTA 68.9% on BrowseComp; strong numbers on OSWorld and WebArena-Verified.

**Claude Sonnet + Vercept (Anthropic).** Anthropic's Vercept acquisition focused on computer-use capabilities. Pushed Claude Sonnet on OSWorld from <15% to 72.5%. Claude Computer Use ships as a tool API.

**Gemini 3 Pro with Browser Use (DeepMind).** Browser Use integration delivers computer-use controls; FSF v3 (April 2026, Lesson 20) specifically tracks autonomy in ML R&D domains.

**WebArena-Verified (ServiceNow, ICLR 2026).** Fixes a well-documented problem: original WebArena had ~11.3% false-negative rate (tasks marked as failed that were actually solved). Verified version re-scores with human-curated success criteria and adds a 258-task Hard subset (ICLR 2026 paper, openreview.net/forum?id=94tlGxmqkN).

### BrowseComp vs. OSWorld vs. WebArena

| Benchmark | What it measures | Time horizon |
|---|---|---|
| BrowseComp | Finding specific facts from open web under time pressure | Minutes |
| OSWorld | Agent operating a full desktop (mouse, keyboard, shell) | Tens of minutes |
| WebArena-Verified | Transactional web tasks in simulated sites | Minutes |
| Hard subset | WebArena-Verified tasks with multi-page state transitions | Tens of minutes |

Different dimensions. A high BrowseComp score says the agent finds facts; it doesn't say it can book a flight. OSWorld scores are closer to "can it work on my desktop." WebArena-Verified is closer to "can it complete a workflow." Any production decision requires the benchmark matching the task distribution.

### Naming the Attack Surface

1. **Indirect prompt injection.** Untrusted page content contains instructions. The agent reads them. The agent executes them. Public examples: Kai Greshake et al. 2024, Tainted Memories paper 2025, HashJack (Cato Networks) 2026.
2. **URL fragment / query injection.** The `#fragment` or query string of a crawled URL contains commands. Never visibly rendered; still in the agent's context.
3. **Memory-binding attacks.** A page instructs the agent to write a persistent memory (Lesson 12 on durable state). Next session, the memory fires its payload without visible trigger.
4. **CSRF-shaped attacks on authenticated sessions.** The Tainted Memories class: the agent is logged in somewhere; an attacker's page issues state-changing requests that the agent executes with the user's cookies.
5. **One-click hijack.** A visually benign button carries a payload the agent will follow. The Comet class.
6. **CSP gaps on the agent host surface.** The rendering layer and tool layer can themselves be attack vectors; the browser-wrapping-browser-agent stack is wide.

### Why "Not Fully Patchable"

The attack is isomorphic to the agent's capability. The agent must read untrusted content to do its job. Any content the agent reads might contain instructions. Any instruction the agent follows might be misaligned with the user's actual request. Defenses (trust boundaries, classifiers, tool whitelists, HITL on consequential actions) raise the cost of attack and shrink its blast radius. They don't close the category.

This is the same reasoning shape as Lob's theorem (Lesson 8): the agent cannot prove the next token is safe; it can only build a system that makes unsafe tokens more detectable.

### The Practical Defense Posture

- **Read/write boundary.** Reads never have consequences. Writes (submitting forms, posting content, invoking a side-effecting tool) require fresh human approval if originating content comes from outside the trust boundary.
- **Per-task tool whitelist.** The agent can browse; it cannot initiate a wire transfer unless that tool is explicitly enabled for this task. Lesson 13 covers budgets.
- **Session isolation.** Browser agent sessions run with limited-scope credentials only. No production auth, no personal email. Every HTTP request is logged for audit.
- **Content sanitizer.** Crawled HTML is stripped of known-bad patterns before being stitched into model context. (Reduces easy attacks; won't block sophisticated payloads.)
- **HITL on consequential actions.** The propose-then-commit pattern (Lesson 15).
- **Canary tokens on memory.** If a memory fires, the user can see it (Lesson 14).

## Use It

`code/main.py` models a micro browser-agent run against three synthetic pages. One page is benign, one has a direct prompt injection block in visible text, one has a URL-fragment injection (invisible but in the agent's context). The script shows (a) what a naive agent would do, (b) what the read/write boundary catches, (c) what a sanitizer catches, (d) what neither catches.

## Ship It

`outputs/skill-browser-agent-trust-boundary.md` scopes a proposed browser agent deployment: which trust zones it touches, what it's authorized to write, and which defenses must be in place before the first run.

## Exercises

1. Run `code/main.py`. Identify which attack the sanitizer catches but the read/write boundary doesn't, and which attack only the read/write boundary catches.

2. Extend the sanitizer to detect a HashJack-style URL fragment injection. Measure false-positive rate on benign URLs with legitimate fragments.

3. Pick a real browser-agent workflow you're familiar with (e.g., "book a flight"). List every read and every write. Mark which writes need HITL and why.

4. Read the WebArena-Verified ICLR 2026 paper. Identify one category of task that the original WebArena scored unreliably and explain how the Verified subset addresses it.

5. Design a memory canary for a browser-agent scenario. What do you store, where, and what triggers an alert?

## Key Terms

| Term | Colloquial usage | Actual meaning |
|---|---|---|
| Indirect prompt injection | "Bad page text" | Untrusted content in a page the agent reads contains instructions the agent executes |
| Tainted Memories | "Memory attack" | Agent writes attacker-supplied instructions into persistent memory; next session fires |
| HashJack | "URL fragment attack" | Payload hidden in URL fragment/query string is in agent context but invisibly rendered |
| One-click hijack | "Bad button" | Visible clickable element carries a follow-on payload the agent will execute |
| BrowseComp | "Web search benchmark" | Finding specific facts from open web; minutes-scale horizon |
| OSWorld | "Desktop benchmark" | Full OS control; multi-step GUI tasks |
| WebArena-Verified | "Fixed web-task benchmark" | ServiceNow's re-scored WebArena with Hard subset |
| Read/write boundary | "Side-effect gate" | Reads never have consequences; writes need fresh approval if content is outside trust |

## Further Reading

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator and deep research merger; BrowseComp SOTA.
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator lineage and architecture that became ChatGPT agent.
- [Zhou et al. — WebArena](https://webarena.dev/) — Original benchmark.
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 fixed-subset paper.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Includes discussion of computer-use agent attack surfaces.
