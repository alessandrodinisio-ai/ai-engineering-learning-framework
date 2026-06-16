# Multimodal Agents & Computer Use (Capstone)

> The frontier product of 2026 is a multimodal agent that reads screenshots, clicks buttons, navigates web UIs, fills forms, and completes workflows end-to-end. SeeClick and CogAgent (2024) proved the GUI grounding primitive. Ferret-UI added mobile. ChartAgent introduced chart-oriented visual tool use. VisualWebArena and AgentVista (2026) are the benchmarks the frontier chases — even Gemini 3 Pro and Claude Opus 4.7 score only ~30% on AgentVista's hard tasks. This capstone pulls every thread from Phase 12 together: perception (high-res VLM), reasoning (LLM with tool use), grounding (coordinate output), long-horizon memory, evaluation.

**Type:** Capstone
**Languages:** Python (stdlib, action schema + agent loop skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 09 (Qwen-VL JSON), Phase 14 (Agent Engineering)
**Time:** ~240 minutes

## Learning Objectives

- Design a multimodal agent loop: perceive → reason → act → observe → repeat.
- Build a GUI grounding output schema (click coordinates, type text, scroll, drag) that a VLM can emit as JSON.
- Compare screenshot-only agents vs accessibility-tree agents vs hybrid agents.
- Stand up a multimodal agent benchmark evaluation on a small VisualWebArena slice.

## The Problem

A flight-booking website workflow: "Find me a flight to Tokyo on April 15, aisle seat, under $800, and book it."

A multimodal agent needs to:

1. Take a browser screenshot.
2. Parse the screenshot + URL + goal into a plan.
3. Emit a structured action: click at (x,y), type "Tokyo" in element E, scroll down, select (radio button).
4. Apply the action to the browser.
5. Observe new state (next screenshot).
6. Repeat until task is done.

Each step is one multimodal VLM call. VLM output must be parseable JSON. Errors compound across steps, so recovery matters.

## The Concept

### GUI Grounding — The Primitive

GUI grounding is: given a screenshot and a natural-language instruction, output the (x, y) coordinates to click (or another action).

SeeClick (arXiv:2401.10935) was the first large-scale open result: fine-tune a VLM on synthetic + real GUI data, output coordinates as plain text tokens. It works.

CogAgent (arXiv:2312.08914) added 1120x1120 high-resolution encoding for dense UIs. Score: ~84% on web navigation.

Ferret-UI (arXiv:2404.05719) focused on mobile UIs, integrating with iOS accessibility datasets.

Output format is typically JSON:

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` helps recovery: if coordinates drift between screenshots, this semantic hint lets the system re-ground.

### Action Schema

A typical action schema has 6-10 action types:

- `click`: (x, y)
- `type`: (text, x?, y?)
- `scroll`: (direction, amount)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (ms)
- `done`: (success, explanation)

The agent emits one action per step. A browser wrapper executes it and returns new state.

### Screenshot-Only vs Accessibility Tree

Two input modes:

- Screenshot-only: full image, no structural info. Most general; works on any app.
- Accessibility tree: structured DOM / iOS accessibility info. Far more reliable for grounding; works where a tree exists.
- Hybrid: both. The tree serves as a reliable grounder for atomic actions, the screenshot provides semantic context.

Production agents use hybrid wherever possible. Browser automation (Selenium + accessibility) always has a tree; desktop apps sometimes do.

### Long-Horizon Memory

A 20-step workflow produces 20 screenshots. VLM context fills fast. Three compression strategies:

- Summary chain: every 5 steps, summarize what happened, drop old screenshots.
- Frame skipping: keep first, last, and every 3rd screenshot.
- Tool-log mode: execute actions, keep a text log of what was done; never look back at old screenshots.

Claude's computer-use API uses the log mode. Simpler, more reliable.

### Visual Tool Use

ChartAgent (arXiv:2510.04514) introduced visual tool use for chart understanding: crop, zoom, OCR, call external detectors. The agent can output "crop to region (100, 200, 300, 400) then call OCR" as a tool call. The tool returns text; VLM continues reasoning.

This pattern generalizes: set-of-mark prompting, region annotation, and external detection tools all fit the same "emit a tool call, receive a structured response" schema.

### 2026 Benchmarks

- ScreenSpot-Pro. GUI grounding on ~1k web screenshots. Open SOTA Qwen2.5-VL-72B ~85%. Frontier ~90%.
- VisualWebArena. End-to-end web tasks (shopping, forums, classifieds). Open SOTA ~20%. Gemini 3 Pro ~27%.
- AgentVista (arXiv:2602.23166). Hardest 2026 benchmark. Real workflows across 12 domains. Frontier models score 27-40%; open models 10-20%.
- WebArena / WebShop. Older benchmarks; saturated by frontier.

### Why It's Still Hard

Agent performance bottlenecks:

1. Fine-grained visual grounding. "Click the tiny X" often fails at mobile resolution.
2. Long-horizon planning. After 10 actions, agents drift from the goal.
3. Error recovery. When a click fails (wrong button), detection + recovery is rarely in training data.
4. Cross-page context. Jumping between tabs or handling long forms loses state.

Research directions: memory architectures, explicit re-planning, multimodal verification (using screenshot matching to determine whether an action succeeded).

### Capstone Build It

Capstone task: build a computer-use agent that:

1. Reads an HTML + screenshot of a simulated booking page.
2. Plans a multi-step sequence: search → select → fill form → submit.
3. Emits JSON actions matching the action schema.
4. Evaluates on a fixed 10-task slice.

This lesson provides scaffold code that is easy to extend to a real browser.

## Use It

`code/main.py` is the capstone scaffold:

- JSON definition of the action schema (10 actions).
- Simulated browser state represented as dicts.
- Agent loop skeleton: receive state, emit action, apply, loop.
- 10-task mini benchmark (synthetic pages), measuring end-to-end success rate.
- Error recovery hook when an action fails.

## Ship It

This lesson produces `outputs/skill-multimodal-agent-designer.md`. Given a computer-use product (domain, action set, evaluation target), it designs the complete agent loop, memory strategy, grounding mode, and expected benchmark scores.

## Exercises

1. Extend the action schema with a `screenshot_region` tool (crop + zoom). Which tasks benefit?

2. Read AgentVista (arXiv:2602.23166). Describe the hardest task category and why frontier models still fail.

3. Long-horizon memory compression: design a summary chain that keeps ≤4 screenshots alive and logs an arbitrary number.

4. Build an error recovery hook: when an action fails (button not found), what does the agent do next?

5. Compare screenshot-only Claude 4.7 vs hybrid screenshot + accessibility-tree Qwen2.5-VL on 10 web tasks. Which wins on what task types?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GUI grounding | "click coordinates" | Model outputs (x,y) for a target given an instruction on a screenshot |
| Action schema | "tool definitions" | JSON description of legal actions (click, type, scroll, drag) |
| Accessibility tree | "structured DOM" | Machine-readable UI hierarchy from browser/iOS APIs |
| Hybrid agent | "screenshot + tree" | Uses both image and structural info; more reliable than either alone |
| Visual tool use | "zoom/crop/detect" | Agent calls external vision tools (OCR, detection) mid-plan |
| Summary chain | "memory compression" | Periodic text summaries replace long screenshot histories |
| VisualWebArena | "end-to-end web benchmark" | 2024 end-to-end web task benchmark |
| AgentVista | "2026 hard benchmark" | 12-domain real workflows; even Gemini 3 Pro scores only ~30% |

## Further Reading

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)
