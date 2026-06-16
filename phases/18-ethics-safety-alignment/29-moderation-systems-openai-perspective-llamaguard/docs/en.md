# Moderation Systems — OpenAI, Perspective, Llama Guard

> Production moderation systems operationalize the safety policies defined in Lessons 12-16. OpenAI Moderation API: `omni-moderation-latest` (2024) is built on GPT-4o and classifies text + images in a single call; 42% better than the previous version on multilingual test sets; the response schema returns 13 category booleans — harassment, harassment/threatening, hate, hate/threatening, illicit, illicit/violent, self-harm, self-harm/intent, self-harm/instructions, sexual, sexual/minors, violence, violence/graphic; free for most developers. Layered pattern: input moderation (pre-generation), output moderation (post-generation), custom moderation (domain rules). Async parallel calls hide latency; return a placeholder response when a flag is hit. Llama Guard 3/4 (Lesson 16): 14 MLCommons harms, code interpreter abuse, 8 languages (v3), multi-image (v4). Perspective API (Google Jigsaw): toxicity scoring predating the "LLM-as-moderator" wave; primarily single-dimension toxicity with severe-toxicity/insult/profanity variants; baseline for content moderation research. Deprecation: Azure Content Moderator deprecated February 2024, retiring February 2027, replaced by Azure AI Content Safety.

**Type:** Build
**Languages:** Python (standard library, three-layer moderation test harness)
**Prerequisites:** Phase 18 Lesson 16 (Llama Guard / Garak / PyRIT)
**Time:** ~60 minutes

## Learning Objectives

- Describe the OpenAI Moderation API category taxonomy and how it differs from Llama Guard 3's MLCommons set.
- Describe the three-moderation-layer pattern (input, output, custom) and name one failure mode for each.
- Describe the Perspective API's position as a "pre-LLM-era" baseline and why it is still used in research.
- Name the Azure deprecation timeline.

## The Problem

Lessons 12-16 describe attacks and defensive tools. Lesson 29 covers the deployed moderation systems that operationalize defenses at the user-facing surface. The three-layer pattern is the 2026 default.

## The Concept

### OpenAI Moderation API

`omni-moderation-latest` (2024). Built on GPT-4o. Classifies text + images in a single call. Free for most developers.

Categories (13 booleans in the response schema):
- harassment, harassment/threatening
- hate, hate/threatening
- self-harm, self-harm/intent, self-harm/instructions
- sexual, sexual/minors
- violence, violence/graphic
- illicit, illicit/violent

Multimodal support applies to `violence`, `self-harm`, `sexual`, but not `sexual/minors`; the rest are text-only.

In the code test harness in `code/main.py`, subcategories like `/threatening`, `/intent`, `/instructions`, `/graphic` are collapsed into their top-level parents for pedagogical brevity. Production code should use the full 13-category schema.

42% better than the previous-generation moderation endpoint on multilingual test sets. One score per category; applications set their own thresholds.

### Llama Guard 3/4

Covered in Lesson 16. 14 MLCommons harm categories (organized differently from OpenAI's 13 response schema booleans). 8 languages supported (v3). Llama Guard 4 (April 2025) is natively multimodal, 12B.

The OpenAI and Llama Guard taxonomies overlap but diverge. OpenAI has "illicit" as a broad category; Llama Guard separates "violent crimes" from "non-violent crimes." Deployers choose based on fit with their policy taxonomy.

### Perspective API (Google Jigsaw)

Toxicity scoring system predating the "LLM-as-moderator" wave (pre-2020). Categories: TOXICITY, SEVERE_TOXICITY, INSULT, PROFANITY, THREAT, IDENTITY_ATTACK. Single-dimension primary score (TOXICITY) with sub-dimension variants.

Widely used as a content moderation research baseline because the API is stable, documented, and has years of calibration data. For modern LLM-adjacent use cases, Llama Guard or OpenAI Moderation is typically a better fit.

### The Three-Layer Pattern

1. **Input moderation.** Classifies the user's prompt before generation. Flags trigger rejection. Latency: one classifier call.
2. **Output moderation.** Classifies the model's output before delivery. Flags trigger replacement with a refusal. Latency: one classifier call after generation.
3. **Custom moderation.** Domain-specific rules (regex, allowlists, business policies). Runs at input or output.

The three layers are sequential by design: input moderation must complete before generation, output moderation runs after generation. Parallelism occurs within a layer — running multiple classifiers concurrently on the same text (e.g., OpenAI Moderation + Llama Guard + Perspective) hides per-classifier latency. As an optional optimization, a placeholder response ("one moment, checking...") can be shown while input moderation completes, deferring first-token streaming. Flag behavior is configurable: reject, sanitize, or escalate to human review.

### Failure Modes

- **Input-only.** Cannot catch output hallucinations (encoding attacks from Lessons 12-14 bypass input classifiers).
- **Output-only.** Allows arbitrary inputs to reach the model; increases cost; exposes internal reasoning to attackers.
- **Custom-only.** Not robust across categories; regex is brittle.

Layering is the default. Defense in depth.

### Azure Deprecation

Azure Content Moderator: deprecated February 2024, retiring February 2027. Replaced by Azure AI Content Safety, which is LLM-based and integrates with Azure OpenAI. This migration is a 2024-2027 production-grade project for Azure deployments.

### Position Within Phase 18

Lesson 16 covers moderation tools in a red-teaming context. Lesson 29 covers deployed moderation. Lesson 30 closes with current dual-use capability evidence.

## Use It

`code/main.py` builds a three-layer moderation test harness: input moderator (keyword + category scores), output moderator (runs the same classifier on output), custom moderator (domain rules). You can pass inputs through and observe which layer catches what.

## Ship It

This lesson produces `outputs/skill-moderation-stack.md`. Given a deployment, it recommends a moderation stack configuration: which classifier at input, which at output, what custom rules, and what arbiter for edge cases.

## Exercises

1. Run `code/main.py`. Pass a benign, a borderline, and a harmful input through all three layers. Report which layer triggers for each.

2. Extend the test harness with a Perspective-API-style toxicity score for a specific category. Compare its threshold behavior to the category scores.

3. Read the OpenAI Moderation API documentation and the Llama Guard 3 category list. Map each OpenAI category to the closest Llama Guard category. Identify three categories that do not map cleanly.

4. Design a moderation stack for a code assistant deployment (e.g., GitHub Copilot). Identify the most and least relevant categories and propose custom rules.

5. Azure Content Moderator retires February 2027. Plan a migration to Azure AI Content Safety. Identify the highest-risk element of the migration.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| OpenAI Moderation | "omni-moderation-latest" | GPT-4o-based, 13-category (text), partial multimodal support classifier |
| Perspective API | "Google Jigsaw toxicity" | Pre-LLM-era toxicity scoring baseline |
| Llama Guard | "MLCommons 14 categories" | Meta's harm classifier (v3: 8B text, 8 languages; v4: 12B multimodal) |
| Input moderation | "pre-generation filter" | Classifier on user prompts before model invocation |
| Output moderation | "post-generation filter" | Classifier on model output before delivery |
| Custom moderation | "domain rules" | Deployment-specific rules (regex, allowlists, policies) |
| Layered moderation | "all three layers" | Standard production deployment pattern |

## Further Reading

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation endpoint
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard repository
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — toxicity scoring
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure replacement
