# ASCII Art and Visual Jailbreaks

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs" (ACL 2024, arXiv:2402.11753). Mask safety-relevant tokens in a harmful request with ASCII art renderings of the same letters, then send the cloaked prompt. GPT-3.5, GPT-4, Gemini, Claude, and Llama-2 all fail to robustly recognize ASCII art tokens. The attack bypasses PPL (perplexity filtering), paraphrasing defenses, and re-tokenization. Related work: the ViTC benchmark measures recognition of non-semantic visual prompts; StructuralSleight generalizes to "uncommon text-encoded structures" (trees, graphs, nested JSON) as an encoding attack family.

**Type:** Build
**Languages:** Python (standard library, ArtPrompt token-cloaking testbed)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)
**Time:** ~60 min

## Learning Objectives

- Describe the ArtPrompt attack: word identification step, ASCII art substitution, final cloaked prompt.
- Explain why standard defenses (PPL, paraphrasing, re-tokenization) fail against ArtPrompt.
- Define ViTC and describe what it measures.
- Describe StructuralSleight as a generalization to arbitrary "uncommon text-encoded structures."

## The Problem

Attacks via paraphrasing and role-play (Lesson 12) and via long context (Lesson 13) operate at the text-pattern level. ArtPrompt operates at the recognition level: the model does not parse the banned token. It parses an image rendered with characters. Safety filters see harmless punctuation. The model sees a word.

## The Concept

### ArtPrompt, Two Steps

Step 1. Word identification. Given a harmful request, the attacker uses an LLM to identify safety-relevant words (e.g., "bomb" in "how to make a bomb").

Step 2. Cloaked prompt generation. Replace each identified word with its ASCII art rendering (a 7x5 or 7x7 character block spelling out the letter shapes). The model receives a grid of punctuation and spaces that a sufficiently capable model recognizes as the word; safety filters see only the grid.

Result: GPT-4, Gemini, Claude, Llama-2, GPT-3.5 all compromised. ASR exceeds 75% on their benchmark subset.

### Why Standard Defenses Fail

- **PPL (perplexity filtering).** ASCII art has high perplexity — but so does all novel input. A threshold that blocks ArtPrompt also blocks legitimate structured input.
- **Paraphrasing.** Paraphrasing the prompt destroys the ASCII art. But in practice, the paraphrasing LLM often preserves or reconstructs the artwork.
- **Re-tokenization.** Splitting tokens differently does not change the fact that the model's vision is recognizing letter shapes.

The underlying problem is that safety filters are token-level or semantic-level; ArtPrompt operates at the visual recognition level.

### ViTC Benchmark

Recognition of non-semantic visual prompts. Measures a model's ability to read ASCII art, wingdings, and other non-text-semantic visual content. ArtPrompt's effectiveness correlates with ViTC accuracy: the better a model reads visual text, the more ArtPrompt works against it. This is a capability–safety tradeoff.

### StructuralSleight

Generalizes ArtPrompt: Uncommon Text-Encoded Structures (UTES). Trees, graphs, nested JSON, CSV-inside-JSON, diff-style code blocks. If a structure is rare in safety training data but the model can parse it, it can smuggle harmful content.

Defense implication: safety must generalize across the structured representations a model can parse. This set is large and growing.

### Image Modality Analogue

Vision LLMs (GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1) enlarge the attack surface. ArtPrompt-style attacks with real images are stronger than ASCII art versions because image encoders produce richer signals.

### Where This Fits in Phase 18

Lessons 12–14 describe three orthogonal attack vectors: iterative refinement (PAIR), context length (MSJ), encoding (ArtPrompt/StructuralSleight). Lesson 15 shifts from model-centric attacks to system-boundary attacks (indirect prompt injection). Lesson 16 describes defensive tooling responses.

## Use It

`code/main.py` builds a toy ArtPrompt. You can cloak a specific word in a harmful query with ASCII art glyphs, verify the cloaked string passes keyword filtering, and (optionally) decode the cloaked string back with a simple recognizer.

## Ship It

This lesson produces `outputs/skill-encoding-audit.md`. Given a jailbreak defense report, it enumerates the encoding attack families covered (ASCII art, base64, leet-speak, UTF-8 homoglyphs, UTES) and the defense layer that catches each.

## Exercises

1. Run `code/main.py`. Verify the cloaked string passes a simple keyword filter. Report the character-level edit distance required.

2. Implement a second encoding: base64 for the same target word. Compare filter bypass rate and recovery difficulty against ArtPrompt.

3. Read Jiang et al. 2024 Section 4.3 (five-model results). Propose a reason why Claude shows higher ArtPrompt resistance than Gemini on the same benchmark.

4. Design a pre-generation defense that detects regions of ASCII art shapes in a prompt. Measure its false-positive rate on legitimate code, tables, and mathematical notation.

5. StructuralSleight lists 10 encoding structures. Sketch a universal defense that handles all 10, and estimate the computational cost per defended prompt.

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| ArtPrompt | "the ASCII art attack" | Two-step jailbreak that masks safety words with ASCII art renderings |
| Cloaking | "hiding the word" | Replacing a banned token with a visual representation the model reads but filters don't |
| UTES | "uncommon structures" | Uncommon Text-Encoded Structures — trees, graphs, nested JSON, etc., used to smuggle content |
| ViTC | "visual text capability" | Benchmark measuring a model's ability to read non-semantic visual encodings |
| Perplexity filtering | "PPL defense" | Rejecting high-perplexity prompts; fails because legitimate structured input also scores high |
| Re-tokenization | "different tokenizer defense" | Pre-processing the prompt with a different tokenizer; fails because recognition is visual |
| Homoglyphs | "lookalike characters" | Unicode characters that look identical to Latin letters; bypass substring checks |

## Further Reading

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — ASCII art jailbreak paper
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — UTES generalization
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — Complementary iterative attack
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) — Complementary length attack
