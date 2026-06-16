# Capstone Project 15 — Constitutional Safety Harness + Red-Team Range

> Anthropic's Constitutional Classifiers, Meta's Llama Guard 4, Google's ShieldGemma-2, NVIDIA's Nemotron 3 Content Safety, and X-Guard for multilingual coverage define the 2026 safety classifier stack. garak, PyRIT, NVIDIA Aegis, and promptfoo are the standard adversarial evaluation tools. NeMo Guardrails v0.12 wires them into a production pipeline. This capstone connects it all: a layered safety harness around a target application, an autonomous red-team agent running 6+ attack families, and a constitutional self-critique run that produces a measurable harmlessness delta.

**Type:** Capstone
**Languages:** Python (safety pipeline, red team), YAML (policy configuration)
**Prerequisites:** Phase 10 (LLMs from Scratch), Phase 11 (LLM Engineering), Phase 13 (Tools), Phase 14 (Agents), Phase 18 (Ethics, Safety & Alignment)
**Phases Involved:** P10 · P11 · P13 · P14 · P18
**Time:** 25 hours

## The Problem

The frontier of LLM safety in 2026 is not whether classifiers work (they broadly do) but how to correctly layer them around a production application without over-refusing or leaving obvious gaps. Llama Guard 4 handles English policy violations. X-Guard (132 languages) handles multilingual jailbreaks. ShieldGemma-2 catches image-based prompt injection. NVIDIA Nemotron 3 Content Safety covers enterprise categories. Anthropic's Constitutional Classifiers are a separate approach used at training time rather than serving time.

The evolution of attacks matters too. PAIR and TAP automate jailbreak discovery. GCG runs gradient-based suffix attacks. Multi-turn and code-switching attacks exploit agent memory. Any deployed LLM needs a red-team range — garak and PyRIT are the standard drivers — plus documented mitigations and CVSS-scored findings.

You will harden a target application (either an 8B instruct model or a RAG chatbot from another capstone), run 6+ attack families against it, and produce a before/after harmlessness measurement.

## The Concept

The safety pipeline has five layers. **Input sanitization**: strip zero-width characters, decode base64/rot13, normalize Unicode. **Policy layer**: NeMo Guardrails v0.12 rails (off-domain, toxicity, PII extraction). **Classifier gate**: run Llama Guard 4 on input, X-Guard on non-English, ShieldGemma-2 on image inputs. **Model**: the target LLM. **Output filter**: run Llama Guard 4 on output, Presidio PII scrubbing, citation enforcement where applicable. **HITL layer**: outputs flagged as high-risk go to a Slack queue.

The red-team range runs on a scheduler. PAIR and TAP autonomously discover jailbreaks. GCG runs gradient-based suffix attacks. ASCII / base64 / rot13 encoding attacks. Multi-turn attacks (persona adoption, memory exploitation). Code-switching attacks (mixing English with Swahili or Thai). Each run produces a structured findings file with CVSS scoring and a disclosure timeline.

The constitutional self-critique run is a training-time intervention. Take 1k harmful-attempt prompts, have the model draft a response, critique it against a written constitution (do-no-harm rules), and retrain on the critique loop. Measure before/after harmlessness delta on a held-out eval.

## Architecture

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## Tech Stack

- Safety classifiers: Llama Guard 4, ShieldGemma-2, NVIDIA Nemotron 3 Content Safety, X-Guard
- Guardrails framework: NeMo Guardrails v0.12 + OPA
- Red-team drivers: garak (NVIDIA), PyRIT (Microsoft Azure), NVIDIA Aegis, promptfoo
- Jailbreak agents: PAIR (Chao et al., 2023), Tree-of-Attacks (TAP), GCG suffix
- Constitutional training: Anthropic-style self-critique loop + SFT on critiques
- PII scrubbing: Presidio
- Target: An 8B instruct model, or a RAG chatbot from another capstone

## Build It

1. **Target setup.** Stand up an 8B instruct model on vLLM (or reuse another capstone's RAG chatbot). This is the application under test.

2. **Safety pipeline wrapper.** Wire the five-layer pipeline around the target. Verify each layer is independently observable (one span per layer in Langfuse).

3. **Classifier coverage.** Load Llama Guard 4, X-Guard (multilingual), ShieldGemma-2 (image). Run each on a small labeled set to establish baselines.

4. **Red-team scheduler.** Schedule garak, PyRIT, a PAIR agent, a TAP agent, a GCG runner, a multi-turn attacker, and a code-switching attacker. Each runs on an independent queue.

5. **Attack suite.** Six attack families: (1) PAIR automated jailbreaks, (2) TAP tree-of-attacks, (3) GCG gradient suffixes, (4) ASCII / base64 / rot13 encoding, (5) multi-turn persona, (6) multilingual code-switching. Report success rate for each family.

6. **Constitutional self-critique.** Curate 1k harmful-attempt prompts. For each, the target drafts a response. A critique LLM scores against a written constitution ("do no harm," "cite evidence," "refuse illegal requests"). Prompts opposed by the critic are rewritten; the target is fine-tuned on the critique-improved pairs. Measure before/after harmlessness on a held-out eval.

7. **Over-refusal measurement.** Track false-positive rate on a benign-prompt suite (e.g., XSTest). The target must remain helpful on benign questions.

8. **CVSS scoring.** For each successful jailbreak, score per CVSS 4.0 (attack vector, complexity, impact). Produce a disclosure timeline and mitigation plan.

9. **Range automation.** All of the above runs on cron; findings write to a queue; over-refusal regression alerts fire to Slack.

## Use It

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## Ship It

`outputs/skill-safety-harness.md` is the deliverable. A production-grade layered safety pipeline plus a reproducible red-team range with before/after harmlessness delta.

| Weight | Criterion | How to Measure |
|:-:|---|---|
| 25 | Attack surface coverage | 6+ attack families exercised, 2+ languages |
| 20 | True-positive / false-positive trade-off | Attack block rate vs XSTest benign pass rate |
| 20 | Self-critique delta | Before/after harmlessness on held-out eval |
| 20 | Documentation and disclosure | CVSS-scored findings with timeline |
| 15 | Automation and reproducibility | Everything runs on cron with alerting |
| **100** | | |

## Exercises

1. Run garak's prompt-injection plugin on a RAG chatbot and compare attack success rates with and without the output filter layer.

2. Add a seventh attack family: indirect prompt injection via retrieved documents. Measure the additional defenses needed.

3. Implement a "refuse-but-help" mode: when guardrails block, the target provides a safer related answer instead of a bare refusal. Measure the XSTest delta.

4. Multilingual coverage gap: find a language where X-Guard underperforms. Propose a fine-tuning dataset targeting it.

5. Run constitutional self-critique on a 30B model and measure whether the delta scales with size.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| Layered safety | "Defense in depth" | Multiple guardrails at input, gate, output, and HITL |
| Llama Guard 4 | "Meta's safety classifier" | 2026 reference input/output content classifier |
| PAIR | "Jailbreak agent" | Paper on LLM-driven jailbreak discovery (Chao et al.) |
| TAP | "Tree-of-Attacks" | Tree-search variant of PAIR |
| GCG | "Greedy Coordinate Gradient" | Gradient-based adversarial suffix attack |
| Constitutional self-critique | "Anthropic-style training" | Target drafts -> critic scores -> rewrite -> retrain |
| XSTest | "Benign probe set" | Benchmark measuring over-refusal regression |
| CVSS 4.0 | "Severity score" | Standard vulnerability scoring for security findings |

## Further Reading

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — training-time reference
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 input/output classifier
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — image + multimodal safety
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — enterprise reference
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132-language multilingual safety
- [garak](https://github.com/NVIDIA/garak) — NVIDIA red-team toolkit
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft red-team framework
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — rail framework
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — jailbreak agent paper
