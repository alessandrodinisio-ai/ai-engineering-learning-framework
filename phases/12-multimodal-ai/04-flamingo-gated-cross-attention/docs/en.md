# Flamingo and Gated Cross-Attention for Few-Shot VLMs

> DeepMind's Flamingo (2022) did two things before anyone else. It proved a single model can handle arbitrarily interleaved image, video, and text sequences. It also proved VLMs can do in-context learning — give a few-shot prompt with three (image, caption) examples and the model captions a new image without a single gradient step. The mechanism: gated cross-attention layers inserted between the frozen LLM's existing layers, with a learnable tanh gate starting from zero, so the LLM's text capability is perfectly preserved at initialization. This lesson walks through Flamingo's Perceiver resampler and gated cross-attention architecture — the ancestor of Gemini's interleaved inputs and Idefics2's visual tokens.

**Type:** Learn
**Languages:** Python (stdlib, gated cross-attention + Perceiver resampler demo)
**Prerequisites:** Phase 12 · 03 (BLIP-2 Q-Former)
**Time:** ~120 min

## Learning Objectives

- Explain how gated cross-attention preserves the frozen LLM's text capability at initialization through tanh(gate) = 0.
- Walk through the Perceiver resampler: N image patches → K fixed "latent" queries via cross-attention.
- Describe how Flamingo handles interleaved image-text sequences with causal masks that respect image position.
- Reproduce a few-shot multimodal prompt structure (3 image-caption examples followed by a query image).

## The Problem

BLIP-2 feeds 32 visual tokens into the frozen LLM's input layer. This works for one image per prompt. But what if you want to feed *many* images, interleaved with text, like "here's image A, caption it; here's image B, caption it; now here's image C, caption it"? The LLM's self-attention has to handle image tokens and text tokens in a single stream, and the question of which positions can attend to which images gets annoying.

Flamingo's answer: don't modify the LLM's input stream at all. Insert extra cross-attention layers between the existing LLM blocks. Text tokens flow through the LLM's causal self-attention as normal. Every few LLM blocks, text tokens additionally cross-attend to image features through a new gated layer. The gate (initialized to zero) means the new layer is a no-op at step 0 — the model behaves identically to the pretrained LLM. As training progresses, the gate opens and visual information flows in.

The second question Flamingo answers: how to handle variable numbers of images per prompt (0, 1, or many)? With a Perceiver resampler — a small cross-attention module that produces a fixed number of visual latent tokens regardless of how many patches you have. The LLM's cross-attention layers see the same shape regardless of how many images are in the prompt.

## The Concept

### The Frozen LLM

Flamingo starts from a frozen Chinchilla 70B LLM. All 70B weights remain untouched. Existing text self-attention and FFN operate normally.

### Perceiver Resampler

For each image in the prompt, the ViT produces N patch tokens. The Perceiver resampler has K fixed learnable latents (Flamingo uses K=64). Each resampler block has two sub-steps:

1. Cross-attention: K latents attend to N patch tokens (Q from latents, K/V from patches).
2. Self-attention among latents + FFN.

After 6 resampler blocks, the output is K=64 visual tokens of dim 1024, regardless of how many patches the ViT produced. A 224x224 image (196 patches) and a 480x480 image (900 patches) both come out as 64 resampler tokens.

For video, the resampler applies temporally: each frame's patches produce 64 latents, a temporal position encoding lets the model distinguish t=0 from t=N. The full video becomes T * 64 visual tokens.

### Gated Cross-Attention

Every M layers of the frozen LLM (Flamingo uses M=4), a new gated cross-attention block is inserted:

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` is a learnable scalar initialized to zero.
- `tanh(0) = 0`, so at initialization the gated branch contributes nothing.
- As `alpha` moves away from zero, the cross-attention contribution grows smoothly.
- The residual connection means even with the gate fully open, it doesn't overwrite the LLM's text representations; it only adds visual information on top.

This is the single most important design choice in Flamingo: visual conditioning is additive, gated, and zero-initialized. At step 0, Flamingo on text-only input is a perfect Chinchilla 70B.

### Masked Cross-Attention for Interleaved Inputs

In a prompt like "<image A> caption A <image B> caption B <image C> ?", each text token should only see images that precede it in the sequence. The cross-attention mask enforces: the text token at position `t` only attends to the resampler tokens of images with index `i < i_t`, where `i_t` is the most recent image before position `t`. Both "only attend to the most recent preceding image" and "attend to all preceding images" are valid choices; Flamingo chose the former.

### In-Context Few-Shot Learning

A Flamingo prompt looks like:

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

The model sees the completion pattern and outputs "bird" (or whatever image3 shows). No gradient steps. The frozen LLM's in-context learning ability transfers through the gated cross-attention — this is the paper's punchline and why it matters.

### Training Data

Flamingo trains on three datasets:

1. MultiModal MassiveWeb (M3W): 43M image-text interleaved web pages with reading order reconstructed.
2. Image-text pairs (ALIGN + LTIP): 4.4B pairs.
3. Video-text pairs (VTP): 27M short videos.

OBELICS (2023) is the open reproduction of the interleaved web corpus; Idefics, Idefics2, and most open "Flamingo-like" models train on it.

### OpenFlamingo and Otter

OpenFlamingo (2023) is the open reproduction. Architecture is identical (Perceiver resampler + gated cross-attention on frozen LLaMA or MPT). Checkpoints at 3B, 4B, 9B. Quality lags behind Flamingo due to smaller base LLMs and less data.

Otter (2023) instruction-tunes OpenFlamingo with MIMIC-IT (a multimodal instruction dataset), proving gated cross-attention works for instruction following too.

### Descendants

- Idefics / Idefics2 / Idefics3: Hugging Face's gated cross-attention lineage, progressively simplified (Idefics2 drops the resampler in favor of direct patch tokens with adaptive pooling).
- Flamingo to Chameleon transition: by 2024 many teams moved to early fusion (Lesson 12.11); Flamingo-style gated cross-attention persists in production where frozen backbones are a requirement.
- Gemini's interleaved inputs: conceptually inherits Flamingo's interleaved format flexibility, though the exact mechanism is proprietary.

### Comparison to BLIP-2

| | BLIP-2 | Flamingo |
|---|---|---|
| Vision bridge | Single Q-Former at input | Gated cross-attention every M layers |
| Visual tokens | 32 per image | 64 per image per cross-attention layer |
| Frozen LLM | Yes | Yes |
| Few-shot in-context | Weak | Strong — paper's core selling point |
| Interleaved inputs | No native support | Yes, by design |
| Training data | 130M pairs | 1.3B pairs + 43M interleaved pages |
| Parameters | 188M trained | ~10B trained (cross-attention layers) |
| Compute | Few A100s for days | Thousands of TPUv4 for weeks |

Pick BLIP-2 for budget-constrained single-image VQA. Pick Flamingo/Idefics2 for interleaved, few-shot, or multi-image reasoning.

## Use It

`code/main.py` demonstrates:

1. A Perceiver resampler with 8 learnable latents on 36 fake patch tokens (pure Python cross-attention).
2. A gated cross-attention step: `alpha = 0` → output equals input (LLM unchanged), then `alpha = 2.0` → visual contribution is mixed in.
3. An interleaved mask builder producing a 2D attention mask for a "(img 1) (text 1) (img 2) (text 2)" sequence.

## Ship It

This lesson produces `outputs/skill-gated-bridge-diagnostic.md`. Given an open VLM's configuration (whether it has a resampler, cross-attention frequency, gating scheme), it identifies Flamingo-lineage elements and explains the freezing strategy. Useful for debugging "why did a fine-tune degrade text performance" (answer: the gate opened too fast and too far).

## Exercises

1. Compute visual parameter count for Flamingo-9B: 9B LLM + 1.4B gated cross-attention layers + 64M resampler. What fraction of total is being trained?

2. Implement the gated residual `y = tanh(alpha) * cross + x` in PyTorch. Demonstrate experimentally that at `alpha=0`, `y==x` holds exactly at initialization.

3. Read OpenFlamingo Section 3.2 (arXiv:2308.01390) on how they handle multiple images when each prompt in a batch has a different number of images. Describe their padding strategy.

4. Why does Flamingo's cross-attention mask let text tokens attend only to the *most recent* preceding image, not all preceding images? Read Flamingo paper Section 2.4 and explain the tradeoff.

5. In-context few-shot: construct a prompt with 4 "image → subject color" examples for a new Flamingo variant. Describe the expected accuracy trajectory as you vary the number of examples from 0 to 8.

## Key Terms

| Term | Common phrasing | What it actually means |
|------|----------------|------------------------|
| Perceiver resampler | "Fixed-latent cross-attention" | Module that produces K fixed tokens from a variable number of input patches |
| Gated cross-attention | "Tanh-gated bridge" | Residual layer `y = tanh(alpha)*cross + x` with alpha learnable, initialized to 0 |
| Interleaved inputs | "Mixed sequence" | Prompt format where images and text are freely interleaved in reading order |
| Frozen LLM | "No LLM gradient" | Text LLM weights don't update; only resampler + cross-attention layers train |
| Few-shot | "In-context examples" | Providing several (image, answer) pairs in the prompt; model generalizes without fine-tuning |
| OBELICS | "Interleaved web corpus" | An open dataset of 141M web pages with images and text in reading order |
| Chinchilla | "70B frozen base" | Flamingo's frozen text LLM, from DeepMind's Chinchilla paper |
| Gate schedule | "How alpha evolves" | The rate at which the cross-attention gate opens during training |
| Cross-attention frequency | "Every M layers" | How often a gated cross-attention block is inserted; Flamingo uses M=4 |
| OpenFlamingo | "Open reproduction" | MosaicML/LAION's 3-9B open checkpoints; architecture identical to Flamingo |

## Further Reading

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — the original paper.
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — open reproduction.
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — interleaved web corpus.
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — general Perceiver architecture.
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — instruction-tuned Flamingo descendant.
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — modern simplification of the Flamingo approach.
