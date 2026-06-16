# T5, BART — Encoder-Decoder Models

> The encoder understands. The decoder generates. Put them back together and you get a model built for input → output tasks: translation, summarization, rewriting, transcription.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT), Phase 7 · 07 (GPT)
**Time:** ~45 min

## The Problem

Decoder-only GPT and encoder-only BERT each trimmed the 2017 architecture for different goals. But many tasks are naturally input-output:

- Translation: English → French.
- Summarization: 5,000-token article → 200-token summary.
- Speech recognition: audio tokens → text tokens.
- Structured extraction: prose → JSON.

For these, encoder-decoder is the cleanest fit. The encoder produces a dense representation of the source. The decoder generates output, cross-attending to that representation at every step. Training is shifted by one on the output side. Same loss as GPT, just conditioned on encoder output.

Two papers define the modern playbook:

1. **T5** (Raffel et al. 2019). "Text-to-Text Transfer Transformer." Every NLP task reframed as text-in, text-out. One architecture, one vocabulary, one loss. Pre-trained on masked span prediction (corrupt spans in input, decode them in output).
2. **BART** (Lewis et al. 2019). "Bidirectional and Auto-Regressive Transformer." A denoising autoencoder: corrupt input in multiple ways (shuffle, mask, delete, rotate), have decoder reconstruct the original.

In 2026, encoder-decoder format persists where input structure matters:

- Whisper (speech → text).
- Google's translation stack.
- Some code completion/repair models with clear context-edit structure.
- Flan-T5 and variants for structured reasoning tasks.

Decoder-only stole the spotlight, but encoder-decoder never disappeared.

## The Concept

![Encoder-decoder with cross-attention](../assets/encoder-decoder.svg)

### The forward loop

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

The key: the encoder runs once per input. The decoder runs autoregressively, but cross-attends to the *same* encoder output at every step. Caching encoder output is free speedup for long inputs.

### T5 pre-training — span corruption

Pick random spans of input (mean length 3 tokens, 15% total). Replace each span with a unique sentinel: `<extra_id_0>`, `<extra_id_1>`, etc. The decoder outputs only the corrupted spans with their sentinel prefixes:

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

Cheaper signal than predicting the full sequence. In the T5 paper's ablations it matches MLM (BERT) and prefix-LM (UniLM).

### BART pre-training — multi-noise denoising

BART tries five corruption functions:

1. Token masking.
2. Token deletion.
3. Text infilling (mask a span, decoder inserts correct length).
4. Sentence permutation.
5. Document rotation.

Combining text infilling + sentence permutation yielded the best downstream numbers. The decoder always reconstructs the original. BART's output is the full sequence, not just corrupted spans — so pre-training compute is higher than T5.

### Inference

Same autoregressive generation as GPT. Greedy / beam / top-p sampling all apply. Beam search (width 4–5) is standard for translation and summarization because the output distribution is narrower than chat.

### When to pick which variant in 2026

| Task | Encoder-decoder? | Why |
|------|------------------|-----|
| Translation | Usually yes | Clear source sequence; fixed output distribution; beam search works |
| Speech-to-text | Yes (Whisper) | Input modality differs from output; encoder shapes audio features |
| Chat / reasoning | No, decoder-only | No persistent "input" — the conversation itself is the sequence |
| Code completion | Usually no | Long-context decoder-only wins; code models like Qwen 2.5 Coder are decoder-only |
| Summarization | Either | BART, PEGASUS beat earlier decoder-only baselines; modern decoder-only LLMs match them |
| Structured extraction | Either | T5 is clean because "text → text" absorbs any output format |

Trend since ~2022: decoder-only took over tasks that belonged to encoder-decoder because (a) instruction-tuned decoder-only LLMs generalize to anything via prompting, (b) one architecture is easier to scale than two, (c) RLHF assumes a decoder. Encoder-decoder holds where input modality differs (speech, image) or beam search quality matters.

## Build It

See `code/main.py`. We implement T5-style span corruption for a toy corpus — the most useful piece of this lesson, since it appears in every encoder-decoder pre-training recipe from here on.

### Step 1: Span corruption

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans totaling ~mask_rate fraction. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

Target format follows T5 convention: `<sent0> span0 <sent1> span1 ...`. The corrupted input interleaves untouched tokens and sentinel tokens at span positions.

### Step 2: Verify round-trip

Given corrupted input and target, reconstruct the original sentence. If your corruption is invertible, the forward pass is well-defined. This is a sanity check — real training never does this, but the test is cheap and catches off-by-one errors in your span bookkeeping.

### Step 3: BART noising

Five functions: `token_mask`, `token_delete`, `text_infill`, `sentence_permute`, `document_rotate`. Combine two and show the result.

## Use It

HuggingFace reference:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5's trick: put the task name in the input text. One model handles dozens of tasks because every task is text-in, text-out. In 2026 this pattern was generalized by instruction-tuned decoder-only models, but T5 codified it first.

## Ship It

See `outputs/skill-seq2seq-picker.md`. This skill chooses between encoder-decoder and decoder-only for a new task based on input-output structure, latency, and quality targets.

## Exercises

1. **Easy.** Run `code/main.py`, apply span corruption to a 30-token sentence, verify that concatenating non-sentinel source tokens with decoded target spans recovers the original.
2. **Medium.** Implement BART's `text_infill` noise: replace random spans with a single `<mask>` token, decoder must infer correct span length plus content. Show an example.
3. **Hard.** Fine-tune `flan-t5-small` on a tiny English → pig-Latin corpus (200 pairs). Measure BLEU on a held-out set of 50 pairs. Compare against fine-tuning `Llama-3.2-1B` with the same data and compute.

## Key Terms

| Term | How people talk about it | What it actually means |
|------|-----------------|-----------------------|
| Encoder-decoder | "seq2seq transformer" | Two stacks: a bidirectional encoder processing input, a causal decoder with cross-attention producing output. |
| Cross-attention | "where source and target talk" | Decoder Q × encoder K/V. The only place encoder information enters the decoder. |
| Span corruption | "T5's pre-training trick" | Replace random spans with sentinel tokens; decoder outputs those spans. |
| Denoising objective | "BART's play" | Apply a noise function to input, train decoder to reconstruct the clean sequence. |
| Sentinel token | "`<extra_id_N>` placeholder" | Special tokens marking corrupted spans in source and re-marking them in target. |
| Flan | "instruction-tuned T5" | T5 fine-tuned on >1,800 tasks; makes encoder-decoder competitive at instruction following. |
| Beam search | "decoding strategy" | Keep top-k partial sequences at each step; standard for translation/summarization. |
| Teacher forcing | "training-time input" | Feed decoder the ground-truth previous output token during training, not the sampled one. |

## Further Reading

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART.
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5.
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper, the canonical encoder-decoder in 2026.
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — Reference implementation.
