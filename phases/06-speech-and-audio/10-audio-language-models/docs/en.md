# Audio-Language Models — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

> Audio-language models in 2026 can reason over speech + environmental sounds + music. Qwen2.5-Omni-7B matches GPT-4o Audio on MMAU-Pro. Audio Flamingo Next beats Gemini 2.5 Pro on LongAudioBench. The gap between open-source and closed-source is essentially closed — except on multi-audio tasks, where everyone is near random.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 12 · 03 (Vision-Language Models), Phase 7 · 10 (Audio Transformers)
**Time:** ~45 minutes

## The Problem

You have 5 seconds of audio: a dog barking, someone yelling "stop!", then silence. Useful questions span multiple dimensions:

- **Transcription.** "What was said?" — ASR territory.
- **Semantic reasoning.** "Is this person in danger?" — requires joint understanding of bark + yell + silence.
- **Music reasoning.** "Which instruments are playing the melody?"
- **Long-audio retrieval.** "In this 90-minute lecture, where does the instructor discuss gradient descent?"

A single model that answers all of these with one prompt is an **audio-language model** (LALM / ALM). It differs from pure ASR: a LALM produces free-form natural language answers, not just transcribed text.

## The Concept

![Audio-language models: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### The Three-Component Template

Every LALM in 2026 is the same skeleton:

1. **Audio encoder.** Whisper encoder · BEATs · CLAP · WavLM · or each model's custom encoder.
2. **Projector.** Linear layer or MLP that bridges audio encoder features into the LLM's token embedding space.
3. **LLM.** Llama / Qwen / Gemma-based decoder. Receives interleaved text + audio tokens; generates text.

Training:

- **Stage 1.** Freeze encoder + LLM; train only the projector on ASR / captioning data.
- **Stage 2.** Full / LoRA fine-tune on instruction-following audio tasks (QA, reasoning, music understanding).
- **Stage 3 (optional).** Speech-in / speech-out by adding a speech decoder. Qwen2.5-Omni and AF3-Chat do this.

### 2026 Model Map

| Model | Backbone | Audio Encoder | Output Modality | Access |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | Text + Speech | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | Text + Speech | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | Text | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | Text | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | Text | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | Text | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | Text | Apache-2.0 |
| Gemini 2.5 Flash/Pro (closed) | Gemini | Proprietary | Text + Speech | API |
| GPT-4o Audio (closed) | GPT-4o | Proprietary | Text + Speech | API |

### Benchmark Reality Check (2026)

**MMAU-Pro.** 1800 QA pairs covering speech / sound / music / mixed. Includes a multi-audio subset.

| Model | Overall | Speech | Sound | Music | Multi-audio |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | SOTA on LongAudioBench | — | — | — | — |

**The multi-audio column is a wake-up call for everyone.** Random chance on 4-way multiple choice = 25%; most models hover there. LALMs still struggle to compare two audio clips.

### Where LALMs Are Useful in 2026

- **Compliance auditing of call center recordings.** "Did the agent mention the required disclosure?"
- **Accessibility.** Describing sound events for deaf users (not just transcription).
- **Content moderation.** Detecting violent language + threatening tone + background context.
- **Podcast / meeting chaptering.** Semantic summarization, not just speaker turns.
- **Music catalog analysis.** "Find all tracks where the B section has a key change."

### Where They're Not (Yet) Useful

- Fine-grained music theory (below chord level).
- Reasoning with speaker attribution in long conversations (degrades beyond 10 minutes).
- Multi-audio comparison (22–26%, barely above random).
- Real-time streaming inference (most are offline batch).

## Build It

### Step 1: Query Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Step 2: The Projector Pattern

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

That's it. The projector is usually 1–3 linear layers. Train it on ASR pairs (audio → transcript), and that's the Stage 1 pretext task.

### Step 3: Run Benchmarks on MMAU / LongAudioBench

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

Report by category (speech / sound / music / multi-audio). Aggregate numbers hide where the model fails.

## Use It

| Task | 2026 Pick |
|------|-----------|
| Free-form audio QA (open-source) | Qwen2.5-Omni-7B |
| Strongest open-source on long audio | Audio Flamingo Next |
| Strongest closed-source | Gemini 2.5 Pro |
| Speech-in / speech-out agent | Qwen2.5-Omni or GPT-4o Audio |
| Music reasoning | Audio Flamingo 3 or 2 (music-specialized AF-CLAP) |
| Call center auditing | Gemini 2.5 Pro via API, with RAG over your policy docs |

## Pitfalls

- **Over-trusting on multi-audio.** If your task needs "which clip has X," random-chance-level performance is real.
- **Long-audio degradation.** Beyond 10 minutes, most models' speaker attribution collapses. Do speaker diarization first (Lesson 6), then summarize.
- **Hallucination on silence.** LALMs using Whisper encoders inherit the same Whisper-style problems. Gate with VAD.
- **Benchmark cherry-picking.** Vendor blog posts highlight their best category. Run MMAU-Pro's multi-audio subset yourself.

## Ship It

Save as `outputs/skill-alm-picker.md`. Pick the LALM + benchmark subset + output modality (text vs speech) for a given audio understanding task.

## Exercises

1. **Easy.** Run `code/main.py` to see a toy projector pattern + fake LALM routing (audio embeddings, text tokens) → output tokens.
2. **Medium.** Score Qwen2.5-Omni-7B on 100 MMAU-Pro speech items. Compare against paper-reported numbers.
3. **Hard.** Build a minimal audio captioning baseline: BEATs encoder + 2-layer projector + frozen Llama-3.2-1B. Fine-tune only the projector on AudioCaps. Compare against SALMONN on Clotho-AQA.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| LALM | Audio ChatGPT | Audio encoder + projector + LLM decoder. |
| Projector | Adapter | Small MLP mapping audio features into LLM embedding space. |
| MMAU | The benchmark | 10k audio QA pairs spanning speech, sound, music. |
| MMAU-Pro | Harder MMAU | 1800 multi-audio / heavy-reasoning questions. |
| LongAudioBench | Long-audio eval | Minutes-long clips with semantic queries. |
| Speech-in / speech-out | Speech-native | Model directly ingests speech and produces speech, no text detour. |

## Further Reading

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — reference architecture.
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — speech-in speech-out.
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — open-source long-audio leader.
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA.
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — dual-encoder pioneer.
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — 2026 live rankings.
