# Vision-Language Models — The ViT-MLP-LLM Pattern

> A vision encoder turns an image into tokens. An MLP projector maps those tokens into the LLM's embedding space. The language model does the rest. That pattern — ViT-MLP-LLM — is every production VLM in 2026.

**Type:** Learn + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 14 (ViT), Phase 4 Lesson 18 (CLIP), Phase 7 Lesson 02 (Self-Attention)
**Time:** ~75 min

## Learning Objectives

- Name the ViT-MLP-LLM architecture and explain what each of the three components contributes
- Compare Qwen3-VL, InternVL3.5, LLaVA-Next, and GLM-4.6V on parameter count, context length, and benchmark performance
- Explain DeepStack: why multi-level ViT features tighten vision-language alignment better than a single final-layer feature
- Measure VLM hallucination in production using Cross-Modal Error Rate (CMER) and act on it

## The Problem

CLIP (Phase 4 Lesson 18) gives you a shared embedding space for images and text — enough for zero-shot classification and retrieval. It can't answer "how many red cars are in this image?" because CLIP doesn't generate text — it only scores similarity.

Vision-language models (VLMs) — Qwen3-VL, InternVL3.5, LLaVA-Next, GLM-4.6V — attach a CLIP-family image encoder to a full language model. The model sees an image plus a question and generates an answer. In 2026, open-source VLMs match or beat GPT-5 and Gemini-2.5-Pro on multimodal benchmarks (MMMU, MMBench, DocVQA, ChartQA, MathVista, OSWorld).

The three components (ViT, projector, LLM) are standard. Models differ in which ViT, which projector, which LLM, training data, and alignment recipe. Once you understand the pattern, swapping any component is mechanical.

## The Concept

### ViT-MLP-LLM Architecture

```mermaid
flowchart LR
    IMG["Image<br/>(H x W x 3)"] --> ViT["Vision encoder<br/>(ViT, CLIP-L,<br/>SigLIP, DINOv3)"]
    ViT --> FEATS["Image tokens<br/>(N, d_vit)"]
    FEATS --> PROJ["Projector<br/>(2-4 layer MLP<br/>or Q-former)"]
    PROJ --> VTOK["Image tokens in<br/>LLM space<br/>(N, d_llm)"]
    TXT["Text prompt"] --> TOK["LLM tokenizer"]
    TOK --> TTOK["Text tokens<br/>(M, d_llm)"]
    VTOK --> CONCAT["Interleave<br/>or concatenate"]
    TTOK --> CONCAT
    CONCAT --> LLM["Decoder LLM<br/>(Qwen3, LLaMA, etc.)"]
    LLM --> OUT["Text answer"]

    style ViT fill:#dbeafe,stroke:#2563eb
    style PROJ fill:#fef3c7,stroke:#d97706
    style LLM fill:#dcfce7,stroke:#16a34a
```

1. **Vision encoder** — A pretrained ViT (CLIP-L/14, SigLIP, DINOv3, or a fine-tuned variant). Produces patch tokens.
2. **Projector** — A small module (2-4 layer MLP, or a Q-former) that maps vision tokens into the LLM's embedding dimension. Most fine-tuning happens here.
3. **LLM** — A decoder-only language model (Qwen3, Llama, Mistral, GLM, InternLM). Reads vision + text tokens sequentially and generates text.

In principle all three components are trainable. In practice the vision encoder and LLM are mostly kept frozen while the projector trains — billions of parameters' worth of signal, cheap.

### DeepStack

Naive projection uses only the final ViT layer. DeepStack (Qwen3-VL) samples features from multiple ViT depths and stacks them. Deeper layers carry high-level semantics; shallower layers carry fine-grained spatial and texture information. Feeding both to the LLM bridges the gap between "what the image contains" (semantics) and "exactly where" (spatial grounding).

### Three Training Phases

Modern VLMs train in stages:

1. **Alignment** — Freeze ViT and LLM. Train only the projector on image-caption pairs. Teaches the projector to map vision space into language space.
2. **Pre-training** — Unfreeze everything. Train on large-scale interleaved image-text data (500M+ pairs). Builds the model's visual knowledge.
3. **Instruction tuning** — Fine-tune on curated (image, question, answer) triplets. Teaches conversational behavior and task formats. This is what turns a "vision-aware language model" into a usable assistant.

Most LoRA fine-tuning targets stage 3 with a small labeled dataset.

### Model Family Comparison (Early 2026)

| Model | Params | Vision Encoder | LLM | Context | Strengths |
|-------|--------|----------------|-----|---------|-----------|
| Qwen3-VL-235B-A22B (MoE) | 235B (22B active) | Custom ViT + DeepStack | Qwen3 | 256K | General SOTA, GUI agent |
| Qwen3-VL-30B-A3B (MoE) | 30B (3B active) | Custom ViT + DeepStack | Qwen3 | 256K | Smaller MoE alternative |
| Qwen3-VL-8B (dense) | 8B | Custom ViT | Qwen3 | 128K | Production dense default |
| InternVL3.5-38B | 38B | InternViT-6B | Qwen3 + GPT-OSS | 128K | Strong on MMBench / MMVet |
| InternVL3.5-241B-A28B | 241B (28B active) | InternViT-6B | Qwen3 | 128K | Competitive with GPT-4o |
| LLaVA-Next 72B | 72B | SigLIP | Llama-3 | 32K | Open, easy to fine-tune |
| GLM-4.6V | ~70B | Custom | GLM | 64K | Open-source, strong OCR |
| MiniCPM-V-2.6 | 8B | SigLIP | MiniCPM | 32K | Edge-friendly |

### Vision Agents

Qwen3-VL-235B achieves top performance on OSWorld — the benchmark for **vision agents** that operate GUIs (desktop, mobile, web). The model sees a screenshot, understands the UI, and issues actions (click, type, scroll). Combined with tools, it closes the loop on common desktop tasks. This is what most 2026 "AI PC" demos run under the hood.

### Agentic Capabilities + RoPE Variants

VLMs need to know **when** a frame occurs in video. Qwen3-VL evolved from T-RoPE (temporal rotary positional embedding) to **text-based temporal alignment** — explicit timestamp text tokens interleaved with video frames. The model sees "`<timestamp 00:32>` frame, prompt" and can reason about temporal relationships.

### The Alignment Problem

12% of image-text pairs in crawled datasets have captions not fully grounded in the image. VLMs trained on this quietly learn to hallucinate — fabricating objects, misreading numbers, inventing relationships. This is the dominant failure mode in production.

Skywork.ai introduced **Cross-Modal Error Rate (CMER)** to track it:

```
CMER = proportion of outputs where text confidence is high but image-text similarity (via a CLIP-family checker) is low
```

High CMER means the model is confidently saying things unsupported by the image. Monitoring CMER as a production KPI reduced hallucination rate by ~35% in their deployments. The trick isn't "fix the model" — it's "route high-CMER outputs to human review."

### Fine-Tuning with LoRA / QLoRA

Full fine-tuning a 70B VLM is out of reach for most teams. LoRA (rank 16-64) on attention + projector layers, or QLoRA with 4-bit base weights, fits on a single A100 / H100. Cost: 5,000-50,000 samples, $100-5,000 compute, 2-10 hours training.

### Spatial Reasoning Remains Weak

Current VLMs score 50-60% on spatial reasoning benchmarks (above/below, left/right, counting, distance). If your use case depends on "which object is on top of which," validate heavily — general VLMs perform below human level. Alternatives that outperform VLMs on pure spatial tasks: a dedicated keypoint / pose estimator, a depth model, or a detection model with box-geometry post-processing.

## Build It

### Step 1: The Projector

The part you train most often. A 2-4 layer MLP with GELU.

```python
import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, vit_dim=768, llm_dim=4096, hidden=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vit_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, llm_dim),
        )

    def forward(self, x):
        return self.net(x)
```

Input is an `(N_patches, d_vit)` token tensor. Output is `(N_patches, d_llm)`. The LLM treats each output row as another token.

### Step 2: End-to-End ViT-MLP-LLM Assembly

A skeleton of a minimal VLM forward pass. Real code uses `transformers`; this is the conceptual layout.

```python
class MinimalVLM(nn.Module):
    def __init__(self, vit, projector, llm, image_token_id):
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.llm = llm
        self.image_token_id = image_token_id  # placeholder token in the text prompt

    def forward(self, image, input_ids, attention_mask):
        # 1. Vision features
        vision_tokens = self.vit(image)                     # (B, N_patches, d_vit)
        vision_embeds = self.projector(vision_tokens)       # (B, N_patches, d_llm)

        # 2. Text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, M, d_llm)

        # 3. Replace image placeholder tokens with vision embeddings
        merged = self._merge(text_embeds, vision_embeds, input_ids)

        # 4. Run the LLM
        return self.llm(inputs_embeds=merged, attention_mask=attention_mask)

    def _merge(self, text_embeds, vision_embeds, input_ids):
        out = text_embeds.clone()
        expected = vision_embeds.size(1)
        for b in range(input_ids.size(0)):
            positions = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(positions) != expected:
                raise ValueError(
                    f"batch item {b} has {len(positions)} image tokens but vision_embeds has {expected} patches."
                    " Every sample in the batch must be pre-padded to the same number of image placeholder tokens.")
            out[b, positions] = vision_embeds[b]
        return out
```

The `<image>` placeholder tokens in the text are replaced with actual image embeddings — the same pattern used by LLaVA, Qwen-VL, and InternVL.

### Step 3: CMER Computation

A lightweight runtime check.

```python
import torch.nn.functional as F


def cross_modal_error_rate(image_emb, text_emb, text_confidence, sim_threshold=0.25, conf_threshold=0.8):
    """
    image_emb, text_emb: embeddings of image and generated text (internally normalized)
    text_confidence:     average per-token probability in [0, 1]
    Returns:             proportion of high-confidence outputs with low image-text alignment
    """
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    sim = (image_emb * text_emb).sum(dim=-1)        # cosine similarity
    high_conf_low_sim = (text_confidence > conf_threshold) & (sim < sim_threshold)
    return high_conf_low_sim.float().mean().item()
```

Treat CMER as a production KPI. Monitor it per endpoint, per prompt type, per customer. Rising CMER means the model is starting to hallucinate on some input distribution.

### Step 4: Toy VLM Classifier (Runnable)

Demonstrates the projector training. Fake "ViT features" go in; a small LLM-style token predicts a class.

```python
class ToyVLM(nn.Module):
    def __init__(self, vit_dim=32, llm_dim=64, num_classes=5):
        super().__init__()
        self.projector = Projector(vit_dim, llm_dim, hidden=64)
        self.head = nn.Linear(llm_dim, num_classes)

    def forward(self, vision_tokens):
        projected = self.projector(vision_tokens)
        pooled = projected.mean(dim=1)
        return self.head(pooled)
```

You can fit this on synthetic (feature, class) pairs in under 200 steps — enough to show the projector pattern works.

## Use It

Three ways production teams use VLMs in 2026:

- **Hosted APIs** — OpenAI Vision, Anthropic Claude Vision, Google Gemini Vision. Zero infrastructure, vendor risk.
- **Open-source self-hosted** — Run Qwen3-VL or InternVL3.5 via `transformers` and `vllm`. Full control, higher upfront investment.
- **Fine-tuned on domain** — Load Qwen2.5-VL-7B or LLaVA-1.6-7B, LoRA on 5k-50k custom samples, serve with `vllm` or `TGI`.

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image

model_id = "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": Image.open("plot.png")},
        {"type": "text", "text": "What does this chart show?"},
    ],
}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
generated = model.generate(**inputs, max_new_tokens=256)
answer = processor.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

`apply_chat_template` hides the `<image>` placeholder tokenization; the model handles the merge internally.

## Ship It

This lesson produces:

- `outputs/prompt-vlm-selector.md` — Given accuracy, latency, context length, and budget, picks Qwen3-VL / InternVL3.5 / LLaVA-Next / API.
- `outputs/skill-cmer-monitor.md` — Produces code to instrument a production VLM endpoint with cross-modal error rate monitoring, per-endpoint dashboards, and alert thresholds.

## Exercises

1. **(Easy)** Run any open-source VLM on five images with three prompts ("What is this?", "Count objects", "Describe the scene"). Manually score each answer as correct / partially correct / hallucination. Compute a rough CMER-like rate.
2. **(Medium)** Fine-tune Qwen2.5-VL-3B or LLaVA-1.6-7B with LoRA (rank 16) on 500 captioned images from a target domain. Compare zero-shot vs fine-tuned MMBench-style accuracy.
3. **(Hard)** Swap the VLM's image encoder from its default SigLIP/CLIP to DINOv3. Retrain only the projector (freeze LLM + freeze DINOv3). Measure whether dense prediction tasks (counting, spatial reasoning) improve.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|----------------------|
| ViT-MLP-LLM | "The VLM pattern" | Vision encoder + projector + language model; every VLM in 2026 |
| Projector | "The bridge" | 2-4 layer MLP (or Q-former) mapping vision tokens into LLM embedding space |
| DeepStack | "Qwen3-VL feature trick" | Stacking multi-level ViT features instead of using only the final layer |
| Image token | "<image> placeholder" | Special token in the text stream replaced by projected vision embeddings |
| CMER | "Hallucination KPI" | Cross-Modal Error Rate; high when text confidence is high but image-text similarity is low |
| Vision agent | "VLM that clicks" | VLM operating GUIs with tool calls (OSWorld, mobile, web) |
| Q-former | "Fixed token count bridge" | BLIP-2-style projector producing a fixed number of vision query tokens |
| Alignment / Pre-training / Instruction tuning | "The three stages" | Standard VLM training pipeline |

## Further Reading

- [Qwen3-VL Technical Report (arXiv 2511.21631)](https://arxiv.org/abs/2511.21631)
- [InternVL3.5 Advancing Open-Source Multimodal Models (arXiv 2508.18265)](https://arxiv.org/html/2508.18265v1)
- [LLaVA-Next series](https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/)
- [BentoML: Best Open-Source VLMs 2026](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)
- [MMMU: Multi-discipline Multimodal Understanding benchmark](https://mmmu-benchmark.github.io/)
- [VLMs in manufacturing (Robotics Tomorrow, March 2026)](https://www.roboticstomorrow.com/story/2026/03/when-machines-learn-to-see-like-experts-the-rise-of-vision-language-models-in-manufacturing/26335/)
