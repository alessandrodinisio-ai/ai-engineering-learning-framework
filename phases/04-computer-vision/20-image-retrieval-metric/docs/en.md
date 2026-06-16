# Image Retrieval and Metric Learning

> Retrieval systems rank candidates by distance in an embedding space. Metric learning is the craft of shaping that space so distances mean what you want them to.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 14 (ViT), Phase 4 Lesson 18 (CLIP)
**Time:** ~45 min

## Learning Objectives

- Explain triplet, contrastive, and proxy-based metric learning losses; pick the right one for a given dataset
- Correctly implement L2 normalization and cosine similarity; examine the distinction between "same item" and "same category" retrieval
- Build a FAISS index, query it with text and images, and report recall@K for a held-out query set
- Use DINOv2, CLIP, and SigLIP as off-the-shelf embedding backbones; know when each wins

## The Problem

Retrieval is everywhere in production vision: deduplication, reverse image search, visual search ("find similar products"), face re-identification, pedestrian re-ID in surveillance, e-commerce instance-level matching. The product question is always: "Given this query image, rank my catalog."

Two design decisions shape the entire system. The embedding — what model produces the vector. The index — how to find nearest neighbors at scale. Both are commoditized in 2026 (DINOv2 for embeddings, FAISS for indexing), which raises the bar: the hard part is defining *what counts as similar* for your application, then shaping the embedding space so distance matches.

That shaping is metric learning. It's a small but high-leverage craft.

## The Concept

### Retrieval at a Glance

```mermaid
flowchart LR
    Q["Query Image<br/>or Text"] --> ENC["Encoder"]
    ENC --> EMB["Query Embedding"]
    EMB --> IDX["FAISS Index"]
    CAT["Catalog Images"] --> ENC2["Encoder (same)"] --> IDX_BUILD["Build Index"]
    IDX_BUILD --> IDX
    IDX --> RANK["Top-k Nearest Neighbors<br/>by Cosine / L2"]
    RANK --> OUT["Ranked Results"]

    style ENC fill:#dbeafe,stroke:#2563eb
    style IDX fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

### Four Loss Families

| Loss | Requires | Pros | Cons |
|------|----------|------|------|
| **Contrastive** | (anchor, positive) + negatives | Simple, works with any pairwise labels | Slow convergence without many negatives |
| **Triplet** | (anchor, positive, negative) | Intuitive; direct control over margin | Hard triplet mining is expensive |
| **NT-Xent / InfoNCE** | pairs + in-batch mined negatives | Scales to large batches | Needs large batches or momentum queues |
| **Proxy-based (ProxyNCA)** | class labels only | Fast, stable, no mining needed | May overfit to proxies on small datasets |

For most production use cases, start from a pretrained backbone and add a metric learning fine-tune only when off-the-shelf embeddings underperform on your test set.

### Triplet Loss Formulation

```
L = max(0, ||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + margin)
```

Pull anchor `a` closer to positive `p`, push away from negative `n`, with a `margin` ensuring separation. This three-image structure generalizes to any similarity ranking.

Mining matters: easy triplets (`n` already far from `a`) contribute zero loss; only hard triplets teach the network. Semi-hard mining (`n` farther than `p` but within the margin) is the 2016 FaceNet recipe and still dominates.

### Cosine Similarity vs L2

Two metrics, two conventions:

- **Cosine**: angle between vectors. Requires L2-normalized embeddings.
- **L2**: Euclidean distance. Works on raw or normalized embeddings, but usually paired with L2 normalization + squared L2.

For most modern networks, they're equivalent: when `||a|| = ||b|| = 1`, `||a - b||^2 = 2 - 2 cos(a, b)`. Pick the convention that matches how your embedding was trained; mixing silently changes the meaning of "nearest."

### Recall@K

The standard retrieval metric:

```
recall@K = fraction of queries with at least one correct match in the top K results
```

Report recall@1, @5, @10 side by side. Recall@10 above 0.95 but recall@1 below 0.5 means the embedding space structure is right but ranking is noisy — try longer fine-tuning or add a re-ranking step.

For deduplication, precision@K matters more because every false positive is a user-visible error. For visual search, recall@K is the product signal.

### FAISS in One Paragraph

Facebook AI Similarity Search. The de facto library for nearest-neighbor search. Three index choices:

- `IndexFlatIP` / `IndexFlatL2` — brute-force, exact, no training needed. Use up to ~1M vectors.
- `IndexIVFFlat` — partitioned into K cells, searches only the nearest cells. Approximate, fast, needs training data.
- `IndexHNSW` — graph-based, fastest for many queries, larger index size.

For 100K vectors, you probably want `IndexFlatIP` on cosine similarity. For 10M, you want `IndexIVFFlat`. For 100M+, combine with product quantization (`IndexIVFPQ`).

### Instance-Level vs Category-Level Retrieval

Two very different problems with the same name:

- **Category-level** — "find cats in my catalog." Class-conditional similarity; off-the-shelf CLIP / DINOv2 embeddings work well.
- **Instance-level** — "find *this exact product* in my catalog." Requires fine-grained discrimination between visually similar objects of the same category; off-the-shelf embeddings underperform; metric learning fine-tuning matters.

Always ask which one you're solving before picking a model.

## Build It

### Step 1: Triplet Loss

```python
import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=0.2):
    d_ap = F.pairwise_distance(anchor, positive, p=2)
    d_an = F.pairwise_distance(anchor, negative, p=2)
    return F.relu(d_ap - d_an + margin).mean()
```

One line. Works on L2-normalized or raw embeddings.

### Step 2: Semi-Hard Mining

Given a batch of embeddings and labels, find the hardest semi-hard negative for each anchor.

```python
def semi_hard_negatives(emb, labels, margin=0.2):
    dist = torch.cdist(emb, emb)
    same_class = labels[:, None] == labels[None, :]
    diff_class = ~same_class
    N = emb.size(0)

    positives = dist.clone()
    positives[~same_class] = float("-inf")
    positives.fill_diagonal_(float("-inf"))
    pos_idx = positives.argmax(dim=1)

    semi_hard = dist.clone()
    semi_hard[same_class] = float("inf")
    d_ap = dist[torch.arange(N), pos_idx].unsqueeze(1)
    semi_hard[dist <= d_ap] = float("inf")
    neg_idx = semi_hard.argmin(dim=1)

    fallback_mask = semi_hard[torch.arange(N), neg_idx] == float("inf")
    if fallback_mask.any():
        hardest = dist.clone()
        hardest[same_class] = float("inf")
        neg_idx = torch.where(fallback_mask, hardest.argmin(dim=1), neg_idx)
    return pos_idx, neg_idx
```

Each anchor gets the hardest in-class positive and a semi-hard negative — farther than the positive but within the margin.

### Step 3: Recall@K

```python
def recall_at_k(query_emb, gallery_emb, query_labels, gallery_labels, k=1):
    sim = query_emb @ gallery_emb.T
    _, top_k = sim.topk(k, dim=-1)
    matches = (gallery_labels[top_k] == query_labels[:, None]).any(dim=-1)
    return matches.float().mean().item()
```

Top-k by inner product on L2-normalized embeddings equals top-k by cosine. Reports the average fraction of queries with at least one correct neighbor.

### Step 4: Putting It Together

```python
import torch
import torch.nn as nn
from torch.optim import Adam

class Encoder(nn.Module):
    def __init__(self, in_dim=128, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

torch.manual_seed(0)
num_classes = 6
protos = F.normalize(torch.randn(num_classes, 128), dim=-1)

def sample_batch(bs=32):
    labels = torch.randint(0, num_classes, (bs,))
    x = protos[labels] + 0.15 * torch.randn(bs, 128)
    return x, labels

enc = Encoder()
opt = Adam(enc.parameters(), lr=3e-3)

for step in range(200):
    x, y = sample_batch(32)
    emb = enc(x)
    pos_idx, neg_idx = semi_hard_negatives(emb, y)
    loss = triplet_loss(emb, emb[pos_idx], emb[neg_idx])
    opt.zero_grad(); loss.backward(); opt.step()
```

After a few hundred steps, embeddings cluster into one group per class.

## Use It

The 2026 production stack:

- **DINOv2 + FAISS** — general-purpose visual retrieval. Works off the shelf.
- **CLIP + FAISS** — when queries are text.
- **Fine-tuned DINOv2 + FAISS** — instance-level retrieval, face re-ID, fashion, e-commerce.
- **Milvus / Weaviate / Qdrant** — managed vector databases wrapping FAISS or HNSW.

For SOTA instance retrieval, the recipe is: DINOv2 backbone, add an embedding head, fine-tune with triplet or InfoNCE loss on instance-labeled pairs, build a FAISS index.

## Ship It

This lesson produces:

- `outputs/prompt-retrieval-loss-picker.md` — a prompt that picks triplet / InfoNCE / ProxyNCA for a given retrieval problem.
- `outputs/skill-recall-at-k-runner.md` — a skill that writes a clean evaluation harness for recall@K with train/val/gallery splits and correct data contracts.

## Exercises

1. **(Easy)** Run the toy example above. Plot embeddings with PCA before and after training to see the six clusters form.
2. **(Medium)** Add a ProxyNCA loss implementation: one learned "proxy" per class, standard cross-entropy on cosine similarity. Compare convergence speed against triplet loss on the toy data.
3. **(Hard)** Take 1,000 ImageNet validation images, embed them with DINOv2 via HuggingFace, build a FAISS flat index, report recall@{1, 5, 10} using the same images as queries (should be 1.0), then report with a held-out split using ImageNet labels as ground truth.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|----------------------|
| Metric learning | "shaping the space" | Training an encoder so distances in its output space reflect a target similarity |
| Triplet loss | "pull and push" | L = max(0, d(a, p) - d(a, n) + margin); the classic metric learning loss |
| Semi-hard mining | "useful negatives" | Negatives farther from the anchor than the positive but within the margin; empirically most informative |
| Proxy-based loss | "class prototypes" | One learned proxy per class; cross-entropy on similarity to proxies; no pairwise mining needed |
| Recall@K | "top-K hit rate" | Fraction of queries with at least one correct result in the top K |
| Instance retrieval | "find this exact thing" | Fine-grained matching; off-the-shelf features typically underperform |
| FAISS | "the nearest-neighbor library" | Facebook's nearest-neighbor library; supports exact and approximate indexes |
| HNSW | "graph index" | Hierarchical Navigable Small World; fast approximate nearest neighbors with small memory overhead |

## Further Reading

- [FaceNet: A Unified Embedding for Face Recognition (Schroff et al., 2015)](https://arxiv.org/abs/1503.03832) — the triplet loss / semi-hard mining paper
- [In Defense of the Triplet Loss for Person Re-Identification (Hermans et al., 2017)](https://arxiv.org/abs/1703.07737) — practical guide to triplet fine-tuning
- [FAISS documentation](https://github.com/facebookresearch/faiss/wiki) — every index, every tradeoff
- [SMoT: Metric Learning Taxonomy (Kim et al., 2021)](https://arxiv.org/abs/2010.06927) — survey of modern losses and their connections

