# Building a Complete Vision Pipeline — Capstone Project

> A production vision system is a chain of models and rules stitched together by data contracts. The pieces are ready from this phase; the capstone connects them end-to-end.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lessons 01–15
**Time:** ~120 min

## Learning Objectives

- Design a production vision pipeline that detects objects, classifies them, and produces structured JSON — handling every failure path
- Plug a detector (Mask R-CNN or YOLO), a classifier (ConvNeXt-Tiny), and a data contract (Pydantic) into a single service
- Benchmark the end-to-end pipeline and find the first bottleneck (usually preprocessing, then the detector)
- Deliver a minimal FastAPI service that accepts image uploads, runs the pipeline, and returns classified detections

## The Problem

Individual vision models are useful; vision products are chains of them. A retail shelf audit is a detector + product classifier + price OCR pipeline. Autonomous driving is 2D detector + 3D detector + segmenter + tracker + planner. Medical pre-screening is segmenter + region classifier + clinician UI.

Connecting these chains is exactly the part that separates "ML prototype" from "product." Every interface between models is a new hiding spot for bugs. Every coordinate transform, every normalization, every mask resize is a candidate for silent failure. A pipeline is only as strong as its weakest interface.

This capstone builds the minimum viable pipeline: detection + classification + structured output + a service layer. Everything else in Phase 4 plugs into this skeleton: swap Mask R-CNN for YOLOv8, add an OCR head, add a segmentation branch, add a tracker. The architecture is stable; the components are swappable.

## The Concept

### Pipeline

```mermaid
flowchart LR
    REQ["HTTP request<br/>+ image bytes"] --> LOAD["Decode<br/>+ preprocess"]
    LOAD --> DET["Detector<br/>(YOLO / Mask R-CNN)"]
    DET --> CROP["Crop + resize<br/>each detection"]
    CROP --> CLS["Classifier<br/>(ConvNeXt-Tiny)"]
    CLS --> AGG["Aggregate<br/>detection + class"]
    AGG --> SCHEMA["Pydantic<br/>validation"]
    SCHEMA --> RESP["JSON response"]

    REQ -.->|error| RESP

    style DET fill:#fef3c7,stroke:#d97706
    style CLS fill:#dbeafe,stroke:#2563eb
    style SCHEMA fill:#dcfce7,stroke:#16a34a
```

Seven stages. Two model stages are expensive; the other five are where bugs hide.

### Data Contracts with Pydantic

Every model boundary becomes a typed object. This turns silent failures into loud failures.

```
Detection(
    box: tuple[float, float, float, float],   # (x1, y1, x2, y2), absolute pixels
    score: float,                              # [0, 1]
    class_id: int,                             # from detector's label map
    mask: Optional[list[list[int]]],           # RLE-encoded if present
)

PipelineResult(
    image_id: str,
    detections: list[Detection],
    classifications: list[Classification],
    inference_ms: float,
)
```

When a detector returns boxes as `(cx, cy, w, h)` instead of `(x1, y1, x2, y2)`, Pydantic validation fails at the boundary and you catch it immediately — rather than debugging a downstream crop that silently returns empty regions.

### Where Latency Goes

Three facts true in nearly every vision pipeline:

1. **Preprocessing is often the largest single block.** Decoding JPEG, color-space conversion, resize — these are CPU-intensive and easy to forget.
2. **The detector dominates GPU time.** 70–90% of GPU time is in the detection forward pass.
3. **Post-processing (NMS, RLE encode/decode) is cheap on GPU, expensive on CPU.** Always profile with real targets.

Knowing this distribution turns optimization into a prioritized checklist.

### Failure Modes

- **Empty detections** — return an empty list, don't crash. Log it.
- **Out-of-bounds boxes** — clamp to image dimensions before cropping.
- **Tiny crops** — skip classification for boxes smaller than the classifier's minimum input.
- **Corrupt uploads** — return 400 with a specific error code, not 500.
- **Model load failure** — fail at service startup, not on the first request.

A production pipeline handles every one of these without writing a generic `try/except` that hides failures. Each failure gets a named code and a response.

### Batching

A production service serves multiple clients. Batching detection and classification across requests can multiply throughput. The cost: added latency from waiting for a batch to fill. Typical approach: collect requests for up to 20 ms, form a batch, process, dispatch responses. `torchserve` and `triton` do this natively; small services with predictable load write their own micro-batcher.

## Build It

### Step 1: Data Contracts

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class Detection(BaseModel):
    box: Tuple[float, float, float, float]
    score: float = Field(ge=0, le=1)
    class_id: int = Field(ge=0)
    mask_rle: Optional[str] = None


class Classification(BaseModel):
    detection_index: int
    class_id: int
    class_name: str
    score: float = Field(ge=0, le=1)


class PipelineResult(BaseModel):
    image_id: str
    detections: List[Detection]
    classifications: List[Classification]
    inference_ms: float
```

Five seconds of code that saves an hour of debugging on any serious pipeline.

### Step 2: A Minimal Pipeline Class

```python
import time
import numpy as np
import torch
from PIL import Image

class VisionPipeline:
    def __init__(self, detector, classifier, class_names,
                 device="cpu", min_crop=32):
        self.detector = detector.to(device).eval()
        self.classifier = classifier.to(device).eval()
        self.class_names = class_names
        self.device = device
        self.min_crop = min_crop

    def preprocess(self, image):
        """
        image: PIL.Image or np.ndarray (H, W, 3) uint8
        Returns: CHW float tensor on device
        """
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor.to(self.device)

    @torch.no_grad()
    def detect(self, image_tensor):
        return self.detector([image_tensor])[0]

    @torch.no_grad()
    def classify(self, crops):
        if len(crops) == 0:
            return []
        batch = torch.stack(crops).to(self.device)
        logits = self.classifier(batch)
        probs = logits.softmax(-1)
        scores, cls = probs.max(-1)
        return list(zip(cls.tolist(), scores.tolist()))

    def run(self, image, image_id="anonymous"):
        t0 = time.perf_counter()
        tensor = self.preprocess(image)
        det = self.detect(tensor)

        crops = []
        detections = []
        valid_indices = []
        for i, (box, score, cls) in enumerate(zip(det["boxes"], det["scores"], det["labels"])):
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            detections.append(Detection(
                box=(x1, y1, x2, y2),
                score=float(score),
                class_id=int(cls),
            ))
            if (x2 - x1) < self.min_crop or (y2 - y1) < self.min_crop:
                continue
            crop = tensor[:, y1:y2, x1:x2]
            crop = torch.nn.functional.interpolate(
                crop.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )[0]
            crops.append(crop)
            valid_indices.append(i)

        class_preds = self.classify(crops)

        classifications = []
        for valid_idx, (cls_id, cls_score) in zip(valid_indices, class_preds):
            classifications.append(Classification(
                detection_index=valid_idx,
                class_id=int(cls_id),
                class_name=self.class_names[cls_id],
                score=float(cls_score),
            ))

        return PipelineResult(
            image_id=image_id,
            detections=detections,
            classifications=classifications,
            inference_ms=(time.perf_counter() - t0) * 1000,
        )
```

Every interface is typed. Every failure path has a specific handling decision.

### Step 3: Plug in a Detector and a Classifier

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import convnext_tiny

# ImageNet pretrained weights — get a realistic pipeline without training
detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
classifier = convnext_tiny(weights="DEFAULT")
class_names = [f"imagenet_class_{i}" for i in range(1000)]

pipe = VisionPipeline(detector, classifier, class_names)

# Smoke test with a synthetic image
test_image = (np.random.rand(400, 600, 3) * 255).astype(np.uint8)
result = pipe.run(test_image, image_id="demo")
print(result.model_dump_json(indent=2)[:500])
```

### Step 4: FastAPI Service

```python
from fastapi import FastAPI, UploadFile, HTTPException
from io import BytesIO

app = FastAPI()
pipe = None  # initialized at startup

@app.on_event("startup")
def load():
    global pipe
    detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT").eval()
    classifier = convnext_tiny(weights="DEFAULT").eval()
    pipe = VisionPipeline(detector, classifier, class_names=[f"c{i}" for i in range(1000)])

@app.post("/detect")
async def detect_endpoint(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="unsupported image type")
    data = await file.read()
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="cannot decode image")
    result = pipe.run(img, image_id=file.filename or "upload")
    return result.model_dump()
```

Run with `uvicorn main:app --host 0.0.0.0 --port 8000`. Test with `curl -F 'file=@dog.jpg' http://localhost:8000/detect`.

### Step 5: Benchmarking the Pipeline

```python
import time

def benchmark(pipe, num_runs=20, image_size=(400, 600)):
    img = (np.random.rand(*image_size, 3) * 255).astype(np.uint8)
    pipe.run(img)  # warmup

    stages = {"preprocess": [], "detect": [], "classify": [], "total": []}
    for _ in range(num_runs):
        t0 = time.perf_counter()
        tensor = pipe.preprocess(img)
        t1 = time.perf_counter()
        det = pipe.detect(tensor)
        t2 = time.perf_counter()
        crops = []
        for box in det["boxes"]:
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            if (x2 - x1) >= pipe.min_crop and (y2 - y1) >= pipe.min_crop:
                crop = tensor[:, y1:y2, x1:x2]
                crop = torch.nn.functional.interpolate(
                    crop.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
                )[0]
                crops.append(crop)
        pipe.classify(crops)
        t3 = time.perf_counter()
        stages["preprocess"].append((t1 - t0) * 1000)
        stages["detect"].append((t2 - t1) * 1000)
        stages["classify"].append((t3 - t2) * 1000)
        stages["total"].append((t3 - t0) * 1000)

    for stage, times in stages.items():
        times.sort()
        print(f"{stage:12s}  p50={times[len(times)//2]:7.1f} ms  p95={times[int(len(times)*0.95)]:7.1f} ms")
```

Typical output on CPU: preprocessing ~3 ms, detection 300–500 ms, classification 20–40 ms, total 350–550 ms. On GPU, detection is 20–40 ms and preprocessing + classification start to matter more relatively.

## Use It

Production templates converge on the same structure, plus:

- **Model versioning** — always log model name and weight hash in the response.
- **Per-request trace ID** — log time per stage per request so slow responses can be correlated with stages.
- **Degradation path** — if the classifier times out, return detections without classifications rather than failing the entire request.
- **Safety filters** — NSFW / PII filters run after classification, before the response leaves the service.
- **Batch endpoint** — a `/detect_batch` that accepts a list of image URLs for bulk processing.

Production services use `torchserve`, `Triton Inference Server`, and `BentoML`, which handle batching, versioning, metrics, and health checks out of the box. Running raw `FastAPI` is fine for prototypes and small-scale products.

## Ship It

This lesson produces:

- `outputs/prompt-vision-service-shape-reviewer.md` — a prompt that reviews vision service code for contract/response shape violations, naming the first bug that would crash.
- `outputs/skill-pipeline-budget-planner.md` — a skill that allocates time budgets to each pipeline stage given a target latency and throughput, flagging which stage will exceed its budget first.

## Exercises

1. **(Easy)** Run this pipeline on 10 images from any open dataset. Report average time per stage and the distribution of detection count per image.
2. **(Medium)** Add a mask output field to `Detection`, encoded as RLE. Verify that even for 10-object images the JSON stays under 1 MB.
3. **(Hard)** Add a micro-batcher before the classifier: collect crops for up to 10 ms, classify all in one GPU call, dispatch results per request. Measure throughput improvement and added latency under 5 concurrent requests per second.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| Pipeline | "system" | Ordered chain of preprocessing, inference, and post-processing steps with typed interfaces between each pair |
| Data contract | "schema" | Pydantic / dataclass definition that every stage's input and output conforms to; catches integration bugs at boundaries |
| Preprocessing | "before model" | Decode, color conversion, resize, normalization; usually the largest CPU time consumer |
| Post-processing | "after model" | NMS, mask resize, thresholding, RLE encoding; cheap on GPU, expensive on CPU |
| Micro-batcher | "collect then forward" | Aggregator that waits a fixed window to gather multiple requests, runs one batched forward |
| Trace ID | "request id" | Per-request identifier logged at every stage for end-to-end tracing of slow requests |
| Failure code | "named error" | Specific error code per failure class rather than generic 500; enables client retry logic |
| Health check | "readiness probe" | Cheap endpoint reporting whether the service can serve; load balancers depend on it |

## Further Reading

- [Full Stack Deep Learning — Deploying Models](https://fullstackdeeplearning.com/course/2022/lecture-5-deployment/) — classic overview of production ML deployment
- [BentoML docs](https://docs.bentoml.com) — serving framework with batching, versioning, and metrics
- [torchserve docs](https://pytorch.org/serve/) — PyTorch's official serving library
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server) — high-throughput serving with batching and multi-model support
