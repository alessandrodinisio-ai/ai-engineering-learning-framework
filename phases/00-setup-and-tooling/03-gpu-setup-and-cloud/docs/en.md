# GPU Setup and Cloud

> Training on CPU works for learning. But serious training needs a GPU.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 min

## Learning Objectives

- Verify local GPU availability with `nvidia-smi` and PyTorch's CUDA API
- Configure Google Colab with a T4 GPU for free cloud experimentation
- Benchmark matrix multiplication on CPU vs GPU to measure the speedup
- Estimate model size your VRAM can hold using the fp16 rule of thumb

## The Problem

Most lessons in Phases 1-3 run fine on CPU. But once you start training CNNs, transformers, or LLMs (Phase 4 onward), you need GPU acceleration. The same training job that takes 8 hours on CPU finishes in 10 minutes on GPU.

You have three options: local GPU, cloud GPU, or Google Colab (free).

## The Concept

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already own it)
   Setup: Install CUDA + cuDNN
   Best for: daily use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: none
   Best for: quick experiments, no local GPU

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: serious training, large models
```

## Build It

### Option 1: Local NVIDIA GPU

Check if you have one:

```bash
nvidia-smi
```

Install PyTorch with CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Option 2: Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. Run `!nvidia-smi` to verify

Upload this course's notebooks directly to Colab.

### Option 3: Cloud GPU

Use Lambda Labs, RunPod, or Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### No GPU? No Problem.

Most lessons run on CPU. Lessons requiring a GPU are clearly marked and include Colab links.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Build It: GPU vs CPU Benchmark

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## Exercises

1. Run the benchmark above and compare CPU vs GPU times
2. If you don't have a GPU, run it on Google Colab and compare
3. Check how much VRAM you have and estimate the largest model it can hold (rule of thumb: fp16 uses 2 bytes per parameter)

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|----------------------|
| CUDA | "GPU programming" | NVIDIA's parallel computing platform that lets you run code on GPUs |
| VRAM | "GPU memory" | Memory on the GPU, separate from system RAM, determines max model size |
| fp16 | "half precision" | 16-bit floating point, halves memory usage with minimal precision loss |
| Tensor Core | "matrix acceleration hardware" | Specialized GPU cores for matrix multiplication, 4-8x faster than general cores |
