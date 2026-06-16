# Real-Time Audio Processing

> A batch pipeline processes a file. A real-time pipeline must process the last 20 ms before the next 20 ms arrives. Every conversational AI, broadcast studio, and phone bot lives or dies by this latency budget.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## The Problem

You want a voice assistant that feels alive. Human conversational turn-taking latency is ~230 ms (silence to response). Beyond 500 ms it feels robotic; beyond 1500 ms it feels broken. The 2026 budget for a full **listen → understand → respond → speak** loop is:

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi (Kyutai, 2024) achieves 200 ms full-duplex. GPT-4o-realtime (2024) is ~320 ms. Cascaded pipelines from 2022 shipped at 2500 ms. The 10× improvement comes from three techniques: (1) streaming everywhere, (2) async pipelines with partial results, (3) interruptible generation.

## The Concept

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frames / chunks / windows.** Real-time audio flows in fixed-length chunks. Common choice: 20 ms (320 samples at 16 kHz). Everything downstream must keep up with this cadence.

**Ring buffer.** A fixed-length circular buffer. Producer thread writes new frames, consumer thread reads. Avoids allocations on the hot path. Size ≈ max latency × sample rate; a 2-second ring at 16 kHz = 32,000 samples.

**VAD (Voice Activity Detection).** Gates downstream work when nobody is talking. Silero VAD 4.0 (2024) runs <1 ms per 30 ms frame on CPU. `webrtcvad` is the older fallback.

**Streaming ASR.** Models that emit partial transcripts as audio arrives. Parakeet-CTC-0.6B streaming mode (NeMo, 2024) achieves 2–5% WER at 320 ms latency. Whisper-Streaming (Macháček et al., 2023) chunks Whisper to approximate streaming at ~2 s latency.

**Interruption (Barge-in).** The user speaks while the assistant is talking — you must (a) detect the interruption, (b) stop TTS, (c) discard remaining LLM output. All within 100 ms, or the user feels the assistant is deaf.

**WebRTC Opus transport.** 20 ms frames, 48 kHz, adaptive bitrate 8–128 kbps. The standard for browser and mobile. LiveKit, Daily.co, and Pion are the 2026 toolstack for building voice apps.

**Jitter buffer.** Network packets arrive out of order / late. A jitter buffer reorders and smooths; too small → audible gaps, too large → latency. Typical 60–80 ms.

### Common Pitfalls

- **Thread contention.** Python's GIL + heavy models starve the audio thread. Use audio libraries with C callbacks (sounddevice, PortAudio) and keep Python off the hot path.
- **Sample-rate conversion latency.** In-pipeline resampling adds 5–20 ms. Either resample ahead of time, or use zero-latency resamplers (PolyPhase, `soxr_hq`).
- **TTS warmup.** Even fast TTS like Kokoro has 100–200 ms warmup on first request. Cache the model and warm it with a dummy run before the first real turn.
- **Echo cancellation.** Without AEC, TTS output re-enters the mic, triggering ASR to recognize the bot's own speech. WebRTC AEC3 is the open-source default.

## Build It

### Step 1: Ring Buffer

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

Capacity determines max buffering latency. 32,000 samples at 16 kHz = 2 s.

### Step 2: VAD Gate

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

In production, swap for Silero VAD:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Step 3: Streaming ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Step 4: Interruption Handler

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

The key is async I/O and a cancellable TTS stream. Calling WebRTC's peerconnection.stop() on the audio track is the classic approach.

## Use It

2026 toolstack:

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) or Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B or Whisper-Streaming |
| LLM first token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro or ElevenLabs Turbo v2.5 |
| Echo cancellation | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API or Moshi |

## Pitfalls

- **Buffering 500 ms "just in case."** Buffering *is* your latency floor. Shrink it.
- **Not pinning threads.** Audio callback runs on a thread with priority below UI = stutter under load.
- **TTS chunks too small.** Chunks under 200 ms make vocoder artifacts audible. 320 ms chunks are the sweet spot.
- **No jitter buffer.** Real networks jitter; without smoothing you get pops.
- **Fire-and-forget error handling.** Audio pipelines must be crash-resistant. One exception kills the entire session.

## Ship It

Save as `outputs/skill-realtime-designer.md`. Design a real-time audio pipeline with concrete latency budgets for each stage.

## Exercises

1. **Easy.** Run `code/main.py`. It simulates a ring buffer + energy VAD; prints per-stage latencies for a fake 10-second stream.
2. **Medium.** Build a passthrough loop with `sounddevice`, processing your mic at 20 ms frames and printing VAD state per frame.
3. **Hard.** Build a full-duplex echo test with `aiortc`: browser → WebRTC → Python → WebRTC → browser. Measure glass-to-glass latency with a 1 kHz pulse.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Ring buffer | The circular queue | Fixed-length, lock-free (or SPSC-locked) FIFO for audio frames. |
| VAD | Silence gate | Model or heuristic marking speech vs. non-speech. |
| Streaming ASR | Real-time STT | Emits partial text as audio arrives; bounded look-ahead. |
| Jitter buffer | Network smoother | Queue that reorders out-of-order packets; typical 60–80 ms. |
| AEC | Echo cancellation | Subtracts the speaker-to-mic feedback path. |
| Barge-in | User interrupts | System detects user speech mid-TTS; must cancel playback. |
| Full-duplex | Both directions simultaneously | User and bot can speak at the same time; Moshi is full-duplex. |

## Further Reading

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — chunked approximation of streaming Whisper.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — full-duplex 200 ms latency.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — production audio agent orchestration.
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — sub-millisecond VAD, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — open-source echo cancellation.
