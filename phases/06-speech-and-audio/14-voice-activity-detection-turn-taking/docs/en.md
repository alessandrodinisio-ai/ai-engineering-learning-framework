# Voice Activity Detection & Turn-Taking — Silero, Cobra, and the Flush Trick

> Every voice agent lives or dies by two decisions: is the user speaking right now, and are they done? VAD answers the first. Turn detection (VAD + silence hangover + semantic endpoint model) answers the second. Get either wrong and your assistant either interrupts the user or talks forever.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 11 (Real-Time Audio), Phase 6 · 12 (Voice Assistant)
**Time:** ~45 minutes

## The Problem

Three distinct judgments a voice agent makes on every 20 ms chunk:

1. **Is this frame speech?** — VAD. Binary, per-frame.
2. **Has the user started a new utterance?** — Onset detection.
3. **Has the user finished speaking?** — Endpoint detection (end-of-turn).

The naive answer (energy threshold) falls apart with any noise — traffic, keyboard clicks, crowd chatter. The 2026 answer: Silero VAD (open-source, deep learning) + a turn detection model (semantic endpoint) + a VAD-calibrated silence hangover.

## The Concept

![VAD cascade: energy → Silero → turn detector → flush trick](../assets/vad-turn-taking.svg)

### Three-Level VAD Cascade

**Level 1: Energy gate.** Cheapest. Threshold RMS at -40 dBFS. Filters obvious silence, but any noise above threshold triggers it.

**Level 2: Silero VAD** (2020-2026, MIT). 1M parameters. Trained on 6000+ languages. ~1 ms per 30 ms chunk on a single CPU thread. 87.7% TPR at 5% FPR. The open-source default.

**Level 3: Semantic turn detector.** LiveKit's turn detection model (2024-2026) or your own small classifier. Distinguishes "mid-sentence pause" from "done speaking." Uses linguistic context (intonation + recent words), not just silence.

### Key Parameters and Their Defaults

- **Threshold.** Silero outputs a probability; threshold at > 0.5 (default) or > 0.3 (sensitive). Lower threshold = fewer clipped first words, more false positives.
- **Min speech duration.** Reject speech shorter than 250 ms — usually a cough or chair noise.
- **Silence hangover (endpoint).** After VAD drops to 0, wait 500-800 ms before declaring end-of-turn. Too short → cuts off users. Too long → feels sluggish.
- **Pre-roll buffer.** Keep 300-500 ms of audio before VAD fires. Prevents "hey" from being clipped.

### The Flush Trick (Kyutai 2025)

Streaming STT models have look-ahead latency (Kyutai STT-1B is 500 ms, STT-2.6B is 2.5 s). Normally you wait that long after speech ends to get a transcript. The flush trick: when VAD fires end-of-speech, **send a flush signal to the STT** forcing it to emit immediately. The STT processes at ~4× real-time, so that 500 ms buffer completes in ~125 ms.

End-to-end: 125 ms VAD + flush STT = conversational latency.

### 2026 VAD Comparison

| VAD | TPR @ 5% FPR | Latency | License |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | Commercial |
| pyannote segmentation | 95% | ~10 ms | MIT-like |

Silero is the right default. Cobra is the compliance / accuracy upgrade. Energy-only VAD has no place in 2026 production.

## Build It

### Step 1: Energy Gate

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### Step 2: Silero VAD in Python

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### Step 3: End-of-Turn State Machine

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### Step 4: Flush Trick Skeleton

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

The STT (Kyutai, Deepgram, AssemblyAI) must support flush for this to work. Whisper streaming doesn't — it's chunk-based and always waits for a chunk boundary.

## Use It

| Scenario | VAD Pick |
|-----------|-----------|
| Open-source, fast, general | Silero VAD |
| Commercial call center | Cobra VAD |
| On-device (mobile) | Silero VAD ONNX |
| Research / diarization | pyannote segmentation |
| Zero-dependency fallback | WebRTC VAD (legacy) |
| Need end-of-turn quality | Silero + LiveKit turn detector stacked |

Rule of thumb: never ship energy-only VAD in production unless you truly have no alternative.

## Pitfalls

- **Fixed threshold.** Works in quiet; breaks in noise. Either calibrate on-device, or use Silero.
- **Silence hangover too short.** Agent interrupts mid-sentence. 500-800 ms is the conversational sweet spot.
- **Hangover too long.** Feels sluggish. A/B test with your target users.
- **No pre-roll buffer.** First 200-300 ms of user audio lost. Always keep a rolling pre-roll.
- **Ignoring semantic endpoint.** "Hmm, let me think..." contains long pauses. Users hate being interrupted mid-thought. Use LiveKit's turn detector or similar.

## Ship It

Save as `outputs/skill-vad-tuner.md`. Select the VAD model, threshold, hangover, pre-roll, and turn detection strategy for a given workload.

## Exercises

1. **Easy.** Run `code/main.py`. It simulates a speech + silence + speech + cough sequence, testing the three-level VAD.
2. **Medium.** Install `silero-vad`, process a 5-minute recording, tune the threshold to minimize both first-word clipping and false triggers. Report precision/recall.
3. **Hard.** Build a mini turn detector: Silero VAD + a 3-layer MLP acting on the last 10 word embeddings (via sentence-transformers). Train on a hand-labeled end-of-turn dataset. Beat Silero-only F1 by 10%.

## Key Terms

| Term | How people talk about it | What it actually means |
|------|-----------------|-----------------------|
| VAD | Speech detector | Binary, per-frame: is this speech? |
| Turn detection | Endpoint detection | VAD + silence hangover + semantic endpoint. |
| Silence hangover | Post-speech wait | Time before declaring end-of-turn; 500-800 ms. |
| Pre-roll | Pre-speech buffer | Keep 300-500 ms of audio before VAD fires. |
| Flush trick | Kyutai's trick | VAD → flush-STT → 125 ms instead of 500 ms latency. |
| Semantic endpoint | "Did they really stop?" | ML classifier that looks at words, not just silence. |
| TPR @ 5% FPR | ROC operating point | Standard VAD benchmark; Silero 87.7%, WebRTC 50%. |

## Further Reading

- [Silero VAD](https://github.com/snakers4/silero-vad) — The reference open-source VAD.
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — Commercial accuracy leader.
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — Sub-200 ms engineering trick.
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — Production semantic endpoint.
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — Legacy baseline.
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — Diarization-grade segmentation.
