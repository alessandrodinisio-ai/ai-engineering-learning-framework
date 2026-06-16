# Voice Agents: Pipecat and LiveKit

> Voice agents are a first-class production category in 2026. Pipecat gives you a Python, frame-based pipeline (VAD → STT → LLM → TTS → transport). LiveKit Agents bridges AI models to users via WebRTC. Production latency targets for high-end stacks land at 450–600 ms end-to-end.

**Type:** Learn
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~60 min

## Learning Objectives

- Describe Pipecat's frame-based pipeline: DOWNSTREAM (source → sink) and UPSTREAM (control).
- Name the stages of a standard voice pipeline and which transports Pipecat supports.
- Explain LiveKit Agents' two voice agent classes (MultimodalAgent, VoicePipelineAgent) and when each is appropriate.
- Summarize 2026 production latency expectations and how they drive architecture choices.

## The Problem

A voice agent is not a text loop with TTS bolted on. The latency budget is brutal (~600 ms), partial audio is the default, turn detection is a model, and transports range from telephony SIP to WebRTC. Either you build a frame-based pipeline (Pipecat) or you lean on a platform (LiveKit).

## The Concept

### Pipecat (pipecat-ai/pipecat)

- Frame-based pipeline framework for Python.
- `Frame` → `FrameProcessor` chain.
- Two flow directions:
  - **DOWNSTREAM** — source → sink (audio in, TTS out).
  - **UPSTREAM** — feedback and control (cancel, metrics, barge-in).
- `PipelineTask` manages lifecycle with events (`on_pipeline_started`, `on_pipeline_finished`, `on_idle_timeout`) and observers for metrics/tracing/RTVI.

Typical pipeline:

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

Transports: Daily, LiveKit, SmallWebRTCTransport, FastAPI WebSocket, WhatsApp.

Pipecat Flows adds structured conversations (state machines). Pipecat Cloud is the managed runtime.

### LiveKit Agents (livekit/agents)

- Bridges AI models to users via WebRTC.
- Key concepts: `Agent`, `AgentSession`, `entrypoint`, `AgentServer`.
- Two voice agent classes:
  - **MultimodalAgent** — direct audio via OpenAI Realtime or equivalent.
  - **VoicePipelineAgent** — STT → LLM → TTS cascade; gives you text-level control.
- Semantic turn detection via a transformer model.
- Native MCP integration.
- Telephony via SIP.
- 50+ models without API keys via LiveKit Inference; 200+ more via plugins.

### Commercial platforms

Vapi (~450–600 ms on an optimized high-end stack) and Retell (~600 ms end-to-end across 180 test calls) build on top of these. When you want a managed voice stack without a WebRTC team, pick a platform.

### Where this pattern breaks

- **Not handling barge-in.** The user interrupts; the agent keeps talking. Needs UPSTREAM cancel frames in Pipecat; LiveKit has equivalents.
- **Ignoring STT confidence.** Feeding low-confidence transcripts to the LLM as gospel. Gate on confidence or request confirmation.
- **TTS mid-sentence truncation.** When the pipeline cancels mid-utterance, TTS needs to know or audio must be clipped.
- **Ignoring the latency budget.** Each component adds 50–200 ms. Sum your chain before launch.

### Typical 2026 latency

- VAD: 20–60 ms
- STT partial results: 100–250 ms
- LLM first token: 150–400 ms
- TTS first audio chunk: 100–200 ms
- Transport RTT: 30–80 ms

End-to-end 450–600 ms is high-end. 800–1200 ms is common. Anything > 1500 ms feels broken.

## Build It

`code/main.py` is a toy frame-based pipeline with:

- `Frame` types (audio, transcript, text, tts_audio, control).
- A `Processor` interface with `process(frame)`.
- A five-stage pipeline (VAD → STT → LLM → TTS → transport) with scripted processors.
- An UPSTREAM cancel frame to demonstrate barge-in.

Run it:

```
python3 code/main.py
```

The trace shows normal flow and a barge-in cancel that stops TTS mid-utterance.

## Use It

- **Pipecat** for full control — custom processors, Python-first, pluggable providers.
- **LiveKit Agents** for WebRTC-first deployments and telephony.
- **Vapi / Retell** for managed voice agents without a WebRTC team.
- **OpenAI Realtime / Gemini Live** for direct audio-in/audio-out (MultimodalAgent).

## Ship It

`outputs/skill-voice-pipeline.md` scaffolds a Pipecat-shaped voice pipeline with VAD + STT + LLM + TTS + transport plus barge-in handling.

## Exercises

1. Add a metrics observer to your toy pipeline: count frames per second per stage. Where does latency accumulate?
2. Implement confidence-gated STT: below threshold, request "Could you repeat that?"
3. Add semantic turn detection: simple rule — if transcript ends with "?", that's end-of-turn.
4. Read Pipecat's transport documentation. Swap the stdlib transport for a SmallWebRTCTransport config (stub).
5. Benchmark OpenAI Realtime vs STT+LLM+TTS cascade on the same query. How much latency does text-level control cost?

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Frame | "Event" | A typed data unit in the pipeline (audio, transcript, text, control) |
| Processor | "Pipeline stage" | A handler with process(frame) |
| DOWNSTREAM | "Forward flow" | Source to sink: audio in, speech out |
| UPSTREAM | "Feedback flow" | Control: cancel, metrics, barge-in |
| VAD | "Voice activity detection" | Detects when the user is speaking |
| Semantic turn detection | "Smart end-of-turn" | Model-based judgment that the user is done speaking |
| MultimodalAgent | "Direct audio agent" | Audio in, audio out; no text in between |
| VoicePipelineAgent | "Cascade agent" | STT + LLM + TTS; text-level control |

## Further Reading

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — Frame-based pipeline, processors, transports
- [LiveKit Agents docs](https://docs.livekit.io/agents/) — WebRTC + voice primitives
- [Vapi](https://vapi.ai/) — Managed voice platform
- [Retell AI](https://www.retellai.com/) — Managed voice, latency-benchmarked
