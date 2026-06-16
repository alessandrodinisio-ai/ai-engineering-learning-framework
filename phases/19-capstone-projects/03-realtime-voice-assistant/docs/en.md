# Capstone Project 03 — Real-Time Voice Assistant (ASR to LLM to TTS)

> A voice agent that feels right has end-to-end latency below 800ms, knows when you are done talking, handles barge-in, and can invoke tools without stalling. In 2026, Retell, Vapi, LiveKit Agents, and Pipecat all hit this bar. They rely on the same shape: streaming ASR, a turn detector, streaming LLM, streaming TTS, all wired through WebRTC with aggressive latency budgets at every hop. Build one, measure WER, MOS, and false-cutoff rate, then run it under packet loss.

**Type:** Capstone
**Languages:** Python (agent + pipeline), TypeScript (web client)
**Prerequisites:** Phase 6 (Speech & Audio), Phase 7 (Transformers), Phase 11 (LLM Engineering), Phase 13 (Tools), Phase 14 (Agents), Phase 17 (Infrastructure)
**Phases involved:** P6 · P7 · P11 · P13 · P14 · P17
**Time:** 30 hours

## The Problem

Voice is the fastest-moving AI experience category in 2025-2026. The technical ceiling drops every quarter. OpenAI Realtime API, Gemini 2.5 Live, Cartesia Sonic-2, ElevenLabs Flash v3, LiveKit Agents 1.0, Pipecat 0.0.70 — all make sub-800ms first-audio-out reachable. The bar is not just latency but interaction feel: not interrupting the user, not being interrupted, recovering after being cut off mid-sentence, calling tools mid-conversation without stalling audio, and surviving jittery mobile networks.

You cannot get there by stitching three REST calls together. The architecture is end-to-end pipeline-style streaming. Build it, and the failure modes reveal themselves: a VAD tuned for phone audio gets triggered by a TV in the background, a turn detector waits for punctuation that never comes, a TTS buffers 400ms before uttering a word. This capstone is about fixing them one by one under load, then publishing a latency-and-quality report.

## The Concept

The pipeline has five streaming stages: **audio input** (WebRTC from browser or PSTN), **ASR** (streaming partial transcripts from Deepgram Nova-3 or faster-whisper), **turn detection** (VAD plus a small turn-detection model that reads partial transcripts for end-of-utterance cues), **LLM** (streaming token output as soon as turn-end is detected), and **TTS** (streaming audio output within ~200ms of the first LLM token).

Three cross-cutting concerns. **Barge-in**: when the user speaks while the agent is talking, TTS is canceled and ASR re-arms immediately. **Tool use**: mid-conversation function calls (weather, calendar) must run on a side channel without stalling audio; if latency exceeds 300ms, the agent emits a filler token ("one moment..."). **Backpressure**: under packet loss, partial transcripts are buffered, the VAD raises its speech-gate threshold, and the agent avoids talking over an unacknowledged message.

Measuring this bar is quantitative. WER below 8% on the Hamming VAD benchmark at 15 dB SNR. First-audio-out p50 below 800ms across 100 live calls. False-cutoff rate below 3%. TTS MOS above 4.2. 50 concurrent calls on a single g5.xlarge. These numbers are the deliverable.

## Architecture

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## Tech Stack

- Transport: LiveKit Agents 1.0 (WebRTC) plus Twilio PSTN gateway; Pipecat 0.0.70 as an alternative framework
- ASR: Deepgram Nova-3 (streaming, first partial under 300ms) or self-hosted faster-whisper Whisper-v3-turbo
- VAD: Silero VAD v5 plus LiveKit turn detector (a small transformer that reads partial transcripts for completion cues)
- LLM: OpenAI GPT-4o-realtime for tight integration, or Gemini 2.5 Flash Live, or cascaded Claude Haiku 4.5 (streaming completion, separate audio path)
- TTS: Cartesia Sonic-2 (lowest time-to-first-byte), ElevenLabs Flash v3, or self-hosted open-source Orpheus
- Tools: weather/calendar/booking via FastMCP side channel; agent emits fillers when tool latency > 300ms
- Observability: OpenTelemetry voice spans, Langfuse voice traces with audio playback
- Deployment: single g5.xlarge (24GB VRAM) for self-hosted Whisper + Orpheus; hosted APIs for lowest latency

## Build It

1. **WebRTC session.** Stand up a LiveKit room and a web client that streams microphone audio. Server-side, attach an agent worker that joins the room.

2. **ASR streaming.** Feed 20ms PCM frames to Deepgram Nova-3 (or faster-whisper on GPU). Subscribe to partial and final transcripts. Log latency of each partial.

3. **VAD and turn detector.** Run Silero VAD v5 on the frame stream. On speech-end events, run the LiveKit turn detector on the latest partial transcript. Commit "end-of-turn" only when VAD reports 500ms silence AND the turn detector completion score > 0.6.

4. **LLM stream.** On turn-end, initiate an LLM call with the ongoing conversation plus the final transcript. Stream tokens. On first token, hand off to TTS.

5. **TTS streaming.** Cartesia Sonic-2 streams audio chunks back. The first chunk must leave the server within 200ms of the first LLM token. Push audio chunks to the LiveKit room; the client plays via WebRTC jitter buffer.

6. **Barge-in.** When VAD detects new user speech while TTS is playing, immediately cancel the TTS stream, discard remaining LLM output, and re-arm ASR. Emit a `tts_canceled` span.

7. **Tool side channel.** Register weather and calendar as function-calling tools. When invoked, fire the call concurrently; if it does not return within 300ms, have the LLM emit a filler ("let me check..."); resume when the tool returns.

8. **Evaluation harness.** Record 100 calls. Compute WER (against holdout transcripts), false-cutoff rate (TTS canceled while user was still mid-sentence), first-audio-out p50, TTS MOS (human-rated or NISQA), and a jitter-loss test (3% packet loss injected).

9. **Stress test.** Drive 50 concurrent calls with synthetic callers on a single g5.xlarge. Measure sustained first-audio-out p95.

## Use It

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## Ship It

`outputs/skill-voice-agent.md` is the deliverable. Given a domain (customer support, scheduling, or kiosk), it stands up a LiveKit agent with the ASR/VAD/LLM/TTS pipeline tuned to the measurement bar. Grading rubric:

| Weight | Criterion | How to measure |
|:-:|---|---|
| 25 | End-to-end latency | First-audio-out p50 below 800ms across 100 recorded calls |
| 20 | Turn-switching quality | False-cutoff rate below 3% on the Hamming VAD benchmark |
| 20 | Tool-use correctness | Mid-conversation tool calls return correct data without stalling audio |
| 20 | Reliability under packet loss | WER and turn-switching stability with 3% packet loss injected |
| 15 | Evaluation harness completeness | Reproducible measurements with public configuration |
| **100** | | |

## Exercises

1. Replace Deepgram Nova-3 with faster-whisper v3 turbo on a g5.xlarge. Measure latency and WER delta. Identify where the CPU vs GPU tradeoff matters.

2. Add a barge-in arbitration policy: what does the agent do when the user interrupts during a tool call? Compare three strategies (hard cancel, finish the tool then stop, queue the next turn).

3. Run an adversarial turn-detection test: have the user pause for a long time mid-sentence. Tune the VAD silence threshold and turn-detector score threshold to minimize false-cutoffs without breaching 900ms.

4. Deploy the same agent to PSTN via Twilio. Compare first-audio-out between PSTN and WebRTC. Explain jitter buffer and codec differences.

5. Add voice activity detection for non-English languages (Japanese, Spanish). Measure Silero VAD v5 false-trigger rate and compare against language-specific fine-tuned versions.

## Key Terms

| Term | What people call it | What it actually is |
|------|---------------------|---------------------|
| Turn detection | "end-of-utterance" | A classifier that determines the user has finished speaking given VAD silence and a partial transcript |
| Barge-in | "interruption handling" | Canceling TTS mid-playback when VAD detects new user speech |
| First-audio-out | "latency" | Time from user stop speaking to the first audio packet leaving the server |
| VAD | "speech gate" | A model that classifies audio frames as speech or silence; Silero VAD v5 is the 2026 default |
| Jitter buffer | "audio smoothing" | A client-side buffer that briefly holds packets to absorb network jitter |
| Filler | "acknowledgment token" | A short phrase the agent emits when a tool is slow, to avoid dead air |
| MOS | "mean opinion score" | A perceptual speech quality rating; NISQA is the automated proxy metric |

## Further Reading

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — reference WebRTC agent framework
- [Pipecat](https://github.com/pipecat-ai/pipecat) — alternative Python-first streaming agent framework
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — reference for integrated voice models
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — streaming ASR reference
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD reference model
- [Cartesia Sonic-2](https://docs.cartesia.ai) — low-latency TTS reference
- [Retell AI architecture](https://docs.retellai.com) — production voice agent architecture
- [Vapi.ai production stack](https://docs.vapi.ai) — alternative production reference
