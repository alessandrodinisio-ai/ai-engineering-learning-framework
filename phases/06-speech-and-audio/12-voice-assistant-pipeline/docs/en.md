# Building a Voice Assistant Pipeline — Phase 6 Capstone

> Stitch everything from lessons 01–11 together. Build a voice assistant that listens, reasons, and speaks. In 2026 this is a solved engineering problem, not a research problem — but the integration details determine whether it ships.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**Time:** ~120 minutes

## The Problem

Build an end-to-end assistant:

1. Capture microphone input (16 kHz mono).
2. Detect start/end of user speech.
3. Stream transcription.
4. Hand the transcript to an LLM that can call tools (timer, weather, calendar).
5. Stream LLM text to a TTS.
6. Play audio back to the user.
7. Stop when the user interrupts mid-response.

Latency target: first TTS audio byte within 800 ms of user finishing, on a laptop CPU. Quality targets: no missed words, no hallucinated captions on silence, no voice cloning leaks, no successful prompt injections.

## The Concept

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### Seven Components

1. **Audio capture.** Mic → 16 kHz mono → 20 ms chunks. Typically `sounddevice` in Python, native AudioUnit/ALSA/WASAPI in production.
2. **VAD (Lesson 11).** Silero VAD @ threshold 0.5, min speech 250 ms, silence hang 500 ms. Emits "start" and "end" signals.
3. **Streaming STT (Lessons 4–5).** Whisper-streaming, Parakeet-TDT, or Deepgram Nova-3 (API). Partial + final transcripts.
4. **LLM with tool calling.** GPT-4o / Claude 3.5 / Gemini 2.5 Flash. Tools defined as JSON schemas. Streams tokens.
5. **Streaming TTS (Lesson 7).** Kokoro-82M (fastest open-source) or Cartesia Sonic (commercial). Start TTS once LLM emits 20 tokens.
6. **Playback.** Speaker output; opus encoding for low-bandwidth networks.
7. **Interruption handler.** If VAD fires during TTS playback, stop playback, cancel LLM, restart STT.

### Three Failure Modes You'll Hit

1. **First word clipped.** VAD starts a beat late. The user's "hey" is lost. Set onset threshold to 0.3, not 0.5.
2. **Mid-response interruption chaos.** LLM keeps generating after user interrupts; assistant talks over the user. Wire VAD → cancel LLM properly.
3. **Silence hallucination.** Whisper outputs "Thanks for watching" on silent warmup frames. Always gate with VAD.

### 2026 Production Reference Stacks

| Stack | Latency | License | Notes |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | Commercial APIs | 2026 industry default |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | Largely open-source | DIY friendly |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | Single model; different architecture, Lesson 15 |
| Vapi / Retell (managed) | 300-500 ms | Commercial | Fastest to ship; limited customization |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | Offline | Open-source | Privacy / edge |

## Build It

### Step 1: Mic Capture with Chunking (Pseudocode)

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### Step 2: VAD-Gated Turn Capture

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### Step 3: Streaming STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### Step 4: Tool Calling in the LLM Loop

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### Step 5: Interruption Handling

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## Use It

See `code/main.py` for a runnable simulation that wires all seven components together with stub models, so you can see the pipeline's shape without hardware. For a real implementation, replace the stubs with:

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` or `openai-whisper`
- `openai` (`gpt-4o`) or `anthropic`
- `kokoro` or `cartesia`
- `sounddevice` for I/O

## Pitfalls

- **Permanently recording PII.** Full-turn audio is PII in most jurisdictions. Retain 30 days max, encrypt at rest.
- **No barge-in handling.** Users will interrupt. Your assistant must stop.
- **Blocking TTS.** Synchronous TTS blocks the event loop. Use async or a separate thread.
- **No tool-call error handling.** Tools fail. The LLM must get the error back + retry once, then degrade gracefully.
- **Over-aggressive hallucination filter.** Too strict, the assistant repeatedly says "I can't help with that." Too loose, it says anything. Calibrate on a held-out set.
- **No wake-word option.** Always-listening is a privacy liability. Add a wake-word gate (Porcupine or openWakeWord).

## Ship It

Save as `outputs/skill-voice-assistant-architect.md`. Given budget + scale + language + compliance constraints, produce a full stack spec.

## Exercises

1. **Easy.** Run `code/main.py`. It simulates a full turn end-to-end with stub modules and prints per-stage latency.
2. **Medium.** Replace the STT stub with a real Whisper model, running on a pre-recorded `.wav`. Measure WER and end-to-end latency.
3. **Hard.** Add tool calling: implement `get_weather` (any API) and `set_timer`. Wire the LLM to the tools, verify the correct function fires when the user says "set a 5 minute timer," and the spoken response confirms it.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Turn | One user + assistant round-trip | A VAD-bounded user utterance + one LLM-TTS response. |
| Barge-in | Interruption | User speaks while assistant is talking; assistant stops. |
| Wake word | "Hey assistant" | Short keyword detector; Porcupine, Snowboy, openWakeWord. |
| End-pointing | Turn boundary | VAD + minimum silence determines the user has finished. |
| Pre-roll | Pre-speech buffer | Keeping 200–400 ms of audio before VAD fires, to avoid clipping the first word. |
| Tool call | Function calling | LLM emits JSON; runtime dispatches; result fed back in-loop. |

## Further Reading

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — production-grade reference.
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — DIY-friendly framework.
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — managed speech-native path.
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — full-duplex reference (Lesson 15).
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — wake-word gating.
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM function calling.
