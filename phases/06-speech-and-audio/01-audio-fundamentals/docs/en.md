# Audio Fundamentals — Waveforms, Sampling, Fourier Transform

> The waveform is the raw signal, the spectrogram is its representation, and mel features are the ML-friendly form. Every modern ASR and TTS pipeline must climb this ladder, and the first step is understanding sampling and Fourier.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Vectors & Matrices), Phase 1 · 14 (Probability Distributions)
**Time:** ~45 minutes

## The Problem

A microphone produces a "pressure varying over time" signal, but your neural network eats tensors. Between the two lies a stack of conventions, and violating any of them causes silent bugs: a model trains fine but WER doubles; TTS goes live with hissing background noise; a voice cloning system memorizes the microphone instead of the speaker.

Every bug in a speech system ultimately traces back to one of three questions:

1. What sample rate was the data recorded at, and what sample rate does the model expect?
2. Has aliasing occurred in the signal?
3. Are you operating on raw samples or some frequency-domain representation?

Get these three right and the rest of Phase 6 falls into place. Get them wrong and even Whisper-Large-v4 will only output garbage.

## The Concept

![Visualization of waveforms, sampling, DFT, and frequency bins](../assets/audio-fundamentals.svg)

**Waveform.** A one-dimensional array of floats in `[-1.0, 1.0]`, indexed by sample number. Convert to seconds by dividing by sample rate: `t = n / sr`. A 10-second clip at 16 kHz is an array of 160,000 floats.

**Sample rate (sr).** How many samples per second. Common sample rates in 2026:

| Sample Rate | Usage |
|------|-----|
| 8 kHz | Telephony, legacy VOIP. Nyquist is only 4 kHz — consonants get chopped. Don't use for ASR. |
| 16 kHz | ASR standard. Whisper, Parakeet, SeamlessM4T v2 all consume 16 kHz. |
| 22.05 kHz | Legacy TTS vocoder training. |
| 24 kHz | Modern TTS (Kokoro, F5-TTS, xTTS v2). |
| 44.1 kHz | CD audio, music. |
| 48 kHz | Film, professional audio, high-fidelity TTS (VALL-E 2, NaturalSpeech 3). |

**Nyquist-Shannon.** The highest frequency a sample rate `sr` can unambiguously represent is `sr/2`. This boundary is the *Nyquist frequency*. Energy above Nyquist undergoes *aliasing* — it folds back into lower frequencies — corrupting the signal. Always apply a low-pass filter before downsampling.

**Bit depth.** 16-bit PCM (signed int16, range ±32,767) is the universal interchange format. Music uses 24-bit; internal DSP processing uses 32-bit float. Libraries like `soundfile` read int16 but expose float32 arrays in `[-1, 1]`.

**Fourier Transform.** Any finite signal is a sum of sinusoids at different frequencies. The Discrete Fourier Transform (DFT) computes `N` complex coefficients from `N` samples — one per frequency bin. `bin k` corresponds to frequency `k · sr / N` Hz; the magnitude is the amplitude at that frequency, and the angle is the phase.

**FFT.** Fast Fourier Transform: an `O(N log N)` algorithm for computing the DFT when `N` is a power of 2. All audio libraries use FFT under the hood. A 1024-point FFT at 16 kHz yields 512 usable frequency bins covering 0–8 kHz at 15.6 Hz resolution.

**Framing + window.** We don't apply a single FFT to the entire audio. Instead, we slice it into overlapping *frames* (typically 25 ms per frame, 10 ms hop), multiply each frame by a window function (Hann, Hamming) to eliminate edge discontinuities, then FFT each frame. This is the Short-Time Fourier Transform (STFT). Lesson 02 continues from here.

## Build It

### Step 1: Read audio and plot the waveform

`code/main.py` uses only the standard library's `wave` module to keep the example zero-dependency. In production you'd use `soundfile` or `torchaudio.load` (both return `(waveform, sr)` tuples):

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### Step 2: Synthesize a sine wave from first principles

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

A 440 Hz sine wave (concert A) at 16 kHz for 1 second is 16,000 floats. Write it out as 16-bit PCM using `wave.open(..., "wb")`.

### Step 3: Compute DFT by hand

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` — fine for verifying correctness with `N=256`, useless for real audio. Real code calls `numpy.fft.rfft` or `torch.fft.rfft`.

### Step 4: Find the dominant frequency

The index `k_star` of the magnitude peak corresponds to frequency `k_star * sr / N`. Run this on the 440 Hz sine wave — you should see a peak at bin `440 * N / sr`.

### Step 5: Demonstrate aliasing

Sample a 7 kHz sine wave at 10 kHz (Nyquist = 5 kHz). The 7 kHz tone exceeds Nyquist and folds to `10 − 7 = 3 kHz`. The FFT peak appears at 3 kHz. This is the classic aliasing demonstration, and exactly why every DAC/ADC includes a steep low-pass filter.

## Use It

The toolkit you'll actually ship with in 2026:

| Task | Library | Why |
|------|---------|-----|
| Read/write WAV/FLAC/OGG | `soundfile` (libsndfile wrapper) | Fastest, stable, returns float32. |
| Resampling | `torchaudio.transforms.Resample` or `librosa.resample` | Built-in correct anti-aliasing. |
| STFT / Mel | `torchaudio` or `librosa` | GPU-friendly; PyTorch ecosystem. |
| Real-time streaming | `sounddevice` or `pyaudio` | Cross-platform PortAudio bindings. |
| Inspect a file | `ffprobe` or `soxi` | CLI, fast, reports sr/channels/encoding. |

Decision rule: **Align sample rates first, then align everything else.** Whisper expects 16 kHz mono float32. Feed it 44.1 kHz stereo and you'll get garbage that looks like a model bug.

## Ship It

Save as `outputs/skill-audio-loader.md`. This skill checks whether audio input matches the downstream model's expectations and resamples correctly when it doesn't.

## Exercises

1. **Easy.** Synthesize a 1-second mix of 220 Hz + 440 Hz + 880 Hz at 16 kHz. Run the DFT and confirm three peaks at the expected bins.
2. **Medium.** Record 3 seconds of your own voice at 48 kHz as WAV. Downsample to 16 kHz using `torchaudio.transforms.Resample` (with anti-aliasing), then downsample to 16 kHz using naive decimation (take every third sample). FFT both. Where does aliasing appear?
3. **Hard.** Build an STFT from scratch using only `math` and the DFT from Step 3. Frame length 400, hop 160, Hann window. Plot the magnitude with `matplotlib.pyplot.imshow`. This is the spectrogram that Lesson 02 covers.

## Key Terms

| Term | How people talk about it | What it actually means |
|------|-----------------|-----------------------|
| Sample rate | How many samples per second | The frequency at which the ADC measures the signal, in Hz. |
| Nyquist | The highest frequency you can represent | `sr/2`; energy above it folds back. |
| Bit depth | Resolution per sample | `int16` = 65,536 quantization levels; `float32` = 24-bit precision within `[-1, 1]`. |
| DFT | Fourier transform of a sequence | `N` samples → `N` complex frequency coefficients. |
| FFT | Fast DFT | `O(N log N)` algorithm requiring `N` to be a power of 2. |
| Bin | Frequency column | `k · sr / N` Hz; resolution = `sr / N`. |
| STFT | What's under the spectrogram | Framed + windowed FFT along time; produces a time × frequency matrix. |
| Aliasing | Weird frequency ghosts | Energy above Nyquist mirrors into lower bins. |

## Further Reading

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — The paper behind the sampling theorem.
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — Free, classic DSP textbook.
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — Hands-on walkthrough with code.
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — Reference explaining why real-world audio isn't a clean sine wave.
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10-minute explanation building intuition for frequency bins.
