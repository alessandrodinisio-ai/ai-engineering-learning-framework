# Capstone Project 12 — Video Understanding Pipeline (Scenes, QA, Search)

> Twelve Labs productionized Marengo + Pegasus. VideoDB shipped a CRUD-for-video API. AI2's Molmo 2 released open-source VLM checkpoints. Gemini long-context natively handles hours of video. TimeLens-100K defined temporal grounding at scale. The 2026 pipeline has converged: scene segmentation, per-scene captioning + embedding, transcript alignment, multi-vector indexing, and a query that answers with (start, end) timestamps plus frame previews. This capstone ingests 100 hours, hits public benchmarks, and measures hallucination on counting and action questions.

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (UI)
**Prerequisites:** Phase 4 (CV), Phase 6 (Speech), Phase 7 (Transformers), Phase 11 (LLM Engineering), Phase 12 (Multimodal), Phase 17 (Infrastructure)
**Phases Involved:** P4 · P6 · P7 · P11 · P12 · P17
**Time:** 30 hours

## The Problem

Long-form video QA is the most bandwidth-intensive multimodal problem at scale in 2026. Gemini 2.5 Pro can natively read a 2-hour video, but ingesting 100 hours into a queryable corpus still requires a scene-level index. The production shape combines scene segmentation (TransNetV2 or PySceneDetect), per-scene VLM captioning (Gemini 2.5, Qwen3-VL-Max, or Molmo 2), transcript alignment (Whisper-v3-turbo with word-level timestamps), and a multi-vector index storing captions, frame embeddings, and transcripts side by side. The query pipeline answers with (start, end) timestamps plus frame previews.

Benchmarks are public (ActivityNet-QA, NeXT-GQA) plus your own 100-question custom set. Hallucination on counting and action questions is a notoriously difficult failure category; this capstone measures it explicitly.

## The Concept

At ingestion time three pipelines run in parallel. **Scene segmentation** splits the video into scenes. **VLM captioning** generates a caption for each scene and a frame embedding from a keyframe. **ASR alignment** produces word-level timestamps. The three streams join on (scene_id, time range). Each scene gets three vector types in a multi-vector index (Qdrant): caption embedding, keyframe embedding, transcript embedding.

At query time, a natural language question fires retrieval against all three vector types; results are merged with RRF; a temporal grounding adapter (TimeLens-style) refines the (start, end) window within the best scene. A VLM synthesizer (Gemini 2.5 Pro or Qwen3-VL-Max) receives the query + best scenes + cropped frames, and answers with cited timestamps and a frame preview.

Measuring hallucination matters. Counting ("How many people entered the room?") and action ("Did the chef pour before stirring?") questions are notoriously unreliable. Report accuracy on these categories separately from descriptive questions.

## Architecture

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## Tech Stack

- Scene segmentation: TransNetV2 (2024-26 SOTA) or PySceneDetect
- ASR: Whisper-v3-turbo via faster-whisper, with word-level timestamps
- VLM captioner + answerer: Gemini 2.5 Pro or Qwen3-VL-Max or Molmo 2
- Temporal grounding: TimeLens-100K trained adapter or VideoITG
- Index: Qdrant with multi-vector support (caption / frame / transcript)
- UI: Next.js 15 with HTML5 video player and scene thumbnails
- Evaluation: ActivityNet-QA, NeXT-GQA, custom 100-question hand-labeled set
- Hallucination benchmark: counting and action subsets with manual labels

## Build It

1. **Ingestion walker.** Accepts a YouTube URL or local MP4. Downsamples to 720p if necessary. Persists `{video_id, file_path}`.

2. **Scene segmentation.** Run TransNetV2 or PySceneDetect, producing `[{scene_id, start_ms, end_ms, keyframe_path}]`. Target 100 hours: roughly 6k-8k scenes.

3. **ASR pass.** Run Whisper-v3-turbo on the audio; export word-level timestamps; slice into per-scene transcript segments.

4. **VLM captioning.** For each scene, call Gemini 2.5 Pro (or Qwen3-VL-Max) with the keyframe and a short caption template. Produce caption + frame embedding.

5. **Multi-vector index.** Qdrant collection with three named vectors. Payload: `{video_id, scene_id, start_ms, end_ms, keyframe_url}`.

6. **Query.** A natural language question fires three dense queries; merge with reciprocal rank fusion; top-k=5 scenes.

7. **Temporal grounding.** Run a TimeLens-style adapter on the best scene to refine the (start, end) window within the scene.

8. **VLM synthesis.** Call Gemini 2.5 Pro with query + top-3 scene clips (as images or short segments) + transcript. Require `(video_id, start_ms, end_ms)` citations.

9. **Evaluation.** Run ActivityNet-QA and NeXT-GQA. Build a 100-question custom set. Report overall accuracy + per-category breakdown (counting, action, descriptive).

## Use It

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## Ship It

`outputs/skill-video-qa.md` is the deliverable. Given a YouTube URL or uploaded video, the pipeline indexes scenes and answers with timestamped citations.

| Weight | Criterion | How to Measure |
|:-:|---|---|
| 25 | Temporal grounding IoU | Intersection-over-union on held-out grounding set |
| 20 | QA accuracy | NeXT-GQA and custom 100-question set |
| 20 | Ingestion throughput | Hours of video processed per dollar spent |
| 20 | UI and citation experience | Timestamp links, thumbnail strip, jump-to-frame |
| 15 | Hallucination rate | Counting and action accuracy reported separately |
| **100** | | |

## Exercises

1. Swap Gemini 2.5 Pro for Qwen3-VL-Max on the captioning pass. Report caption quality delta on a 50-scene human-eval sample.

2. Collapse per-scene frame embeddings into a single pooled vector instead of multi-vector. Measure retrieval regression.

3. Build a "strict counting" mode: the synthesizer extracts each counted instance with a timestamp, and the user clicks to verify. Measure whether user verification reduces hallucination.

4. Benchmark ingestion cost: video-hours-per-dollar across three VLM choices. Pick the sweet spot.

5. Add speaker-diarized transcripts: run pyannote speaker diarization on audio and embed per-speaker transcripts. Demonstrate "What did Alice say about X?" queries.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| Scene segmentation | "Shot detection" | Splitting video into scenes at shot boundaries |
| Multi-vector index | "Caption + frame + transcript" | A Qdrant collection with one named vector per representation |
| Temporal grounding | "When exactly did it happen" | Refining (start, end) window for a query answer |
| Frame embedding | "Visual representation" | A vector embedding of a keyframe; used for scene visual similarity |
| RRF fusion | "Reciprocal rank fusion" | A merge strategy across multiple ranked lists; classic hybrid retrieval trick |
| Counting hallucination | "Got the count wrong" | A known failure mode of VLMs on "how many X" questions |
| ActivityNet-QA | "Video QA benchmark" | Long-form video QA accuracy benchmark |

## Further Reading

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — open-source VLM checkpoints
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — temporal grounding at scale
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — hosted reference
- [VideoDB](https://videodb.io) — CRUD-for-video API reference
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — commercial reference
- [TransNetV2](https://github.com/soCzech/TransNetV2) — scene segmentation model
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — classic open-source alternative
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — reference evaluation benchmark
