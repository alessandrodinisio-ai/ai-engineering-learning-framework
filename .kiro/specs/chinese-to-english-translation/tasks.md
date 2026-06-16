# Implementation Plan: Chinese-to-English Translation

## Overview

Translate the AI Engineering Learning Framework from Simplified Chinese to English using a batch-oriented approach. The work is organized into sequential batches: top-level documents first, then each phase's lesson documents, then glossary files and final cleanup. Each batch is verified against the site build pipeline (`node site/build.js`) before proceeding. TypeScript/fast-check is used for property-based validation tests.

## Tasks

- [x] 1. Set up translation infrastructure and terminology reference
  - [x] 1.1 Create the terminology mapping reference from TRANSLATION.md
    - Extract the Chinese → English terminology table from TRANSLATION.md
    - Document standard section heading mappings (学习目标 → Learning Objectives, 问题 → The Problem, 概念 → The Concept, 动手做 → Build It, 用起来 → Use It, 上线 → Ship It, 练习 → Exercises, 关键术语 → Key Terms, 延伸阅读 → Further Reading, 陷阱 → Pitfalls, 关联 → Connections)
    - Document standard metadata label mappings (类型 → Type, 语言 → Languages, 前置条件 → Prerequisites, 时间 → Time)
    - _Requirements: 9.3, 2.2, 2.5_

  - [x] 1.2 Update `site/build.js` to reference `docs/en.md` instead of `docs/zh.md`
    - Change `extractLessonMeta` function to read from `docs/en.md`
    - Update the console log message that mentions `docs/zh.md`
    - Update the `GITHUB_BASE` URL from the Chinese fork to the English project URL
    - Update `SITE_ORIGIN` from `https://aieng-zh.cn` to the appropriate English site URL
    - Translate Chinese text strings in `writeLlms()` and `syncCounts()` to English
    - _Requirements: 3.4, 6.1, 7.1_

  - [x] 1.3 Set up fast-check test framework for translation validation
    - Initialize a test directory with fast-check and a test runner (vitest or jest)
    - Create test utilities for segment parsing, CJK detection, and structural validation
    - _Requirements: Property tests infrastructure_

- [x] 2. Batch 0 — Translate top-level documents
  - [x] 2.1 Translate README.md
    - Translate all Chinese prose (headings, paragraphs, blockquotes, list items) to English
    - Preserve mermaid diagram blocks byte-for-byte
    - Preserve badge URLs, HTML alignment tags, and ASCII art dividers (░▒ lines) unchanged
    - Preserve all code blocks, inline code spans, URLs, and file paths unchanged
    - Preserve the lesson table structure (pipe-delimited rows with at least 4 columns)
    - Replace Chinese fork repository URLs with English project URLs
    - Verify no CJK characters remain in prose sections
    - _Requirements: 1.1, 1.2, 1.4, 1.5, 1.7, 6.3, 7.1, 8.1–8.8_

  - [x] 2.2 Translate ROADMAP.md
    - Translate all Chinese phase names to English (e.g., "配置与工具" → "Setup & Tooling")
    - Translate all Chinese lesson names in table rows to English
    - Preserve phase header format matching regex `^##\s+Phase\s+(\d+).*?—\s*(✅|🚧|⬚)`
    - Preserve lesson row format matching `^\|\s*\d+\s*\|\s*(.+?)\s*\|\s*(✅|🚧|⬚)\s*\|`
    - Preserve status characters (✅, 🚧, ⬚) and estimated time values unchanged
    - Translate preamble text (legend, total duration note) to English
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.2, 8.7_

  - [x] 2.3 Translate CONTRIBUTING.md
    - Translate all Chinese prose (contribution guidelines, style rules, PR workflow) to English
    - Update internal references from `zh.md` to `en.md`
    - Preserve code blocks, URLs, and structural formatting unchanged
    - _Requirements: 1.1, 1.2, 7.2_

  - [x] 2.4 Translate CHANGELOG.md, CODE_OF_CONDUCT.md, LESSON_TEMPLATE.md, and FORKING.md
    - Translate all Chinese prose to English in each file
    - Preserve code blocks, URLs, file paths, and markdown structure unchanged
    - For LESSON_TEMPLATE.md, ensure standard section headings are in the approved English set
    - _Requirements: 1.1, 1.2, 1.6_

  - [x] 2.5 Replace TRANSLATION.md content
    - Replace content with a note (≤10 sentences) explaining this was translated from the Chinese fork
    - Credit the Chinese fork repository
    - Preserve the professional terminology mapping table as a bilingual reference
    - _Requirements: 7.3_

  - [x] 2.6 Run build validation for Batch 0
    - Run `node site/build.js` and verify exit code 0
    - Verify `site/data.js` is non-empty and contains valid JavaScript
    - Verify phase count in `site/data.js` matches ROADMAP `## Phase` headers
    - Fix any structural issues and re-run until build passes
    - _Requirements: 6.1, 6.5, 10.3, 10.4_

  - [x] 2.7 Write property tests for top-level document translation
    - **Property 2: No residual CJK in prose sections**
    - **Property 6: ROADMAP structural integrity**
    - **Property 8: Document structure count invariant**
    - **Validates: Requirements 1.7, 2.7, 5.3, 5.4, 6.2, 9.4, 10.2**

- [x] 3. Batch 1 — Translate Phase 0 lessons (Setup & Tooling)
  - [x] 3.1 Translate all lesson documents in `phases/00-setup-and-tooling/*/docs/zh.md`
    - Translate Chinese prose to English in each lesson file
    - Use standard English section headings (Learning Objectives, The Problem, The Concept, etc.)
    - Use standard English metadata labels (Type, Languages, Prerequisites, Time)
    - Translate Chinese comments in code blocks while preserving all code syntax
    - Preserve fenced code blocks, inline code, URLs, mermaid diagrams, and file paths unchanged
    - Translate the blockquote motto (lesson summary) at the top of each lesson
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 8.1–8.8, 9.1–9.5_

  - [x] 3.2 Rename Phase 0 lesson files from `zh.md` to `en.md`
    - Save translated content as `docs/en.md` in each lesson directory
    - Remove the original `docs/zh.md` files
    - Skip any lesson directories that lack `docs/zh.md`
    - _Requirements: 3.1, 3.2, 3.5_

  - [x] 3.3 Update Phase 0 internal references from `docs/zh.md` to `docs/en.md`
    - Scan all markdown files, JS, PY, and HTML files referencing Phase 0 lesson docs
    - Replace any `docs/zh.md` references with `docs/en.md`
    - _Requirements: 3.3_

  - [x] 3.4 Run build validation for Batch 1
    - Run `node site/build.js` and verify exit code 0
    - Verify code block counts in translated files match originals
    - Fix any structural issues and re-run until build passes
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 4. Checkpoint — Verify Phase 0 translation
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Batch 2 — Translate Phase 1 lessons (Math Foundations)
  - [x] 5.1 Translate all lesson documents in `phases/01-math-foundations/*/docs/zh.md`
    - Apply same translation rules as Phase 0 (standard headings, metadata, code preservation)
    - Pay special attention to mathematical terminology consistency (backpropagation, gradient descent, etc.)
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 5.2 Rename Phase 1 lesson files from `zh.md` to `en.md` and update references
    - Save as `docs/en.md`, remove `docs/zh.md`, update all references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 5.3 Run build validation for Batch 2
    - Run `node site/build.js` and verify exit code 0
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 6. Batch 3 — Translate Phase 2 lessons (ML Fundamentals)
  - [x] 6.1 Translate all lesson documents in `phases/02-ml-fundamentals/*/docs/zh.md`
    - Apply standard translation rules, maintain ML terminology consistency
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 6.2 Rename Phase 2 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 6.3 Run build validation for Batch 3
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 7. Batch 4 — Translate Phase 3 lessons (Deep Learning Core)
  - [x] 7.1 Translate all lesson documents in `phases/03-deep-learning-core/*/docs/zh.md`
    - Apply standard translation rules, maintain deep learning terminology consistency
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 7.2 Rename Phase 3 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 7.3 Run build validation for Batch 4
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 8. Batch 5 — Translate Phase 4 lessons (Computer Vision)
  - [x] 8.1 Translate all lesson documents in `phases/04-computer-vision/*/docs/zh.md`
    - Apply standard translation rules
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 8.2 Rename Phase 4 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 8.3 Run build validation for Batch 5
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 9. Checkpoint — Verify Phases 1–4 translation
  - Ensure all tests pass, ask the user if questions arise.

- si
  - [x] 10.1 Translate all lesson documents in `phases/05-nlp-foundations-to-advanced/*/docs/zh.md`
    - Apply standard translation rules, maintain NLP terminology consistency (tokenizer, embedding, attention mechanism)
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 10.2 Rename Phase 5 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 10.3 Run build validation for Batch 6
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 11. Batch 7 — Translate Phase 6 lessons (Speech & Audio)
  - [x] 11.1 Translate all lesson documents in `phases/06-speech-and-audio/*/docs/zh.md`
    - Apply standard translation rules
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 11.2 Rename Phase 6 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 11.3 Run build validation for Batch 7
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 12. Batch 8 — Translate Phase 7 lessons (Transformers Deep Dive)
  - [x] 12.1 Translate all lesson documents in `phases/07-transformers-deep-dive/*/docs/zh.md`
    - Apply standard translation rules, maintain transformer terminology (self-attention, multi-head attention, KV cache)
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 12.2 Rename Phase 7 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 12.3 Run build validation for Batch 8
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 13. Batch 9 — Translate Phase 8 lessons (Generative AI)
  - [x] 13.1 Translate all lesson documents in `phases/08-generative-ai/*/docs/en.md`
    - Apply standard translation rules, maintain generative AI terminology (fine-tuning, RLHF, diffusion)
    - NOTE: Files already renamed to en.md but content is still Chinese
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 13.2 Rename Phase 8 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 13.3 Run build validation for Batch 9
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 14. Batch 10 — Translate Phase 9 lessons (Reinforcement Learning)
  - [x] 14.1 Translate all lesson documents in `phases/09-reinforcement-learning/*/docs/en.md`
    - Apply standard translation rules
    - NOTE: Files already renamed to en.md but content is still Chinese
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 14.2 Rename Phase 9 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 14.3 Run build validation for Batch 10
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 15. Checkpoint — Verify Phases 5–9 translation
  - All phases 5–9 verified: zero CJK in prose sections.

- [x] 16. Batch 11 — Translate Phase 10 lessons (LLMs from Scratch)
  - [x] 16.1 Translate all lesson documents in `phases/10-llms-from-scratch/*/docs/en.md`
    - All files translated. Only 02-building-a-tokenizer has Chinese in code test strings (acceptable - test data).
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 16.2 Rename Phase 10 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 16.3 Run build validation for Batch 11
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 17. Batch 12 — Translate Phase 11 lessons (LLM Engineering)
  - [x] 17.1 Translate all lesson documents in `phases/11-llm-engineering/*/docs/en.md`
    - All 12 files translated in previous sessions.
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 17.2 Rename Phase 11 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 17.3 Run build validation for Batch 12
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 18. Batch 13 — Translate Phase 12 lessons (Multimodal AI)
  - [x] 18.1 Translate all lesson documents in `phases/12-multimodal-ai/*/docs/en.md`
    - All files translated in previous sessions.
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 18.2 Rename Phase 12 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 18.3 Run build validation for Batch 13
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 19. Batch 14 — Translate Phase 13 lessons (Tools & Protocols)
  - [x] 19.1 Translate all lesson documents in `phases/13-tools-and-protocols/*/docs/en.md`
    - All 23 files translated (7 remaining files completed 2026-06-15).
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 19.2 Rename Phase 13 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 19.3 Run build validation for Batch 14
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 20. Batch 15 — Translate Phase 14 lessons (Agent Engineering)
  - [x] 20.1 Translate all lesson documents in `phases/14-agent-engineering/*/docs/en.md`
    - All 42 files translated (2026-06-15).
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 20.2 Rename Phase 14 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 20.3 Run build validation for Batch 15
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 21. Checkpoint — Verify Phases 10–14 translation
  - All phases verified: zero CJK in prose sections.

- [x] 22. Batch 16 — Translate Phase 15 lessons (Autonomous Systems)
  - [x] 22.1 Translate all lesson documents in `phases/15-autonomous-systems/*/docs/en.md`
    - Apply standard translation rules
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 22.2 Rename Phase 15 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 22.3 Run build validation for Batch 16
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 23. Batch 17 — Translate Phase 16 lessons (Multi-Agent & Swarms)
  - [x] 23.1 Translate all lesson documents in `phases/16-multi-agent-and-swarms/*/docs/en.md`
    - All 25 files translated (13 remaining completed in prior sessions).
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 23.2 Rename Phase 16 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 23.3 Run build validation for Batch 17
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 24. Batch 18 — Translate Phase 17 lessons (Infrastructure & Production)
  - [x] 24.1 Translate all lesson documents in `phases/17-infrastructure-and-production/*/docs/en.md`
    - All 28 files translated (2026-06-15).
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 24.2 Rename Phase 17 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 24.3 Run build validation for Batch 18
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 25. Batch 19 — Translate Phase 18 lessons (Ethics, Safety & Alignment)
  - [x] 25.1 Translate all lesson documents in `phases/18-ethics-safety-alignment/*/docs/en.md`
    - All 30 files translated (2026-06-15).
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 25.2 Rename Phase 18 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 25.3 Run build validation for Batch 19
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 26. Batch 20 — Translate Phase 19 lessons (Capstone Projects)
  - [x] 26.1 Translate all lesson documents in `phases/19-capstone-projects/*/docs/en.md`
    - All 85 files translated (2026-06-15).
    - _Requirements: 2.1–2.7, 8.1–8.8, 9.1–9.5_

  - [x] 26.2 Rename Phase 19 lesson files from `zh.md` to `en.md` and update references
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 26.3 Run build validation for Batch 20
    - _Requirements: 10.2, 10.3, 10.4_

- [x] 27. Checkpoint — Verify Phases 15–19 translation
  - All phases verified: zero CJK in prose sections (2026-06-15).

- [x] 28. Final batch — Translate glossary files
  - [x] 28.1 Translate `glossary/terms.md`
    - Translate "What people say" and "What it actually means" fields from Chinese to English
    - Preserve bold field labels (`**What people say:**`, `**What it actually means:**`) unchanged
    - Preserve "Why it's called that" field unchanged where it exists; do not add to terms that lack it
    - Preserve all term heading names (`### Agent`, `### Attention`, etc.) unchanged
    - Preserve alphabetical letter headings (`## A`, `## B`), bullet structure, and horizontal rule separators
    - _Requirements: 4.1, 4.2, 4.3, 4.5, 6.4_

  - [x] 28.2 Translate `glossary/myths.md`
    - Preserve all existing English content unchanged
    - Translate only Chinese prose passages to English if present
    - _Requirements: 4.4_

  - [x] 28.3 Run build validation for final batch
    - Run `node site/build.js` and verify exit code 0
    - Verify glossary term count in `site/data.js` matches `### Term` headings in `glossary/terms.md`
    - _Requirements: 6.4, 6.5, 10.3, 10.4_

- [x] 29. Global reference cleanup and final verification
  - [x] 29.1 Scan entire repository for remaining `docs/zh.md` references
    - Search all `.md`, `.js`, `.py`, `.html` files for any remaining `docs/zh.md` strings
    - Replace all found references with `docs/en.md`
    - Verify zero occurrences of `docs/zh.md` remain in the repository
    - _Requirements: 3.3, 7.2_

  - [x] 29.2 Update `catalog.json` and any other metadata files
    - Ensure catalog.json references are consistent with English content
    - Update any remaining Chinese text in project metadata
    - _Requirements: 7.1_

  - [x] 29.3 Run final full build validation
    - Run `node site/build.js` and verify exit code 0 with non-empty `site/data.js`
    - Verify phase count matches ROADMAP `## Phase` headers
    - Verify glossary term count matches `### Term` headings
    - Verify all 8 top-level documents exist and contain no CJK in prose sections
    - Verify all lesson directories have `docs/en.md` and no `docs/zh.md`
    - _Requirements: 6.1, 6.5, 10.3, 10.5_

  - [x] 29.4 Write property tests for full translation validation
    - **Property 1: Non-translatable element preservation**
    - **Property 4: Lesson document standard structure**
    - **Property 5: Glossary structural integrity**
    - **Property 7: Reference update completeness**
    - **Property 9: Terminology consistency**
    - **Validates: Requirements 1.2, 2.2, 2.3, 2.5, 2.6, 3.3, 4.1–4.5, 8.1–8.8, 9.3**

- [x] 30. Final checkpoint — Ensure all tests pass
  - All lesson files translated to English. Zero CJK in prose. Build passes. Only remaining Chinese is in code test data (Phase 10, Lesson 02 tokenizer tests).

## Notes

### ⚠️ Strategy Update (2026-06-14)

**Large-batch sub-agent translation does NOT work.** Each lesson file is 400–800+ lines of dense
technical prose. Sub-agents hit context window limits after translating 3–6 files per session.
Attempting to translate 20+ files in a single sub-agent call results in partial completion or
silent failures.

**New strategy: micro-batches of 3–5 files per sub-agent call.**

- Each sub-agent should translate at most 3–5 files (ideally ~1500 lines total)
- Verify after each micro-batch with the CJK detection script
- Prioritize completing one phase fully before moving to the next
- Target: ~15–20 files per main session (3–4 sub-agent calls)

**Current progress (2026-06-15):**
- Files renamed zh.md→en.md: ✅ ALL DONE (0 zh.md remain)
- Files with content actually translated to English: **~500 of 501**
- Files still containing Chinese prose: **0**
  - Only exception: Phase 10, Lesson 02 has Chinese in code test strings (tokenizer test data — acceptable)
- Phase 00 mermaid diagrams: ✅ translated
- Glossary: ✅ translated
- Build: ✅ passes (exit 0)
- Remaining tasks: property tests (29.4), final checkpoint (30)

**Translation complete.** All prose content is in English.

---

### Original notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each batch corresponds to one processing unit: top-level docs, one phase, or glossary
- Build validation after every batch catches structural regressions immediately
- The terminology mapping from task 1.1 must be applied consistently across ALL subsequent batches
- If a lesson directory lacks `docs/zh.md`, it is silently skipped (Requirement 3.5)
- If build fails after a batch, fix the structural issue and re-run before proceeding (Requirement 10.4)
- Property tests validate universal correctness properties defined in the design document
- Checkpoints are placed at reasonable intervals (~5 phases each) to verify cumulative progress

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1", "2.2", "2.3", "2.4", "2.5"] },
    { "id": 2, "tasks": ["2.6", "2.7"] },
    { "id": 3, "tasks": ["3.1"] },
    { "id": 4, "tasks": ["3.2"] },
    { "id": 5, "tasks": ["3.3", "3.4"] },
    { "id": 6, "tasks": ["5.1"] },
    { "id": 7, "tasks": ["5.2"] },
    { "id": 8, "tasks": ["5.3"] },
    { "id": 9, "tasks": ["6.1"] },
    { "id": 10, "tasks": ["6.2"] },
    { "id": 11, "tasks": ["6.3"] },
    { "id": 12, "tasks": ["7.1"] },
    { "id": 13, "tasks": ["7.2"] },
    { "id": 14, "tasks": ["7.3"] },
    { "id": 15, "tasks": ["8.1"] },
    { "id": 16, "tasks": ["8.2"] },
    { "id": 17, "tasks": ["8.3"] },
    { "id": 18, "tasks": ["10.1"] },
    { "id": 19, "tasks": ["10.2"] },
    { "id": 20, "tasks": ["10.3"] },
    { "id": 21, "tasks": ["11.1"] },
    { "id": 22, "tasks": ["11.2"] },
    { "id": 23, "tasks": ["11.3"] },
    { "id": 24, "tasks": ["12.1"] },
    { "id": 25, "tasks": ["12.2"] },
    { "id": 26, "tasks": ["12.3"] },
    { "id": 27, "tasks": ["13.1"] },
    { "id": 28, "tasks": ["13.2"] },
    { "id": 29, "tasks": ["13.3"] },
    { "id": 30, "tasks": ["14.1"] },
    { "id": 31, "tasks": ["14.2"] },
    { "id": 32, "tasks": ["14.3"] },
    { "id": 33, "tasks": ["16.1"] },
    { "id": 34, "tasks": ["16.2"] },
    { "id": 35, "tasks": ["16.3"] },
    { "id": 36, "tasks": ["17.1"] },
    { "id": 37, "tasks": ["17.2"] },
    { "id": 38, "tasks": ["17.3"] },
    { "id": 39, "tasks": ["18.1"] },
    { "id": 40, "tasks": ["18.2"] },
    { "id": 41, "tasks": ["18.3"] },
    { "id": 42, "tasks": ["19.1"] },
    { "id": 43, "tasks": ["19.2"] },
    { "id": 44, "tasks": ["19.3"] },
    { "id": 45, "tasks": ["20.1"] },
    { "id": 46, "tasks": ["20.2"] },
    { "id": 47, "tasks": ["20.3"] },
    { "id": 48, "tasks": ["22.1"] },
    { "id": 49, "tasks": ["22.2"] },
    { "id": 50, "tasks": ["22.3"] },
    { "id": 51, "tasks": ["23.1"] },
    { "id": 52, "tasks": ["23.2"] },
    { "id": 53, "tasks": ["23.3"] },
    { "id": 54, "tasks": ["24.1"] },
    { "id": 55, "tasks": ["24.2"] },
    { "id": 56, "tasks": ["24.3"] },
    { "id": 57, "tasks": ["25.1"] },
    { "id": 58, "tasks": ["25.2"] },
    { "id": 59, "tasks": ["25.3"] },
    { "id": 60, "tasks": ["26.1"] },
    { "id": 61, "tasks": ["26.2"] },
    { "id": 62, "tasks": ["26.3"] },
    { "id": 63, "tasks": ["28.1", "28.2"] },
    { "id": 64, "tasks": ["28.3"] },
    { "id": 65, "tasks": ["29.1", "29.2"] },
    { "id": 66, "tasks": ["29.3", "29.4"] }
  ]
}
```
