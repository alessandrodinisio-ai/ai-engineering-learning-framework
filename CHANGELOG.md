# Changelog

What's new in the curriculum. Latest entries first.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/). Each entry names the phase, lesson, and what changed so learners can jump straight to the update.

## [Unreleased]

### Added
- `scripts/scaffold-lesson.sh` — scaffold script that creates the full `phases/NN-phase/NN-lesson/` directory structure and generates a `docs/en.md` skeleton pre-filled from `LESSON_TEMPLATE.md`.
- `.github/PULL_REQUEST_TEMPLATE.md` — contributor checklist (code runs, no comments in code, implement from scratch first, atomic commits per lesson, ROADMAP rows use markdown links).
- `.github/ISSUE_TEMPLATE/bug_report.md` and `new_lesson_proposal.md` — structured forms for bug reports and lesson proposals.
- This `CHANGELOG.md`.

## 2026-04 — Phase 4: Computer Vision complete

### Added
- All 28 lessons in Phase 4, from image basics through multimodal vision (VLM, 3D, video, self-supervised).
- Phase 4 rows in `ROADMAP.md` now use markdown links to lesson directories so the website can render them.

### Fixed
- Polish pass on 15+ Phase 4 lessons:
  - `phase-4/02`: Shape calculator clarified adaptive pooling, flatten, and linear RF/stride handling.
  - `phase-4/03`: Backbone selector description lists all covered families; added detection-head guidance for OCR, medical, and industrial scenarios.
  - `phase-4/04`: Classification diagnostics use quantized thresholds per failure mode; declare `n/a` for undefined metrics; add guard for fewer than 3 classes.
  - `phase-4/06`: Detection metric reader uses `AP@0.5` (not `mAP@0.5`); per-class recall declared optional; anchor designer clarifies stride truncation and single-anchor-per-level path.
  - `phase-4/10`: Sampler selector declares `unet_forward_ms` as input; ControlNet guard rule promoted to rule 0.
  - `phase-4/14`: ViT checker aligned with rejection rules — transplant attempts are audited, not approved.
  - `phase-4/24`: Open-vocabulary stack selector has explicit rule priority and license-filter semantics; concept designer resolves step-5/rule-80 conflict.
  - `phase-4/25`: VLM doc `_merge` throws a descriptive `ValueError` on placeholder mismatch; CMER normalizes internally.
  - `phase-4/27`: `synthetic_frames` clips GT boxes to frame H/W bounds.
  - `phase-4/28`: `rope_3d` validates dimension splits; removed unused `F` import from DiT block example.

## 2026-Q1 and earlier

### Added
- Phase 0 (Setup & Tooling): all 12 lessons.
- Phase 1 (Math Foundations): all 22 lessons.
- Phase 2 (ML Fundamentals): all 18 lessons.
- Phase 3 (Deep Learning Core): core lessons covering perceptron, backpropagation, optimizers.
- Built-in Claude Code skills: `find-your-level` (placement quiz) and `check-understanding` (per-phase quiz).
- Website `aiengineeringfromscratch.com`: course catalog, per-lesson pages, roadmap, 277-term glossary.
- Initial scaffolding for all 20 phases (`phases/00-*` through `phases/19-*`).
- `LESSON_TEMPLATE.md`, `CONTRIBUTING.md`, `ROADMAP.md`, `README.md`.

[Unreleased]: https://github.com/rohitg00/ai-engineering-from-scratch/compare/HEAD...HEAD
