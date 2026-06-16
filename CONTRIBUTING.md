# Contributing Guide

Lessons, translations, fixes, artifacts — all welcome. One PR does one thing: reviews are faster, and contributor credit gets attributed correctly.

## Important: README and ROADMAP Feed the Website

`site/build.js` parses `README.md`, `ROADMAP.md`, and `glossary/terms.md` to generate `site/data.js`. Any PR that touches these files must keep the following two format types intact:

- Phase headings use either `### Phase N: Name \`X lessons\`` form, or
  `<details><summary><b>Phase N — Name</b> ... <code>X lessons</code> ... <em>Description</em></summary>` form.
- Lesson tables use the `| # | Lesson | Type | Lang |` column structure (capstone project tables use
  `| # | Project | Combines | Lang |`). The `Lang` column can be plain text (`Python, TypeScript`) or the older emoji flags
  (`🐍 🟦 🦀 🟣 ⚛️`); both are equivalent to the parser.
- ROADMAP status characters (`✅`, `🚧`, `⬚`) appear on phase headings and lesson rows.
  Do not replace them with text — the parser identifies them by these exact characters.

After editing these files, run `node site/build.js`; if your changes are structurally safe, `git diff site/data.js`
should show only a timestamp change.

## Ways to Contribute

### 1. Add a New Lesson

Each lesson lives in `phases/XX-phase-name/NN-lesson-name/` with this structure:

```
NN-lesson-name/
├── code/           at least one runnable implementation
├── notebook/       Jupyter notebook for experimentation (optional)
├── docs/
│   └── en.md       lesson document (required)
└── outputs/        prompts, skills, or agents produced by this lesson (if applicable)
```

**Lesson document format** (`en.md`):

```markdown
# Lesson Title

> One-line motto — the core idea in one sentence.

## The Problem

Why does this matter? What can't you do without this?

## The Concept

Explain with diagrams, visuals, and intuition. Code comes later.

## Build It

Step-by-step implementation from scratch.

## Use It

Now use a real framework or library to do the same thing.

## Ship It

The prompt, skill, agent, or tool this lesson produces.

## Exercises

1. Exercise one
2. Exercise two
3. Challenge exercise
```

### 2. Add a Translation

Create a new file in any lesson's `docs/` folder:

```
docs/
├── en.md    (English — always required)
├── zh.md    (Chinese)
├── ja.md    (Japanese)
├── es.md    (Spanish)
├── hi.md    (Hindi)
└── ...
```

Keep the same structure as the English version. Translate content, not code.

### 3. Add an Artifact

If a lesson should produce a reusable prompt, skill, agent, or MCP server:

1. Create it in the lesson's `outputs/` folder
2. Add a reference in the top-level `outputs/` index

**Prompt format:**

```markdown
---
name: prompt-name
description: What this prompt does
phase: 14
lesson: 01
---

[System prompt or template here]
```

**Skill format:**

```markdown
---
name: skill-name
description: What this skill teaches
version: 1.0.0
phase: 14
lesson: 01
tags: [agents, loops]
---

[Skill content here]
```

### 4. Fix Bugs or Improve Existing Lessons

- Fix code that doesn't run
- Improve explanations
- Add better diagrams
- Update outdated information

### 5. Add Exercises or Projects

More exercises and projects are always welcome, especially those that tie multiple phases together.

## Standards

- **Code must run.** Every code file should execute without errors using the listed dependencies.
- **No comments in code.** Code should be self-explanatory. Explanations belong in the docs.
- **Use the right language.** Don't force Python where TypeScript or Rust fits better.
- **Build from scratch first.** Always implement the concept from first principles before showing the framework version.
- **Stay practical.** Theory serves practice, not the other way around.
- **No AI-generated filler.** Write like a human. Be direct. Cut the fluff.

## Pull Request Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b add-lesson-phase3-gradient-descent`)
3. Make your changes
4. Ensure all code runs
5. Submit a pull request with a clear description

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Be kind, helpful, and constructive.

## Style

- Direct prose. Cut the fluff. Match this handbook's tone, not marketing copy.
- No decorative emoji in headings. The emoji flags in the Lang column are the only exception, and only because the parser maps them.
- Code runs as-is with the dependencies listed in the lesson.
- Build from scratch first, frameworks second.
