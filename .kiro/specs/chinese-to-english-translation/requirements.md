# Requirements Document

## Introduction

This spec covers the systematic translation of the AI Engineering Learning Framework project from Simplified Chinese back to English. The project is a Chinese fork of the English "AI Engineering from Scratch" curriculum. All prose documentation (~8 top-level files, ~500 lesson documents in `phases/*/docs/zh.md`, glossary entries, and ROADMAP lesson names) must be translated to English while preserving code, URLs, file paths, Markdown structure, mermaid diagrams, and the site build pipeline (`site/build.js`).

## Glossary

- **Lesson_Document**: A Markdown file at `phases/<phase>/<lesson>/docs/zh.md` containing the Chinese lesson prose
- **Top_Level_Document**: One of the root-level Markdown files (README.md, ROADMAP.md, CONTRIBUTING.md, CHANGELOG.md, CODE_OF_CONDUCT.md, LESSON_TEMPLATE.md, FORKING.md, TRANSLATION.md)
- **Translation_System**: The automated or manual process that converts Chinese prose to English
- **Site_Build_Pipeline**: The `site/build.js` script that parses README.md, ROADMAP.md, and glossary/terms.md to generate `site/data.js`
- **Structural_Marker**: Markdown syntax elements (code fences, table delimiters, HTML tags, mermaid blocks, frontmatter) that must not be modified during translation
- **Section_Heading**: Fixed lesson document headings (The Problem, The Concept, Build It, Use It, Ship It, Exercises, Key Terms, Further Reading)
- **Glossary_File**: Files under `glossary/` (terms.md, myths.md) containing term definitions

## Requirements

### Requirement 1: Translate Top-Level Documentation Files

**User Story:** As a contributor, I want the top-level documentation files translated from Chinese to English, so that English-speaking developers can understand project guidelines without a language barrier.

#### Acceptance Criteria

1. WHEN the Translation_System processes a Top_Level_Document, THE Translation_System SHALL replace all Chinese prose (including headings, paragraphs, blockquotes, and list items) with English prose that conveys the same meaning
2. WHEN the Translation_System encounters fenced code blocks, inline code spans, URLs, file paths, or product/library names inside a Top_Level_Document, THE Translation_System SHALL preserve them unchanged
3. WHEN a fenced code block in a Top_Level_Document contains Chinese comments or Chinese placeholder text, THE Translation_System SHALL translate those Chinese strings to English while preserving all code syntax and indentation
4. WHEN the Translation_System processes README.md, THE Translation_System SHALL preserve the mermaid diagram blocks (from opening ```mermaid to closing ```) byte-for-byte, including node text and theme configuration
5. WHEN the Translation_System processes README.md, THE Translation_System SHALL preserve all badge URLs, HTML alignment tags, and the ASCII art dividers (░▒ lines) unchanged
6. THE Translation_System SHALL translate all eight Top_Level_Documents: README.md, ROADMAP.md, CONTRIBUTING.md, CHANGELOG.md, CODE_OF_CONDUCT.md, LESSON_TEMPLATE.md, FORKING.md, and TRANSLATION.md
7. WHEN the Translation_System completes a Top_Level_Document, THE Translation_System SHALL verify that no CJK Unicode characters (U+4E00–U+9FFF) remain outside of fenced code blocks, URLs, or product names

### Requirement 2: Translate Lesson Documents

**User Story:** As a learner, I want each lesson document available in English, so that I can follow the curriculum in English.

#### Acceptance Criteria

1. WHEN the Translation_System processes a Lesson_Document, THE Translation_System SHALL translate all Chinese prose to English
2. WHEN the Translation_System processes a Lesson_Document, THE Translation_System SHALL use the standard English Section_Headings: "Learning Objectives", "The Problem", "The Concept", "Build It", "Use It", "Ship It", "Exercises", "Key Terms", "Further Reading", "Pitfalls", "Connections"
3. WHEN the Translation_System encounters fenced code blocks or inline code spans (backticked terms) within a Lesson_Document, THE Translation_System SHALL preserve all code unchanged including variable names, function names, imports, and commands
4. WHEN the Translation_System encounters comments within code blocks that contain Chinese explanatory text, THE Translation_System SHALL translate those comments to English
5. WHEN the Translation_System encounters metadata fields (Type, Languages, Prerequisites, Time), THE Translation_System SHALL output them in English format ("**Type:**", "**Languages:**", "**Prerequisites:**", "**Time:**")
6. WHEN the Translation_System processes a Lesson_Document, THE Translation_System SHALL preserve the mermaid diagram syntax and node display text unchanged
7. WHEN the Translation_System encounters a blockquote motto at the top of a Lesson_Document (the one-line core idea), THE Translation_System SHALL translate it to English, as the Site_Build_Pipeline extracts it as the lesson summary

### Requirement 3: Rename Lesson Files from zh.md to en.md

**User Story:** As a developer, I want lesson files named `en.md` instead of `zh.md`, so that the file naming reflects the English content and aligns with the upstream project convention.

#### Acceptance Criteria

1. WHEN the Translation_System completes translation of a Lesson_Document at `docs/zh.md`, THE Translation_System SHALL save the English output as `docs/en.md` in the same directory
2. WHEN the Translation_System creates `docs/en.md`, THE Translation_System SHALL remove the original `docs/zh.md` file from the same directory
3. WHEN a reference inside any project file (Markdown documents, Python source files, JavaScript build scripts, and HTML files) points to `docs/zh.md`, THE Translation_System SHALL update that reference to point to `docs/en.md`
4. WHEN the Translation_System updates references in `site/build.js`, THE Translation_System SHALL replace the hardcoded `docs/zh.md` path in the `extractLessonMeta` function and console output with `docs/en.md`
5. IF a lesson directory does not contain a `docs/zh.md` file, THEN THE Translation_System SHALL skip that directory and continue processing the next lesson without error

### Requirement 4: Translate the Glossary Files

**User Story:** As a learner, I want the glossary terms and myths explained in English, so that I can reference definitions while studying.

#### Acceptance Criteria

1. WHEN the Translation_System processes glossary/terms.md, THE Translation_System SHALL translate the "What people say" and "What it actually means" fields from Chinese to English while preserving the bold field labels (e.g., `**What people say:**`, `**What it actually means:**`) unchanged
2. WHEN the Translation_System processes glossary/terms.md, THE Translation_System SHALL preserve the "Why it's called that" field unchanged on terms where it exists, and SHALL NOT add the field to terms that lack it
3. WHEN the Translation_System processes glossary/terms.md, THE Translation_System SHALL preserve all term heading names (e.g., `### Agent`, `### Attention`) unchanged since they are already in English
4. WHEN the Translation_System processes glossary/myths.md, THE Translation_System SHALL preserve all existing English content unchanged, translating only any Chinese prose passages to English if present
5. WHEN the Translation_System processes a Glossary_File, THE Translation_System SHALL preserve the alphabetical letter headings (e.g., `## A`, `## B`), the `###` term headings, bullet-point field structure, horizontal rule separators, and all other Markdown formatting

### Requirement 5: Translate the ROADMAP Lesson Names

**User Story:** As a contributor, I want the ROADMAP to show English lesson names, so that lesson tracking is consistent with the English content.

#### Acceptance Criteria

1. WHEN the Translation_System processes ROADMAP.md, THE Translation_System SHALL translate all Chinese lesson names in table rows to English while preserving any markdown links (e.g., `[English Name](path)`) with the path unchanged
2. WHEN the Translation_System processes ROADMAP.md, THE Translation_System SHALL translate Chinese phase names (e.g., "配置与工具") to English (e.g., "Setup & Tooling") while preserving the phase header format `## Phase N: Name — <status> (~X hours)`
3. WHEN the Translation_System processes ROADMAP.md, THE Translation_System SHALL preserve the status characters (✅, 🚧, ⬚) and the table structure (pipe-delimited rows, header separator lines, column alignment)
4. WHEN the Translation_System processes ROADMAP.md, THE Translation_System SHALL preserve the estimated time values (e.g., `~75 min`, `~120 min`) unchanged
5. WHEN the Translation_System processes ROADMAP.md, THE Translation_System SHALL translate the preamble text (legend, total duration note, and any introductory paragraphs) from Chinese to English

### Requirement 6: Maintain Site Build Pipeline Compatibility

**User Story:** As a maintainer, I want the site build pipeline to continue working after translation, so that the website generates correctly from the translated files.

#### Acceptance Criteria

1. WHEN the Translation_System completes all translations, THE Site_Build_Pipeline SHALL parse README.md without errors and produce a valid `site/data.js` file that contains syntactically correct JavaScript defining PHASES, GLOSSARY, and ARTIFACTS arrays
2. WHEN the Translation_System translates ROADMAP.md, THE Translation_System SHALL preserve each phase header line matching the regex `^##\s+Phase\s+(\d+).*?—\s*(✅|🚧|⬚)` and each lesson row matching `^\|\s*\d+\s*\|\s*(.+?)\s*\|\s*(✅|🚧|⬚)\s*\|` so that `site/build.js` can extract phase numbers, status emojis, and lesson names
3. WHEN the Translation_System translates README.md, THE Translation_System SHALL preserve the lesson table header line matching the regex `^\|\s*#\s*\|\s*Lesson` and each lesson row as a pipe-delimited line with at least 4 columns (`| # | Lesson Name or [Link](url) | Type | Lang |`) so that `site/build.js` can parse lesson entries
4. WHEN the Translation_System translates glossary/terms.md, THE Translation_System SHALL preserve each term as a `### Term` heading (matching `^###\s+(.+)`), followed by a line matching `\*\*What people say:\*\*` and a line matching `\*\*What it actually means:\*\*` so that `site/build.js` can extract all glossary entries
5. WHEN the Translation_System completes all translations, THE Site_Build_Pipeline SHALL produce a `site/data.js` where the number of parsed phases equals the number of `## Phase N` headers in ROADMAP.md and the number of parsed glossary terms equals the number of `### Term` headings in glossary/terms.md

### Requirement 7: Update Repository Metadata and References

**User Story:** As a maintainer, I want repository references updated to reflect that this is now an English-language project, so that links, descriptions, and internal references are consistent.

#### Acceptance Criteria

1. WHEN the Translation_System processes README.md, THE Translation_System SHALL replace all occurrences of the Chinese fork repository URL (including badge links, stargazer links, and git clone commands) with the English project repository URL, and SHALL replace the Chinese-language website URL with the English project website URL or remove it if no English equivalent exists
2. WHEN the Translation_System processes CONTRIBUTING.md, THE Translation_System SHALL translate all Chinese prose (contribution guidelines, style rules, PR workflow instructions, and section headings) to English, and SHALL update internal references from `zh.md` as the primary documentation file to `en.md`
3. WHEN the Translation_System encounters the TRANSLATION.md file, THE Translation_System SHALL replace its content with a note of no more than 10 sentences explaining that this project was translated from the Chinese fork back to English, crediting the Chinese fork repository, and SHALL preserve the professional terminology mapping table as a bilingual reference for contributors

### Requirement 8: Preserve Non-Translatable Elements

**User Story:** As a developer, I want all non-prose elements preserved exactly during translation, so that code, automation, and tooling continue to work.

#### Acceptance Criteria

1. THE Translation_System SHALL preserve all file paths, directory names, and folder structures as character-identical to the source document
2. THE Translation_System SHALL preserve all URLs (http/https links) unchanged, including query parameters and fragment identifiers
3. THE Translation_System SHALL preserve all product names, library names, and tool names unchanged, where a product/library/tool name is defined as any proper noun that refers to a software project, package, framework, CLI tool, or service (e.g., PyTorch, NumPy, uv, pnpm, cargo, Claude, MCP, LangGraph, vLLM, Docker)
4. THE Translation_System SHALL preserve all Markdown structural elements unchanged: code fences (```), table delimiter rows (pipe-and-dash lines), HTML tags, frontmatter blocks (--- delimiters), and image references (![alt](path) syntax)
5. THE Translation_System SHALL preserve all inline code spans (text enclosed in single backticks) unchanged
6. THE Translation_System SHALL preserve all mermaid diagram blocks — defined as content between ` ```mermaid ` and ` ``` ` fences — including their syntax, node labels, and edge labels unchanged
7. THE Translation_System SHALL preserve all emoji status characters (✅, 🚧, ⬚) in their original positions within the document structure
8. WHEN the Translation_System encounters mixed content where translatable prose and non-translatable elements appear in the same line or table cell, THE Translation_System SHALL translate only the prose portions while preserving the non-translatable elements character-identical

### Requirement 9: Translation Quality Standards

**User Story:** As a reader, I want the English translation to be clear, direct, and technically accurate, so that the content reads naturally and conveys the same meaning as the original.

#### Acceptance Criteria

1. THE Translation_System SHALL produce English prose that varies sentence length and rhythm, avoids repetitive clause structures, and contains no padding phrases (e.g., "It is worth noting that", "at this point in time", "In order to", "It should be mentioned that")
2. THE Translation_System SHALL maintain the direct, concise writing style of the original: sentences SHALL average no more than 20 words, contain no filler words ("basically", "actually", "essentially", "in terms of"), and use no promotional or hyperbolic language
3. WHEN the original Chinese text contains technical terms that have a standard English equivalent listed in the project terminology table or widely established in the field (backpropagation, gradient descent, attention mechanism, tokenizer, embedding, fine-tuning), THE Translation_System SHALL use the standard English terminology consistently across all documents
4. THE Translation_System SHALL preserve the paragraph count, paragraph order, section order, heading hierarchy, and bullet list order of each lesson without adding explanatory text or removing original statements
5. WHEN the Translation_System encounters ambiguous Chinese text where multiple English interpretations are possible, THE Translation_System SHALL select the interpretation that is most consistent with the technical context of the surrounding content and the lesson's subject matter
6. IF the original Chinese text contains a domain-specific term that has no widely established English equivalent, THEN THE Translation_System SHALL retain the pinyin or original term in parentheses alongside a descriptive English translation on first occurrence

### Requirement 10: Batch Processing and Progress Tracking

**User Story:** As a maintainer, I want to translate the project in manageable batches with progress tracking, so that the work is organized and verifiable.

#### Acceptance Criteria

1. THE Translation_System SHALL process files in sequential batches starting with the eight Top_Level_Documents as batch zero, followed by one batch per phase in order from Phase 0 through Phase 19, for a total of 21 batches
2. WHEN a batch of Lesson_Documents within a single phase is completed, THE Translation_System SHALL verify that the count of fenced code blocks (delimited by triple backticks) in each translated file matches the count in the corresponding original file
3. WHEN a batch is completed, THE Translation_System SHALL run `node site/build.js` and verify that the process exits with code 0 and produces a non-empty `site/data.js` file without error output
4. IF the Site_Build_Pipeline produces a non-zero exit code or error output after a batch translation, THEN THE Translation_System SHALL fix the structural issue and re-run `node site/build.js` until it exits with code 0 before proceeding to the next batch
5. WHEN a batch is completed and verified, THE Translation_System SHALL record the batch identifier, the count of files translated, and the verification result in a progress log so that completed work is auditable
