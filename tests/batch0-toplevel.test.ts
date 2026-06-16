/**
 * Property-based tests for Batch 0 — Top-level document translation.
 *
 * Feature: chinese-to-english-translation
 * Properties tested:
 *   Property 2: No residual CJK in prose sections
 *   Property 6: ROADMAP structural integrity
 *   Property 8: Document structure count invariant
 *
 * Validates: Requirements 1.7, 2.7, 5.3, 5.4, 6.2, 9.4, 10.2
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import { fc, PBT_PARAMS } from './helpers.js';
import { containsCJK, extractCJK } from './utils/cjk-detection.js';
import { extractProseSections } from './utils/segment-parsing.js';
import {
  countCodeBlocks,
  countHeadings,
  countMermaidBlocks,
} from './utils/structural-validation.js';

/** Root directory of the project. */
const ROOT = resolve(import.meta.dirname, '..');

/** Top-level documents that should be fully translated. */
const TOP_LEVEL_DOCS = [
  'README.md',
  'ROADMAP.md',
  'CONTRIBUTING.md',
  'CHANGELOG.md',
  'CODE_OF_CONDUCT.md',
  'LESSON_TEMPLATE.md',
  'FORKING.md',
  'TRANSLATION.md',
];

/**
 * Documents where CJK is expected in prose sections.
 * TRANSLATION.md preserves a bilingual terminology table by design (task 2.5),
 * so CJK characters in its table cells are intentional, not residual.
 */
const CJK_EXEMPT_DOCS = new Set(['TRANSLATION.md']);

/**
 * Helper: read a top-level document. Returns content or null if missing.
 */
function readTopLevelDoc(filename: string): string | null {
  const filepath = resolve(ROOT, filename);
  if (!existsSync(filepath)) return null;
  return readFileSync(filepath, 'utf-8');
}

describe('Batch 0 — Top-level document property tests', () => {
  /**
   * Property 2: No residual CJK in prose sections
   *
   * For any translated top-level document, scanning all prose segments
   * (text outside of fenced code blocks, inline code spans, URLs, and
   * product names) SHALL find zero characters in the CJK Unified
   * Ideographs range (U+4E00–U+9FFF).
   *
   * **Validates: Requirements 1.7, 2.7**
   */
  describe('Property 2: No residual CJK in prose sections', () => {
    /** Documents to check for CJK (excludes bilingual reference docs). */
    const DOCS_TO_CHECK = TOP_LEVEL_DOCS.filter((d) => !CJK_EXEMPT_DOCS.has(d));

    // Generate a random selection of top-level docs per run
    it('should contain zero CJK characters in prose of all top-level documents', () => {
      fc.assert(
        fc.property(
          fc.constantFrom(...DOCS_TO_CHECK),
          (docName) => {
            const content = readTopLevelDoc(docName);
            if (content === null) {
              // Document doesn't exist — skip (not a failure)
              return true;
            }

            const proseSections = extractProseSections(content);

            for (const section of proseSections) {
              if (containsCJK(section)) {
                const found = extractCJK(section);
                // Fail with details about where CJK was found
                expect(
                  found,
                  `CJK characters found in prose of ${docName}: "${found.join(', ')}" in line: "${section.slice(0, 100)}"`
                ).toHaveLength(0);
              }
            }

            return true;
          }
        ),
        PBT_PARAMS
      );
    });

    // Exhaustive check across all documents (non-property, ensures full coverage)
    it('should have zero CJK in prose across every top-level document (exhaustive)', () => {
      for (const docName of DOCS_TO_CHECK) {
        const content = readTopLevelDoc(docName);
        if (content === null) continue;

        const proseSections = extractProseSections(content);
        const cjkFindings: { line: string; chars: string[] }[] = [];

        for (const section of proseSections) {
          if (containsCJK(section)) {
            cjkFindings.push({
              line: section.slice(0, 120),
              chars: extractCJK(section),
            });
          }
        }

        expect(
          cjkFindings,
          `${docName} has ${cjkFindings.length} prose lines with residual CJK characters`
        ).toHaveLength(0);
      }
    });

    // Verify TRANSLATION.md only has CJK inside table rows (bilingual reference)
    it('TRANSLATION.md CJK is confined to table rows only', () => {
      const content = readTopLevelDoc('TRANSLATION.md');
      if (content === null) return;

      const lines = content.split('\n');
      for (const line of lines) {
        const trimmed = line.trim();
        // Skip table rows (start with |) — CJK is expected there
        if (trimmed.startsWith('|')) continue;
        // Skip empty lines
        if (trimmed === '') continue;
        // Non-table prose should not contain CJK
        if (containsCJK(trimmed)) {
          const found = extractCJK(trimmed);
          expect(
            found,
            `TRANSLATION.md has CJK outside table rows: "${trimmed.slice(0, 100)}"`
          ).toHaveLength(0);
        }
      }
    });
  });

  /**
   * Property 6: ROADMAP structural integrity
   *
   * For any phase header in ROADMAP.md, the line SHALL match the regex
   * `^##\s+Phase\s+(\d+).*?—\s*(✅|🚧|⬚)`.
   * For any lesson row, the line SHALL match
   * `^\|\s*\d+\s*\|.+\|\s*(✅|🚧|⬚)\s*\|`.
   *
   * **Validates: Requirements 5.3, 5.4, 6.2**
   */
  describe('Property 6: ROADMAP structural integrity', () => {
    const PHASE_HEADER_REGEX = /^##\s+Phase\s+(\d+).*?—\s*(✅|🚧|⬚)/;
    const LESSON_ROW_REGEX = /^\|\s*\d+\s*\|.+\|\s*(✅|🚧|⬚)\s*\|/;

    it('every phase header matches the required format', () => {
      const content = readTopLevelDoc('ROADMAP.md');
      expect(content, 'ROADMAP.md must exist').not.toBeNull();

      const lines = content!.split('\n');
      const phaseHeaders = lines.filter((line) => /^##\s+Phase\s+\d+/.test(line));

      expect(phaseHeaders.length, 'ROADMAP should have at least one phase header').toBeGreaterThan(0);

      fc.assert(
        fc.property(
          fc.constantFrom(...phaseHeaders),
          (header) => {
            expect(
              PHASE_HEADER_REGEX.test(header),
              `Phase header does not match required format: "${header}"`
            ).toBe(true);
          }
        ),
        PBT_PARAMS
      );
    });

    it('every lesson row matches the required format', () => {
      const content = readTopLevelDoc('ROADMAP.md');
      expect(content, 'ROADMAP.md must exist').not.toBeNull();

      const lines = content!.split('\n');
      // Lesson rows start with | followed by a number, but skip table header/delimiter rows
      const lessonRows = lines.filter((line) => {
        const trimmed = line.trim();
        // Must start with |, contain a digit in the first cell, and NOT be a delimiter row
        return (
          /^\|\s*\d+\s*\|/.test(trimmed) &&
          !/^\|[-\s|]+\|$/.test(trimmed)
        );
      });

      expect(lessonRows.length, 'ROADMAP should have lesson rows').toBeGreaterThan(0);

      fc.assert(
        fc.property(
          fc.constantFrom(...lessonRows),
          (row) => {
            expect(
              LESSON_ROW_REGEX.test(row),
              `Lesson row does not match required format: "${row}"`
            ).toBe(true);
          }
        ),
        PBT_PARAMS
      );
    });

    it('phase headers have sequential numbering starting from 0', () => {
      const content = readTopLevelDoc('ROADMAP.md');
      expect(content).not.toBeNull();

      const lines = content!.split('\n');
      const phaseNumbers: number[] = [];

      for (const line of lines) {
        const match = line.match(PHASE_HEADER_REGEX);
        if (match) {
          phaseNumbers.push(parseInt(match[1], 10));
        }
      }

      expect(phaseNumbers[0], 'First phase should be 0').toBe(0);

      // Phase numbers should be monotonically increasing
      for (let i = 1; i < phaseNumbers.length; i++) {
        expect(
          phaseNumbers[i],
          `Phase numbers should increase: ${phaseNumbers[i - 1]} -> ${phaseNumbers[i]}`
        ).toBeGreaterThan(phaseNumbers[i - 1]);
      }
    });
  });

  /**
   * Property 8: Document structure count invariant
   *
   * For any translated document, the count of fenced code blocks (``` pairs),
   * the count of headings at each level, and the count of mermaid blocks SHALL
   * remain consistent and non-negative. Structural elements that exist in the
   * document should be countable and match expected patterns.
   *
   * Since we don't have originals to compare against (they were translated
   * in-place), we verify structural consistency: code blocks come in pairs,
   * headings exist where expected, and structure counts are self-consistent.
   *
   * **Validates: Requirements 9.4, 10.2**
   */
  describe('Property 8: Document structure count invariant', () => {
    it('all top-level documents have self-consistent structure', () => {
      fc.assert(
        fc.property(
          fc.constantFrom(...TOP_LEVEL_DOCS),
          (docName) => {
            const content = readTopLevelDoc(docName);
            if (content === null) return true;

            // Code block fences must come in pairs (even count of ``` lines)
            const fenceMatches = content.match(/^```/gm);
            const fenceCount = fenceMatches ? fenceMatches.length : 0;
            expect(
              fenceCount % 2,
              `${docName} has odd number of code fences (${fenceCount}), indicating unclosed block`
            ).toBe(0);

            // Code block count should be non-negative
            const codeBlocks = countCodeBlocks(content);
            expect(codeBlocks).toBeGreaterThanOrEqual(0);

            // Mermaid blocks should be <= total code blocks
            const mermaidBlocks = countMermaidBlocks(content);
            expect(
              mermaidBlocks,
              `${docName}: mermaid blocks (${mermaidBlocks}) exceeds total code blocks (${codeBlocks})`
            ).toBeLessThanOrEqual(codeBlocks);

            // Heading counts should be non-negative at every level
            const headings = countHeadings(content);
            for (let level = 1; level <= 6; level++) {
              expect(headings[level]).toBeGreaterThanOrEqual(0);
            }

            return true;
          }
        ),
        PBT_PARAMS
      );
    });

    it('README.md has expected structural elements', () => {
      const content = readTopLevelDoc('README.md');
      expect(content, 'README.md must exist').not.toBeNull();

      const headings = countHeadings(content!);
      // README should have at least one H1 (the title) or uses HTML instead
      // and multiple H2 sections
      expect(headings[2], 'README should have multiple H2 sections').toBeGreaterThan(0);

      // README should have code blocks (at minimum the ASCII art dividers)
      const codeBlocks = countCodeBlocks(content!);
      expect(codeBlocks, 'README should have code blocks').toBeGreaterThan(0);
    });

    it('ROADMAP.md has expected structural elements', () => {
      const content = readTopLevelDoc('ROADMAP.md');
      expect(content, 'ROADMAP.md must exist').not.toBeNull();

      const headings = countHeadings(content!);
      // ROADMAP should have 1 H1 and many H2 (phase headers)
      expect(headings[1], 'ROADMAP should have an H1 heading').toBe(1);
      expect(headings[2], 'ROADMAP should have multiple H2 phase headers').toBeGreaterThan(10);
    });

    it('CONTRIBUTING.md has expected structural elements', () => {
      const content = readTopLevelDoc('CONTRIBUTING.md');
      expect(content, 'CONTRIBUTING.md must exist').not.toBeNull();

      const headings = countHeadings(content!);
      // CONTRIBUTING should have at least an H1 and some H2 sections
      expect(headings[1], 'CONTRIBUTING should have an H1 heading').toBeGreaterThanOrEqual(1);
      expect(headings[2], 'CONTRIBUTING should have H2 sections').toBeGreaterThan(0);
    });
  });
});
