/**
 * Structural validation utilities for markdown document analysis.
 *
 * Provides helpers to count and verify structural elements like headings,
 * code blocks, paragraphs, and list items. Used by property-based tests
 * to verify the document structure count invariant (Property 8).
 */

export interface DocumentStructure {
  /** Count of headings at each level (1–6). */
  headings: Record<number, number>;
  /** Total number of fenced code blocks. */
  codeBlockCount: number;
  /** Total number of mermaid blocks. */
  mermaidBlockCount: number;
  /** Total number of non-empty paragraphs (prose lines separated by blank lines). */
  paragraphCount: number;
  /** Total number of bullet list items. */
  bulletListItems: number;
  /** Total number of numbered list items. */
  numberedListItems: number;
}

/**
 * Counts fenced code blocks in markdown content.
 * A code block is delimited by lines starting with ```.
 */
export function countCodeBlocks(markdown: string): number {
  const matches = markdown.match(/^```/gm);
  // Each code block has an opening and closing fence
  return matches ? Math.floor(matches.length / 2) : 0;
}

/**
 * Counts mermaid diagram blocks in markdown content.
 */
export function countMermaidBlocks(markdown: string): number {
  const matches = markdown.match(/^```mermaid/gm);
  return matches ? matches.length : 0;
}

/**
 * Counts headings at each level (1–6) in markdown content.
 * Only counts ATX-style headings (lines starting with #).
 * Headings inside code blocks are excluded.
 */
export function countHeadings(markdown: string): Record<number, number> {
  const counts: Record<number, number> = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0 };

  // Remove code blocks first
  const withoutCode = markdown.replace(/^```[^\n]*\n[\s\S]*?^```/gm, '');

  const lines = withoutCode.split('\n');
  for (const line of lines) {
    const match = line.match(/^(#{1,6})\s+/);
    if (match) {
      const level = match[1].length;
      counts[level]++;
    }
  }

  return counts;
}

/**
 * Counts bullet list items (lines starting with - or * followed by space).
 * Items inside code blocks are excluded.
 */
export function countBulletListItems(markdown: string): number {
  const withoutCode = markdown.replace(/^```[^\n]*\n[\s\S]*?^```/gm, '');
  const matches = withoutCode.match(/^[\t ]*[-*]\s+/gm);
  return matches ? matches.length : 0;
}

/**
 * Counts numbered list items (lines starting with digits followed by . or )).
 * Items inside code blocks are excluded.
 */
export function countNumberedListItems(markdown: string): number {
  const withoutCode = markdown.replace(/^```[^\n]*\n[\s\S]*?^```/gm, '');
  const matches = withoutCode.match(/^[\t ]*\d+[.)]\s+/gm);
  return matches ? matches.length : 0;
}

/**
 * Counts paragraphs in markdown content.
 * A paragraph is a group of consecutive non-empty, non-structural lines
 * separated by blank lines. Headings, list items, and code blocks are excluded.
 */
export function countParagraphs(markdown: string): number {
  // Remove code blocks
  const withoutCode = markdown.replace(/^```[^\n]*\n[\s\S]*?^```/gm, '\n');
  const lines = withoutCode.split('\n');

  let count = 0;
  let inParagraph = false;

  for (const line of lines) {
    const trimmed = line.trim();
    const isBlank = trimmed === '';
    const isHeading = /^#{1,6}\s+/.test(trimmed);
    const isListItem = /^[-*]\s+/.test(trimmed) || /^\d+[.)]\s+/.test(trimmed);
    const isTableRow = /^\|/.test(trimmed);
    const isHR = /^---+$/.test(trimmed) || /^\*\*\*+$/.test(trimmed);
    const isBlockquote = /^>/.test(trimmed);

    const isProse = !isBlank && !isHeading && !isListItem && !isTableRow && !isHR && !isBlockquote;

    if (isProse && !inParagraph) {
      count++;
      inParagraph = true;
    } else if (!isProse) {
      inParagraph = false;
    }
  }

  return count;
}

/**
 * Analyzes a markdown document and returns a complete structural summary.
 */
export function analyzeStructure(markdown: string): DocumentStructure {
  return {
    headings: countHeadings(markdown),
    codeBlockCount: countCodeBlocks(markdown),
    mermaidBlockCount: countMermaidBlocks(markdown),
    paragraphCount: countParagraphs(markdown),
    bulletListItems: countBulletListItems(markdown),
    numberedListItems: countNumberedListItems(markdown),
  };
}

/**
 * Approved English section headings for lesson documents.
 */
export const APPROVED_SECTION_HEADINGS = new Set([
  'Learning Objectives',
  'The Problem',
  'The Concept',
  'Build It',
  'Use It',
  'Ship It',
  'Exercises',
  'Key Terms',
  'Further Reading',
  'Pitfalls',
  'Connections',
]);

/**
 * Checks if all ## headings in a lesson document use approved section headings.
 * Returns an array of non-approved headings found.
 */
export function findNonApprovedHeadings(markdown: string): string[] {
  const withoutCode = markdown.replace(/^```[^\n]*\n[\s\S]*?^```/gm, '');
  const lines = withoutCode.split('\n');
  const invalid: string[] = [];

  for (const line of lines) {
    const match = line.match(/^##\s+(.+)/);
    if (match) {
      const heading = match[1].trim();
      if (!APPROVED_SECTION_HEADINGS.has(heading)) {
        invalid.push(heading);
      }
    }
  }

  return invalid;
}
