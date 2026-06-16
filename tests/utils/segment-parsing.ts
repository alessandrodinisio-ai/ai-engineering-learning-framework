/**
 * Segment parsing utilities for markdown translation validation.
 *
 * Identifies code blocks, inline code, URLs, and prose sections within
 * markdown content. Used by property-based tests to verify that
 * non-translatable elements are preserved unchanged.
 */

export type SegmentType =
  | 'code_block'
  | 'mermaid_block'
  | 'inline_code'
  | 'url'
  | 'prose';

export interface Segment {
  type: SegmentType;
  content: string;
  startIndex: number;
  endIndex: number;
}

/**
 * Extracts all fenced code blocks (``` ... ```) from markdown content.
 * Returns an array of objects with the block content and position.
 */
export function extractCodeBlocks(markdown: string): Segment[] {
  const segments: Segment[] = [];
  const regex = /^```[^\n]*\n[\s\S]*?^```/gm;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(markdown)) !== null) {
    const content = match[0];
    const isMermaid = content.startsWith('```mermaid');
    segments.push({
      type: isMermaid ? 'mermaid_block' : 'code_block',
      content,
      startIndex: match.index,
      endIndex: match.index + content.length,
    });
  }

  return segments;
}

/**
 * Extracts all inline code spans (single backtick) from markdown content.
 * Does not match inside fenced code blocks.
 */
export function extractInlineCode(markdown: string): Segment[] {
  // First, remove fenced code blocks to avoid matching backticks inside them
  const withoutBlocks = markdown.replace(/^```[^\n]*\n[\s\S]*?^```/gm, (m) =>
    ' '.repeat(m.length)
  );

  const segments: Segment[] = [];
  const regex = /`([^`\n]+)`/g;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(withoutBlocks)) !== null) {
    segments.push({
      type: 'inline_code',
      content: match[0],
      startIndex: match.index,
      endIndex: match.index + match[0].length,
    });
  }

  return segments;
}

/**
 * Extracts all URLs (http/https) from markdown content.
 */
export function extractURLs(markdown: string): Segment[] {
  const segments: Segment[] = [];
  const regex = /https?:\/\/[^\s)>\]]+/g;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(markdown)) !== null) {
    segments.push({
      type: 'url',
      content: match[0],
      startIndex: match.index,
      endIndex: match.index + match[0].length,
    });
  }

  return segments;
}

/**
 * Extracts prose sections from markdown (everything that is NOT inside
 * code blocks, inline code, or URLs). Returns the text content of each
 * prose section.
 */
export function extractProseSections(markdown: string): string[] {
  // Replace code blocks with whitespace placeholders
  let prose = markdown.replace(/^```[^\n]*\n[\s\S]*?^```/gm, (m) =>
    '\n'.repeat(m.split('\n').length - 1)
  );

  // Replace inline code with whitespace placeholders
  prose = prose.replace(/`[^`\n]+`/g, (m) => ' '.repeat(m.length));

  // Replace URLs with whitespace placeholders
  prose = prose.replace(/https?:\/\/[^\s)>\]]+/g, (m) => ' '.repeat(m.length));

  // Split into non-empty lines and return
  return prose
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

/**
 * Checks if a line is a fenced code block delimiter (``` with optional language).
 */
export function isCodeFence(line: string): boolean {
  return /^```/.test(line.trim());
}

/**
 * Determines if a given position in the markdown is inside a code block.
 */
export function isInsideCodeBlock(markdown: string, position: number): boolean {
  const before = markdown.slice(0, position);
  const fences = before.match(/^```/gm);
  // If odd number of fences before position, we're inside a code block
  return fences !== null && fences.length % 2 === 1;
}
