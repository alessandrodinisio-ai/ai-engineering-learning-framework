import { describe, it, expect } from 'vitest';
import { fc } from './helpers.js';
import {
  containsCJK,
  countCJK,
  extractCJK,
  isCJKChar,
} from './utils/cjk-detection.js';
import {
  extractCodeBlocks,
  extractInlineCode,
  extractURLs,
  extractProseSections,
} from './utils/segment-parsing.js';
import {
  countCodeBlocks,
  countHeadings,
  countMermaidBlocks,
  analyzeStructure,
  APPROVED_SECTION_HEADINGS,
} from './utils/structural-validation.js';

describe('Test utilities setup verification', () => {
  describe('CJK detection', () => {
    it('detects CJK characters', () => {
      expect(containsCJK('你好世界')).toBe(true);
      expect(containsCJK('Hello World')).toBe(false);
      expect(containsCJK('Mix 混合 text')).toBe(true);
    });

    it('counts CJK characters', () => {
      expect(countCJK('你好')).toBe(2);
      expect(countCJK('Hello')).toBe(0);
      expect(countCJK('Hello 你好 World')).toBe(2);
    });

    it('extracts CJK sequences', () => {
      expect(extractCJK('Hello 你好 World 世界')).toEqual(['你好', '世界']);
      expect(extractCJK('No CJK here')).toEqual([]);
    });

    it('identifies individual CJK characters', () => {
      expect(isCJKChar('你')).toBe(true);
      expect(isCJKChar('A')).toBe(false);
      expect(isCJKChar('ab')).toBe(false);
    });
  });

  describe('Segment parsing', () => {
    it('extracts code blocks', () => {
      const md = '# Title\n\n```python\nprint("hello")\n```\n\nSome text\n\n```js\nconsole.log("hi")\n```\n';
      const blocks = extractCodeBlocks(md);
      expect(blocks).toHaveLength(2);
      expect(blocks[0].type).toBe('code_block');
      expect(blocks[0].content).toContain('print("hello")');
      expect(blocks[1].content).toContain('console.log("hi")');
    });

    it('identifies mermaid blocks', () => {
      const md = '# Diagram\n\n```mermaid\nflowchart TD\n  A-->B\n```\n';
      const blocks = extractCodeBlocks(md);
      expect(blocks).toHaveLength(1);
      expect(blocks[0].type).toBe('mermaid_block');
    });

    it('extracts inline code', () => {
      const md = 'Use `npm install` to install `vitest` for testing.';
      const spans = extractInlineCode(md);
      expect(spans).toHaveLength(2);
      expect(spans[0].content).toBe('`npm install`');
      expect(spans[1].content).toBe('`vitest`');
    });

    it('extracts URLs', () => {
      const md = 'Visit https://github.com/user/repo and http://example.com for more.';
      const urls = extractURLs(md);
      expect(urls).toHaveLength(2);
      expect(urls[0].content).toBe('https://github.com/user/repo');
      expect(urls[1].content).toBe('http://example.com');
    });

    it('extracts prose sections without code or URLs', () => {
      const md = '# Hello\n\nSome prose here.\n\n```python\ncode()\n```\n\nMore prose with `inline` code.';
      const prose = extractProseSections(md);
      expect(prose).toContain('# Hello');
      expect(prose).toContain('Some prose here.');
      expect(prose.some((p) => p.includes('code()'))).toBe(false);
    });
  });

  describe('Structural validation', () => {
    it('counts code blocks', () => {
      const md = '```py\ncode\n```\n\ntext\n\n```js\nmore\n```\n';
      expect(countCodeBlocks(md)).toBe(2);
    });

    it('counts mermaid blocks', () => {
      const md = '```mermaid\nflowchart\n```\n\n```python\ncode\n```\n';
      expect(countMermaidBlocks(md)).toBe(1);
    });

    it('counts headings by level', () => {
      const md = '# H1\n## H2\n## H2 again\n### H3\n';
      const counts = countHeadings(md);
      expect(counts[1]).toBe(1);
      expect(counts[2]).toBe(2);
      expect(counts[3]).toBe(1);
    });

    it('analyzes full document structure', () => {
      const md = '# Title\n\nParagraph one.\n\n## Section\n\n```js\ncode\n```\n\nParagraph two.\n';
      const structure = analyzeStructure(md);
      expect(structure.headings[1]).toBe(1);
      expect(structure.headings[2]).toBe(1);
      expect(structure.codeBlockCount).toBe(1);
    });

    it('has the approved section headings set', () => {
      expect(APPROVED_SECTION_HEADINGS.has('Learning Objectives')).toBe(true);
      expect(APPROVED_SECTION_HEADINGS.has('The Problem')).toBe(true);
      expect(APPROVED_SECTION_HEADINGS.has('Random Heading')).toBe(false);
    });
  });

  describe('fast-check integration', () => {
    it('runs a basic property test', () => {
      fc.assert(
        fc.property(fc.string(), (s) => {
          // CJK detection should never throw
          const result = containsCJK(s);
          return typeof result === 'boolean';
        })
      );
    });
  });
});
