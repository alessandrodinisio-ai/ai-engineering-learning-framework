/**
 * CJK character detection utilities for translation validation.
 *
 * The CJK Unified Ideographs block (U+4E00–U+9FFF) covers the vast majority
 * of Simplified Chinese characters used in the source documents.
 */

/** Regex matching a single CJK Unified Ideograph (U+4E00–U+9FFF). */
export const CJK_REGEX = /[\u4E00-\u9FFF]/;

/** Regex matching one or more consecutive CJK characters (global). */
export const CJK_GLOBAL_REGEX = /[\u4E00-\u9FFF]+/g;

/**
 * Returns true if the string contains at least one CJK character.
 */
export function containsCJK(text: string): boolean {
  return CJK_REGEX.test(text);
}

/**
 * Returns all CJK character sequences found in the given text.
 */
export function extractCJK(text: string): string[] {
  return text.match(CJK_GLOBAL_REGEX) ?? [];
}

/**
 * Returns true if the character at the given index is a CJK character.
 */
export function isCJKChar(char: string): boolean {
  if (char.length !== 1) return false;
  const code = char.charCodeAt(0);
  return code >= 0x4e00 && code <= 0x9fff;
}

/**
 * Counts the number of CJK characters in a string.
 */
export function countCJK(text: string): number {
  let count = 0;
  for (const char of text) {
    if (isCJKChar(char)) count++;
  }
  return count;
}
