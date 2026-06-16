/**
 * Shared test helpers and re-exports for property-based tests.
 */
import fc from 'fast-check';

export { fc };

/**
 * Default fast-check parameters for translation property tests.
 * Minimum 100 iterations per the design document.
 */
export const PBT_PARAMS: fc.Parameters<unknown> = {
  numRuns: 100,
};
