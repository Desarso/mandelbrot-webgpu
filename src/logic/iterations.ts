/**
 * How many iterations a view at a given zoom is likely to need.
 *
 * There is no exact answer: the count a view needs depends on what is in it,
 * not only on how deep it is. A filament near a high-period minibrot can need
 * ten times what its neighbour does at the same scale. What is true in general
 * is that the requirement grows steadily with depth, and that guessing far too
 * low is much worse than guessing high — too low renders the interior as a
 * flat blob and hides the structure entirely, whereas too high only costs
 * time.
 *
 * The curve below is fitted to the hand-chosen locations in `locations.ts`,
 * which are views someone looked at and picked an iteration count for:
 *
 *   span     depth   chosen   curve
 *   2.8       0.0      500      420
 *   0.02      2.1     1000     1160
 *   1e-6      6.4     4000     2900
 *   3e-11    11.0     6000     4800
 *   6e-42    41.7    20000    19800
 *
 * It runs deliberately below the hand-picked values in the middle of the range
 * and meets them at the ends. Auto is a starting point, not a replacement for
 * turning the slider up when a particular view wants more.
 */

import Decimal from "decimal.js";

/**
 * Range of the manual slider. Not a hard ceiling on what the renderer will
 * do — auto is deliberately unbounded, because clamping the count at depth
 * does not make the frame cheap, it makes it wrong: every pixel hits the cap
 * and the view reads as solid interior.
 */
export const MAX_ITERATIONS = 200000;

/** Widest span the view supports, i.e. depth 0. */
const HOME_SPAN = 2.8;

const BASE = 420;
const SCALE = 350;
const EXPONENT = 1.08;

/**
 * Decimal orders of magnitude of zoom past the home view. `span` is a Decimal
 * because at depth it is far outside the range of a double.
 */
export function zoomDepth(span: Decimal): number {
  if (span.lessThanOrEqualTo(0)) return 0;
  // Decimal.log(10) rather than toNumber(): 1e-400 is simply 0 as a double.
  const depth = new Decimal(HOME_SPAN).div(span).log(10).toNumber();
  return Number.isFinite(depth) ? Math.max(0, depth) : 0;
}

/**
 * Suggested iteration count for a span, clamped to `limit`.
 *
 * Rounded to two significant figures: the underlying estimate is not accurate
 * enough to justify showing "5,847", and a round number reads as the guess it
 * is rather than as a measurement.
 */
export function iterationsForSpan(span: Decimal, limit = Infinity): number {
  const raw = BASE + SCALE * Math.pow(zoomDepth(span), EXPONENT);
  const magnitude = Math.pow(10, Math.max(0, Math.floor(Math.log10(raw)) - 1));
  const rounded = Math.round(raw / magnitude) * magnitude;
  return Math.min(limit, Math.max(100, rounded));
}
