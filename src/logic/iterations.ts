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
 * The curve is fitted to measurements, not guessed. `/gpu.html?…&sweep=1`
 * renders one view at a range of iteration counts and reports how many sampled
 * pixels are still unresolved at each; the count stops mattering once raising
 * it stops converting them. At the default centre, 7004 sampled pixels:
 *
 *   span     depth   still interior at 200 / 800 / 3200 / 12800   settles at
 *   1e-5      5.45   3543      667       598        589              ~1600
 *   1e-12    12.45   7004     7004       329        274              ~6400
 *   1e-25    25.45   7004     7004      7004       7004(*)          ~20000
 *
 *   (*) at 1e-25 nothing at all escapes below ~16000: the whole frame is one
 *       flat interior colour. Undershooting does not produce a rough image,
 *       it produces no image.
 *
 * Those three fit 99 * depth^1.64 almost exactly. What the fit cannot capture
 * is that the requirement depends on the location as much as the depth: the
 * period-1215 minibrot at 6e-42 settles at 14000, where this curve predicts
 * three times that. Finding a nucleus sets the count from its period instead,
 * which is the principled answer for those views.
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

/** Enough for the widest views, where the curve itself is near zero. */
const FLOOR = 500;
const SCALE = 99;
const EXPONENT = 1.64;

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
  const raw = Math.max(FLOOR, SCALE * Math.pow(zoomDepth(span), EXPONENT));
  const magnitude = Math.pow(10, Math.max(0, Math.floor(Math.log10(raw)) - 1));
  const rounded = Math.round(raw / magnitude) * magnitude;
  return Math.min(limit, Math.max(100, rounded));
}
