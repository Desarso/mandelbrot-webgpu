/**
 * Choosing an iteration count by measuring the view, not by predicting it.
 *
 * How many iterations a view needs depends on where it is, not only on how
 * deep it is: the period-1215 minibrot at 6e-42 is fully resolved by 14,000,
 * while a plain filament at 1e-25 still shows nothing at 13,000. No function of
 * the zoom alone can know that, so this does not try to. It renders, asks the
 * renderer how many samples used the whole budget, and adjusts.
 *
 * The measurement on its own does not answer the question — a sample that hit
 * the cap is either genuinely inside the set or merely out of iterations, and
 * nothing distinguishes them at a single budget. What does distinguish them is
 * the *change*: doubling the budget converts the second kind and leaves the
 * first alone. So the rule is
 *
 *   raise while raising keeps resolving pixels, lower while lowering costs none
 *
 * which converges on the smallest count that renders the same picture as an
 * unlimited one. It is deterministic: the same view always walks the same path
 * to the same answer.
 */

/** What one render told us. */
export interface Probe {
  iterations: number;
  /** Fraction of samples that used the whole budget, 0..1. */
  capped: number;
}

export type Decision =
  | { action: "try"; iterations: number }
  | { action: "settle"; iterations: number };

/**
 * A change in capped fraction smaller than this counts as no change.
 *
 * This is the quality knob, and it wants to be small. What a budget buys near
 * the top of the range is the thin fringe along the boundary -- a fraction of
 * a percent of the frame, and the most interesting part of it. At one part in
 * 500 the home view gave away three quarters of its budget for a difference
 * the search could not see but the eye can. One part in 5000.
 */
const MEANINGFUL = 0.0002;

/** Steps are powers of two, so a search spans a wide range in few renders. */
const STEP = 2;

/**
 * Above this, treat the frame as telling us nothing except "not enough yet".
 *
 * Escape is a cliff, not a slope. Measured at 1e-25, the whole sampled frame
 * was still capped at 2000 iterations, and at 4000, and at 13000 — and then
 * at 20000 all but 318 of 7004 pixels escaped at once. Between those points
 * doubling the budget changes the measurement by exactly nothing, which the
 * "raising bought no improvement" rule reads as sufficiency. It is the
 * opposite: nothing has escaped yet because the budget is nowhere near
 * enough, and giving it back renders a black screen.
 */
const MOSTLY_CAPPED = 0.98;

/** Never propose fewer than this; cheap views should still look right. */
const MIN_ITERATIONS = 100;

/**
 * Decides what to render next, given the probes taken so far for one view.
 *
 * `probes` is in the order they were taken and must be non-empty. Returns
 * either the next count to try, or the count to settle on.
 */
export function nextIterations(probes: readonly Probe[], limit: number): Decision {
  const clamp = (n: number) =>
    Math.max(MIN_ITERATIONS, Math.min(limit, Math.round(n)));
  const tried = (n: number) => probes.some((p) => p.iterations === n);

  // Capped fraction falls as the budget rises and never rises, so sorting by
  // budget also sorts by how resolved the frame is.
  const sorted = [...probes].sort((a, b) => a.iterations - b.iterations);
  const top = sorted[sorted.length - 1];
  const below = sorted.length >= 2 ? sorted[sorted.length - 2] : null;

  // Almost nothing has escaped. Keep raising regardless of what the last step
  // did or did not buy, because below the escape threshold it buys nothing
  // right up until it buys everything.
  if (top.capped >= MOSTLY_CAPPED) {
    const higher = clamp(top.iterations * STEP);
    if (higher > top.iterations && !tried(higher)) {
      return { action: "try", iterations: higher };
    }
    return { action: "settle", iterations: top.iterations };
  }

  // Is the largest budget tried actually enough? Either nothing hit the cap,
  // or the step up to it converted almost nothing, which means the budget had
  // stopped being the binding constraint before we got there.
  const enough =
    top.capped === 0 || (below !== null && below.capped - top.capped <= MEANINGFUL);

  if (!enough) {
    const higher = clamp(top.iterations * STEP);
    if (higher > top.iterations && !tried(higher)) {
      return { action: "try", iterations: higher };
    }
    return { action: "settle", iterations: top.iterations };
  }

  // It is enough, so the question becomes how much can be given back. Take the
  // cheapest budget that renders the same picture as the most expensive one,
  // and probe below it until that stops being true.
  const best = top.capped;
  const sufficient = sorted.filter(
    (p) => p.capped <= best + MEANINGFUL && p.capped < MOSTLY_CAPPED
  );
  if (sufficient.length === 0) return { action: "settle", iterations: top.iterations };
  const cheapest = sufficient[0];

  const lower = clamp(cheapest.iterations / STEP);
  if (lower < cheapest.iterations && !tried(lower)) {
    return { action: "try", iterations: lower };
  }
  return { action: "settle", iterations: cheapest.iterations };
}
