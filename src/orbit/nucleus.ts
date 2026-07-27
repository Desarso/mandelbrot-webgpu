/**
 * Finding minibrot nuclei — the practical way to reach a deep zoom that has
 * something in it.
 *
 * Zooming by hand past ~1e-30 almost always lands in a smooth region where
 * every pixel escapes within an iteration or two of every other. The
 * interesting places are the nuclei: centres of period-p minibrots, where
 * f_p(c) = 0 for the p-th iterate of z -> z^2 + c starting at z = 0. Newton
 * converges on those quadratically, and the accompanying size estimate says how
 * far to zoom in once you are there.
 *
 * All arithmetic is decimal.js so the result carries as many digits as the
 * caller asks for; a double runs out at ~15.
 */

import Decimal from "decimal.js";
import { Complex } from "../logic/Complex";

export interface Nucleus {
  centerX: Decimal;
  centerY: Decimal;
  /** Period of the minibrot. */
  period: number;
  /** Approximate diameter, i.e. a sensible span to view it at. */
  size: Decimal;
  /** Newton steps used; -1 when it never converged. */
  steps: number;
}

const ZERO = () => new Complex(new Decimal(0), new Decimal(0));
const ONE = () => new Complex(new Decimal(1), new Decimal(0));

function magnitudeSquared(z: Complex): Decimal {
  return z.real.times(z.real).plus(z.imag.times(z.imag));
}

/**
 * Guesses the period of the minibrot near `c`.
 *
 * Walks the orbit and records every iteration where |z| reaches a new minimum.
 * Those indices are the candidate periods (this is the cheap form of the atom
 * domain idea): a point close to a period-p nucleus has its orbit return near
 * the origin every p steps.
 */
export function detectPeriod(c: Complex, maxPeriod: number): number[] {
  let z = ZERO();
  let smallest: Decimal | null = null;
  const candidates: number[] = [];

  for (let i = 1; i <= maxPeriod; i++) {
    z = z.square().add(c);
    const magnitude = magnitudeSquared(z);
    if (magnitude.greaterThan(4)) break;
    if (smallest === null || magnitude.lessThan(smallest)) {
      smallest = magnitude;
      candidates.push(i);
    }
  }
  // Largest first: deeper periods sit at smaller scales, which is what a deep
  // zoom wants.
  return candidates.reverse();
}

/**
 * Newton's method on f_p(c) = 0.
 *
 * f_p is the p-th iterate evaluated at z_0 = 0, and its derivative with respect
 * to c follows the companion recurrence dz <- 2*z*dz + 1.
 */
export function newtonNucleus(
  start: Complex,
  period: number,
  maxSteps = 64
): { c: Complex; steps: number; converged: boolean } {
  let c = start;

  for (let step = 1; step <= maxSteps; step++) {
    let z = ZERO();
    let dz = ZERO();

    for (let i = 0; i < period; i++) {
      // dz must be updated with the previous z, before z advances.
      dz = new Complex(new Decimal(2), new Decimal(0)).times(z).times(dz).add(ONE());
      z = z.square().add(c);
      if (magnitudeSquared(z).greaterThan(new Decimal(1e12))) {
        return { c, steps: step, converged: false };
      }
    }

    if (magnitudeSquared(dz).isZero()) {
      return { c, steps: step, converged: false };
    }

    const delta = z.divide(dz);
    c = c.minus(delta);

    // Converged once the step is far below the working precision.
    const tolerance = new Decimal(10).pow(-(Decimal.precision - 6));
    if (magnitudeSquared(delta).lessThan(tolerance.times(tolerance))) {
      return { c, steps: step, converged: true };
    }
  }
  return { c, steps: maxSteps, converged: false };
}

/**
 * Diameter of the period-p minibrot at nucleus `c`.
 *
 * Standard estimate: accumulate the derivative product l = prod 2*z_i and the
 * correction b = 1 + sum 1/l, then size = 1/(b * l^2). Sanity check: the
 * period-2 bulb at c = -1 comes out at 0.5, its actual diameter.
 */
export function sizeEstimate(c: Complex, period: number): Decimal {
  let z = ZERO();
  let l = ONE();
  let b = ONE();
  const two = new Complex(new Decimal(2), new Decimal(0));

  for (let i = 1; i < period; i++) {
    z = z.square().add(c);
    l = two.times(z).times(l);
    if (magnitudeSquared(l).isZero()) return new Decimal(0);
    b = b.add(ONE().divide(l));
  }

  const denominator = b.times(l).times(l);
  if (magnitudeSquared(denominator).isZero()) return new Decimal(0);
  return ONE().divide(denominator).abs();
}

export interface FindOptions {
  maxPeriod?: number;
  /** Working precision in decimal digits. */
  digits?: number;
  /** Skip nuclei bigger than this; use to force a deeper result. */
  maxSize?: Decimal;
}

/**
 * Finds a minibrot nucleus near `start`.
 *
 * Every candidate period is tried and the *smallest* converged nucleus wins,
 * rather than the first: Newton diverges for many candidates, and taking the
 * first that happens to converge tends to land on a big shallow bulb. Starting
 * from a point already on the boundary at depth gives the deepest results —
 * from far away, Newton simply converges to the nearest large nucleus.
 */
export function findNucleus(
  startX: Decimal,
  startY: Decimal,
  options: FindOptions = {}
): Nucleus | null {
  const maxPeriod = options.maxPeriod ?? 4000;
  const digits = options.digits ?? 60;

  const previousPrecision = Decimal.precision;
  Decimal.set({ precision: digits });

  try {
    const start = new Complex(startX, startY);
    const periods = detectPeriod(start, maxPeriod);
    let best: Nucleus | null = null;

    for (const period of periods) {
      // Period 1 is the main cardioid; nothing to zoom into.
      if (period < 2) continue;

      const result = newtonNucleus(start, period);
      if (!result.converged) continue;

      const size = sizeEstimate(result.c, period);
      if (size.isZero() || !size.isFinite()) continue;
      if (options.maxSize && size.greaterThan(options.maxSize)) continue;

      if (!best || size.lessThan(best.size)) {
        best = {
          centerX: result.c.real,
          centerY: result.c.imag,
          period,
          size,
          steps: result.steps,
        };
      }
    }
    return best;
  } finally {
    Decimal.set({ precision: previousPrecision });
  }
}
