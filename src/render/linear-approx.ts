/**
 * Linear approximation: skipping ranges of reference iterations.
 *
 * The perturbation step is dz <- 2*X_k*dz + dz^2 + d. While |dz| stays far
 * below |X_k| the quadratic term is negligible and the map is linear, so a
 * whole range of iterations collapses into
 *
 *     dz_{n+m} = A * dz_n + B * d
 *
 * with A = prod 2*X_k and B built by B <- 2*X_k*B + 1. At depth dz starts
 * astronomically small and stays linear for thousands of iterations, so this
 * replaces most of the per-pixel work with a handful of multiplies.
 *
 * Steps are built in levels: level 0 covers BASE_STEP iterations, each higher
 * level merges two steps of the level below. A pixel greedily takes the largest
 * step whose validity radius still contains its |dz|.
 *
 * Everything carries an explicit binary exponent. A over a 4096-iteration step
 * routinely reaches 10^700 and would otherwise overflow to Infinity, while the
 * matching radius underflows to zero — the product is what stays sane, not the
 * factors. Radii are kept as log2 for the same reason.
 */

/** Iterations covered by a level-0 step. */
export const BASE_STEP = 8;

/**
 * How far below 2*|X| the delta must stay at each individual iteration. The
 * neglected term is dz^2 against 2*X*dz, so the relative error contributed by
 * one iteration is about |dz| / (2*|X|).
 */
const TOLERANCE = 2 ** -16;

/**
 * Radius penalty applied on every merge, in log2.
 *
 * Errors accumulate along a step: m iterations at TOLERANCE each give roughly
 * m * TOLERANCE overall. A merged step covers twice as many iterations as its
 * halves, so halving its entry radius holds the total error roughly constant
 * across levels. Without this a 512-iteration step carries 512x the per-step
 * error, which visibly shifts escape counts at shallow zoom — deep views hide
 * it only because their deltas sit far below any radius anyway.
 */
const MERGE_PENALTY_LOG2 = 1;

/** Sentinel log2-radius meaning "this step is never usable". */
export const NEVER = -1e30;

/** Floats per packed entry: Ax, Ay, Ae, Bx, By, Be, radiusLog2, pad. */
export const ENTRY_FLOATS = 8;

export interface LinearApproxTable {
  data: Float32Array;
  levelOffsets: number[];
  levelCounts: number[];
  levels: number;
  entryCount: number;
}

/** A complex number as (x, y) * 2^e, mantissa normalised to [1, 2). */
interface Scaled {
  x: number;
  y: number;
  e: number;
}

function normalise(x: number, y: number, e: number): Scaled {
  const magnitude = Math.max(Math.abs(x), Math.abs(y));
  if (magnitude === 0 || !Number.isFinite(magnitude)) return { x: 0, y: 0, e: 0 };
  const shift = Math.floor(Math.log2(magnitude));
  const scale = 2 ** -shift;
  return { x: x * scale, y: y * scale, e: e + shift };
}

function multiply(a: Scaled, b: Scaled): Scaled {
  return normalise(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x, a.e + b.e);
}

function add(a: Scaled, b: Scaled): Scaled {
  if (a.x === 0 && a.y === 0) return b;
  if (b.x === 0 && b.y === 0) return a;
  const difference = a.e - b.e;
  if (difference > 80) return a;
  if (difference < -80) return b;
  if (difference >= 0) {
    const scale = 2 ** -difference;
    return normalise(a.x + b.x * scale, a.y + b.y * scale, a.e);
  }
  const scale = 2 ** difference;
  return normalise(a.x * scale + b.x, a.y * scale + b.y, b.e);
}

function log2Magnitude(v: Scaled): number {
  const m = Math.hypot(v.x, v.y);
  return m === 0 ? -Infinity : v.e + Math.log2(m);
}

const ONE: Scaled = { x: 1, y: 0, e: 0 };
const ZERO: Scaled = { x: 0, y: 0, e: 0 };

interface Step {
  a: Scaled;
  b: Scaled;
  /** log2 of the largest |dz| that may enter this step. */
  radiusLog2: number;
}

/**
 * Builds the table from reference orbit samples.
 *
 * @param orbit reduced samples, 6 floats per entry (hi, lo, exp per component)
 * @param length valid sample count
 * @param maxDelta largest |d| any pixel will use, so the radius can account for
 *   the B*d term without knowing the pixel
 */
export function buildLinearApprox(
  orbit: Float32Array,
  length: number,
  maxDelta: number,
  maxLevels = 14
): LinearApproxTable {
  // Reference samples are O(1), so plain doubles hold them; only the products
  // above need exponents.
  const count = Math.max(0, length - 1);
  const refX = new Float64Array(count + 1);
  const refY = new Float64Array(count + 1);
  for (let i = 0; i <= count; i++) {
    refX[i] = (orbit[i * 6 + 0] + orbit[i * 6 + 1]) * 2 ** orbit[i * 6 + 2];
    refY[i] = (orbit[i * 6 + 3] + orbit[i * 6 + 4]) * 2 ** orbit[i * 6 + 5];
  }

  const maxDeltaLog2 = maxDelta > 0 ? Math.log2(maxDelta) : -Infinity;
  const levels: Step[][] = [];

  // ---- level 0 -------------------------------------------------------------
  const level0Count = Math.floor(count / BASE_STEP);
  const level0: Step[] = [];
  const toleranceLog2 = Math.log2(TOLERANCE);

  for (let s = 0; s < level0Count; s++) {
    const start = s * BASE_STEP;
    let a = ONE;
    let b = ZERO;

    // Linearity has to hold at *every* step, not just on entry. At step k the
    // travelling delta is dz_k = A_k*dz_0 + B_k*d, so require
    //
    //     |A_k*dz_0| + |B_k*d| < tolerance * |2*X_k|
    //
    // and solve for |dz_0|. Dropping the B_k*d part is what breaks shallow
    // views: there d is large enough to carry the delta out of the linear
    // regime on its own, no matter how small the entry delta is.
    let radiusLog2 = Infinity;

    for (let k = 0; k < BASE_STEP; k++) {
      const twoXValue = Math.hypot(2 * refX[start + k], 2 * refY[start + k]);
      if (twoXValue === 0) {
        radiusLog2 = NEVER;
        break;
      }

      const budgetLog2 = toleranceLog2 + Math.log2(twoXValue);
      const injectedLog2 = log2Magnitude(b) + maxDeltaLog2;
      if (injectedLog2 >= budgetLog2) {
        // The delta term alone already breaks linearity here.
        radiusLog2 = NEVER;
        break;
      }
      const headroomLog2 =
        injectedLog2 === -Infinity
          ? budgetLog2
          : budgetLog2 + Math.log2(1 - 2 ** (injectedLog2 - budgetLog2));

      radiusLog2 = Math.min(radiusLog2, headroomLog2 - log2Magnitude(a));

      const twoX = normalise(2 * refX[start + k], 2 * refY[start + k], 0);
      a = multiply(twoX, a);
      b = add(multiply(twoX, b), ONE);
    }

    level0.push({
      a,
      b,
      radiusLog2: Number.isFinite(radiusLog2) ? radiusLog2 : NEVER,
    });
  }
  levels.push(level0);

  // ---- merge pairs ---------------------------------------------------------
  for (let level = 1; level < maxLevels; level++) {
    const previous = levels[level - 1];
    const mergedCount = Math.floor(previous.length / 2);
    if (mergedCount < 1) break;

    const merged: Step[] = [];
    for (let s = 0; s < mergedCount; s++) {
      const first = previous[2 * s];
      const second = previous[2 * s + 1];

      const a = multiply(second.a, first.a);
      const b = add(multiply(second.a, first.b), second.b);

      // Entering the second half, |dz| has become |A1*dz + B1*d|. That has to
      // land inside the second radius, worst case over the largest |d| in the
      // image. All of it in log2 because |A1| can be 10^700.
      const b1dLog2 = log2Magnitude(first.b) + maxDeltaLog2;
      let radiusLog2 = NEVER;
      if (b1dLog2 < second.radiusLog2) {
        // headroom = r2 - |B1*d|, still in log2.
        const headroomLog2 =
          second.radiusLog2 + Math.log2(1 - 2 ** (b1dLog2 - second.radiusLog2));
        const mapped = headroomLog2 - log2Magnitude(first.a);
        radiusLog2 = Math.min(first.radiusLog2, mapped) - MERGE_PENALTY_LOG2;
      }
      if (!Number.isFinite(radiusLog2)) radiusLog2 = NEVER;

      merged.push({ a, b, radiusLog2 });
    }
    levels.push(merged);
  }

  // ---- pack ----------------------------------------------------------------
  let entryCount = 0;
  for (const level of levels) entryCount += level.length;

  const data = new Float32Array(Math.max(1, entryCount) * ENTRY_FLOATS);
  const levelOffsets: number[] = [];
  const levelCounts: number[] = [];
  let offset = 0;

  for (const level of levels) {
    levelOffsets.push(offset);
    levelCounts.push(level.length);
    level.forEach((step, index) => {
      const target = (offset + index) * ENTRY_FLOATS;
      data[target + 0] = step.a.x;
      data[target + 1] = step.a.y;
      data[target + 2] = step.a.e;
      data[target + 3] = step.b.x;
      data[target + 4] = step.b.y;
      data[target + 5] = step.b.e;
      data[target + 6] = step.radiusLog2;
      data[target + 7] = 0;
    });
    offset += level.length;
  }

  return {
    data,
    levelOffsets,
    levelCounts,
    levels: levels.length,
    entryCount,
  };
}

/** CPU mirror of the shader's step, for tests. Inputs/outputs are Scaled. */
export function applyStep(
  table: LinearApproxTable,
  level: number,
  index: number,
  dz: Scaled,
  delta: Scaled
): Scaled {
  const base = (table.levelOffsets[level] + index) * ENTRY_FLOATS;
  const a: Scaled = {
    x: table.data[base + 0],
    y: table.data[base + 1],
    e: table.data[base + 2],
  };
  const b: Scaled = {
    x: table.data[base + 3],
    y: table.data[base + 4],
    e: table.data[base + 5],
  };
  return add(multiply(a, dz), multiply(b, delta));
}

export function stepRadiusLog2(
  table: LinearApproxTable,
  level: number,
  index: number
): number {
  return table.data[(table.levelOffsets[level] + index) * ENTRY_FLOATS + 6];
}

export const scaled = { normalise, multiply, add, log2Magnitude };
export type { Scaled };
