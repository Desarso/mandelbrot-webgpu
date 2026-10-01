/**
 * The skip-table builder as it was first written: one object per complex
 * value, every operation allocating. Kept as the oracle for the allocation-free
 * builder in src/render/bla.ts, which has to produce the same table bit for
 * bit -- it performs the same floating-point operations in the same order.
 */

import {
  BASE_STEP,
  ENTRY_FLOATS,
  NEVER,
  add,
  compose,
  log2Magnitude,
  multiply,
  normalise,
  scale,
  type BlaTable,
  type BuildOptions,
  type Scaled,
  type Step,
} from "../../src/render/bla";

const TOLERANCE = 2 ** -16;
const MERGE_PENALTY_LOG2 = 1;
const ONE: Scaled = { x: 1, y: 0, e: 0 };
const ZERO: Scaled = { x: 0, y: 0, e: 0 };

/**
 * Builds the skip table from reference orbit samples.
 *
 * @param orbit reduced samples, 6 floats per entry (hi, lo, exp per component)
 * @param length valid sample count
 * @param maxDelta largest |d| any pixel will use, so radii can account for the
 *   terms `d` injects without knowing the pixel
 */
export function buildBlaReference(
  orbit: Float32Array,
  length: number,
  maxDelta: number,
  options: BuildOptions = {}
): BlaTable {
  const maxLevels = options.maxLevels ?? 14;
  const quadratic = options.quadratic ?? true;

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
  const toleranceLog2 = Math.log2(TOLERANCE);
  const levels: Step[][] = [];

  // ---- level 0 -------------------------------------------------------------
  const level0Count = Math.floor(count / BASE_STEP);
  const level0: Step[] = [];

  for (let s = 0; s < level0Count; s++) {
    const start = s * BASE_STEP;
    let a = ONE;
    let b = ZERO;
    let c = ZERO;
    let d = ZERO;
    let e = ZERO;

    // Linearity has to hold at *every* iteration, not just on entry. At step k
    // the travelling delta is A_k*w + B_k*d (+ higher order), so require the
    // first neglected term to stay below tolerance * |2*X_k|.
    //
    // Without second-order terms the neglected term is quadratic and the bound
    // is on |w| directly. With them it is cubic, so the same tolerance permits
    // an entry radius larger by roughly 1/sqrt(tolerance).
    let radiusLog2 = Infinity;

    for (let k = 0; k < BASE_STEP; k++) {
      const twoXValue = Math.hypot(2 * refX[start + k], 2 * refY[start + k]);
      if (twoXValue === 0) {
        radiusLog2 = NEVER;
        break;
      }

      // Bound on the travelling delta |w_k| at this iteration.
      //
      // Truncating at first order leaves |w_k|^2 against the retained
      // |2*X_k*w_k|, so |w_k| <= tolerance * |2*X_k|. Truncating at second
      // order leaves |w_k|^3, so |w_k| <= sqrt(tolerance * |2*X_k|) — the same
      // tolerance, but a square root larger.
      const linearBound = toleranceLog2 + Math.log2(twoXValue);
      const budgetLog2 = quadratic ? linearBound / 2 : linearBound;

      // Part of that budget is already spent by the B_k*d term, which exists
      // no matter how small the entry delta is.
      const injectedLog2 = log2Magnitude(b) + maxDeltaLog2;
      if (injectedLog2 >= budgetLog2) {
        radiusLog2 = NEVER;
        break;
      }
      const headroomLog2 =
        injectedLog2 === -Infinity
          ? budgetLog2
          : budgetLog2 + Math.log2(1 - 2 ** (injectedLog2 - budgetLog2));

      // w_k = A_k * w, so divide through by |A_k|.
      radiusLog2 = Math.min(radiusLog2, headroomLog2 - log2Magnitude(a));

      const twoX = normalise(2 * refX[start + k], 2 * refY[start + k], 0);
      // Second-order updates use the *previous* A and B, so compute them first.
      const nextC = add(multiply(twoX, c), multiply(a, a));
      const nextD = add(multiply(twoX, d), scale(multiply(a, b), 2));
      const nextE = add(multiply(twoX, e), multiply(b, b));
      a = multiply(twoX, a);
      b = add(multiply(twoX, b), ONE);
      if (quadratic) {
        c = nextC;
        d = nextD;
        e = nextE;
      }
    }

    level0.push({
      a,
      b,
      c,
      d,
      e,
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
      const composed = compose(first, second);

      // Entering the second half the delta has become |A1*w + B1*d|, which has
      // to land inside the second radius, worst case over the largest |d|.
      const b1dLog2 = log2Magnitude(first.b) + maxDeltaLog2;
      let radiusLog2 = NEVER;
      if (b1dLog2 < second.radiusLog2) {
        const headroomLog2 =
          second.radiusLog2 + Math.log2(1 - 2 ** (b1dLog2 - second.radiusLog2));
        const mapped = headroomLog2 - log2Magnitude(first.a);
        radiusLog2 = Math.min(first.radiusLog2, mapped) - MERGE_PENALTY_LOG2;
      }
      if (!Number.isFinite(radiusLog2)) radiusLog2 = NEVER;

      merged.push({ ...composed, radiusLog2 });
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
      const put = (slot: number, v: Scaled) => {
        data[target + slot * 3 + 0] = v.x;
        data[target + slot * 3 + 1] = v.y;
        data[target + slot * 3 + 2] = v.e;
      };
      put(0, step.a);
      put(1, step.b);
      put(2, step.c);
      put(3, step.d);
      put(4, step.e);
      data[target + 15] = step.radiusLog2;
    });
    offset += level.length;
  }

  return {
    data,
    levelOffsets,
    levelCounts,
    levels: levels.length,
    entryCount,
    quadratic,
  };
}

