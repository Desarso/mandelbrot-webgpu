/**
 * Bilinear approximation, to second order.
 *
 * The perturbation step is `w <- 2*X_k*w + w^2 + d`, linear in the delta `w`
 * and in the pixel offset `d` apart from that one quadratic term. Expanding a
 * whole range of iterations as a polynomial in `(w, d)` and truncating gives
 *
 *     w_out = A*w + B*d + C*w^2 + D*w*d + E*d^2
 *
 * Keeping only `A` and `B` is the usual "BLA" — bilinear in the two inputs —
 * and its validity ends where the neglected `w^2` starts to matter. Carrying
 * the second-order coefficients moves that boundary out by roughly the square
 * root of the tolerance, so far more pixels take far longer skips.
 *
 * Per-iteration recurrences, from substituting the expansion into the step:
 *
 *     A' = 2X*A            B' = 2X*B + 1
 *     C' = 2X*C + A^2      D' = 2X*D + 2AB      E' = 2X*E + B^2
 *
 * Composing two ranges substitutes one polynomial into the other and drops
 * anything above second order; see `compose`.
 *
 * Everything carries an explicit binary exponent: `A` over a few thousand
 * iterations reaches 10^700 and would otherwise overflow to Infinity, while
 * the matching radius underflows to zero. Radii are stored as log2.
 */

/** Iterations covered by a level-0 step. */
export const BASE_STEP = 8;

/**
 * How far below `2*|X|` the delta must stay at each iteration.
 *
 * With second-order terms the leading neglected term is cubic, so the same
 * tolerance buys a much larger radius than it did for the linear form.
 */
export const TOLERANCE = 2 ** -16;

/**
 * Radius penalty per merge, in log2. Errors accumulate along a step, and a
 * merged step covers twice as many iterations as its halves.
 */
export const MERGE_PENALTY_LOG2 = 1;

/** Sentinel log2-radius meaning "this step is never usable". */
export const NEVER = -1e30;

/**
 * Floats per packed entry: five complex coefficients as (x, y, exponent),
 * then radiusLog2, then padding to a multiple of four.
 */
export const ENTRY_FLOATS = 20;

export interface BlaTable {
  data: Float32Array;
  levelOffsets: number[];
  levelCounts: number[];
  levels: number;
  entryCount: number;
  /** True when second-order coefficients are populated. */
  quadratic: boolean;
}

/** A complex number as (x, y) * 2^e, mantissa normalised to [1, 2). */
export interface Scaled {
  x: number;
  y: number;
  e: number;
}

export function normalise(x: number, y: number, e: number): Scaled {
  const magnitude = Math.max(Math.abs(x), Math.abs(y));
  if (magnitude === 0 || !Number.isFinite(magnitude)) return { x: 0, y: 0, e: 0 };
  const shift = Math.floor(Math.log2(magnitude));
  const scale = 2 ** -shift;
  return { x: x * scale, y: y * scale, e: e + shift };
}

export function multiply(a: Scaled, b: Scaled): Scaled {
  return normalise(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x, a.e + b.e);
}

export function add(a: Scaled, b: Scaled): Scaled {
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

export function scale(a: Scaled, factor: number): Scaled {
  return normalise(a.x * factor, a.y * factor, a.e);
}

export function log2Magnitude(v: Scaled): number {
  const m = Math.hypot(v.x, v.y);
  return m === 0 ? -Infinity : v.e + Math.log2(m);
}

const ONE: Scaled = { x: 1, y: 0, e: 0 };
const ZERO: Scaled = { x: 0, y: 0, e: 0 };

/** One skip: the truncated polynomial plus the entry radius it is valid for. */
export interface Step {
  a: Scaled;
  b: Scaled;
  c: Scaled;
  d: Scaled;
  e: Scaled;
  radiusLog2: number;
}

/**
 * Substitutes `first` into `second`, keeping terms up to second order.
 *
 * With `w1 = A1*w + B1*d + C1*w^2 + D1*w*d + E1*d^2`, the composite is
 * `A2*w1 + B2*d + C2*w1^2 + D2*w1*d + E2*d^2`, and `w1^2` only contributes at
 * second order through its own linear part.
 */
export function compose(first: Step, second: Step): Omit<Step, "radiusLog2"> {
  const a1 = first.a;
  const b1 = first.b;
  const a2 = second.a;
  const c2 = second.c;
  const d2 = second.d;

  return {
    a: multiply(a2, a1),
    b: add(multiply(a2, b1), second.b),
    // C = A2*C1 + C2*A1^2
    c: add(multiply(a2, first.c), multiply(c2, multiply(a1, a1))),
    // D = A2*D1 + 2*C2*A1*B1 + D2*A1
    d: add(
      add(multiply(a2, first.d), scale(multiply(c2, multiply(a1, b1)), 2)),
      multiply(d2, a1)
    ),
    // E = A2*E1 + C2*B1^2 + D2*B1 + E2
    e: add(
      add(multiply(a2, first.e), multiply(c2, multiply(b1, b1))),
      add(multiply(d2, b1), second.e)
    ),
  };
}

export interface BuildOptions {
  maxLevels?: number;
  /** Set false to drop the second-order terms, for A/B comparison. */
  quadratic?: boolean;
}

// ---------------------------------------------------------------------------
// The builder below runs the same arithmetic as normalise/multiply/add/scale
// and compose, in the same order, but on numbers held in Float64Arrays rather
// than one object per value. A 200,000-iteration orbit took 300ms to tabulate
// with an allocation per operation, all of it on the main thread. The output
// is bit-identical; tests/render/bla-reference.ts keeps the object version as
// the oracle that holds it to that.
//
// A step is STEP_WORDS doubles: A, B, C, D, E as (x, y, e), then radiusLog2.

const STEP_WORDS = 16;
const A = 0;
const B = 3;
const C = 6;
const D = 9;
const E = 12;
const RADIUS = 15;

/**
 * `2 ** k` for every exponent the builder can ask for, made with that same
 * expression so a lookup returns exactly what computing it would.
 */
const POW2_BIAS = 1200;
const POW2 = new Float64Array(2 * POW2_BIAS + 1);
for (let k = -POW2_BIAS; k <= POW2_BIAS; k++) POW2[k + POW2_BIAS] = 2 ** k;

const bits = new Float64Array(1);
const words = new Uint32Array(bits.buffer);
const HIGH = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1 ? 1 : 0;

/**
 * `Math.floor(Math.log2(m))` for finite m > 0, mostly without calling log2.
 *
 * The answer is the binary exponent of m, which can be read straight out of
 * its bits -- except that Math.log2 rounds, and when m is within a few ulps of
 * a power of two its result can land on the neighbouring integer. Mantissas
 * within 2^-36 of either end of [1, 2), and subnormals, go through Math.log2
 * itself, so the result is always what the original expression gives. Away
 * from the ends log2 sits at least 2e-11 from an integer, a hundred times its
 * largest rounding error for any double.
 */
function floorLog2(m: number): number {
  bits[0] = m;
  const high = words[HIGH];
  const low = words[1 - HIGH];
  const biased = (high >>> 20) & 0x7ff;
  const top = high & 0xfffff;
  const next = low >>> 16;
  const nearOne = top === 0 && next === 0;
  const nearTwo = top === 0xfffff && next === 0xffff;
  if (biased === 0 || nearOne || nearTwo) return Math.floor(Math.log2(m));
  return biased - 1023;
}

/** normalise(), writing (x, y, e) to dst[at..at+2]. */
function normaliseTo(dst: Float64Array, at: number, x: number, y: number, e: number) {
  const magnitude = Math.max(Math.abs(x), Math.abs(y));
  if (magnitude === 0 || !Number.isFinite(magnitude)) {
    dst[at] = 0;
    dst[at + 1] = 0;
    dst[at + 2] = 0;
    return;
  }
  const shift = floorLog2(magnitude);
  const factor = POW2[POW2_BIAS - shift];
  dst[at] = x * factor;
  dst[at + 1] = y * factor;
  dst[at + 2] = e + shift;
}

/** multiply(). `dst` may be either operand. */
function multiplyTo(
  dst: Float64Array, at: number,
  p: Float64Array, i: number,
  q: Float64Array, j: number
) {
  const px = p[i], py = p[i + 1], qx = q[j], qy = q[j + 1];
  normaliseTo(dst, at, px * qx - py * qy, px * qy + py * qx, p[i + 2] + q[j + 2]);
}

/** add(). `dst` may be either operand. */
function addTo(
  dst: Float64Array, at: number,
  p: Float64Array, i: number,
  q: Float64Array, j: number
) {
  const px = p[i], py = p[i + 1], pe = p[i + 2];
  const qx = q[j], qy = q[j + 1], qe = q[j + 2];
  const difference = pe - qe;
  let takeP = false;
  let takeQ = false;
  if (px === 0 && py === 0) takeQ = true;
  else if (qx === 0 && qy === 0) takeP = true;
  else if (difference > 80) takeP = true;
  else if (difference < -80) takeQ = true;

  if (takeQ) {
    dst[at] = qx;
    dst[at + 1] = qy;
    dst[at + 2] = qe;
  } else if (takeP) {
    dst[at] = px;
    dst[at + 1] = py;
    dst[at + 2] = pe;
  } else if (difference >= 0) {
    const factor = POW2[POW2_BIAS - difference];
    normaliseTo(dst, at, px + qx * factor, py + qy * factor, pe);
  } else {
    const factor = POW2[POW2_BIAS + difference];
    normaliseTo(dst, at, px * factor + qx, py * factor + qy, qe);
  }
}

/** scale(). `dst` may be the operand. */
function scaleTo(dst: Float64Array, at: number, p: Float64Array, i: number, factor: number) {
  normaliseTo(dst, at, p[i] * factor, p[i + 1] * factor, p[i + 2]);
}

/** log2Magnitude(). */
function log2At(p: Float64Array, i: number): number {
  const m = Math.hypot(p[i], p[i + 1]);
  return m === 0 ? -Infinity : p[i + 2] + Math.log2(m);
}

/**
 * Builds the skip table from reference orbit samples.
 *
 * @param orbit reduced samples, 6 floats per entry (hi, lo, exp per component)
 * @param length valid sample count
 * @param maxDelta largest |d| any pixel will use, so radii can account for the
 *   terms `d` injects without knowing the pixel
 */
export function buildBla(
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
  const levels: Float64Array[] = [];

  // Working registers: the step being built, 2X, scratch, and the constant 1.
  const r = new Float64Array(STEP_WORDS + 18);
  const TWO_X = STEP_WORDS;
  const T0 = STEP_WORDS + 3;
  const T1 = STEP_WORDS + 6;
  const NEXT_C = STEP_WORDS + 9;
  const NEXT_D = STEP_WORDS + 12;
  const ONE_AT = STEP_WORDS + 15;
  r[ONE_AT] = 1;

  // ---- level 0 -------------------------------------------------------------
  const level0Count = Math.floor(count / BASE_STEP);
  const level0 = new Float64Array(level0Count * STEP_WORDS);

  for (let s = 0; s < level0Count; s++) {
    const start = s * BASE_STEP;
    // A = 1; B, C, D, E = 0.
    r.fill(0, 0, STEP_WORDS);
    r[A] = 1;

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
      const injectedLog2 = log2At(r, B) + maxDeltaLog2;
      if (injectedLog2 >= budgetLog2) {
        radiusLog2 = NEVER;
        break;
      }
      const headroomLog2 =
        injectedLog2 === -Infinity
          ? budgetLog2
          : budgetLog2 + Math.log2(1 - 2 ** (injectedLog2 - budgetLog2));

      // w_k = A_k * w, so divide through by |A_k|.
      radiusLog2 = Math.min(radiusLog2, headroomLog2 - log2At(r, A));

      normaliseTo(r, TWO_X, 2 * refX[start + k], 2 * refY[start + k], 0);
      // Second-order updates use the *previous* A and B, so compute them first.
      if (quadratic) {
        // C' = 2X*C + A^2
        multiplyTo(r, T0, r, TWO_X, r, C);
        multiplyTo(r, T1, r, A, r, A);
        addTo(r, NEXT_C, r, T0, r, T1);
        // D' = 2X*D + 2AB
        multiplyTo(r, T0, r, TWO_X, r, D);
        multiplyTo(r, T1, r, A, r, B);
        scaleTo(r, T1, r, T1, 2);
        addTo(r, NEXT_D, r, T0, r, T1);
        // E' = 2X*E + B^2, straight into E: nothing below reads the old E.
        multiplyTo(r, T0, r, TWO_X, r, E);
        multiplyTo(r, T1, r, B, r, B);
        addTo(r, E, r, T0, r, T1);
      }
      // A' = 2X*A, B' = 2X*B + 1
      multiplyTo(r, A, r, TWO_X, r, A);
      multiplyTo(r, T0, r, TWO_X, r, B);
      addTo(r, B, r, T0, r, ONE_AT);
      if (quadratic) {
        r.copyWithin(C, NEXT_C, NEXT_C + 3);
        r.copyWithin(D, NEXT_D, NEXT_D + 3);
      }
    }

    r[RADIUS] = Number.isFinite(radiusLog2) ? radiusLog2 : NEVER;
    level0.set(r.subarray(0, STEP_WORDS), s * STEP_WORDS);
  }
  levels.push(level0);

  // ---- merge pairs ---------------------------------------------------------
  // Scratch for compose(): the composite, then products in flight.
  const m = new Float64Array(STEP_WORDS + 9);
  const P0 = STEP_WORDS;
  const P1 = STEP_WORDS + 3;
  const P2 = STEP_WORDS + 6;
  for (let level = 1; level < maxLevels; level++) {
    const previous = levels[level - 1];
    const mergedCount = Math.floor(previous.length / STEP_WORDS / 2);
    if (mergedCount < 1) break;

    const merged = new Float64Array(mergedCount * STEP_WORDS);
    for (let s = 0; s < mergedCount; s++) {
      const f = 2 * s * STEP_WORDS;
      const g = f + STEP_WORDS;
      const p = previous;

      // compose(first, second), term by term.
      multiplyTo(m, A, p, g + A, p, f + A);
      multiplyTo(m, P0, p, g + A, p, f + B);
      addTo(m, B, m, P0, p, g + B);
      // C = A2*C1 + C2*A1^2
      multiplyTo(m, P0, p, g + A, p, f + C);
      multiplyTo(m, P1, p, f + A, p, f + A);
      multiplyTo(m, P1, p, g + C, m, P1);
      addTo(m, C, m, P0, m, P1);
      // D = A2*D1 + 2*C2*A1*B1 + D2*A1
      multiplyTo(m, P0, p, g + A, p, f + D);
      multiplyTo(m, P1, p, f + A, p, f + B);
      multiplyTo(m, P1, p, g + C, m, P1);
      scaleTo(m, P1, m, P1, 2);
      addTo(m, P0, m, P0, m, P1);
      multiplyTo(m, P1, p, g + D, p, f + A);
      addTo(m, D, m, P0, m, P1);
      // E = A2*E1 + C2*B1^2 + D2*B1 + E2
      multiplyTo(m, P0, p, g + A, p, f + E);
      multiplyTo(m, P1, p, f + B, p, f + B);
      multiplyTo(m, P1, p, g + C, m, P1);
      addTo(m, P0, m, P0, m, P1);
      multiplyTo(m, P2, p, g + D, p, f + B);
      addTo(m, P2, m, P2, p, g + E);
      addTo(m, E, m, P0, m, P2);

      // Entering the second half the delta has become |A1*w + B1*d|, which has
      // to land inside the second radius, worst case over the largest |d|.
      const firstRadius = p[f + RADIUS];
      const secondRadius = p[g + RADIUS];
      const b1dLog2 = log2At(p, f + B) + maxDeltaLog2;
      let radiusLog2 = NEVER;
      if (b1dLog2 < secondRadius) {
        const headroomLog2 = secondRadius + Math.log2(1 - 2 ** (b1dLog2 - secondRadius));
        const mapped = headroomLog2 - log2At(p, f + A);
        radiusLog2 = Math.min(firstRadius, mapped) - MERGE_PENALTY_LOG2;
      }
      if (!Number.isFinite(radiusLog2)) radiusLog2 = NEVER;

      m[RADIUS] = radiusLog2;
      merged.set(m.subarray(0, STEP_WORDS), s * STEP_WORDS);
    }
    levels.push(merged);
  }

  // ---- pack ----------------------------------------------------------------
  let entryCount = 0;
  for (const level of levels) entryCount += level.length / STEP_WORDS;

  const data = new Float32Array(Math.max(1, entryCount) * ENTRY_FLOATS);
  const levelOffsets: number[] = [];
  const levelCounts: number[] = [];
  let offset = 0;

  for (const level of levels) {
    const steps = level.length / STEP_WORDS;
    levelOffsets.push(offset);
    levelCounts.push(steps);
    for (let index = 0; index < steps; index++) {
      // Five coefficients then the radius: the packed layout is the step's
      // own first sixteen words, rounded to f32.
      data.set(
        level.subarray(index * STEP_WORDS, index * STEP_WORDS + STEP_WORDS),
        (offset + index) * ENTRY_FLOATS
      );
    }
    offset += steps;
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

/** Reads a packed entry back, for tests and the CPU mirror. */
export function readStep(table: BlaTable, level: number, index: number): Step {
  const base = (table.levelOffsets[level] + index) * ENTRY_FLOATS;
  const get = (slot: number): Scaled => ({
    x: table.data[base + slot * 3 + 0],
    y: table.data[base + slot * 3 + 1],
    e: table.data[base + slot * 3 + 2],
  });
  return {
    a: get(0),
    b: get(1),
    c: get(2),
    d: get(3),
    e: get(4),
    radiusLog2: table.data[base + 15],
  };
}

/** CPU mirror of the shader's skip, for tests. */
export function applyStep(step: Step, w: Scaled, delta: Scaled): Scaled {
  let out = add(multiply(step.a, w), multiply(step.b, delta));
  out = add(out, multiply(step.c, multiply(w, w)));
  out = add(out, scale(multiply(step.d, multiply(w, delta)), 1));
  out = add(out, multiply(step.e, multiply(delta, delta)));
  return out;
}

export function stepRadiusLog2(
  table: BlaTable,
  level: number,
  index: number
): number {
  return table.data[(table.levelOffsets[level] + index) * ENTRY_FLOATS + 15];
}
