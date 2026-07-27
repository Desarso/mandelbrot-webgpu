import { describe, expect, it } from "vitest";
import {
  BASE_STEP,
  NEVER,
  applyStep,
  buildLinearApprox,
  scaled,
  stepRadiusLog2,
  type Scaled,
} from "../../src/render/linear-approx";

/** Builds reference orbit samples in the reduced format the GPU emits. */
function makeOrbit(cx: number, cy: number, count: number) {
  const orbit = new Float32Array((count + 1) * 6);
  const xs = [0];
  const ys = [0];
  let x = 0;
  let y = 0;
  for (let i = 1; i <= count; i++) {
    const nx = x * x - y * y + cx;
    y = 2 * x * y + cy;
    x = nx;
    xs.push(x);
    ys.push(y);
  }
  for (let i = 0; i <= count; i++) {
    const hiX = Math.fround(xs[i]);
    const hiY = Math.fround(ys[i]);
    orbit[i * 6 + 0] = hiX;
    orbit[i * 6 + 1] = Math.fround(xs[i] - hiX);
    orbit[i * 6 + 2] = 0;
    orbit[i * 6 + 3] = hiY;
    orbit[i * 6 + 4] = Math.fround(ys[i] - hiY);
    orbit[i * 6 + 5] = 0;
  }
  return { orbit, xs, ys };
}

/**
 * Full perturbation iteration in scaled arithmetic — the thing the
 * approximation must reproduce. Plain doubles would underflow at these deltas.
 */
function iterate(
  xs: number[],
  ys: number[],
  from: number,
  steps: number,
  dz: Scaled,
  delta: Scaled
): Scaled {
  let z = dz;
  for (let k = 0; k < steps; k++) {
    const twoX = scaled.normalise(2 * xs[from + k], 2 * ys[from + k], 0);
    z = scaled.add(
      scaled.add(scaled.multiply(twoX, z), scaled.multiply(z, z)),
      delta
    );
  }
  return z;
}

function relativeError(a: Scaled, b: Scaled): number {
  const difference = scaled.add(a, { x: -b.x, y: -b.y, e: b.e });
  const scale = scaled.log2Magnitude(b);
  if (!Number.isFinite(scale)) return 0;
  return 2 ** (scaled.log2Magnitude(difference) - scale);
}

// Inside the period-3 bulb: the orbit is bounded and cycles through three
// distinct non-zero values, so |X| varies without ever hitting zero. An
// escaping orbit would fill the fixture with Infinity, and a nucleus orbit
// returns to exactly zero, which legitimately invalidates any step spanning it.
const CX = -0.12;
const CY = 0.74;

describe("linear approximation table", () => {
  it("builds levels that halve in count", () => {
    const { orbit } = makeOrbit(CX, CY, 4096);
    const table = buildLinearApprox(orbit, 4097, 1e-20);

    expect(table.levels).toBeGreaterThan(4);
    for (let level = 1; level < table.levels; level++) {
      expect(table.levelCounts[level]).toBe(
        Math.floor(table.levelCounts[level - 1] / 2)
      );
    }
    expect(table.levelCounts[0]).toBe(Math.floor(4096 / BASE_STEP));
  });

  it("survives the coefficient overflow that kills plain doubles", () => {
    // A over a long step reaches astronomical magnitudes. Stored naively it
    // becomes Infinity and every later radius turns into NaN.
    const { orbit } = makeOrbit(CX, CY, 8192);
    const table = buildLinearApprox(orbit, 8193, 1e-40);

    for (let i = 0; i < table.data.length; i++) {
      expect(Number.isFinite(table.data[i])).toBe(true);
    }
    // The attracting 3-cycle has |multiplier| < 1, so A shrinks relentlessly:
    // over thousands of steps it lands far outside anything a double can hold.
    let biggest = 0;
    for (let level = 0; level < table.levels; level++) {
      for (let index = 0; index < table.levelCounts[level]; index++) {
        const base = (table.levelOffsets[level] + index) * 8;
        biggest = Math.max(biggest, Math.abs(table.data[base + 2]));
      }
    }
    expect(biggest).toBeGreaterThan(1100);
  });

  it("reproduces full iteration inside the validity radius", () => {
    const { orbit, xs, ys } = makeOrbit(CX, CY, 4096);
    const delta: Scaled = scaled.normalise(1e-24, -7e-25, 0);
    const table = buildLinearApprox(orbit, 4097, 2e-24);

    let checked = 0;
    let worst = 0;
    for (let level = 0; level < Math.min(table.levels, 7); level++) {
      const steps = BASE_STEP << level;
      for (let index = 0; index < Math.min(table.levelCounts[level], 10); index++) {
        const radiusLog2 = stepRadiusLog2(table, level, index);
        if (radiusLog2 <= NEVER / 2) continue;

        // Enter right at the allowed radius: the worst legal case.
        const dz = scaled.normalise(1, 0.3, Math.round(radiusLog2));
        const approx = applyStep(table, level, index, dz, delta);
        const exact = iterate(xs, ys, index * steps, steps, dz, delta);

        worst = Math.max(worst, relativeError(approx, exact));
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(20);
    expect(worst).toBeLessThan(1e-3);
  });

  it("is essentially exact when the delta is the only input", () => {
    // With dz = 0 there is no quadratic term at all, so B*d should be spot on.
    // Step 1, not 0: the first step spans X_0 = 0, which zeroes A and is a
    // genuinely degenerate case the table marks unusable anyway.
    const { orbit, xs, ys } = makeOrbit(CX, CY, 512);
    const table = buildLinearApprox(orbit, 513, 1e-30);
    const delta = scaled.normalise(1e-30, 0, 0);
    const zero: Scaled = { x: 0, y: 0, e: 0 };

    const approx = applyStep(table, 0, 1, zero, delta);
    const exact = iterate(xs, ys, BASE_STEP, BASE_STEP, zero, delta);
    expect(relativeError(approx, exact)).toBeLessThan(1e-6);
  });

  it("marks the step spanning X_0 = 0 unusable", () => {
    const { orbit } = makeOrbit(CX, CY, 512);
    const table = buildLinearApprox(orbit, 513, 1e-30);
    // 2*X_0 is zero, so A collapses and no delta is small enough to make the
    // linear map meaningful there.
    expect(stepRadiusLog2(table, 0, 0)).toBeLessThanOrEqual(NEVER / 2);
    expect(stepRadiusLog2(table, 0, 1)).toBeGreaterThan(NEVER / 2);
  });

  it("never widens the radius as steps merge", () => {
    const { orbit } = makeOrbit(CX, CY, 4096);
    const table = buildLinearApprox(orbit, 4097, 1e-20);

    for (let level = 1; level < table.levels; level++) {
      for (let index = 0; index < Math.min(table.levelCounts[level], 20); index++) {
        expect(stepRadiusLog2(table, level, index)).toBeLessThanOrEqual(
          stepRadiusLog2(table, level - 1, index * 2) + 1e-9
        );
      }
    }
  });

  it("refuses steps whose radius the delta already exceeds", () => {
    const { orbit } = makeOrbit(CX, CY, 2048);
    // A huge maxDelta leaves no headroom, so merged steps must be rejected.
    const table = buildLinearApprox(orbit, 2049, 1e6);
    let refused = 0;
    const top = table.levels - 1;
    for (let index = 0; index < table.levelCounts[top]; index++) {
      if (stepRadiusLog2(table, top, index) <= NEVER / 2) refused++;
    }
    expect(refused).toBeGreaterThan(0);
  });
});
