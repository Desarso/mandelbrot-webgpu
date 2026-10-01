import { describe, expect, it } from "vitest";
import { buildBla } from "../../src/render/bla";
import { buildBlaReference } from "./bla-reference";

/** Reference orbit samples in the reduced format the GPU emits. */
function makeOrbit(cx: number, cy: number, count: number) {
  const orbit = new Float32Array((count + 1) * 6);
  let x = 0;
  let y = 0;
  for (let i = 1; i <= count; i++) {
    const nx = x * x - y * y + cx;
    y = 2 * x * y + cy;
    x = nx;
    // Split as hi + lo with an exponent, like emitSample, so the decode path
    // in the builder is exercised too.
    const ex = x === 0 ? 0 : Math.floor(Math.log2(Math.abs(x)));
    const ey = y === 0 ? 0 : Math.floor(Math.log2(Math.abs(y)));
    const mx = x / 2 ** ex;
    const my = y / 2 ** ey;
    orbit[i * 6 + 0] = Math.fround(mx);
    orbit[i * 6 + 1] = Math.fround(mx - Math.fround(mx));
    orbit[i * 6 + 2] = ex;
    orbit[i * 6 + 3] = Math.fround(my);
    orbit[i * 6 + 4] = Math.fround(my - Math.fround(my));
    orbit[i * 6 + 5] = ey;
  }
  return orbit;
}

const sameBits = (a: Float32Array, b: Float32Array) => {
  const x = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const y = new Uint32Array(b.buffer, b.byteOffset, b.length);
  if (x.length !== y.length) return `length ${x.length} vs ${y.length}`;
  for (let i = 0; i < x.length; i++) {
    if (x[i] !== y[i]) return `word ${i}: ${a[i]} vs ${b[i]}`;
  }
  return "same";
};

describe("allocation-free skip-table builder", () => {
  // Bounded orbits of several characters: inside a bulb, near a minibrot,
  // on the real axis (y = 0 exactly), one through zero (c = 0 gives a zero
  // reference and NEVER radii), and an escaping one that grows huge.
  const cases: Array<[string, number, number, number]> = [
    ["period-3 bulb", -0.12, 0.74, 4096],
    ["near the 1215 minibrot", -0.6007057551602345, 0.44123987024167955, 6000],
    ["real axis", -1.7548776662466927, 0, 3000],
    ["zero orbit", 0, 0, 512],
    ["escaping", -0.75, 0.05, 2000],
    ["short", -0.1, 0.65, 37],
  ];
  for (const [name, cx, cy, count] of cases) {
    for (const maxDelta of [1e-40, 1e-12, 1e-3, 0]) {
      for (const quadratic of [true, false]) {
        it(`matches the reference bit for bit: ${name}, |d| ${maxDelta}, ${quadratic ? "second" : "first"} order`, () => {
          const orbit = makeOrbit(cx, cy, count);
          const fast = buildBla(orbit, count + 1, maxDelta, { quadratic });
          const slow = buildBlaReference(orbit, count + 1, maxDelta, { quadratic });
          expect(sameBits(fast.data, slow.data)).toBe("same");
          expect(fast.levelOffsets).toEqual(slow.levelOffsets);
          expect(fast.levelCounts).toEqual(slow.levelCounts);
          expect(fast.levels).toBe(slow.levels);
          expect(fast.entryCount).toBe(slow.entryCount);
        });
      }
    }
  }

  it("honours maxLevels like the reference", () => {
    const orbit = makeOrbit(-0.12, 0.74, 4096);
    for (const maxLevels of [1, 3, 20]) {
      const fast = buildBla(orbit, 4097, 1e-30, { maxLevels });
      const slow = buildBlaReference(orbit, 4097, 1e-30, { maxLevels });
      expect(sameBits(fast.data, slow.data)).toBe("same");
      expect(fast.levels).toBe(slow.levels);
    }
  });
});
