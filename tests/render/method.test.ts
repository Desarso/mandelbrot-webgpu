import { describe, expect, it } from "vitest";
import Decimal from "decimal.js";
import { Method, binaryExponent, methodForScale } from "../../src/render/webgpu-renderer";

const upp = (span: string, height = 1080) => new Decimal(span).div(height);

describe("methodForScale", () => {
  it("iterates c directly when the view is wider than f32 can blur", () => {
    expect(methodForScale(upp("2.8"))).toBe(Method.Direct);
    expect(methodForScale(upp("0.02"))).toBe(Method.Direct);
  });

  it("switches to perturbation before f32 loses the pixel grid", () => {
    // 6e-8 is roughly f32's resolution near |c| ~ 1; direct iteration has to
    // be gone well before the pixel spacing gets there.
    expect(methodForScale(upp("1e-3"))).toBe(Method.Plain);
    expect(methodForScale(upp("6e-6"))).toBe(Method.Plain);
  });

  it("keeps the plain delta across the range where it is fastest", () => {
    expect(methodForScale(upp("1e-8"))).toBe(Method.Plain);
    expect(methodForScale(upp("3e-11"))).toBe(Method.Plain);
    expect(methodForScale(upp("1e-18"))).toBe(Method.Plain);
  });

  it("gives the delta its own exponent well above the f32 floor", () => {
    // A plain f32 denormalises at 1.2e-38; the handover is 13 decades early.
    expect(methodForScale(upp("1e-25"))).toBe(Method.Hdr);
    expect(methodForScale(upp("6e-42"))).toBe(Method.Hdr);
    expect(methodForScale(upp("1e-300"))).toBe(Method.Hdr);
  });

  it("depends on pixel spacing, not span alone", () => {
    // The same span on a taller viewport resolves finer and can need a
    // stronger method.
    expect(methodForScale(new Decimal("1e-2").div(100))).toBe(Method.Direct);
    expect(methodForScale(new Decimal("1e-2").div(100000))).toBe(Method.Plain);
  });
});

describe("binaryExponent", () => {
  const original = (v: Decimal) => Math.floor(Number(v.abs().log(2).toFixed(6)));

  function makeRandom(seed: number) {
    let state = seed >>> 0;
    return () => {
      state = (state * 1664525 + 1013904223) >>> 0;
      return state / 2 ** 32;
    };
  }

  it("agrees with the arbitrary-precision logarithm it replaces", () => {
    const random = makeRandom(1215);
    // The precisions the renderer sets for 8, 16 and 64 limbs. The oracle is
    // slow at the top end, hence fewer cases there.
    for (const [precision, cases] of [[78, 1000], [155, 150], [618, 15]]) {
      Decimal.set({ precision });
      for (let i = 0; i < cases; i++) {
        // Spread over everything from the home view to far past 1e-2000, with
        // long arbitrary digit strings like a real deep coordinate.
        const exponent = -Math.floor(random() * 2100);
        let digits = "";
        for (let k = 0; k < 40; k++) digits += Math.floor(random() * 10);
        const sign = random() < 0.5 ? "-" : "";
        const v = new Decimal(`${sign}${1 + Math.floor(random() * 9)}.${digits}e${exponent}`);
        expect(binaryExponent(v)).toBe(original(v));
      }
    }
  });

  it("agrees right at the edges, where rounding to six places decides", () => {
    Decimal.set({ precision: 78 });
    const two = new Decimal(2);
    for (let k = -2000; k <= 10; k += 97) {
      const power = two.pow(k);
      for (const nudge of ["0", "1e-30", "-1e-30", "1e-12", "-1e-12", "3e-7", "-3e-7",
        "3.4657e-7", "-3.4657e-7", "3.4658e-7", "-3.4658e-7", "1e-6", "-1e-6"]) {
        const v = power.times(new Decimal(1).plus(nudge));
        expect(binaryExponent(v)).toBe(original(v));
      }
    }
  });
});
