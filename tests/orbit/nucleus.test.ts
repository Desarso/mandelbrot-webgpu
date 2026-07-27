import Decimal from "decimal.js";
import { describe, expect, it } from "vitest";
import { Complex } from "../../src/logic/Complex";
import {
  detectPeriod,
  findNucleus,
  newtonNucleus,
  sizeEstimate,
} from "../../src/orbit/nucleus";

const c = (re: string, im: string) => new Complex(new Decimal(re), new Decimal(im));

/** Iterates c and returns the escape iteration, or -1 if it stays bounded. */
function escapeIteration(point: Complex, maxIterations: number): number {
  let z = new Complex(new Decimal(0), new Decimal(0));
  for (let i = 1; i <= maxIterations; i++) {
    z = z.square().add(point);
    if (z.real.times(z.real).plus(z.imag.times(z.imag)).greaterThan(4)) return i;
  }
  return -1;
}

describe("size estimate", () => {
  it("gives the period-2 bulb its actual diameter", () => {
    // The bulb at c = -1 has radius 1/4.
    expect(Number(sizeEstimate(c("-1", "0"), 2))).toBeCloseTo(0.5, 9);
  });

  it("shrinks as the period grows", () => {
    const small = findNucleus(new Decimal("-1.75"), new Decimal("0"), {
      maxPeriod: 64,
    });
    expect(small).not.toBeNull();
    expect(Number(small!.size)).toBeLessThan(0.5);
    expect(Number(small!.size)).toBeGreaterThan(0);
  });
});

describe("newton on the nucleus equation", () => {
  it("lands exactly on the period-2 nucleus at c = -1", () => {
    Decimal.set({ precision: 60 });
    const result = newtonNucleus(c("-1.01", "0.01"), 2);
    expect(result.converged).toBe(true);
    expect(Number(result.c.real)).toBeCloseTo(-1, 12);
    expect(Number(result.c.imag)).toBeCloseTo(0, 12);
  });

  it("lands on the period-3 nucleus at c = -1.7548776662...", () => {
    Decimal.set({ precision: 60 });
    const result = newtonNucleus(c("-1.75", "0"), 3);
    expect(result.converged).toBe(true);
    // Known root of the period-3 nucleus polynomial.
    expect(result.c.real.toSignificantDigits(10).toString()).toBe("-1.754877666");
  });

  it("converges to far more digits than a double holds", () => {
    Decimal.set({ precision: 120 });
    const result = newtonNucleus(c("-1.75", "0"), 3);
    expect(result.converged).toBe(true);

    // f_3(c) must vanish to the working precision, which a double cannot show.
    let z = new Complex(new Decimal(0), new Decimal(0));
    for (let i = 0; i < 3; i++) z = z.square().add(result.c);
    const residual = z.real.times(z.real).plus(z.imag.times(z.imag)).sqrt();
    expect(Number(residual.log(10))).toBeLessThan(-100);
  });
});

describe("period detection", () => {
  it("offers the period as a candidate near a known nucleus", () => {
    Decimal.set({ precision: 40 });
    const candidates = detectPeriod(c("-1.7548", "0"), 200);
    expect(candidates).toContain(3);
  });
});

describe("findNucleus", () => {
  it("returns a point that is genuinely inside the set", () => {
    const found = findNucleus(new Decimal("-0.75"), new Decimal("0.1"), {
      maxPeriod: 512,
      digits: 60,
    });
    expect(found).not.toBeNull();
    Decimal.set({ precision: 60 });
    expect(
      escapeIteration(new Complex(found!.centerX, found!.centerY), 5000)
    ).toBe(-1);
  });

  it("produces a nucleus whose neighbourhood still has structure", () => {
    const found = findNucleus(new Decimal("-0.75"), new Decimal("0.1"), {
      maxPeriod: 512,
      digits: 60,
    });
    expect(found).not.toBeNull();
    Decimal.set({ precision: 60 });

    // Somewhere within a few diameters the set must give way — that boundary
    // detail is what a deep view needs. A smooth region stays bounded
    // everywhere. Which multiple escapes first depends on the direction, so
    // probe a range rather than assuming one.
    const escapes = [1, 2, 4, 8].map((multiple) =>
      escapeIteration(
        new Complex(found!.centerX.plus(found!.size.times(multiple)), found!.centerY),
        20000
      )
    );
    expect(escapes.some((iteration) => iteration > 0)).toBe(true);
  });

  it("finds a deep nucleus when started from a point already at depth", () => {
    // Newton converges to the nearest nucleus, so depth comes from the
    // starting point. From a shallow point it lands on a big bulb; from a
    // boundary point at depth it reaches a very small minibrot.
    const shallow = findNucleus(new Decimal("-0.75"), new Decimal("0.1"), {
      maxPeriod: 512,
      digits: 60,
    })!;
    const deep = findNucleus(
      new Decimal("-0.600705755160234496572763605385"),
      new Decimal("0.441239870241679552586993134940"),
      { maxPeriod: 4000, digits: 100 }
    );

    expect(deep).not.toBeNull();
    expect(deep!.period).toBeGreaterThan(shallow.period);
    expect(Number(deep!.size)).toBeLessThan(1e-30);

    // And the nucleus really is a nucleus: f_p(c) vanishes.
    Decimal.set({ precision: 100 });
    let z = new Complex(new Decimal(0), new Decimal(0));
    const c0 = new Complex(deep!.centerX, deep!.centerY);
    for (let i = 0; i < deep!.period; i++) z = z.square().add(c0);
    const residual = z.real.times(z.real).plus(z.imag.times(z.imag)).sqrt();
    expect(Number(residual)).toBeLessThan(1e-30);
  });
});
