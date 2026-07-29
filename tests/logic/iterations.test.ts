import { describe, expect, it } from "vitest";
import Decimal from "decimal.js";
import { iterationsForSpan, zoomDepth } from "../../src/logic/iterations";

const LIMIT = 200000;
const at = (span: string) => iterationsForSpan(new Decimal(span), LIMIT);

describe("zoomDepth", () => {
  it("is zero at the home view and never negative", () => {
    expect(zoomDepth(new Decimal("2.8"))).toBeCloseTo(0, 6);
    expect(zoomDepth(new Decimal("100"))).toBe(0);
  });

  it("counts decimal orders of magnitude", () => {
    expect(zoomDepth(new Decimal("2.8e-10"))).toBeCloseTo(10, 6);
    expect(zoomDepth(new Decimal("2.8e-40"))).toBeCloseTo(40, 6);
  });

  it("survives spans far outside double range", () => {
    // 1e-400 is 0 as a double; going through Decimal.log keeps it meaningful.
    expect(zoomDepth(new Decimal("1e-400"))).toBeGreaterThan(390);
    expect(Number.isFinite(zoomDepth(new Decimal("1e-2000")))).toBe(true);
  });

  it("does not divide by zero", () => {
    expect(zoomDepth(new Decimal(0))).toBe(0);
  });
});

describe("iterationsForSpan", () => {
  it("grows monotonically once past the shallow floor", () => {
    const spans = ["1e-6", "3e-11", "1e-20", "6e-42"];
    const counts = spans.map(at);
    for (let i = 1; i < counts.length; i++) {
      expect(counts[i]).toBeGreaterThan(counts[i - 1]);
    }
    // Wide views all sit on the floor rather than dropping toward zero.
    expect(at("2.8")).toBe(at("0.02"));
  });

  it("stays in the region the hand-picked locations use", () => {
    // Those are in locations.ts: someone looked at each view and chose a
    // count. Auto should land near them, not an order of magnitude away.
    expect(at("2.8")).toBeGreaterThanOrEqual(300);
    expect(at("2.8")).toBeLessThanOrEqual(700);
  });

  it("matches what the sweeps measured as sufficient", () => {
    // From /gpu.html?...&sweep=1 at the default centre: the count at which
    // raising it stops resolving further pixels. Within 25% is close enough
    // for a starting guess; being badly under means a blank frame.
    for (const [span, needed] of [
      ["1e-5", 1600],
      ["1e-12", 6400],
      ["1e-25", 20000],
    ] as const) {
      const value = at(span);
      expect(value).toBeGreaterThan(needed * 0.75);
      expect(value).toBeLessThan(needed * 1.5);
    }
  });

  it("respects a limit when given one", () => {
    expect(at("1e-300")).toBeLessThanOrEqual(LIMIT);
    expect(at("1e-3000")).toBeLessThanOrEqual(LIMIT);
    expect(iterationsForSpan(new Decimal("1e-500"), 5000)).toBe(5000);
  });

  it("is unbounded without one, because a clamped count renders wrong", () => {
    // Capping at depth does not make the frame cheap, it makes every pixel hit
    // the cap and the view read as solid interior.
    const deep = iterationsForSpan(new Decimal("1e-600"));
    expect(deep).toBeGreaterThan(LIMIT);
    expect(iterationsForSpan(new Decimal("1e-2000"))).toBeGreaterThan(deep);
    expect(Number.isFinite(iterationsForSpan(new Decimal("1e-2000")))).toBe(true);
  });

  it("returns whole, round numbers", () => {
    for (const span of ["2.8", "1e-6", "6e-42", "1e-120"]) {
      const value = at(span);
      expect(Number.isInteger(value)).toBe(true);
      // Two significant figures: the estimate is not precise enough to
      // justify more, and a round number reads as a guess.
      expect(value).toBe(Number(value.toPrecision(2)));
    }
  });
});
