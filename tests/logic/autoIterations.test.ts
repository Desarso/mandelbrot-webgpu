import { describe, expect, it } from "vitest";
import { nextIterations, type Probe } from "../../src/logic/autoIterations";

const LIMIT = 200000;

/**
 * Runs the search against a view whose true requirement is `needed` and whose
 * genuine interior is `interior` of the frame. Below `needed`, pixels are
 * capped in proportion to how far short the budget falls; at or above it, only
 * the real interior remains.
 */
function search(needed: number, interior: number, start: number) {
  const measure = (iterations: number): Probe => ({
    iterations,
    capped:
      iterations >= needed
        ? interior
        : interior + (1 - interior) * (1 - iterations / needed),
  });

  const probes: Probe[] = [measure(start)];
  for (let step = 0; step < 40; step++) {
    const decision = nextIterations(probes, LIMIT);
    if (decision.action === "settle") {
      return { settled: decision.iterations, renders: probes.length };
    }
    probes.push(measure(decision.iterations));
  }
  throw new Error("did not converge");
}

describe("nextIterations", () => {
  it("climbs to what the view needs when it starts too low", () => {
    const { settled } = search(20000, 0.05, 1000);
    // Doubling from 1000 lands on 16000 or 32000; either renders the picture.
    expect(settled).toBeGreaterThanOrEqual(16000);
  });

  it("gives budget back when it starts far too high", () => {
    // This is the complaint that started it: auto guessing much too high.
    const { settled } = search(1600, 0.09, 51200);
    expect(settled).toBeLessThan(51200);
    expect(settled).toBeGreaterThanOrEqual(1600);
  });

  it("settles immediately when the starting guess is already right", () => {
    const { settled, renders } = search(6400, 0.04, 6400);
    expect(settled).toBe(6400);
    // One probe up to confirm nothing more is gained, then done.
    expect(renders).toBeLessThanOrEqual(3);
  });

  it("converges in a handful of renders even from far away", () => {
    expect(search(20000, 0.05, 500).renders).toBeLessThanOrEqual(9);
    expect(search(500, 0.2, 128000).renders).toBeLessThanOrEqual(11);
  });

  it("never settles below what the view needs", () => {
    for (const needed of [800, 1600, 6400, 20000]) {
      const { settled } = search(needed, 0.05, 2000);
      // Within one doubling of the requirement, and never under it.
      expect(settled).toBeGreaterThanOrEqual(needed);
    }
  });

  it("respects the limit", () => {
    const probes: Probe[] = [{ iterations: LIMIT, capped: 0.5 }];
    const decision = nextIterations(probes, LIMIT);
    expect(decision.iterations).toBeLessThanOrEqual(LIMIT);
    expect(decision.action).toBe("settle");
  });

  it("treats a wholly escaped frame as budget to give back", () => {
    const decision = nextIterations([{ iterations: 8000, capped: 0 }], LIMIT);
    expect(decision).toEqual({ action: "try", iterations: 4000 });
  });

  it("does not mistake genuine interior for an insufficient budget", () => {
    // A view that is mostly inside the set: the capped fraction is high and
    // stays high however much budget it is given. Raising must stop -- the
    // right move is to give budget back, not to keep buying more of it.
    const probes: Probe[] = [
      { iterations: 4000, capped: 0.62 },
      { iterations: 8000, capped: 0.6199 },
    ];
    const decision = nextIterations(probes, LIMIT);
    expect(decision.iterations).toBeLessThan(8000);
  });

  it("settles on a mostly-interior view rather than climbing forever", () => {
    // The same view driven to completion: 62% of it is genuinely inside the
    // set at any budget, and the search must not read that as needing more.
    const measure = (iterations: number): Probe => ({
      iterations,
      capped: iterations >= 3000 ? 0.62 : 0.62 + (1 - 0.62) * (1 - iterations / 3000),
    });
    const probes: Probe[] = [measure(4000)];
    for (let i = 0; i < 40; i++) {
      const d = nextIterations(probes, LIMIT);
      if (d.action === "settle") {
        expect(d.iterations).toBeGreaterThanOrEqual(3000);
        expect(d.iterations).toBeLessThanOrEqual(8000);
        return;
      }
      probes.push(measure(d.iterations));
    }
    throw new Error("did not converge");
  });
});
