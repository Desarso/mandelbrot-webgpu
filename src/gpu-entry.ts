import Decimal from "decimal.js";
import { acquireGpu } from "./gpu/device";
import { WebGpuRenderer } from "./render/webgpu-renderer";
import { DEFAULT_COLORS } from "./logic/colorSettings";

const canvas = document.getElementById("c") as HTMLCanvasElement;
const stats = document.getElementById("stats")!;

const params = new URL(window.location.href).searchParams;
const centerX = params.get("cx") ?? "-0.743643887037158704752191506114774";
const centerY = params.get("cy") ?? "0.131825904205311970493132056385139";
const span = params.get("span") ?? "1e-8";
const iterations = Number.parseInt(params.get("i") ?? "1200", 10);
const palette = Number.parseInt(params.get("p") ?? "1", 10);
/** `&la=0` disables linear approximation, for A/B comparison. */
const useApprox = params.get("la") !== "0";
const num = (key: string, fallback: number) => {
  const v = Number.parseFloat(params.get(key) ?? "");
  return Number.isFinite(v) ? v : fallback;
};
const colorOverrides = {
  mode: num("mode", DEFAULT_COLORS.mode),
  colorDensity: num("density", DEFAULT_COLORS.colorDensity),
  slopeDepth: num("slope", DEFAULT_COLORS.slopeDepth),
  lightAngle: num("light", DEFAULT_COLORS.lightAngle),
  supersample: num("ss", DEFAULT_COLORS.supersample),
  slopeLighting: params.get("lighting") !== "0",
};

async function main() {
  Decimal.set({ precision: 200 });
  const ctx = await acquireGpu();
  const renderer = new WebGpuRenderer(ctx, canvas);
  await renderer.init();

  const spanDecimal = new Decimal(span);
  const result = await renderer.render({
    centerX: new Decimal(centerX),
    centerY: new Decimal(centerY),
    unitsPerPixel: spanDecimal.div(canvas.height),
    width: canvas.width,
    height: canvas.height,
    maxIterations: iterations,
    colors: { ...DEFAULT_COLORS, palette, ...colorOverrides },
    useApprox,
  });

  // `&verify=1`: render the same view with and without linear approximation and
  // report how far apart the images are. An approximation that changes the
  // picture is not an approximation, it is a bug.
  if (params.get("verify") === "1") {
    const points: [number, number][] = [];
    for (let y = 8; y < canvas.height; y += 17) {
      for (let x = 8; x < canvas.width; x += 19) points.push([x, y]);
    }

    const withApprox = await renderer.debugReadPixels(points);
    const approxStats = result;

    const plain = await renderer.render({
      centerX: new Decimal(centerX),
      centerY: new Decimal(centerY),
      unitsPerPixel: spanDecimal.div(canvas.height),
      width: canvas.width,
      height: canvas.height,
      maxIterations: iterations,
      colors: { ...DEFAULT_COLORS, palette, ...colorOverrides },
      useApprox: false,
    });
    const withoutApprox = await renderer.debugReadPixels(points);

    let differing = 0;
    let worstChannel = 0;
    for (let i = 0; i < points.length; i++) {
      let delta = 0;
      for (let ch = 0; ch < 3; ch++) {
        delta = Math.max(delta, Math.abs(withApprox[i][ch] - withoutApprox[i][ch]));
      }
      if (delta > 0) differing++;
      worstChannel = Math.max(worstChannel, delta);
    }

    stats.textContent = [
      `span           ${span}   (zoom ${new Decimal("2.8").div(spanDecimal).toExponential(2)}x)`,
      `iterations     ${iterations}`,
      `sampled        ${points.length} pixels`,
      ``,
      `with LA        ${approxStats.renderMs.toFixed(1)}ms, ${(approxStats.skipRatio * 100).toFixed(1)}% skipped`,
      `without LA     ${plain.renderMs.toFixed(1)}ms`,
      `speedup        ${(plain.renderMs / approxStats.renderMs).toFixed(1)}x`,
      ``,
      `pixels differing  ${differing} / ${points.length}`,
      `worst channel     ${worstChannel} / 255`,
      worstChannel <= 2 ? "MATCH" : "MISMATCH",
    ].join("\n");
    console.log("[verify]\n" + stats.textContent);
    return;
  }

  if (palette === 7) {
    const pts: [number, number][] = [[360, 240], [560, 240], [700, 240], [20, 240]];
    const px = await renderer.debugReadPixels(pts);
    stats.textContent = px
      .map(([r, g, b, a], i) => {
        return `pixel ${pts[i][0]},${pts[i][1]}  delta0.e=${r - 128}  dz.e=${g - 128}  |dz.m.x|≈${((b / 255) * 2).toFixed(3)}  lastRef≈${Math.round((a / 255) * 2000)}  (JS says refLength=${result.orbitLength})`;
      })
      .join("\n");
    console.log("[gpu debug]\n" + stats.textContent);
    return;
  }

  if (palette === 6) {
    const pts: [number, number][] = [
      [360, 240], [410, 240], [460, 240], [560, 240], [700, 240], [20, 240],
    ];
    const px = await renderer.debugReadPixels(pts);
    stats.textContent = px
      .map(([r, g, b], i) => {
        const n = Math.round((r / 255) * iterations);
        return `pixel ${pts[i][0]},${pts[i][1]}  rgb=${r},${g},${b}  n≈${n}  escaped=${g > 128}  refIter≈${Math.round((b / 255) * result.orbitLength)}`;
      })
      .join("\n");
    console.log("[gpu debug]\n" + stats.textContent);
    return;
  }

  // Compare the reduced orbit the renderer actually reads against an exact
  // BigInt orbit computed here, so a data problem cannot masquerade as a
  // shader-logic problem.
  const checkCount = Math.min(iterations, 600);
  const probe = await renderer.debugReadOrbit(checkCount + 1);
  const bits = 32n * 31n;
  const scaleDec = (t: string) => {
    const m = /^([+-]?)(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/.exec(t)!;
    const [, sg, w = "", fr = "", ex] = m;
    const p = BigInt(ex ?? "0") - BigInt(fr.length);
    let num = BigInt(w + fr || "0") << bits;
    let den = 1n;
    if (p >= 0n) num *= 10n ** p;
    else den = 10n ** -p;
    const s = num / den;
    return sg === "-" ? -s : s;
  };
  const bmul = (a: bigint, b: bigint) => {
    const neg = a < 0n !== b < 0n;
    const m = ((a < 0n ? -a : a) * (b < 0n ? -b : b)) >> bits;
    return neg ? -m : m;
  };
  const toNum = (v: bigint) => {
    if (v === 0n) return 0;
    const neg = v < 0n;
    const mag = neg ? -v : v;
    const sh = BigInt(Math.max(0, mag.toString(2).length - 53));
    const val = Number(mag >> sh) * 2 ** Number(sh - bits);
    return neg ? -val : val;
  };

  let ex = 0n;
  let ey = 0n;
  const cxb = scaleDec(centerX);
  const cyb = scaleDec(centerY);
  const lines: string[] = [];
  let firstBad = -1;
  let worst = 0;
  for (let i = 1; i <= checkCount; i++) {
    const xx = bmul(ex, ex);
    const yy = bmul(ey, ey);
    const xy = bmul(ex, ey);
    ex = xx - yy + cxb;
    ey = xy + xy + cyb;
    const gx = (probe[i * 6 + 0] + probe[i * 6 + 1]) * 2 ** probe[i * 6 + 2];
    const gy = (probe[i * 6 + 3] + probe[i * 6 + 4]) * 2 ** probe[i * 6 + 5];
    const err = Math.max(Math.abs(gx - toNum(ex)), Math.abs(gy - toNum(ey)));
    if (err > worst) worst = err;
    if (err > 1e-10 && firstBad < 0) {
      firstBad = i;
      lines.push(`  first mismatch at X${i}:`);
      lines.push(`    gpu   ${gx.toExponential(10)}  ${gy.toExponential(10)}`);
      lines.push(`    exact ${toNum(ex).toExponential(10)}  ${toNum(ey).toExponential(10)}`);
    }
  }
  lines.push(
    `  compared ${checkCount} samples (batch size 128), worst error ${worst.toExponential(2)}` +
      (firstBad < 0 ? " — all ok" : ` — FIRST BAD at ${firstBad}`)
  );

  stats.textContent = [
    `adapter        ${ctx.capabilities.adapterInfo}`,
    `centre         ${centerX}`,
    `               ${centerY}`,
    `span           ${span}   (zoom ${new Decimal("2.8").div(spanDecimal).toExponential(2)}x)`,
    `precision      ${result.limbs} limbs = ${result.decimalDigits} decimal digits`,
    `orbit          ${result.orbitLength} samples, escaped=${result.orbitEscaped}, ${result.orbitMs.toFixed(1)}ms`,
    `render         ${result.renderMs.toFixed(1)}ms for ${canvas.width}x${canvas.height}`,
    `approximation  ${(result.skipRatio * 100).toFixed(1)}% of iterations skipped ` +
      `(${result.approxSteps.toLocaleString()} steps, ${result.skippedIterations.toLocaleString()} iterations)`,
    `rebases        ${result.rebases.toLocaleString()}`,
    `orbit check    (gpu reduced samples vs exact BigInt)`,
    ...lines,
  ].join("\n");
  console.log("[gpu]", stats.textContent);
}

main().catch((error) => {
  stats.textContent = `ERROR: ${error?.stack ?? error}`;
  stats.className = "fail";
  console.error("[gpu]", error);
});
