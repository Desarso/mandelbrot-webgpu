# Mandelbrot

A browser Mandelbrot renderer with an arbitrary-precision reference orbit
computed on the GPU, and perturbation rendering whose per-pixel deltas carry
their own exponent. Solid + Vite. GPLv3 — see [LICENSE.md](LICENSE.md).

```bash
pnpm install
pnpm dev
```

## Two backends

The app picks **WebGPU** when the browser exposes it and falls back to
**WebGL2** otherwise. The current backend and its precision are shown in the
View section of the control panel.

| | WebGL2 | WebGPU |
|---|---|---|
| Reference orbit | decimal.js on the CPU, uploaded as an `RG32F` texture | arbitrary-precision fixed point, GPU-resident |
| Per-pixel delta | plain `f32` | mantissa + explicit exponent |
| Depth limit | span `1e-34` (measured: `1e-34` renders, `1e-36` collapses) | bounded by limb count, not the renderer |
| Linear approximation | no | yes — 21.7x at 2.8e40x, verified bit-identical |
| Precision | ~15 digits | 8–256 limbs = 67–2450 decimal digits, chosen from the zoom |

The WebGL depth limit is not the orbit — it is the per-pixel delta underflowing
to subnormal `f32`. That is the specific thing the WebGPU path fixes.

## How the WebGPU path works

**Fixed point, not floating point.** Every reference sample satisfies `|z| < 4`
or the orbit has already escaped, and `|c| < 2`, so no exponent field is needed.
Values are two's-complement integers over `LIMBS` u32 words scaled by
`2^-(32*(LIMBS-1))`. Addition and subtraction are plain carry chains with no
alignment step. (This is a deliberate departure from the brief's
`GpuBigFloat {sign, exponent, limbs}` — rationale in LICENSE.md.)

**One workgroup.** `z <- z² + c` is strictly serial, so the only parallelism
available is *inside* each big-number operation. Running the whole recurrence in
a single workgroup gives barriers instead of the grid-wide sync WebGPU lacks,
and lets one dispatch advance hundreds of iterations with no CPU round trip.
Multiplication spreads its columns across 256 threads; carry resolution is
serial.

**Nothing big comes back.** The CPU uploads the centre once, encodes batches of
dispatches, and reads a 4-word status buffer per batch. The high-precision state
never leaves the GPU. Samples are emitted in a reduced HDR format
(`mantissaHi, mantissaLo, exponent`) for the renderer to consume.

**Per-pixel deltas carry an exponent.** At a zoom of `1e-60` the delta starts
around `1e-60` and grows to `O(1)` before escaping — a range no `f32` holds. The
shader carries it as a mantissa/exponent pair, renormalised each iteration.

**Linear approximation skips most of the work.** While the delta stays far below
the reference, `dz <- 2*X*dz + dz^2 + d` is linear, so a whole range collapses
into `dz_{n+m} = A*dz_n + B*d`. Steps are precomputed in levels (8, 16, 32, …
iterations) and a pixel greedily takes the largest one whose validity radius
still contains its delta. Measured at 2.8e40x: **21.7x faster, 96.9% of
iterations skipped, and the image is bit-identical** to the unapproximated
render. At shallow zoom the deltas are too large to qualify and it correctly
declines to fire — those views are already fast.

`A` over a 4096-iteration step routinely reaches 10^700, so coefficients and
radii carry explicit exponents; stored as plain doubles they become `Infinity`
and every radius turns into `NaN`.

Orbits are cached and reused while the view stays near the point they were built
at, so panning does not pay for regeneration.

## Finding somewhere worth looking

Hand-zooming past ~1e-30 almost always lands in a smooth region where every
pixel escapes within an iteration or two of every other — correct, and dull.
**Find minibrot** runs Newton on `f_p(c) = 0` for the p-th iterate, which
converges on the centre of a nearby period-p minibrot, then uses the standard
size estimate to frame it and raises the iteration count to match the period.

Newton converges to the *nearest* nucleus, so depth comes from where you start:
from a shallow point it lands on a big bulb, from a boundary point at depth it
reaches minibrots of size 1e-43 and smaller.

```
src/
  arithmetic/    fixed-point representation + exact BigInt oracle
  gpu/           device setup, WGSL arithmetic library, self-test
  orbit/         batched reference-orbit driver
  render/        backends (webgl, webgpu), perturbation shader
  logic/         view state, input, URL encoding, colour settings
tests/           vitest, CPU oracle
```

## Verification

Three layers, because each catches things the others cannot.

```bash
pnpm test          # 30 checks: CPU oracle vs BigInt, no GPU needed
```

Open **`/selftest.html`** for the GPU layer — 18 checks that run the actual WGSL
against the BigInt oracle on real hardware: `mul32` over 4102 cases including
every boundary value, big add/sub/multiply at 8/16/64 limbs, and reference
orbits compared sample-for-sample. Batch sizes 1, 7 and 128 are checked
separately; a single-dispatch run cannot catch a batch-boundary bug, and one
lived there until it was.

Open **`/gpu.html?cx=…&cy=…&span=…&i=…`** to render one view on the WebGPU path
with timings, plus a live comparison of the reduced orbit against an exact
BigInt orbit computed in the page. `&p=6` and `&p=7` switch to debug views
(iteration count / escape flag, and delta magnitudes).

Add **`&verify=1`** to render the same view twice — with and without linear
approximation — and report the speedup alongside how many sampled pixels
differ. An approximation that changes the picture is not an approximation, and
this is what caught two real bugs in it.

Verified against ground truth: at spans `1e-40` and `1e-60` the GPU escape
counts match a BigInt oracle. Reference orbits match to ~1e-15, the limit of the
reduced sample format.

## Sharing views

The URL always holds the current state: `?v=` view, `?c=` colour, `?i=`
iterations. Coordinates are trimmed to the digits the zoom can resolve, then
written as a base36 mantissa — about 35% shorter than decimal. (LZ compression
measured *longer*: fractal coordinate digits are incompressible and its encoder
packs 6 bits per character.)

## Controls

Drag to pan, scroll to zoom (toward the cursor), `H` to hide the panel.
Interaction renders at 25% resolution and restores full resolution 160 ms after
you stop. Only resolution is reduced, never the iteration count — capping
iterations makes deep views collapse to a solid interior colour.

## Not done

- **NTT multiplication.** The schoolbook multiplier is `O(n²)`; fine through the
  256-limb top profile (~2450 digits), which is where zoom depth stops being
  limited by precision and starts being limited by iteration count.
- **Orbit compression.** Orbits are stored in full: 6 floats per sample, so
  200k iterations is ~4.8 MB.
- **Bilinear approximation.** Only the linear form is implemented.
- Browser matrix beyond Chrome on macOS; device-loss recovery.

Known weakness: at a minibrot nucleus the reference orbit returns to zero every
period, which forces a rebase each time. After a rebase the delta is `O(1)` and
too large for any approximation step, so linear approximation stops firing
exactly at the coordinates **Find minibrot** takes you to. Those views are
correct but slow.
