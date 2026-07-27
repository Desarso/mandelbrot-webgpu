# Mandelbrot

A browser Mandelbrot renderer with an arbitrary-precision reference orbit
computed on the GPU, and perturbation rendering whose per-pixel deltas carry
their own exponent. Solid + Vite. GPLv3 — see [LICENSE.md](LICENSE.md).

Live at **[mandelbrot.gabrielmalek.com](https://mandelbrot.gabrielmalek.com)**,
with a walkthrough of the techniques at
[/tech.html](https://mandelbrot.gabrielmalek.com/tech.html).

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
| Approximation | no | second-order BLA — 21x at 2.8e40x, verified bit-identical |
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

**Second-order approximation skips most of the work.** While the delta stays
far below the reference, `w <- 2*X*w + w^2 + d` is nearly linear, so a whole
range collapses into a truncated polynomial

    w_out = A*w + B*d + C*w^2 + D*w*d + E*d^2

Keeping only `A` and `B` is the usual bilinear approximation; carrying the
second-order terms pushes the first neglected term from quadratic to cubic, so
the same error budget admits an entry delta larger by roughly its square root.
Steps are built in doubling levels (8, 16, 32, … iterations) and a pixel
greedily takes the largest one its delta fits inside. Measured at 2.8e40x:
**21x faster, 96.9% of iterations skipped, bit-identical** to the
unapproximated render. At shallow zoom the deltas are too large to qualify and
it correctly declines to fire — those views are already fast.

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

## Choosing a method by depth

Perturbation is the right algorithm at 1e-40 and the wrong one at 1e-2: zoomed
out the delta is the same size as `z`, so the rebase test fires nearly every
iteration. Three tiers, picked from units-per-pixel:

| units/pixel | method | what it does |
| --- | --- | --- |
| `> 1e-5` | direct | plain f32 `z <- z² + c`, no reference orbit at all |
| `> 1e-25` | plain | perturbation with a plain f32 delta |
| otherwise | hdr | perturbation with an explicit exponent, plus the skip table |

Measured at 720x480, render time only: 4e-3 goes 39ms + 37ms of orbit to 15ms;
1e-11 goes 90ms to 25ms; 2e-21 goes 193ms to 132ms. Past 1e-25 the skip table
starts carrying the frame and the exponent-carrying delta wins again, which is
where the handover sits — still thirteen decades above the f32 floor.

## Computing a frame and colouring it are separate

`compute` iterates every sub-sample and stores what the colouring needs
(iteration count and `|z|²`, or the height field in distance mode).
`shadePass` reads that and applies the palette, relief lighting and gamma. The
stored field is keyed on everything that changes the numbers, so changing a
palette does not recompute anything.

At 1e-25 with 20,000 iterations: a full frame is 247ms, a recolour is 1.0ms.

Distance mode got faster outright as well. The relief lighting needs the height
of neighbouring samples, and used to get them by iterating each one again —
three full evaluations per sub-sample. The gradient now reads the stored field:
20.6ms against roughly 90ms at 1e-8.

## Staying responsive

- **Banded rendering.** A frame is four horizontal bands with a yield between
  them, so a gesture arriving mid-frame costs one band rather than a screen.
  Band size adapts to measured cost, targeting 50ms per submission — a command
  that runs too long is killed by the driver watchdog, which takes the device
  and the tab with it. Fencing between bands is the obvious implementation and
  is badly wrong: it drains the pipeline, and eight fenced bands took 569ms
  against 215ms unsplit.
- **Reprojection.** A pan or zoom moves the picture far more often than it
  changes it, so the last completed frame is re-blitted under the new view
  transform while the real one computes. One full-screen triangle, exact
  wherever the views overlap at the same scale.

## Not done

Measured and rejected, rather than skipped:

- **NTT multiplication.** Almost every view runs at eight limbs — 6e-42 needs
  67 decimal digits — and schoolbook is 64 limb products there. NTT wins in the
  thousands of bits, not the hundreds; it would be far slower everywhere this
  renderer actually operates.
- **Orbit compression.** 24 bytes a sample, so 200k iterations is ~4.8 MB.
  Generating the orbit costs 1.6s at that length; storing it costs nothing
  worth reclaiming.
- **Series approximation** of the initial segment. Bilinear approximation is
  strictly more general — it applies at any point in the orbit rather than only
  the start — and is already implemented, to second order.
- **Web Worker split.** The only synchronous main-thread work of consequence is
  building the skip table, 17-25ms once per orbit. Everything else is GPU work
  behind an `await`. An OffscreenCanvas rewrite to move 20ms is not worth it.

Genuinely not done:

- Browser matrix beyond Chrome on macOS.
- Device loss is detected and reported, but not recovered from: the page has to
  be reloaded.

## Pages and translation

`/` is the renderer, `/about.html` explains what the Mandelbrot set is for a
general reader, and `/tech.html` explains how it is computed. The two were
deliberately split: the second is about the renderer, the first is about why
the thing it renders is worth looking at.

Every user-visible string lives in `src/i18n/strings.ts`. `scripts/translate.mjs`
feeds those through [BulkTranslatorGo](https://github.com/Desarso/BulkTranslator)
and writes one JSON file per language into `src/i18n/locales/`, which Vite emits
as lazy chunks — a visitor downloads one translation, not eighty-two. The
language comes from `navigator.languages`, and can be overridden with `?lang=`
or the picker in the Advanced tab; a missing key falls back to English.

```bash
go install github.com/Desarso/BulkTranslator/BulkTranslatorGo@latest
node scripts/translate.mjs            # all languages
node scripts/translate.mjs de ja      # just these
node scripts/translate.mjs --missing  # only keys added since last run
```

Two things about that endpoint are worth knowing, because both fail silently.
It returns only the *first sentence* of anything multi-sentence, so the script
splits on sentence boundaries and rejoins. And it sometimes drops a language
from a multi-destination request entirely — one batch of eight came back as
untranslated English — so the script checks each result against the source and
refuses to write a locale that looks untranslated.

## Deployment

A two-stage `Dockerfile` builds with pnpm and serves `dist` with nginx. Coolify's
"static" build pack serves the repository as-is and never runs a build, which
ships raw `.tsx`; owning the build here avoids depending on that.
