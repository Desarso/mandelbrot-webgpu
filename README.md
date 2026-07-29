# mandelbrot-webgpu

A Mandelbrot renderer that runs in a browser tab and does not stop at double
precision. The reference orbit is computed in arbitrary precision **on the
GPU**, and per-pixel deltas carry their own exponent, so zoom depth is bounded
by patience rather than by `f64`.

Live at **[mandelbrot.gabrielmalek.com](https://mandelbrot.gabrielmalek.com)**.

- **[/about.html](https://mandelbrot.gabrielmalek.com/about.html)** — what the
  Mandelbrot set is and why it is worth looking at. No code.
- **[/tech.html](https://mandelbrot.gabrielmalek.com/tech.html)** — how it is
  computed, and who each technique is due to.

Both pages, and the whole interface, exist in 83 languages.

```bash
pnpm install
pnpm dev
```

GPLv3 — see [LICENSE](LICENSE), and [LICENSE.md](LICENSE.md) for why.

---

## What makes deep zooming hard

Colouring a pixel means iterating `z ← z² + c` until `|z| > 16` and counting the
steps. Straightforward, until the view gets small. At a span of `1e-16` two
adjacent pixels differ by less than a `double` can represent, so the whole
screen computes the same number and the image collapses into flat bands. Naive
rendering stops there — about fifty doublings past the full view.

Everything below exists to get past that.

## Perturbation: iterate once, reuse everywhere

Compute one orbit `X_n` at the view centre in high precision. Every other pixel
is that orbit plus a small offset, and the offset obeys an exact identity:

```
w ← 2·X·w + w² + δ
```

`w` is the pixel's deviation from the reference and `δ` its offset in the
complex plane. This is not an approximation — it is the original recurrence
rewritten. The expensive high-precision work happens once per frame; each of
the million pixels runs cheap arithmetic against it.

The catch is that `w` is only small while the pixel stays near the reference.
When it does not, the low-precision arithmetic loses all its significant
figures and the pixel comes out visibly wrong — the "glitches" that a decade of
increasingly elaborate correction schemes tried to repair. This renderer uses
**rebasing** instead: whenever `|z| < |w|`, restart the pixel against the
beginning of the same orbit. Three lines, no glitch list, no second reference.

## Arbitrary precision on the GPU

**Fixed point, not floating point.** Every reference sample satisfies `|z| < 16`
or the orbit has already escaped, and `|c| < 2`, so no exponent field is needed.
Values are two's-complement integers over `LIMBS` u32 words scaled by
`2^-(32·(LIMBS-1))`. Addition and subtraction are plain carry chains with no
alignment step.

**WGSL has no 64-bit integers.** A 32×32→64 multiply is emulated with four
16-bit partial products, and the column sums are accumulated in a way that
cannot overflow before the carry pass.

**One workgroup, not a grid.** `z ← z² + c` is strictly serial, so the only
parallelism available is *inside* each big-number operation. Running the whole
recurrence in a single workgroup gives barriers instead of the grid-wide
synchronisation WebGPU does not have, and lets one dispatch advance hundreds of
iterations with no CPU round trip. Multiplication spreads its columns across 256
threads; carry resolution is serial.

**Nothing big comes back.** The CPU uploads the centre once, encodes batches of
dispatches, and reads a four-word status buffer per batch. The high-precision
state never leaves the GPU. Samples are emitted in a reduced format
(`mantissaHi, mantissaLo, exponent`) for the renderer to consume.

Limb count is chosen from the zoom: 8 limbs (67 decimal digits) covers
everything down to about `1e-42`, and the top profile is 256 limbs.

## Deltas that carry their own exponent

At a zoom of `1e-60` the per-pixel delta starts around `1e-60` and grows to
`O(1)` before escaping — a dynamic range no `f32` holds. It is carried as an
explicit mantissa/exponent pair and renormalised every iteration. This, not the
orbit, is what set the depth limit of the older WebGL path: its `f32` deltas
simply underflowed to zero.

Two subtleties, both of which produced silently wrong images before they were
found:

- Renormalising with a single `exp2(-shift)` overflows as soon as the mantissa
  drops below `2^-128`, which happens whenever the reference passes near zero.
  The rescale is done in two halves.
- A value wrapped as `Hdr(x, 0)` keeps its magnitude in the *mantissa*. The
  comparison used by the rebase test looks at exponents first, so at a minibrot
  nucleus — where the reference returns to within `1e-24` of zero every period —
  it concluded `|z| ≫ |w|` at exactly the moment `|z| < |w|`. Rebasing never
  fired, the delta stopped growing, every pixel hit the iteration cap, and the
  frame came out solid black.

## Skipping most of the work

While the delta stays far below the reference, the recurrence is nearly linear,
so a whole range of iterations collapses into one truncated polynomial:

```
w_out = A·w + B·δ + C·w² + D·w·δ + E·δ²
```

Keeping only `A` and `B` is the usual bilinear approximation; carrying the
second-order terms pushes the first neglected term from quadratic to cubic, so
the same error budget admits an entry delta larger by roughly its square root.
Steps are built in doubling levels (8, 16, 32, … iterations) and a pixel
greedily takes the largest one its delta fits inside.

Measured at 2.8e40×: **21× faster, 96.9% of iterations skipped, bit-identical**
to the unapproximated render. At shallow zoom the deltas are too large to
qualify and it correctly declines to fire — those views are already fast.

`A` over a 4096-iteration step routinely reaches 10^700, so coefficients and
radii carry explicit exponents; stored as plain doubles they become `Infinity`
and every radius turns into `NaN`.

## Choosing a method by depth

Perturbation is the right algorithm at `1e-40` and the wrong one at `1e-2`:
zoomed out the delta is the same size as `z`, so the rebase test fires almost
every iteration — sixty million times on the home view — and a high-precision
reference orbit gets built for a view that cannot resolve it.

| units/pixel | method | what it does |
| --- | --- | --- |
| `> 1e-5` | direct | plain `f32` `z ← z² + c`, no reference orbit at all |
| `> 1e-25` | plain | perturbation with a plain `f32` delta |
| otherwise | hdr | perturbation with an explicit exponent, plus the skip table |

Measured at 720×480, render time only: `4e-3` goes from 39 ms + 37 ms of orbit
to **15 ms**; `1e-11` from 90 ms to **25 ms**; `2e-21` from 193 ms to **132 ms**.
Past `1e-25` the skip table starts carrying the frame and the exponent-carrying
delta wins again, which is where the handover sits — still thirteen decades
above the `f32` floor.

## Computing a frame and colouring it are separate

`compute` iterates every sub-sample and stores what the colouring will need: the
iteration count and `|z|²`, or the height field in distance mode. `shadePass`
reads that and applies the palette, the relief lighting and the gamma. The
stored field is keyed on everything that changes the *numbers* — position,
scale, iteration count, mode, sample grid, method — so changing a palette
recomputes nothing.

At `1e-25` with 20,000 iterations: a full frame is **247 ms**, a recolour is
**1.0 ms**.

Distance mode got faster outright as well. The relief lighting needs the height
of neighbouring samples and used to obtain them by iterating each one again —
three full evaluations per sub-sample. The gradient now reads the stored field:
**20.6 ms** against roughly 90 ms at `1e-8`.

## Colouring

Two modes. **Iteration** is the classic escape-count banding, with optional
smooth shading, `sqrt`/`log` remapping and mirrored bands. **Distance** uses the
derivative of the orbit to estimate how far each pixel lies from the set, which
resolves filaments that banding misses, then treats that distance as a height
field: a screen-space gradient gives a normal, and diffuse plus Blinn-Phong
lighting produces the embossed, pseudo-3D look. Palette mixing and lighting run
in linear light with a single gamma encode at the end.

## Staying responsive

- **Banded rendering.** A frame is four horizontal bands with a yield between
  them, so a gesture arriving mid-frame costs one band rather than a screen.
  Band size adapts to measured cost, targeting 50 ms per submission — a command
  that runs too long is killed by the driver watchdog, which takes the device
  and the tab with it. Fencing between bands is the obvious implementation and
  is badly wrong: it drains the pipeline, and eight fenced bands took 569 ms
  against 215 ms unsplit.
- **Reprojection.** A pan or zoom moves the picture far more often than it
  changes it, so the last completed frame is re-blitted under the new view
  transform while the real one computes. One full-screen triangle, and exact
  wherever the views overlap at the same scale. It samples a separate history
  texture that only ever receives whole frames — the render target can be caught
  mid-update, and presenting that puts a seam on screen.
- **Reduced resolution while interacting**, restored 160 ms after you stop. Only
  resolution is reduced, never the iteration count: capping iterations makes deep
  views collapse to a solid interior colour.

## Finding somewhere worth looking

Hand-zooming past ~`1e-30` almost always lands in a smooth region where every
pixel escapes within an iteration or two of every other — correct, and dull.
**Find minibrot** runs Newton on `f_p(c) = 0` for the p-th iterate, which
converges on the centre of a nearby period-p minibrot, then uses the standard
size estimate to frame it and raises the iteration count to match the period.

Newton converges to the *nearest* nucleus, so depth comes from where you start:
from a shallow point it lands on a big bulb, from a boundary point at depth it
reaches minibrots of size `1e-43` and smaller. The Places tab is a set of
destinations found this way.

## Verification

Fractal renderers fail quietly — a subtly wrong image still looks like a
fractal. Three layers, because each catches things the others cannot.

```bash
pnpm test          # 51 checks: CPU oracle vs BigInt, no GPU needed
```

Open **`/selftest.html`** for the GPU layer — 18 checks that run the actual WGSL
against the BigInt oracle on real hardware: `mul32` over 4102 cases including
every boundary value, big add/sub/multiply at 8/16/64 limbs, and reference
orbits compared sample-for-sample. Batch sizes 1, 7 and 128 are checked
separately; a single-dispatch run cannot catch a batch-boundary bug, and one
lived there until it was.

Open **`/gpu.html?cx=…&cy=…&span=…&i=…`** to render one view on the WebGPU path
with timings, plus a live comparison of the reduced orbit against an exact
BigInt orbit computed in the page.

| flag | what it does |
| --- | --- |
| `&mode=2` | diagnostic view: iterations, escape flag, `log2` of the delta, rebase count |
| `&m=0`, `1`, `2` | forces direct / plain / hdr, for A/B comparison |
| `&la=0` | disables the approximation |
| `&verify=1` | renders with and without approximation and diffs the pixels |
| `&bench=1` | times each configuration repeatedly with the orbit warm |

`&verify=1` exists because an approximation that changes the picture is not an
approximation; it caught two real bugs. `&bench=1` exists because single renders
across page loads are not comparable — orbit generation alone ranged from 292 ms
to 15 s on the same machine within a few minutes, and three conclusions drawn
from single measurements turned out to be noise.

## Sharing views

The URL always holds the current state: `?v=` view, `?c=` colour, `?i=`
iterations, `?lang=` language. Coordinates are trimmed to the digits the zoom
can resolve, then written as a base36 mantissa — about 35% shorter than decimal.
(LZ compression measured *longer*: fractal coordinate digits are incompressible
and its encoder packs 6 bits per character.)

## Pages and translation

Every user-visible string lives in `src/i18n/strings.ts`.
`scripts/translate.mjs` feeds those through
[BulkTranslatorGo](https://github.com/Desarso/BulkTranslator) and writes one
JSON file per language into `src/i18n/locales/`, which Vite emits as lazy chunks
— a visitor downloads one translation, not eighty-two. The language comes from
`navigator.languages`, and can be overridden with `?lang=` or the picker in the
Advanced tab; a missing key falls back to English. RTL scripts set `dir="rtl"`.

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

## Layout

```
src/
  arithmetic/    fixed-point representation + exact BigInt oracle
  gpu/           device setup, WGSL arithmetic library, self-test
  orbit/         nucleus finding (Newton, period detection, size estimate)
  render/        backends (webgl, webgpu), perturbation shader, BLA table
  logic/         view state, input, URL encoding, colour settings, locations
  i18n/          source strings, generated locales, language detection
scripts/         translation generator
tests/           vitest, CPU oracle
```

The app picks **WebGPU** when the browser exposes it and falls back to
**WebGL2** otherwise; the WebGL path runs out of precision around `1e-34`. The
current backend and its precision are shown in the View section of the panel,
and a notice explains how to enable WebGPU when it is available but unused.

## Measured and rejected

Techniques that were evaluated and found not to help here:

- **NTT multiplication.** Almost every view runs at eight limbs — `6e-42` needs
  67 decimal digits — where schoolbook is 64 limb products. NTT wins in the
  thousands of bits, not the hundreds.
- **Orbit compression.** 24 bytes a sample, so 200k iterations is ~4.8 MB.
  Generating the orbit costs 1.6 s at that length; storing it costs nothing
  worth reclaiming.
- **Series approximation** of the initial segment. Bilinear approximation is
  strictly more general — it applies at any point in the orbit rather than only
  the start — and is already implemented, to second order.
- **Web Worker split.** The only synchronous main-thread work of consequence is
  building the skip table, 17–25 ms once per orbit. Everything else is GPU work
  behind an `await`. An OffscreenCanvas rewrite to move 20 ms is not worth it.

Genuinely not done: browser matrix beyond Chrome on macOS, and recovery from a
lost GPU device (it is detected and reported, but the page must be reloaded).

## Credits

Almost none of the mathematics here is original. It was worked out over about
fifteen years, mostly by people posting derivations on a forum and giving the
code away.

| Idea | Due to |
| --- | --- |
| Perturbation | K. I. Martin, *Superfractalthing Maths* (2013) |
| Glitch detection | Pauldelbrot, on fractalforums (2014) |
| Rebasing | Zhuoran, on fractalforums.org (2021) |
| Bilinear approximation | Zhuoran (2021), written up in detail by Claude Heiland-Allen |
| Distance estimation | Hart, Sandin and Kauffman (1989); the Mandelbrot form popularised by Iñigo Quílez |
| Nucleus finding | Douady–Hubbard theory; practical Newton and size estimates from Claude Heiland-Allen and Robert Munafo's *Mu-Ency* |
| GPU arbitrary precision | [FractalShark](https://github.com/mattsaccount364/FractalShark) by Matt Renzelmann, whose approximation code came by way of [Fractal Zoomer](https://github.com/hrkalona/Fractal-Zoomer) by Kalonakis Christos |
| The colouring | Maths Town's *The Hardest Trip*; cosine palettes from Iñigo Quílez |

[Kalles Fraktaler](https://mathr.co.uk/kf/kf.html) (Karl Runmo, later Claude
Heiland-Allen) and FractalShark are the reference implementations for anyone
doing this seriously; both are far more capable than this. The threads at
[fractalforums.org](https://fractalforums.org/) are where most of the above was
derived in public, usually years before it appeared in any paper.

## Deployment

A two-stage `Dockerfile` builds with pnpm and serves `dist` with nginx. Coolify's
"static" build pack serves the repository as-is and never runs a build, which
ships raw `.tsx`; owning the build here avoids depending on that.
