# Licensing

## Mode: GPL port

This project is developed in **GPL port mode**. It studies and adapts
implementation details from [FractalShark](https://github.com/mattsaccount364/FractalShark)
(© Matt Renzelmann and contributors, GPL-3.0), and is therefore licensed under the
**GNU General Public License v3.0 or later**.

That choice is deliberate and recorded here before any source was consulted, as
required by the project brief.

### Attribution

- **FractalShark** — GPL-3.0 — https://github.com/mattsaccount364/FractalShark
  Source of the overall architecture for arbitrary-precision reference-orbit
  generation on the GPU, perturbation rendering, reference rebasing, and
  linear-approximation acceleration.

The CUDA implementation is not translated mechanically. WebGPU has no native
`u64`, no grid-wide barriers and no cooperative groups, so the same mathematical
pipeline is reimplemented with browser-native algorithms (emulated 64-bit
arithmetic over `u32` pairs, single-workgroup barriers instead of grid sync).

### Prior art also consulted

- K. I. Martin, *Superfractalthing Maths* — perturbation theory for the
  Mandelbrot set.
- Zhuoran, *Another solution to perturbation glitches* (fractalforums) — the
  rebasing rule used in the per-pixel iteration.
- Claude Heiland-Allen, *Perturbation glitches* and *bilinear approximation*
  write-ups.

## Deviation from the brief: fixed-point instead of floating-point limbs

The brief specifies a `GpuBigFloat { sign, exponent, limbs }`. The reference
orbit does not need an exponent: every sample satisfies `|X_n| < 4` or the orbit
has already escaped, and the centre coordinate satisfies `|C| < 2`. The engine
therefore stores orbit state as **two's-complement fixed point** with
`32 × (LIMBS − 1)` fractional bits.

This removes exponent alignment from the inner loop entirely — addition and
subtraction become plain limbwise carry chains — while covering exactly the
range the recurrence can produce. Values that *do* need a wide exponent are the
per-pixel deltas, and those are carried by the HDR sample format
(`mantissaHi/mantissaLo/exponent`) on the render side, as the brief describes.
