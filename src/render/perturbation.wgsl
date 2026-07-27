// Per-pixel perturbation rendering against a GPU-generated reference orbit.
//
// The reference samples X_n are O(1) and fit in f32 directly. The per-pixel
// delta does not: at a zoom of 1e-60 it starts around 1e-60 and grows to O(1)
// before escaping, a dynamic range no f32 can hold. So the delta is carried as
// an explicit mantissa/exponent pair ("Hdr" below) and renormalised every
// iteration. That, not the orbit, is what sets the depth limit of the old
// WebGL path -- its f32 deltas simply underflow to zero.

struct Uniforms {
    resolution: vec2<f32>,
    // Complex units per pixel, as mantissa * 2^exponent.
    scaleMantissa: f32,
    scaleExponent: i32,
    // View centre relative to the reference point, same encoding.
    offsetMantissa: vec2<f32>,
    offsetExponent: i32,
    maxIterations: u32,
    refLength: u32,
    palette: u32,
    colorCycle: f32,
    colorOffset: f32,
    mapping: u32,
    mirror: u32,
    smoothShading: u32,
    interior: vec3<f32>,
    stopCount: u32,
    // Linear approximation: levels of precomputed skips. laLevels == 0 disables.
    laLevels: u32,
    laBaseStep: u32,
    // Distance-estimation colouring.
    mode: u32,              // 0 iteration bands, 1 distance estimation
    colorDensity: f32,
    colorPhase: f32,
    slopeDepth: f32,
    lightDir: vec3<f32>,    // normalised, z is elevation out of the screen
    ambientLight: f32,
    diffuseStrength: f32,
    specularStrength: f32,
    slopeLighting: u32,
    supersample: u32,
    invGamma: f32,
    /// 0 direct f32, 1 plain-f32 perturbation, 2 HDR perturbation.
    method: u32,
    /// View centre as plain f32, used only by the direct method.
    centre: vec2<f32>,
    /// First screen row this dispatch covers, for tiled rendering.
    rowOffset: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> orbit: array<f32>;   // hi, lo, exp per component
@group(0) @binding(1) var<uniform> u: Uniforms;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<storage, read> stops: array<vec4<f32>>;
/** Ax, Ay, Ae, Bx, By, Be, radiusLog2, pad — per skip entry. */
@group(0) @binding(4) var<storage, read> la: array<f32>;
/** levelOffsets then levelCounts, laLevels each. */
@group(0) @binding(5) var<storage, read> laIndex: array<u32>;
/** [0] iterations skipped, [1] LA steps taken, [2] rebases, [3] plain steps. */
@group(0) @binding(6) var<storage, read_write> stats: array<atomic<u32>>;

const TAU: f32 = 6.283185307179586;
const ESCAPE_R: f32 = 16.0;
const ESCAPE_R2: f32 = 256.0;

// ------------------------------------------------------- mantissa/exponent pair

struct Hdr {
    m: vec2<f32>,
    e: i32,
};

fn hdrZero() -> Hdr { return Hdr(vec2<f32>(0.0), 0); }

fn hdrNorm(a: Hdr) -> Hdr {
    let mx = max(abs(a.m.x), abs(a.m.y));
    if (mx == 0.0 || mx != mx || mx > 3.0e38) { return hdrZero(); }
    let shift = i32(floor(log2(mx)));

    // Rescale in two halves. A single exp2(-shift) overflows f32 as soon as
    // the mantissa drops below 2^-128, which happens whenever the reference
    // orbit passes near zero — every period at a minibrot nucleus. The
    // resulting Infinity destroyed the delta and the pixel could never escape.
    let half = exp2(f32(-shift) * 0.5);
    return Hdr(a.m * half * half, a.e + shift);
}

fn hdrAdd(a: Hdr, b: Hdr) -> Hdr {
    if (a.m.x == 0.0 && a.m.y == 0.0) { return b; }
    if (b.m.x == 0.0 && b.m.y == 0.0) { return a; }
    let d = a.e - b.e;
    // Past ~24 bits of separation the smaller term cannot change the result.
    if (d > 40) { return a; }
    if (d < -40) { return b; }
    if (d >= 0) {
        return hdrNorm(Hdr(a.m + b.m * exp2(f32(-d)), a.e));
    }
    return hdrNorm(Hdr(a.m * exp2(f32(d)) + b.m, b.e));
}

fn hdrMul(a: Hdr, b: Hdr) -> Hdr {
    let m = vec2<f32>(
        a.m.x * b.m.x - a.m.y * b.m.y,
        a.m.x * b.m.y + a.m.y * b.m.x
    );
    return hdrNorm(Hdr(m, a.e + b.e));
}

/// Multiplies by a plain complex number whose magnitude is around 1.
fn hdrMulPlain(a: Hdr, b: vec2<f32>) -> Hdr {
    let m = vec2<f32>(
        a.m.x * b.x - a.m.y * b.y,
        a.m.x * b.y + a.m.y * b.x
    );
    return hdrNorm(Hdr(m, a.e));
}

/// True when |a| < |b|, comparing exponent first.
fn hdrLess(a: Hdr, b: Hdr) -> bool {
    let am = dot(a.m, a.m);
    let bm = dot(b.m, b.m);
    if (am == 0.0) { return bm != 0.0; }
    if (bm == 0.0) { return false; }
    if (a.e != b.e) { return a.e < b.e; }
    return am < bm;
}

/// Collapses to a plain vec2 when the exponent is in f32 range; otherwise 0,
/// which is the right answer for an escape test on a tiny value.
fn hdrValue(a: Hdr) -> vec2<f32> {
    if (a.e < -120 || a.e > 120) { return vec2<f32>(0.0); }
    return a.m * exp2(f32(a.e));
}

// ------------------------------------------------------------- reference orbit

fn refSample(i: u32) -> vec2<f32> {
    let base = i * 6u;
    let x = (orbit[base + 0u] + orbit[base + 1u]) * exp2(orbit[base + 2u]);
    let y = (orbit[base + 3u] + orbit[base + 4u]) * exp2(orbit[base + 5u]);
    return vec2<f32>(x, y);
}

/// The reference sample as a *normalised* Hdr.
///
/// hdrLess compares exponents before mantissas, and hdrAdd returns an operand
/// verbatim when the exponents differ by more than 40, so anything that reaches
/// a comparison must carry its magnitude in the exponent rather than the
/// mantissa. Wrapping a raw sample as Hdr(x, 0) breaks that: deep inside a
/// minibrot the reference passes within ~1e-24 of zero every period, and the
/// unnormalised form claimed exponent 0 there — so |z| < |dz| never fired and
/// the pixel never rebased.
fn refHdr(i: u32) -> Hdr {
    return hdrNorm(Hdr(refSample(i), 0));
}

// -------------------------------------------------------------------- colour

fn cosPalette(t: f32, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return a + b * cos(TAU * (c * t + d));
}

fn ultraFractal(t: f32) -> vec3<f32> {
    var p = array<f32, 6>(0.0, 0.16, 0.42, 0.6425, 0.8575, 1.0);
    var c = array<vec3<f32>, 6>(
        vec3<f32>(0.000, 0.027, 0.392),
        vec3<f32>(0.125, 0.420, 0.796),
        vec3<f32>(0.929, 1.000, 1.000),
        vec3<f32>(1.000, 0.667, 0.000),
        vec3<f32>(0.000, 0.008, 0.000),
        vec3<f32>(0.000, 0.027, 0.392)
    );
    var col = c[0];
    for (var i = 0; i < 5; i = i + 1) {
        col = mix(col, c[i + 1], smoothstep(p[i], p[i + 1], t));
    }
    return col;
}

fn customPalette(t: f32) -> vec3<f32> {
    let count = max(u.stopCount, 1u);
    if (count == 1u) { return stops[0].rgb; }
    let scaled = t * f32(count);
    let index = u32(floor(scaled)) % count;
    let next = (index + 1u) % count;
    return mix(stops[index].rgb, stops[next].rgb, fract(scaled));
}

fn palette(tIn: f32) -> vec3<f32> {
    let t = clamp(tIn, 0.0, 1.0);
    if (u.palette == 1u) { return ultraFractal(t); }
    if (u.palette == 2u) {
        return cosPalette(t, vec3<f32>(0.50, 0.30, 0.20), vec3<f32>(0.50, 0.35, 0.25),
                             vec3<f32>(1.0), vec3<f32>(0.00, 0.10, 0.20));
    }
    if (u.palette == 3u) {
        return cosPalette(t, vec3<f32>(0.45, 0.50, 0.60), vec3<f32>(0.35, 0.40, 0.40),
                             vec3<f32>(1.0, 1.0, 0.9), vec3<f32>(0.60, 0.70, 0.85));
    }
    if (u.palette == 4u) { return vec3<f32>(0.5 - 0.5 * cos(TAU * t)); }
    if (u.palette == 5u) { return customPalette(t); }
    return cosPalette(t, vec3<f32>(0.5), vec3<f32>(0.5), vec3<f32>(1.0),
                         vec3<f32>(0.00, 0.33, 0.67));
}

fn wrapCoordinate(t: f32) -> f32 {
    if (u.mirror == 1u) {
        let m = t - 2.0 * floor(t / 2.0);
        if (m > 1.0) { return 2.0 - m; }
        return m;
    }
    return fract(t);
}

// ------------------------------------------------- linear approximation steps

const LA_NEVER: f32 = -1e29;
/**
 * Safety margin, in log2, between a pixel's delta and a step's stated radius.
 *
 * At the radius itself the truncation error is only as small as the tolerance,
 * and marginal steps visibly shift escape counts. The margin is a pure win at
 * depth — deltas there sit tens of orders below any radius — and simply stops
 * shallow views taking the borderline steps they gain nothing from.
 */
const LA_MARGIN_LOG2: f32 = 24.0;

/**
 * A precomputed skip: the range's map truncated to second order,
 * `w_out = A*w + B*d + C*w^2 + D*w*d + E*d^2`, plus the entry radius it holds
 * for. The second-order terms are what let the radius be roughly the square
 * root of the first-order one instead of proportional to it.
 */
struct Skip {
    a: Hdr,
    b: Hdr,
    c: Hdr,
    d: Hdr,
    e: Hdr,
    radiusLog2: f32,
};

const SKIP_FLOATS: u32 = 20u;

fn loadCoefficient(base: u32, slot: u32) -> Hdr {
    let at = base + slot * 3u;
    return Hdr(vec2<f32>(la[at], la[at + 1u]), i32(la[at + 2u]));
}

fn loadSkip(entry: u32) -> Skip {
    let base = entry * SKIP_FLOATS;
    return Skip(
        loadCoefficient(base, 0u),
        loadCoefficient(base, 1u),
        loadCoefficient(base, 2u),
        loadCoefficient(base, 3u),
        loadCoefficient(base, 4u),
        la[base + 15u]
    );
}

/// Applies the truncated polynomial.
fn applySkip(skip: Skip, w: Hdr, d: Hdr) -> Hdr {
    var out = hdrAdd(hdrMul(skip.a, w), hdrMul(skip.b, d));
    out = hdrAdd(out, hdrMul(skip.c, hdrMul(w, w)));
    out = hdrAdd(out, hdrMul(skip.d, hdrMul(w, d)));
    out = hdrAdd(out, hdrMul(skip.e, hdrMul(d, d)));
    return out;
}

/// log2 of |v|, for comparing against a step's validity radius.
fn hdrLog2(v: Hdr) -> f32 {
    let m = dot(v.m, v.m);
    if (m == 0.0) { return -1e30; }
    return f32(v.e) + 0.5 * log2(m);
}

/**
 * Largest valid skip starting at iteration n, or 0 if none applies.
 *
 * Steps are aligned: a level-L step covers laBaseStep << L iterations and only
 * starts at multiples of that. Bigger levels are tried first, so a pixel with a
 * tiny delta jumps thousands of iterations at once.
 */
fn takeSkip(
    at: u32,
    dz: ptr<function, Hdr>,
    deriv: ptr<function, Hdr>,
    withDerivative: bool,
    delta0: Hdr
) -> u32 {
    if (u.laLevels == 0u) { return 0u; }
    let dzLog2 = hdrLog2(*dz);

    // A level-L step starts only at multiples of laBaseStep << L, so the
    // highest level that can possibly align here is fixed by the trailing zeros
    // of at / laBaseStep. Walking down from the top level every time wasted most
    // of its work on steps that were never aligned to begin with.
    let unit = at / u.laBaseStep;
    var level: i32 = i32(u.laLevels) - 1;
    if (unit != 0u) {
        level = min(level, i32(countTrailingZeros(unit)));
    }

    loop {
        if (level < 0) { break; }
        let count = laIndex[u.laLevels + u32(level)];
        let index = unit >> u32(level);

        if (index < count) {
            let skip = loadSkip(laIndex[u32(level)] + index);
            // Safety margin: the delta must sit well below the radius, not just
            // inside it. Right at the boundary the linear map is only as good
            // as the tolerance, and marginal steps visibly shift escape counts.
            // Deep views sit tens of orders below the radius, so they lose
            // nothing; shallow views simply stop taking the marginal steps.
            if (skip.radiusLog2 > LA_NEVER && dzLog2 + LA_MARGIN_LOG2 <= skip.radiusLog2) {
                *dz = applySkip(skip, *dz, delta0);
                // The orbit derivative obeys the same linear recurrence with
                // d = 1, so the very same A and B advance it over the range.
                if (withDerivative) {
                    *deriv = hdrAdd(hdrMul(skip.a, *deriv), skip.b);
                }
                return u.laBaseStep << u32(level);
            }
        }
        level = level - 1;
    }
    return 0u;
}

// ------------------------------------------------------------------ iteration

/**
 * Result of iterating one point.
 *
 * `logDeriv` is log2 of |dz/dc| for the *full* orbit, carried in log space
 * because the derivative reaches astronomical magnitudes at depth — it is the
 * denominator of the distance estimate, so only its logarithm is ever needed.
 */
struct Sample {
    escaped: bool,
    n: u32,
    z: vec2<f32>,
    z2: f32,
    logDeriv: f32,
    skipped: u32,
    skips: u32,
    rebases: u32,
    /// log2 |delta| when the loop ended, for diagnostics.
    dzLog2: f32,
    /// Final reference index, for diagnostics.
    refIter: u32,
};

const HDR_ONE = Hdr(vec2<f32>(1.0, 0.0), 0);

fn iterate(delta0: Hdr, wantDerivative: bool) -> Sample {
    var dz = hdrZero();
    // D_{k+1} = 2*z_k*D_k + 1, the derivative of the whole orbit w.r.t. c.
    var deriv = hdrZero();
    var z = vec2<f32>(0.0);
    var refIter: u32 = 0u;
    let lastRef = max(u.refLength - 1u, 1u);
    var n: u32 = 0u;
    var z2: f32 = 0.0;
    var escaped = false;

    var skipped: u32 = 0u;
    var skips: u32 = 0u;
    var rebases: u32 = 0u;

    while (n < u.maxIterations) {
        // The skip table describes the reference orbit's own map from index j to
        // j + span, so it is indexed by refIter, not by n. Keying it on n would
        // switch approximation off permanently after the first rebase — and at a
        // minibrot nucleus the orbit returns to zero every period, so every
        // pixel rebases many times.
        if ((refIter % u.laBaseStep) == 0u &&
            refIter + u.laBaseStep <= lastRef &&
            n + u.laBaseStep <= u.maxIterations) {
            let span = takeSkip(refIter, &dz, &deriv, wantDerivative, delta0);
            if (span > 0u) {
                n = n + span;
                refIter = refIter + span;
                skipped = skipped + span;
                skips = skips + 1u;

                let jumped = hdrAdd(refHdr(refIter), dz);
                z = hdrValue(jumped);
                z2 = dot(z, z);
                if (z2 > ESCAPE_R2) { escaped = true; break; }
                if (hdrLess(jumped, dz) || refIter >= lastRef) {
                    dz = jumped;
                    refIter = 0u;
                    rebases = rebases + 1u;
                }
                continue;
            }
        }

        // dz <- 2*X_k*dz + dz^2 + delta0
        let twoX = 2.0 * refSample(refIter);
        dz = hdrAdd(hdrAdd(hdrMulPlain(dz, twoX), hdrMul(dz, dz)), delta0);
        if (wantDerivative) {
            deriv = hdrAdd(hdrMulPlain(deriv, 2.0 * z), HDR_ONE);
        }
        refIter = refIter + 1u;

        let zHdr = hdrAdd(refHdr(refIter), dz);
        z = hdrValue(zHdr);
        n = n + 1u;

        z2 = dot(z, z);
        if (z2 > ESCAPE_R2) { escaped = true; break; }

        if (hdrLess(zHdr, dz) || refIter >= lastRef) {
            dz = zHdr;
            refIter = 0u;
            rebases = rebases + 1u;
        }
    }

    var logDeriv = 0.0;
    if (wantDerivative) { logDeriv = hdrLog2(deriv); }

    return Sample(escaped, n, z, z2, logDeriv, skipped, skips, rebases,
                  hdrLog2(dz), refIter);
}

// --------------------------------------------------- distance-estimation field

/// log2 of one pixel's width in the complex plane.
fn logPixelSize() -> f32 {
    return f32(u.scaleExponent) + log2(max(abs(u.scaleMantissa), 1e-30));
}

/**
 * Height field: how many octaves the distance to the set sits below one pixel.
 *
 * distance = 0.5 * |z| * ln|z| / |dz/dc|, normalised against the pixel size so
 * the banding stays the same visual scale at any zoom. Everything is done in
 * log2: the derivative alone can reach 10^700 at depth.
 */
fn heightOf(s: Sample) -> f32 {
    if (!s.escaped) { return 0.0; }
    let magnitude = max(sqrt(s.z2), 1.0000001);
    let logDistance =
        -1.0 + log2(magnitude) + log2(max(log(magnitude), 1e-30)) - s.logDeriv;
    return -(logDistance - logPixelSize());
}

fn delta0For(pixel: vec2<f32>) -> Hdr {
    let fromCentre = pixel - 0.5 * u.resolution;
    let pixelDelta = hdrNorm(Hdr(fromCentre * u.scaleMantissa, u.scaleExponent));
    let centreOffset = hdrNorm(Hdr(u.offsetMantissa, u.offsetExponent));
    return hdrAdd(pixelDelta, centreOffset);
}

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn emptySample() -> Sample {
    return Sample(false, 0u, vec2<f32>(0.0), 0.0, 0.0, 0u, 0u, 0u, 0.0, 0u);
}

/// Direct z <- z^2 + c in plain f32.
///
/// Perturbation only pays when |delta| << |z|. Zoomed out that is false, the
/// rebase test fires almost every iteration, and we also pay for an
/// arbitrary-precision reference orbit the view cannot even resolve. Iterating
/// c directly is both simpler and much faster, and f32 has precision to spare
/// until the pixel spacing approaches its resolution near |c| ~ 1.
fn iterateDirect(c: vec2<f32>, wantDerivative: bool) -> Sample {
    var z = vec2<f32>(0.0);
    var deriv = vec2<f32>(0.0);
    var n: u32 = 0u;
    var z2: f32 = 0.0;
    var escaped = false;

    while (n < u.maxIterations) {
        if (wantDerivative) { deriv = 2.0 * cmul(z, deriv) + vec2<f32>(1.0, 0.0); }
        z = vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        n = n + 1u;
        z2 = dot(z, z);
        if (z2 > ESCAPE_R2) { escaped = true; break; }
    }

    var logDeriv = 0.0;
    if (wantDerivative) { logDeriv = 0.5 * log2(max(dot(deriv, deriv), 1e-38)); }
    return Sample(escaped, n, z, z2, logDeriv, 0u, 0u, 0u, 0.0, 0u);
}

/// Perturbation with a plain f32 delta.
///
/// Same recurrence as iterate(), but without the explicit exponent. Every
/// iteration of the HDR path pays a log2 and two exp2 for renormalisation;
/// while the delta stays above f32's smallest normal there is nothing to
/// renormalise, so that cost is pure overhead.
fn iteratePlain(delta0: vec2<f32>, wantDerivative: bool) -> Sample {
    var dz = vec2<f32>(0.0);
    var deriv = vec2<f32>(0.0);
    var z = vec2<f32>(0.0);
    var refIter: u32 = 0u;
    let lastRef = max(u.refLength - 1u, 1u);
    var n: u32 = 0u;
    var z2: f32 = 0.0;
    var escaped = false;
    var rebases: u32 = 0u;

    while (n < u.maxIterations) {
        // 2*X*dz + dz^2 + delta0, factored to one complex multiply.
        dz = cmul(2.0 * refSample(refIter) + dz, dz) + delta0;
        if (wantDerivative) { deriv = 2.0 * cmul(z, deriv) + vec2<f32>(1.0, 0.0); }
        refIter = refIter + 1u;

        z = refSample(refIter) + dz;
        n = n + 1u;

        z2 = dot(z, z);
        if (z2 > ESCAPE_R2) { escaped = true; break; }

        if (z2 < dot(dz, dz) || refIter >= lastRef) {
            dz = z;
            refIter = 0u;
            rebases = rebases + 1u;
        }
    }

    var logDeriv = 0.0;
    if (wantDerivative) { logDeriv = 0.5 * log2(max(dot(deriv, deriv), 1e-38)); }
    return Sample(escaped, n, z, z2, logDeriv, 0u, 0u, rebases,
                  0.5 * log2(max(dot(dz, dz), 1e-38)), refIter);
}

/// Dispatches to the method the renderer picked for this zoom depth. The
/// branch is uniform across the dispatch, so it costs nothing per pixel.
fn iterateAny(pixel: vec2<f32>, wantDerivative: bool) -> Sample {
    if (u.method == 0u) {
        let fromCentre = pixel - 0.5 * u.resolution;
        let c = u.centre + fromCentre * u.scaleMantissa * exp2(f32(u.scaleExponent));
        return iterateDirect(c, wantDerivative);
    }
    if (u.method == 1u) {
        return iteratePlain(hdrValue(delta0For(pixel)), wantDerivative);
    }
    return iterate(delta0For(pixel), wantDerivative);
}

// --------------------------------------------------------------------- shading

fn iterationColour(s: Sample) -> vec3<f32> {
    var mu = f32(s.n);
    if (u.smoothShading == 1u) {
        mu = mu - log2(0.5 * log(s.z2) / log(ESCAPE_R));
    }
    var cycle = u.colorCycle;
    if (u.mapping == 1u) {
        mu = sqrt(max(mu, 0.0));
        cycle = u.colorCycle / 8.0;
    } else if (u.mapping == 2u) {
        mu = log2(max(mu, 1.0));
        cycle = u.colorCycle / 64.0;
    }
    return palette(wrapCoordinate(mu / max(cycle, 0.001) + u.colorOffset));
}

/// sRGB-ish decode, so palette stops are mixed and lit in linear light.
fn toLinear(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2));
}

/**
 * Colour for one sample, with the pseudo-3D relief.
 *
 * The normal comes from the screen-space gradient of the height field, which
 * is a normal-map illusion rather than displaced geometry: nothing moves, the
 * shading just reads as a surface. The bands flow and fold while zooming
 * because the distance field itself changes, not because the palette scrolls.
 */
fn shade(s: Sample, hCentre: f32, hRight: f32, hUp: f32) -> vec3<f32> {
    if (!s.escaped) { return toLinear(u.interior); }

    let base = toLinear(palette(fract(hCentre * u.colorDensity + u.colorPhase)));
    if (u.slopeLighting == 0u) { return base; }

    let dx = hRight - hCentre;
    let dy = hUp - hCentre;
    let normal = normalize(vec3<f32>(-dx * u.slopeDepth, -dy * u.slopeDepth, 1.0));

    let diffuse = max(dot(normal, u.lightDir), 0.0);
    var lit = base * (u.ambientLight + u.diffuseStrength * diffuse);

    if (u.specularStrength > 0.0) {
        // Blinn-Phong against a viewer straight down the z axis.
        let halfway = normalize(u.lightDir + vec3<f32>(0.0, 0.0, 1.0));
        let specular = pow(max(dot(normal, halfway), 0.0), 32.0);
        lit = lit + vec3<f32>(u.specularStrength * specular);
    }
    return lit;
}

// --------------------------------------------------------------------- entry

@compute @workgroup_size(8, 8)
fn render(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(u32(u.resolution.x), u32(u.resolution.y));
    let row = gid.y + u.rowOffset;
    if (gid.x >= size.x || row >= size.y) { return; }

    let distanceMode = u.mode == 1u;
    let grid = max(u.supersample, 1u);
    let step = 1.0 / f32(grid);

    var accumulated = vec3<f32>(0.0);
    var skipped: u32 = 0u;
    var skips: u32 = 0u;
    var rebases: u32 = 0u;
    var plain: u32 = 0u;

    for (var sy: u32 = 0u; sy < grid; sy = sy + 1u) {
        for (var sx: u32 = 0u; sx < grid; sx = sx + 1u) {
            let jitter = vec2<f32>(
                (f32(sx) + 0.5) * step,
                (f32(sy) + 0.5) * step
            );
            let pixel = vec2<f32>(
                f32(gid.x),
                u.resolution.y - 1.0 - f32(row)
            ) + jitter;

            let centre = iterateAny(pixel, distanceMode);
            skipped = skipped + centre.skipped;
            skips = skips + centre.skips;
            rebases = rebases + centre.rebases;
            plain = plain + (centre.n - centre.skipped);

            // mode 2 is a diagnostic view: red = iterations used, green =
            // escaped, blue = where in the reference orbit it ended up.
            if (u.mode == 2u) {
                textureStore(output, vec2<i32>(i32(gid.x), i32(row)), vec4<f32>(
                    f32(centre.n) / f32(max(u.maxIterations, 1u)),
                    select(0.0, 1.0, centre.escaped),
                    // log2|dz| mapped from [-300, 44] into 0..1.
                    clamp((centre.dzLog2 + 300.0) / 344.0, 0.0, 1.0),
                    // Alpha: rebases, saturating at 255.
                    clamp(f32(centre.rebases) / 255.0, 0.0, 1.0)
                ));
                return;
            }

            if (!distanceMode) {
                if (centre.escaped) {
                    accumulated = accumulated + toLinear(iterationColour(centre));
                } else {
                    accumulated = accumulated + toLinear(u.interior);
                }
                continue;
            }

            let hCentre = heightOf(centre);
            var hRight = hCentre;
            var hUp = hCentre;
            if (u.slopeLighting == 1u && centre.escaped) {
                // Neighbouring samples, re-evaluated rather than reused: the
                // gradient of the height field is what the lighting needs.
                let right = iterate(delta0For(pixel + vec2<f32>(1.0, 0.0)), true);
                let up = iterate(delta0For(pixel + vec2<f32>(0.0, 1.0)), true);
                skipped = skipped + right.skipped + up.skipped;
                skips = skips + right.skips + up.skips;
                rebases = rebases + right.rebases + up.rebases;
                plain = plain + (right.n - right.skipped) + (up.n - up.skipped);
                if (right.escaped) { hRight = heightOf(right); }
                if (up.escaped) { hUp = heightOf(up); }
            }
            accumulated = accumulated + shade(centre, hCentre, hRight, hUp);
        }
    }

    let samples = f32(grid * grid);
    let linearColour = accumulated / samples;
    // Encode out of linear light at the very end.
    let encoded = pow(max(linearColour, vec3<f32>(0.0)), vec3<f32>(u.invGamma));

    textureStore(output, vec2<i32>(i32(gid.x), i32(row)),
                 vec4<f32>(clamp(encoded, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));

    atomicAdd(&stats[0], skipped);
    atomicAdd(&stats[1], skips);
    atomicAdd(&stats[2], rebases);
    atomicAdd(&stats[3], plain);
}
