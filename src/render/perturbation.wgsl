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
};

@group(0) @binding(0) var<storage, read> orbit: array<f32>;   // hi, lo, exp per component
@group(0) @binding(1) var<uniform> u: Uniforms;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<storage, read> stops: array<vec4<f32>>;

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
    if (mx == 0.0 || mx != mx) { return hdrZero(); }
    let shift = i32(floor(log2(mx)));
    return Hdr(a.m * exp2(f32(-shift)), a.e + shift);
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

// --------------------------------------------------------------------- entry

@compute @workgroup_size(8, 8)
fn render(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(u32(u.resolution.x), u32(u.resolution.y));
    if (gid.x >= size.x || gid.y >= size.y) { return; }

    // Pixel offset from the centre, then from the reference point.
    let pixel = vec2<f32>(f32(gid.x), u.resolution.y - 1.0 - f32(gid.y)) + 0.5;
    let fromCentre = pixel - 0.5 * u.resolution;

    let pixelDelta = hdrNorm(Hdr(fromCentre * u.scaleMantissa, u.scaleExponent));
    let centreOffset = hdrNorm(Hdr(u.offsetMantissa, u.offsetExponent));
    let delta0 = hdrAdd(pixelDelta, centreOffset);

    var dz = hdrZero();
    var z = vec2<f32>(0.0);
    var refIter: u32 = 0u;
    let lastRef = max(u.refLength - 1u, 1u);
    var n: u32 = 0u;
    var z2: f32 = 0.0;
    var escaped = false;

    while (n < u.maxIterations) {
        // dz <- 2*X_k*dz + dz^2 + delta0
        let twoX = 2.0 * refSample(refIter);
        dz = hdrAdd(hdrAdd(hdrMulPlain(dz, twoX), hdrMul(dz, dz)), delta0);
        refIter = refIter + 1u;

        let zHdr = hdrAdd(Hdr(refSample(refIter), 0), dz);
        z = hdrValue(zHdr);
        n = n + 1u;

        z2 = dot(z, z);
        if (z2 > ESCAPE_R2) { escaped = true; break; }

        // Rebase when the perturbation outgrows the orbit, or the orbit ends.
        if (hdrLess(zHdr, dz) || refIter >= lastRef) {
            dz = zHdr;
            refIter = 0u;
        }
    }

    // Debug view: red = iterations used, green = escaped.
    if (u.palette == 6u) {
        textureStore(output, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(
            f32(n) / f32(max(u.maxIterations, 1u)),
            select(0.0, 1.0, escaped),
            f32(refIter) / f32(max(u.refLength, 1u)),
            1.0
        ));
        return;
    }

    // Debug view: how delta0 and dz evolved.
    if (u.palette == 7u) {
        textureStore(output, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(
            (f32(delta0.e) + 128.0) / 255.0,
            (f32(dz.e) + 128.0) / 255.0,
            abs(dz.m.x) * 0.5,
            f32(lastRef) / 2000.0
        ));
        return;
    }

    var colour = u.interior;
    if (escaped) {
        var mu = f32(n);
        if (u.smoothShading == 1u) {
            mu = mu - log2(0.5 * log(z2) / log(ESCAPE_R));
        }
        var cycle = u.colorCycle;
        if (u.mapping == 1u) {
            mu = sqrt(max(mu, 0.0));
            cycle = u.colorCycle / 8.0;
        } else if (u.mapping == 2u) {
            mu = log2(max(mu, 1.0));
            cycle = u.colorCycle / 64.0;
        }
        colour = palette(wrapCoordinate(mu / max(cycle, 0.001) + u.colorOffset));
    }

    textureStore(output, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(colour, 1.0));
}
