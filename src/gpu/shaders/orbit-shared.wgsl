// Arbitrary-precision reference orbit, with its working state in workgroup
// memory.
//
// The same recurrence and the same arithmetic as orbit.wgsl, operation for
// operation, so every sample it emits is bit-identical. What changes is where
// the time went. orbit.wgsl runs x*x, y*y and x*y as three separate
// multiplications, each fenced five times with storage and workgroup barriers,
// against operands that live in device memory, and the serial parts -- carry
// resolution, negation, addition -- run on one thread at device-memory
// latency. Here:
//
//   - the columns of all three products accumulate in one parallel pass, since
//     none of them depends on another;
//   - each product resolves its carries on its own thread, concurrently,
//     and x' and y' are assembled side by side;
//   - every operand lives in workgroup memory, so the serial parts run at
//     shared-memory latency and only workgroup barriers are needed.
//
// Four barriers per iteration instead of seventeen.
//
// The working set is 23 * LIMBS words, which has to fit the device's workgroup
// storage. The renderer checks that and falls back to orbit.wgsl when it does
// not -- in practice only the 256-limb profile on a device that offers no more
// than the 16 KiB minimum.
//
// Concatenated after orbit-bindings.wgsl and bigfixed.wgsl, for LIMBS and
// mul32. It never touches `scratch`.

/** Invocations per workgroup. Six columns per limb is all there is to share. */
override ORBIT_WG: u32 = 256u;

/** |x| then |y|, LIMBS words each. */
override MAGNITUDE_WORDS: u32 = 2u * LIMBS;
/** Three products, 2 * LIMBS columns each, 96 bits per column. */
override COLUMN_WORDS: u32 = 18u * LIMBS;
/** x*x, y*y and x*y in fixed point, LIMBS words each. */
override PRODUCT_WORDS: u32 = 3u * LIMBS;

var<workgroup> magnitude: array<u32, MAGNITUDE_WORDS>;
var<workgroup> columns: array<u32, COLUMN_WORDS>;
var<workgroup> product: array<u32, PRODUCT_WORDS>;
/** Sign of x and of y, 1 when negative. */
var<workgroup> negativeX: u32;
var<workgroup> negativeY: u32;
/** Set once the orbit escapes or the sample buffer is full. */
var<workgroup> stop: u32;
var<workgroup> resumeAt: u32;
/** x' and y' as floats, for the escape test. */
var<workgroup> approxZ: vec2<f32>;

/** Product slots. x' is built in place of x*x and y' in place of x*y. */
const P_XX: u32 = 0u;
const P_YY: u32 = 1u;
const P_XY: u32 = 2u;

fn productBase(p: u32) -> u32 { return p * LIMBS; }

fn isNegativeProduct(base: u32) -> bool {
    return (product[base + LIMBS - 1u] & 0x80000000u) != 0u;
}

/// Every column of x*x, y*y and x*y, shared out across the workgroup. Each
/// column is owned by one invocation, so nothing races. The per-column sum is
/// the one in accumulateColumns.
fn accumulateAll(tid: u32) {
    let perProduct = 2u * LIMBS;
    for (var t: u32 = tid; t < 3u * perProduct; t = t + ORBIT_WG) {
        let p = t / perProduct;
        let k = t - p * perProduct;
        // x*x reads |x| twice, y*y reads |y| twice, x*y one of each.
        let aBase = select(0u, LIMBS, p == P_YY);
        let bBase = select(0u, LIMBS, p != P_XX);

        var sum0: u32 = 0u;
        var sum1: u32 = 0u;
        var sum2: u32 = 0u;

        let lo = select(0u, k - (LIMBS - 1u), k >= LIMBS);
        let hi = min(k, LIMBS - 1u);
        for (var i: u32 = lo; i <= hi; i = i + 1u) {
            let m = mul32(magnitude[aBase + i], magnitude[bBase + (k - i)]);

            let n0 = sum0 + m.lo;
            let c0 = select(0u, 1u, n0 < sum0);
            sum0 = n0;

            let n1 = sum1 + m.hi;
            let c1 = select(0u, 1u, n1 < sum1);
            let n1c = n1 + c0;
            let c2 = select(0u, 1u, n1c < n1);
            sum1 = n1c;

            sum2 = sum2 + c1 + c2;
        }

        columns[t * 3u + 0u] = sum0;
        columns[t * 3u + 1u] = sum1;
        columns[t * 3u + 2u] = sum2;
    }
}

/// Carry pass for product `p`, then the sign. Same as resolveColumns followed
/// by mulFixed's sign fix-up. Squares are never negative.
fn resolveProduct(p: u32) {
    var carry0: u32 = 0u;
    var carry1: u32 = 0u;
    let columnCount = 2u * LIMBS;
    let first = p * columnCount * 3u;
    let dst = productBase(p);

    for (var k: u32 = 0u; k < columnCount; k = k + 1u) {
        let c0 = columns[first + k * 3u + 0u];
        let c1 = columns[first + k * 3u + 1u];
        let c2 = columns[first + k * 3u + 2u];

        let s0 = c0 + carry0;
        let k0 = select(0u, 1u, s0 < c0);

        let s1a = c1 + carry1;
        let k1 = select(0u, 1u, s1a < c1);
        let s1 = s1a + k0;
        let k2 = select(0u, 1u, s1 < s1a);

        if (k + 1u >= LIMBS && k < 2u * LIMBS - 1u) {
            product[dst + (k + 1u - LIMBS)] = s0;
        }
        carry0 = s1;
        carry1 = c2 + k1 + k2;
    }

    if (p == P_XY && (negativeX ^ negativeY) == 1u) {
        var carry: u32 = 1u;
        for (var i: u32 = 0u; i < LIMBS; i = i + 1u) {
            let inverted = ~product[dst + i];
            let sum = inverted + carry;
            carry = select(0u, 1u, sum < inverted);
            product[dst + i] = sum;
        }
    }
}

/// x' = x*x - y*y + cx, in place of x*x.
///
/// One pass where orbit.wgsl takes two (subInto, then addSeedInto). Limb i of
/// the sum depends only on limb i of the difference and on its own carry, so
/// running both chains side by side gives the same words.
fn combineX() {
    let dst = productBase(P_XX);
    let src = productBase(P_YY);
    var borrow: u32 = 0u;
    var carry: u32 = 0u;
    for (var i: u32 = 0u; i < LIMBS; i = i + 1u) {
        let a = product[dst + i];
        let b = product[src + i];
        let d = a - b;
        let b0 = select(0u, 1u, a < b);
        let e = d - borrow;
        let b1 = select(0u, 1u, d < borrow);
        borrow = b0 + b1;

        let c = seed[i];
        let s = e + c;
        let c0 = select(0u, 1u, s < e);
        let t = s + carry;
        let c1 = select(0u, 1u, t < s);
        carry = c0 + c1;

        product[dst + i] = t;
    }
}

/// y' = 2*x*y + cy, in place of x*y. Doubling and adding the seed in one
/// pass, for the same reason as combineX.
fn combineY() {
    let dst = productBase(P_XY);
    var carryDouble: u32 = 0u;
    var carry: u32 = 0u;
    for (var i: u32 = 0u; i < LIMBS; i = i + 1u) {
        let a = product[dst + i];
        let s0 = a + a;
        let d0 = select(0u, 1u, s0 < a);
        let t0 = s0 + carryDouble;
        let d1 = select(0u, 1u, t0 < s0);
        carryDouble = d0 + d1;

        let c = seed[LIMBS + i];
        let s = t0 + c;
        let c0 = select(0u, 1u, s < t0);
        let t = s + carry;
        let c1 = select(0u, 1u, t < s);
        carry = c0 + c1;

        product[dst + i] = t;
    }
}

/// Splits a two's-complement product slot into a magnitude at `magBase` and a
/// sign, which is the form the next iteration's multiplications want.
fn takeMagnitude(src: u32, magBase: u32) -> u32 {
    let negative = isNegativeProduct(src);
    if (!negative) {
        for (var i: u32 = 0u; i < LIMBS; i = i + 1u) {
            magnitude[magBase + i] = product[src + i];
        }
        return 0u;
    }
    var carry: u32 = 1u;
    for (var i: u32 = 0u; i < LIMBS; i = i + 1u) {
        let inverted = ~product[src + i];
        let sum = inverted + carry;
        carry = select(0u, 1u, sum < inverted);
        magnitude[magBase + i] = sum;
    }
    return 1u;
}

/// The HDR triple for one component, from its magnitude and sign. Identical
/// to emitSample, which negates in place to reach the same magnitude.
fn emitMagnitude(magBase: u32, negative: u32, out: u32) {
    var top: i32 = -1;
    for (var i: i32 = i32(LIMBS) - 1; i >= 0; i = i - 1) {
        if (magnitude[magBase + u32(i)] != 0u) { top = i; break; }
    }

    if (top < 0) {
        samples[out + 0u] = 0.0;
        samples[out + 1u] = 0.0;
        samples[out + 2u] = 0.0;
        return;
    }

    let hiWord = magnitude[magBase + u32(top)];
    var w1: u32 = 0u;
    var w2: u32 = 0u;
    if (top >= 1) { w1 = magnitude[magBase + u32(top) - 1u]; }
    if (top >= 2) { w2 = magnitude[magBase + u32(top) - 2u]; }

    let msbIndex = firstLeadingBit(hiWord);
    let shift = 31u - msbIndex;

    var high: u32 = hiWord << shift;
    var low: u32 = w1 << shift;
    if (shift != 0u) {
        high = high | (w1 >> (32u - shift));
        low = low | (w2 >> (32u - shift));
    }

    let msbPosition = i32(u32(top) * 32u + msbIndex);
    let exponent = msbPosition - i32(32u * (LIMBS - 1u)) - 31;
    let sign = select(1.0, -1.0, negative == 1u);

    let mantissaHi = f32(high >> 8u) * 256.0;
    let mantissaLo = f32(high & 0xffu) + f32(low) * exp2(-32.0);

    samples[out + 0u] = sign * mantissaHi;
    samples[out + 1u] = sign * mantissaLo;
    samples[out + 2u] = f32(exponent);
}

/// approxValue, reading a product slot.
fn approxProduct(base: u32) -> f32 {
    let hi = f32(i32(product[base + LIMBS - 1u]));
    var lo: f32 = 0.0;
    if (LIMBS >= 2u) { lo = f32(product[base + LIMBS - 2u]) * exp2(-32.0); }
    return hi + lo;
}

@compute @workgroup_size(ORBIT_WG)
fn advanceOrbit(@builtin(local_invocation_id) local: vec3<u32>) {
    let tid = local.x;

    // Resume from the status buffer, so many dispatches can be encoded into
    // one submission (see orbit.wgsl). A full sample buffer counts as done:
    // iterating past it would leave `state` ahead of the last sample written,
    // and the orbit could no longer be extended from where it stopped.
    if (tid == 0u) {
        stop = select(0u, 1u, status[1] == 1u || status[0] >= params.maxSamples);
        resumeAt = status[0] - 1u;
        if (stop == 0u) {
            for (var i: u32 = 0u; i < LIMBS; i = i + 1u) {
                product[productBase(P_XX) + i] = state[i];
                product[productBase(P_XY) + i] = state[LIMBS + i];
            }
            negativeX = takeMagnitude(productBase(P_XX), 0u);
            negativeY = takeMagnitude(productBase(P_XY), LIMBS);
        }
    }
    if (workgroupUniformLoad(&stop) == 1u) { return; }
    let startIndex = workgroupUniformLoad(&resumeAt);

    for (var iter: u32 = 0u; iter < params.iterations; iter = iter + 1u) {
        // Also the barrier that publishes the previous iteration's magnitudes.
        if (workgroupUniformLoad(&stop) == 1u) { break; }

        accumulateAll(tid);
        workgroupBarrier();

        if (tid < 3u) { resolveProduct(tid); }
        workgroupBarrier();

        // x' and y' are independent, so each gets its own invocation.
        let index = startIndex + iter + 1u;
        if (tid == 0u) {
            combineX();
            negativeX = takeMagnitude(productBase(P_XX), 0u);
            emitMagnitude(0u, negativeX, index * 6u + 0u);
            approxZ.x = approxProduct(productBase(P_XX));
        } else if (tid == 1u) {
            combineY();
            negativeY = takeMagnitude(productBase(P_XY), LIMBS);
            emitMagnitude(LIMBS, negativeY, index * 6u + 3u);
            approxZ.y = approxProduct(productBase(P_XY));
        }
        workgroupBarrier();

        if (tid == 0u) {
            status[0] = index + 1u;
            // |z|^2 > 256, as escapedNow() and ESCAPE_R2 in perturbation.wgsl.
            let z = approxZ;
            if (z.x * z.x + z.y * z.y > 256.0) {
                stop = 1u;
                status[1] = 1u;
                status[2] = index;
            }
            if (index + 1u >= params.maxSamples) { stop = 1u; }
        }
    }

    // Invocation 0 wrote the last iteration's result; publish it to the rest.
    workgroupBarrier();
    for (var i: u32 = tid; i < LIMBS; i = i + ORBIT_WG) {
        state[i] = product[productBase(P_XX) + i];
        state[LIMBS + i] = product[productBase(P_XY) + i];
    }
}
