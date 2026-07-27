#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

uniform int u_maxIterations;
uniform float u_unitsPerPixel;
uniform vec2 u_resolution;

// Perturbation reference orbit for a point near the middle of the view:
// X_0 = 0, X_1 = c, X_2 = c^2 + c, ... iterated in arbitrary precision on the
// CPU and uploaded as an RG32F texture so it can be as long as the iteration
// limit. u_refLength is how many samples are valid, u_refWidth the texture row
// length, and u_refOffset the view centre relative to the reference point.
uniform sampler2D u_referenceOrbit;
uniform int u_refLength;
uniform int u_refWidth;
uniform vec2 u_refOffset;

// Colouring
#define MAX_STOPS 8
uniform int u_palette;
uniform float u_colorCycle;   // iterations per full pass through the palette
uniform float u_colorOffset;  // 0..1 rotation of the palette
uniform bool u_smooth;
uniform int u_mapping;        // 0 linear, 1 sqrt, 2 log
uniform bool u_mirror;        // ping-pong the palette instead of wrapping
uniform vec3 u_interior;      // colour for points that never escape
uniform vec3 u_stops[MAX_STOPS];
uniform int u_stopCount;

out vec4 fragColor;

const float ESCAPE_R = 16.0;
const float ESCAPE_R2 = ESCAPE_R * ESCAPE_R;
const float TAU = 6.283185307179586;

vec2 refOrbit(int i) {
    return texelFetch(u_referenceOrbit, ivec2(i % u_refWidth, i / u_refWidth), 0).xy;
}

vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec3 cosPalette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(TAU * (c * t + d));
}

// The classic Ultra Fractal gradient: deep blue -> white -> gold -> black.
vec3 ultraFractal(float t) {
    float p[6] = float[6](0.0, 0.16, 0.42, 0.6425, 0.8575, 1.0);
    vec3 c[6] = vec3[6](vec3(0.000, 0.027, 0.392), vec3(0.125, 0.420, 0.796), vec3(0.929, 1.000, 1.000), vec3(1.000, 0.667, 0.000), vec3(0.000, 0.008, 0.000), vec3(0.000, 0.027, 0.392));

    vec3 col = c[0];
    for(int i = 0; i < 5; i++) {
        col = mix(col, c[i + 1], smoothstep(p[i], p[i + 1], t));
    }
    return col;
}

// Evenly spaced user stops, interpolated around a closed loop.
vec3 customPalette(float t) {
    int count = max(u_stopCount, 1);
    if(count == 1) {
        return u_stops[0];
    }
    float scaled = t * float(count);
    int index = int(floor(scaled)) % count;
    int next = (index + 1) % count;
    return mix(u_stops[index], u_stops[next], fract(scaled));
}

vec3 palette(float t) {
    t = clamp(t, 0.0, 1.0);
    if(u_palette == 1) {
        return ultraFractal(t);
    } else if(u_palette == 2) {     // ember
        return cosPalette(t, vec3(0.50, 0.30, 0.20), vec3(0.50, 0.35, 0.25), vec3(1.00, 1.00, 1.00), vec3(0.00, 0.10, 0.20));
    } else if(u_palette == 3) {     // ice
        return cosPalette(t, vec3(0.45, 0.50, 0.60), vec3(0.35, 0.40, 0.40), vec3(1.00, 1.00, 0.90), vec3(0.60, 0.70, 0.85));
    } else if(u_palette == 4) {     // mono
        return vec3(0.5 - 0.5 * cos(TAU * t));
    } else if(u_palette == 5) {     // custom stops
        return customPalette(t);
    }
    return cosPalette(t, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.00, 0.33, 0.67));  // spectrum
}

// Wrap the palette coordinate, either repeating or ping-ponging.
float wrapCoordinate(float t) {
    if(u_mirror) {
        float m = mod(t, 2.0);
        return m > 1.0 ? 2.0 - m : m;
    }
    return fract(t);
}

void main() {
    // Offset of this pixel from the reference point, in the complex plane.
    vec2 delta0 = (gl_FragCoord.xy - 0.5 * u_resolution) * u_unitsPerPixel + u_refOffset;

    vec2 dz = vec2(0.0);
    vec2 z = vec2(0.0);
    int refIter = 0;
    int lastRef = max(u_refLength - 1, 1);
    int n = 0;
    float z2 = 0.0;
    bool escaped = false;

    while(n < u_maxIterations) {
        // dz_{k+1} = 2 * X_k * dz_k + dz_k^2 + delta0   (all complex products)
        dz = 2.0 * cmul(refOrbit(refIter), dz) + cmul(dz, dz) + delta0;
        refIter++;
        z = refOrbit(refIter) + dz;
        n++;

        z2 = dot(z, z);
        if(z2 > ESCAPE_R2) {
            escaped = true;
            break;
        }

        // Rebase to the start of the reference orbit once the perturbation has
        // grown bigger than the orbit itself, or once the reference escapes and
        // there are no more samples to follow. Restarting at X_0 = 0 is exact:
        // z = X_0 + z, and the next step picks c back up through X_1.
        if(z2 < dot(dz, dz) || refIter >= lastRef) {
            dz = z;
            refIter = 0;
        }
    }

    if(!escaped) {
        fragColor = vec4(u_interior, 1.0);
        return;
    }

    float mu = float(n);
    if(u_smooth) {
        mu -= log2(0.5 * log(z2) / log(ESCAPE_R));
    }

    // Compress the iteration count so deep zooms keep usable banding.
    float cycle = u_colorCycle;
    if(u_mapping == 1) {
        mu = sqrt(max(mu, 0.0));
        cycle = u_colorCycle / 8.0;
    } else if(u_mapping == 2) {
        mu = log2(max(mu, 1.0));
        cycle = u_colorCycle / 64.0;
    }

    fragColor = vec4(palette(wrapCoordinate(mu / max(cycle, 0.001) + u_colorOffset)), 1.0);
}
