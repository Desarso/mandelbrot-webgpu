/**
 * WebGL2 rendering path: reference orbit iterated on the CPU with decimal.js,
 * uploaded as an RG32F texture, and a fragment shader doing per-pixel
 * perturbation in plain f32.
 *
 * Depth is capped around a span of 1e-34: below that the f32 per-pixel deltas
 * fall into subnormal range and the image collapses. The WebGPU path carries an
 * explicit exponent per delta and has no such floor.
 */

import Decimal from "decimal.js";
import { Complex } from "../logic/Complex";
import { hexToRgb, MAX_STOPS, type ColorSettings } from "../logic/colorSettings";
import fragmentShaderSource from "../logic/fragmentShader.glsl?raw";
import type { BackendStats, DrawRequest, RenderBackend } from "./backend";

const vertexShaderSource = `#version 300 es
in vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const ORBIT_TEXTURE_WIDTH = 1024;
const MAX_REF_ITER = 200000;

const UNIFORM_NAMES = [
  "u_maxIterations",
  "u_unitsPerPixel",
  "u_resolution",
  "u_referenceOrbit",
  "u_refLength",
  "u_refWidth",
  "u_refOffset",
  "u_palette",
  "u_colorCycle",
  "u_colorOffset",
  "u_smooth",
  "u_mapping",
  "u_mirror",
  "u_interior",
  "u_stops",
  "u_stopCount",
] as const;

type UniformName = (typeof UNIFORM_NAMES)[number];

export class WebGlBackend implements RenderBackend {
  readonly name = "webgl" as const;
  readonly minSpan = 1e-34;

  private gl: WebGL2RenderingContext;
  private uniforms = new Map<UniformName, WebGLUniformLocation | null>();

  private refX = new Decimal(0);
  private refY = new Decimal(0);
  private refLength = 0;
  private refCapacity = 0;
  private orbit = new Float32Array(0);
  private orbitTexture: WebGLTexture | null = null;
  private orbitDirty = true;

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false });
    if (!gl) throw new Error("WebGL2 is not supported by this browser");
    this.gl = gl;

    const program = this.buildProgram(vertexShaderSource, fragmentShaderSource);
    gl.useProgram(program);

    const missing: string[] = [];
    for (const name of UNIFORM_NAMES) {
      const location =
        gl.getUniformLocation(program, name) ??
        gl.getUniformLocation(program, `${name}[0]`);
      if (location === null) missing.push(name);
      this.uniforms.set(name, location);
    }
    if (missing.length) {
      throw new Error(`Shader is missing uniforms: ${missing.join(", ")}`);
    }

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
      gl.STATIC_DRAW
    );
    const positionLocation = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
  }

  async draw(request: DrawRequest): Promise<BackendStats> {
    const gl = this.gl;
    const started = performance.now();

    const magnitude = Math.max(0, -Math.log10(request.unitsPerPixel.toNumber()));
    Decimal.set({ precision: Math.max(30, Math.ceil(magnitude) + 20) });

    const maxIterations = Math.min(MAX_REF_ITER - 1, request.maxIterations);
    const orbitMs = this.ensureReferenceOrbit(request, maxIterations + 1);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.orbitTexture);
    gl.uniform1i(this.uniform("u_referenceOrbit"), 0);

    gl.uniform1i(this.uniform("u_maxIterations"), maxIterations);
    gl.uniform1f(this.uniform("u_unitsPerPixel"), request.unitsPerPixel.toNumber());
    gl.uniform2f(this.uniform("u_resolution"), request.width, request.height);
    gl.uniform1i(this.uniform("u_refLength"), this.refLength);
    gl.uniform1i(this.uniform("u_refWidth"), ORBIT_TEXTURE_WIDTH);
    gl.uniform2f(
      this.uniform("u_refOffset"),
      request.centerX.minus(this.refX).toNumber(),
      request.centerY.minus(this.refY).toNumber()
    );

    const colors = request.colors;
    const stops = colors.stops.slice(0, MAX_STOPS);
    const packedStops = new Float32Array(MAX_STOPS * 3);
    stops.forEach((stop, index) => packedStops.set(hexToRgb(stop), index * 3));

    gl.uniform1i(this.uniform("u_palette"), colors.palette);
    gl.uniform1f(this.uniform("u_colorCycle"), Math.max(1, colors.cycle));
    gl.uniform1f(this.uniform("u_colorOffset"), colors.offset);
    gl.uniform1i(this.uniform("u_smooth"), colors.smooth ? 1 : 0);
    gl.uniform1i(this.uniform("u_mapping"), colors.mapping);
    gl.uniform1i(this.uniform("u_mirror"), colors.mirror ? 1 : 0);
    gl.uniform3fv(this.uniform("u_interior"), hexToRgb(colors.interior));
    gl.uniform3fv(this.uniform("u_stops"), packedStops);
    gl.uniform1i(this.uniform("u_stopCount"), Math.max(1, stops.length));

    gl.viewport(0, 0, request.width, request.height);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    return {
      precision: "f64 orbit / f32 pixels",
      orbitLength: this.refLength,
      orbitMs,
      renderMs: performance.now() - started - orbitMs,
      // The WebGL path has no approximation and no counters.
      skipRatio: 0,
      rebases: 0,
    };
  }

  dispose() {
    // The context and its resources go away with the canvas.
    this.orbitTexture = null;
  }

  private ensureReferenceOrbit(request: DrawRequest, count: number): number {
    const halfSpan = request.unitsPerPixel.times(
      Math.min(request.width, request.height) / 2
    );
    const drift = request.centerX
      .minus(this.refX)
      .abs()
      .plus(request.centerY.minus(this.refY).abs());

    const stale =
      this.orbitDirty ||
      count > this.refCapacity ||
      drift.greaterThan(halfSpan.times(0.5));
    // Rebuilding mid-gesture is the zoom stutter; a stale reference is still
    // exact, just less efficient.
    if (!stale || (request.interacting && !this.orbitDirty)) return 0;

    const started = performance.now();
    this.refX = request.centerX;
    this.refY = request.centerY;
    this.buildReferenceOrbit(count);
    this.uploadOrbit();
    this.orbitDirty = false;
    return performance.now() - started;
  }

  private buildReferenceOrbit(count: number) {
    const rows = Math.ceil(count / ORBIT_TEXTURE_WIDTH);
    const capacity = rows * ORBIT_TEXTURE_WIDTH;
    if (this.orbit.length !== capacity * 2) {
      this.orbit = new Float32Array(capacity * 2);
    }
    this.refCapacity = capacity;

    const c = new Complex(this.refX, this.refY);
    const bailout = new Decimal(4);
    let z = new Complex(new Decimal(0), new Decimal(0));

    this.orbit[0] = 0;
    this.orbit[1] = 0;

    let length = 1;
    for (let i = 1; i < count; i++) {
      z = z.square().add(c);
      this.orbit[i * 2] = z.real.toNumber();
      this.orbit[i * 2 + 1] = z.imag.toNumber();
      length = i + 1;
      if (z.dot(z).greaterThan(bailout)) break;
    }
    this.refLength = length;
  }

  private uploadOrbit() {
    const gl = this.gl;
    if (!this.orbitTexture) {
      this.orbitTexture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.orbitTexture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    } else {
      gl.bindTexture(gl.TEXTURE_2D, this.orbitTexture);
    }
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RG32F,
      ORBIT_TEXTURE_WIDTH,
      this.refCapacity / ORBIT_TEXTURE_WIDTH,
      0,
      gl.RG,
      gl.FLOAT,
      this.orbit
    );
  }

  private uniform(name: UniformName): WebGLUniformLocation | null {
    return this.uniforms.get(name) ?? null;
  }

  private buildProgram(vertex: string, fragment: string): WebGLProgram {
    const gl = this.gl;
    const program = gl.createProgram();
    if (!program) throw new Error("Failed to create WebGL program");

    const vs = this.compileShader(gl.VERTEX_SHADER, vertex);
    const fs = this.compileShader(gl.FRAGMENT_SHADER, fragment);
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error(`Program link failed: ${log}`);
    }
    return program;
  }

  private compileShader(type: number, source: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type);
    if (!shader) throw new Error("Failed to create shader");
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Shader compile failed: ${log}`);
    }
    return shader;
  }
}

export type { ColorSettings };
