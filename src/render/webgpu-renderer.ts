/**
 * WebGPU rendering path: arbitrary-precision reference orbit on the GPU, then
 * a perturbation compute pass whose per-pixel deltas carry their own exponent.
 *
 * Unlike the WebGL path there is no f32 underflow floor, so zoom depth is
 * bounded by the precision profile (limb count) rather than by the renderer.
 */

import Decimal from "decimal.js";
import { compileShader, readBuffer, storageBuffer, type GpuContext } from "../gpu/device";
import bigfixedSource from "../gpu/shaders/bigfixed.wgsl?raw";
import orbitBindings from "../gpu/shaders/orbit-bindings.wgsl?raw";
import orbitSource from "../gpu/shaders/orbit.wgsl?raw";
import perturbationSource from "./perturbation.wgsl?raw";
import { hexToRgb, MAX_STOPS, type ColorSettings } from "../logic/colorSettings";
import { parseFixed } from "../arithmetic/types";

const orbitModule = [orbitBindings, bigfixedSource, orbitSource].join("\n");

/** Precision profiles, chosen from the zoom depth. */
const LIMB_PROFILES = [8, 16, 32, 64, 128, 256] as const;

/**
 * Reference-orbit iterations per dispatch. Each batch costs one small status
 * readback, so larger batches mean fewer CPU round trips; keep it bounded so a
 * single submission stays responsive.
 */
const ORBIT_BATCH = 512;

export interface RenderRequest {
  centerX: Decimal;
  centerY: Decimal;
  /** Complex units per device pixel. */
  unitsPerPixel: Decimal;
  width: number;
  height: number;
  maxIterations: number;
  colors: ColorSettings;
}

export interface RenderStats {
  limbs: number;
  decimalDigits: number;
  orbitLength: number;
  orbitEscaped: boolean;
  orbitMs: number;
  renderMs: number;
}

/**
 * Picks a limb count with enough fractional bits to resolve one pixel, plus a
 * safety margin. `unitsPerPixel` of 1e-40 needs ~133 bits before margin.
 */
export function limbsForScale(unitsPerPixel: Decimal): number {
  const decimals = Math.max(0, -Math.log10(unitsPerPixel.toNumber()));
  const bitsNeeded = decimals * Math.LOG2E * Math.LN10 + 64;
  for (const limbs of LIMB_PROFILES) {
    if (32 * (limbs - 1) >= bitsNeeded) return limbs;
  }
  return LIMB_PROFILES[LIMB_PROFILES.length - 1];
}

/** Splits a Decimal into an f32 mantissa and a binary exponent. */
function splitExponent(value: Decimal): { mantissa: number; exponent: number } {
  if (value.isZero()) return { mantissa: 0, exponent: 0 };
  const exponent = Math.floor(Number(value.abs().log(2).toFixed(6)));
  const mantissa = Number(value.div(new Decimal(2).pow(exponent)).toFixed(12));
  return { mantissa, exponent };
}

/**
 * Splits a complex offset into two mantissas sharing one exponent, which is
 * what the shader's Hdr type expects.
 */
function splitComplex(x: Decimal, y: Decimal) {
  const magnitude = Decimal.max(x.abs(), y.abs());
  if (magnitude.isZero()) return { x: 0, y: 0, exponent: 0 };
  const exponent = Math.floor(Number(magnitude.log(2).toFixed(6)));
  const divisor = new Decimal(2).pow(exponent);
  return {
    x: Number(x.div(divisor).toFixed(12)),
    y: Number(y.div(divisor).toFixed(12)),
    exponent,
  };
}

function scratchWords(limbs: number): number {
  return 7 * limbs + 2 * limbs * 3;
}

export class WebGpuRenderer {
  private ctx: GpuContext;
  private canvas: HTMLCanvasElement;
  private context: GPUCanvasContext;
  private format: GPUTextureFormat;

  private orbitPipelines = new Map<number, GPUComputePipeline>();
  private renderPipeline: GPUComputePipeline | null = null;
  private blitPipeline: GPURenderPipeline | null = null;

  private target: GPUTexture | null = null;
  private targetSize = { width: 0, height: 0 };
  private sampler: GPUSampler;

  private uniformBuffer: GPUBuffer;
  private stopsBuffer: GPUBuffer;
  private orbitBuffer: GPUBuffer | null = null;
  private orbitCapacity = 0;

  /** Cached reference orbit: regenerating it per frame would kill panning. */
  private refX = new Decimal(0);
  private refY = new Decimal(0);
  private refLimbs = 0;
  private refIterations = 0;
  private refLength = 0;
  private refEscaped = false;
  private refValid = false;

  constructor(ctx: GpuContext, canvas: HTMLCanvasElement) {
    this.ctx = ctx;
    this.canvas = canvas;

    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("Could not get a webgpu canvas context");
    this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device: ctx.device,
      format: this.format,
      alphaMode: "opaque",
    });

    this.sampler = ctx.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });
    this.uniformBuffer = ctx.device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.stopsBuffer = storageBuffer(ctx.device, MAX_STOPS * 4, "palette-stops");
  }

  async init() {
    const { device } = this.ctx;

    const renderModule = await compileShader(device, perturbationSource, "perturbation");
    this.renderPipeline = device.createComputePipeline({
      label: "perturbation",
      layout: "auto",
      compute: { module: renderModule, entryPoint: "render" },
    });

    const blitModule = await compileShader(
      device,
      `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;

struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    var p = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0)
    );
    var out: VsOut;
    out.pos = vec4<f32>(p[i], 0.0, 1.0);
    out.uv = vec2<f32>((p[i].x + 1.0) * 0.5, (1.0 - p[i].y) * 0.5);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(src, smp, in.uv);
}
`,
      "blit"
    );
    this.blitPipeline = device.createRenderPipeline({
      label: "blit",
      layout: "auto",
      vertex: { module: blitModule, entryPoint: "vs" },
      fragment: {
        module: blitModule,
        entryPoint: "fs",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-strip" },
    });
  }

  private orbitPipeline(limbs: number): GPUComputePipeline {
    const cached = this.orbitPipelines.get(limbs);
    if (cached) return cached;
    // The module compiled cleanly during the self-test, so plain creation is
    // safe here; errors would already have surfaced at init.
    const module = this.ctx.device.createShaderModule({
      label: `orbit-${limbs}`,
      code: orbitModule,
    });
    const pipeline = this.ctx.device.createComputePipeline({
      label: `orbit-${limbs}`,
      layout: "auto",
      compute: { module, entryPoint: "advanceOrbit", constants: { LIMBS: limbs } },
    });
    this.orbitPipelines.set(limbs, pipeline);
    return pipeline;
  }

  /** Generates the reference orbit at the view centre, entirely on the GPU. */
  private async generateOrbit(
    request: RenderRequest,
    limbs: number
  ): Promise<{ length: number; escaped: boolean; ms: number }> {
    const { device } = this.ctx;
    const started = performance.now();
    const maxSamples = request.maxIterations + 1;

    if (this.orbitCapacity < maxSamples) {
      this.orbitBuffer?.destroy();
      this.orbitBuffer = storageBuffer(
        device,
        maxSamples * 6,
        "orbit-samples",
        GPUBufferUsage.COPY_SRC
      );
      this.orbitCapacity = maxSamples;
    }

    const pipeline = this.orbitPipeline(limbs);
    const state = storageBuffer(device, limbs * 2, "orbit-state");
    const seed = storageBuffer(device, limbs * 2, "orbit-seed");
    const scratch = storageBuffer(device, scratchWords(limbs), "orbit-scratch");
    const status = storageBuffer(device, 4, "orbit-status", GPUBufferUsage.COPY_SRC);
    const params = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const seedData = new Uint32Array(limbs * 2);
    seedData.set(parseFixed(request.centerX.toFixed(), limbs), 0);
    seedData.set(parseFixed(request.centerY.toFixed(), limbs), limbs);
    device.queue.writeBuffer(seed, 0, seedData);
    device.queue.writeBuffer(state, 0, new Uint32Array(limbs * 2));
    device.queue.writeBuffer(status, 0, new Uint32Array(4));
    device.queue.writeBuffer(this.orbitBuffer!, 0, new Float32Array(6));

    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state } },
        { binding: 1, resource: { buffer: seed } },
        { binding: 2, resource: { buffer: scratch } },
        { binding: 3, resource: { buffer: this.orbitBuffer! } },
        { binding: 4, resource: { buffer: status } },
        { binding: 5, resource: { buffer: params } },
      ],
    });

    // `sampleCount` counts written samples; sample 0 (z = 0) comes from the CPU.
    // The shader emits at `startIndex + iter + 1`, so startIndex must be the
    // index of the last sample already written, i.e. sampleCount - 1. Passing
    // the count itself skips one sample per batch and shifts all the rest.
    let sampleCount = 1;
    let escaped = false;

    while (sampleCount - 1 < request.maxIterations) {
      const done = sampleCount - 1;
      const iterations = Math.min(ORBIT_BATCH, request.maxIterations - done);
      device.queue.writeBuffer(
        params,
        0,
        new Uint32Array([iterations, done, maxSamples, 0])
      );

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(1);
      pass.end();
      device.queue.submit([encoder.finish()]);

      // Only the 4-word status comes back; the big state never leaves the GPU.
      const raw = new Uint32Array(await readBuffer(device, status, 16));
      if (raw[0] <= sampleCount) break; // no progress: escaped or done
      sampleCount = raw[0];
      if (raw[1] === 1) {
        escaped = true;
        break;
      }
    }

    [state, seed, scratch, status, params].forEach((b) => b.destroy());
    return {
      length: Math.max(2, sampleCount),
      escaped,
      ms: performance.now() - started,
    };
  }

  private ensureTarget(width: number, height: number) {
    if (this.target && this.targetSize.width === width && this.targetSize.height === height) {
      return;
    }
    this.target?.destroy();
    this.target = this.ctx.device.createTexture({
      label: "render-target",
      size: { width, height },
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });
    this.targetSize = { width, height };
  }

  /** Reads the reduced orbit samples back, for comparison against an oracle. */
  async debugReadOrbit(count: number): Promise<Float32Array> {
    if (!this.orbitBuffer) return new Float32Array(0);
    const words = Math.min(count * 6, this.orbitCapacity * 6);
    return new Float32Array(await readBuffer(this.ctx.device, this.orbitBuffer, words * 4));
  }

  /** Reads pixels back from the render target (rgba8unorm). */
  async debugReadPixels(points: [number, number][]): Promise<number[][]> {
    if (!this.target) return [];
    const { device } = this.ctx;
    const { width, height } = this.targetSize;
    // copyTextureToBuffer requires bytesPerRow to be a multiple of 256.
    const bytesPerRow = Math.ceil((width * 4) / 256) * 256;

    const staging = device.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { texture: this.target },
      { buffer: staging, bytesPerRow, rowsPerImage: height },
      { width, height }
    );
    device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    return points.map(([x, y]) => {
      const i = y * bytesPerRow + x * 4;
      return [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]];
    });
  }

  async render(request: RenderRequest): Promise<RenderStats> {
    const { device } = this.ctx;
    if (!this.renderPipeline || !this.blitPipeline) {
      throw new Error("WebGpuRenderer.init() was not awaited");
    }

    const limbs = limbsForScale(request.unitsPerPixel);
    Decimal.set({ precision: Math.ceil((32 * (limbs - 1)) / 3.32) + 10 });

    // Reuse the reference orbit while the view stays near the point it was
    // built at. Regenerating costs tens of milliseconds, so doing it every
    // frame would make panning unusable at depth.
    const halfSpan = request.unitsPerPixel.times(
      Math.min(request.width, request.height) / 2
    );
    const drift = request.centerX
      .minus(this.refX)
      .abs()
      .plus(request.centerY.minus(this.refY).abs());
    const stale =
      !this.refValid ||
      limbs !== this.refLimbs ||
      request.maxIterations > this.refIterations ||
      drift.greaterThan(halfSpan.times(0.5));

    let orbitMs = 0;
    if (stale) {
      this.refX = request.centerX;
      this.refY = request.centerY;
      const orbit = await this.generateOrbit(request, limbs);
      this.refLimbs = limbs;
      this.refIterations = request.maxIterations;
      this.refLength = orbit.length;
      this.refEscaped = orbit.escaped;
      this.refValid = true;
      orbitMs = orbit.ms;
    }

    const started = performance.now();
    this.ensureTarget(request.width, request.height);

    const scale = splitExponent(request.unitsPerPixel);
    const offset = splitComplex(
      request.centerX.minus(this.refX),
      request.centerY.minus(this.refY)
    );

    const colors = request.colors;
    const stopData = new Float32Array(MAX_STOPS * 4);
    colors.stops.slice(0, MAX_STOPS).forEach((stop, i) => {
      stopData.set(hexToRgb(stop), i * 4);
      stopData[i * 4 + 3] = 1;
    });
    device.queue.writeBuffer(this.stopsBuffer, 0, stopData);

    // std140-ish layout matching the Uniforms struct.
    const uniforms = new ArrayBuffer(128);
    const f32 = new Float32Array(uniforms);
    const i32 = new Int32Array(uniforms);
    const u32 = new Uint32Array(uniforms);
    f32[0] = request.width;
    f32[1] = request.height;
    f32[2] = scale.mantissa;
    i32[3] = scale.exponent;
    f32[4] = offset.x;
    f32[5] = offset.y;
    i32[6] = offset.exponent;
    u32[7] = request.maxIterations;
    u32[8] = this.refLength;
    u32[9] = colors.palette;
    f32[10] = Math.max(1, colors.cycle);
    f32[11] = colors.offset;
    u32[12] = colors.mapping;
    u32[13] = colors.mirror ? 1 : 0;
    u32[14] = colors.smooth ? 1 : 0;
    // vec3 must start on a 16-byte boundary.
    const interior = hexToRgb(colors.interior);
    f32[16] = interior[0];
    f32[17] = interior[1];
    f32[18] = interior[2];
    u32[19] = Math.max(1, Math.min(MAX_STOPS, colors.stops.length));
    device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);

    const bind = device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.orbitBuffer! } },
        { binding: 1, resource: { buffer: this.uniformBuffer } },
        { binding: 2, resource: this.target!.createView() },
        { binding: 3, resource: { buffer: this.stopsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(
      Math.ceil(request.width / 8),
      Math.ceil(request.height / 8)
    );
    pass.end();

    const blitBind = device.createBindGroup({
      layout: this.blitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.target!.createView() },
        { binding: 1, resource: this.sampler },
      ],
    });
    const view = this.context.getCurrentTexture().createView();
    const draw = encoder.beginRenderPass({
      colorAttachments: [
        { view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } },
      ],
    });
    draw.setPipeline(this.blitPipeline);
    draw.setBindGroup(0, blitBind);
    draw.draw(4);
    draw.end();

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    return {
      limbs,
      decimalDigits: Math.floor((32 * (limbs - 1)) / 3.32),
      orbitLength: this.refLength,
      orbitEscaped: this.refEscaped,
      orbitMs,
      renderMs: performance.now() - started,
    };
  }
}
