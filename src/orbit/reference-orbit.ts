/**
 * Drives the GPU reference-orbit engine.
 *
 * The high-precision state (x, y, c) never leaves the GPU. The CPU only
 * uploads the centre once, encodes batches of dispatches, and reads back a
 * 3-word status buffer to decide whether to keep going. The orbit samples the
 * renderer consumes come back in the reduced HDR format, not full precision.
 */

import { GpuContext, readBuffer, storageBuffer } from "../gpu/device";
import { parseFixed } from "../arithmetic/types";
import orbitShader from "../gpu/shaders/orbit.wgsl?raw";

/** Words of scratch a given precision needs: 7 slots + 2*LIMBS 96-bit columns. */
export function scratchWords(limbs: number): number {
  return 7 * limbs + 2 * limbs * 3;
}

export interface OrbitRequest {
  centerX: string;
  centerY: string;
  limbs: number;
  maxIterations: number;
  /** Iterations per dispatch; tuned so a submission stays responsive. */
  batchSize?: number;
  signal?: AbortSignal;
  onProgress?: (generated: number, total: number) => void;
}

export interface OrbitResult {
  /** [hi, lo, exp] for real then imaginary, per sample. */
  samples: Float32Array;
  length: number;
  escaped: boolean;
  escapeIteration: number;
  elapsedMs: number;
}

export class ReferenceOrbitEngine {
  private ctx: GpuContext;
  private pipelines = new Map<number, GPUComputePipeline>();

  constructor(ctx: GpuContext) {
    this.ctx = ctx;
  }

  /** Pipelines are specialised per precision through an override constant. */
  private pipelineFor(limbs: number): GPUComputePipeline {
    const cached = this.pipelines.get(limbs);
    if (cached) return cached;

    const module = this.ctx.device.createShaderModule({
      label: `orbit-${limbs}`,
      code: orbitShader,
    });
    const pipeline = this.ctx.device.createComputePipeline({
      label: `orbit-${limbs}`,
      layout: "auto",
      compute: {
        module,
        entryPoint: "advanceOrbit",
        constants: { LIMBS: limbs },
      },
    });
    this.pipelines.set(limbs, pipeline);
    return pipeline;
  }

  async generate(request: OrbitRequest): Promise<OrbitResult> {
    const { device } = this.ctx;
    const limbs = request.limbs;
    const maxSamples = request.maxIterations + 1;
    const batchSize = request.batchSize ?? 256;
    const started = performance.now();

    const pipeline = this.pipelineFor(limbs);

    const stateBuffer = storageBuffer(device, limbs * 2, "orbit-state");
    const seedBuffer = storageBuffer(device, limbs * 2, "orbit-seed");
    const scratch = storageBuffer(device, scratchWords(limbs), "orbit-scratch");
    const samples = storageBuffer(
      device,
      maxSamples * 6,
      "orbit-samples",
      GPUBufferUsage.COPY_SRC
    );
    const status = storageBuffer(device, 4, "orbit-status", GPUBufferUsage.COPY_SRC);
    const params = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // z starts at 0; c is the view centre.
    const seed = new Uint32Array(limbs * 2);
    seed.set(parseFixed(request.centerX, limbs), 0);
    seed.set(parseFixed(request.centerY, limbs), limbs);
    device.queue.writeBuffer(seedBuffer, 0, seed);
    device.queue.writeBuffer(stateBuffer, 0, new Uint32Array(limbs * 2));
    device.queue.writeBuffer(status, 0, new Uint32Array(4));
    // Sample 0 is exactly zero.
    device.queue.writeBuffer(samples, 0, new Float32Array(6));

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: stateBuffer } },
        { binding: 1, resource: { buffer: seedBuffer } },
        { binding: 2, resource: { buffer: scratch } },
        { binding: 3, resource: { buffer: samples } },
        { binding: 4, resource: { buffer: status } },
        { binding: 5, resource: { buffer: params } },
      ],
    });

    // Counts written samples; sample 0 (z = 0) is written by the CPU. The
    // shader emits at `startIndex + iter + 1`, so it must be handed the index
    // of the last written sample, not the count.
    let sampleCount = 1;
    let escaped = false;
    let escapeIteration = 0;

    while (sampleCount - 1 < request.maxIterations) {
      if (request.signal?.aborted) break;

      const done = sampleCount - 1;
      const iterations = Math.min(batchSize, request.maxIterations - done);
      device.queue.writeBuffer(
        params,
        0,
        new Uint32Array([iterations, done, maxSamples, 0])
      );

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
      device.queue.submit([encoder.finish()]);

      // Only the tiny status buffer comes back, never the big state.
      const raw = new Uint32Array(await readBuffer(device, status, 16));
      if (raw[0] <= sampleCount) break;
      sampleCount = raw[0];
      if (raw[1] === 1) {
        escaped = true;
        escapeIteration = raw[2];
        break;
      }
      request.onProgress?.(sampleCount - 1, request.maxIterations);
      if (sampleCount >= maxSamples) break;
    }

    const length = Math.max(1, sampleCount);
    const data = new Float32Array(await readBuffer(device, samples, maxSamples * 24));

    stateBuffer.destroy();
    seedBuffer.destroy();
    scratch.destroy();
    samples.destroy();
    status.destroy();
    params.destroy();

    return {
      samples: data,
      length,
      escaped,
      escapeIteration,
      elapsedMs: performance.now() - started,
    };
  }
}
