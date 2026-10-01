/**
 * The two reference-orbit kernels, and which one a device can run.
 *
 * Both compute the same samples bit for bit; the self-test holds them to that.
 * orbit-shared.wgsl keeps its working set in workgroup memory and is several
 * times faster, but that working set grows with the limb count, and the
 * 256-limb profile needs more than the 16 KiB WebGPU guarantees. orbit.wgsl
 * runs at any precision on any device, so it stays as the fallback.
 */

import { storageBuffer } from "./device";
import bigfixedSource from "./shaders/bigfixed.wgsl?raw";
import orbitBindings from "./shaders/orbit-bindings.wgsl?raw";
import orbitSource from "./shaders/orbit.wgsl?raw";
import orbitSharedSource from "./shaders/orbit-shared.wgsl?raw";

// Bindings must precede the arithmetic library: WGSL has no forward references.
export const orbitModule = [orbitBindings, bigfixedSource, orbitSource].join("\n");
export const orbitSharedModule = [orbitBindings, bigfixedSource, orbitSharedSource].join("\n");

export type OrbitKernel = "storage" | "shared";

/** Words of `scratch` the storage kernel needs. */
export function scratchWords(limbs: number): number {
  return 7 * limbs + 2 * limbs * 3;
}

/**
 * Workgroup memory the shared kernel declares: two magnitudes, three products'
 * columns at three words each, three products, plus a few scalars.
 */
export function sharedOrbitBytes(limbs: number): number {
  return 23 * limbs * 4 + 64;
}

/**
 * Invocations for the shared kernel. Each owns whole columns, and there are
 * only 6 * LIMBS of them, so 256 invocations at eight limbs would mostly wait
 * at barriers.
 */
export function sharedOrbitWorkgroup(limbs: number): number {
  return Math.min(256, Math.ceil((6 * limbs) / 32) * 32);
}

export function orbitKernelFor(device: GPUDevice, limbs: number): OrbitKernel {
  const fits =
    sharedOrbitBytes(limbs) <= device.limits.maxComputeWorkgroupStorageSize &&
    sharedOrbitWorkgroup(limbs) <= device.limits.maxComputeInvocationsPerWorkgroup;
  return fits ? "shared" : "storage";
}

export function createOrbitPipeline(
  device: GPUDevice,
  module: GPUShaderModule,
  kernel: OrbitKernel,
  limbs: number
): GPUComputePipeline {
  const constants: Record<string, number> =
    kernel === "shared"
      ? { LIMBS: limbs, ORBIT_WG: sharedOrbitWorkgroup(limbs) }
      : { LIMBS: limbs };
  return device.createComputePipeline({
    label: `orbit-${kernel}-${limbs}`,
    layout: "auto",
    compute: { module, entryPoint: "advanceOrbit", constants },
  });
}

/** Buffers one orbit run needs besides the samples, which the caller owns. */
export interface OrbitBuffers {
  state: GPUBuffer;
  seed: GPUBuffer;
  /** Only the storage kernel has one. */
  scratch: GPUBuffer | null;
  status: GPUBuffer;
  params: GPUBuffer;
}

export function createOrbitBuffers(
  device: GPUDevice,
  kernel: OrbitKernel,
  limbs: number
): OrbitBuffers {
  return {
    state: storageBuffer(device, limbs * 2, "orbit-state"),
    seed: storageBuffer(device, limbs * 2, "orbit-seed"),
    scratch:
      kernel === "storage"
        ? storageBuffer(device, scratchWords(limbs), "orbit-scratch")
        : null,
    status: storageBuffer(device, 4, "orbit-status", GPUBufferUsage.COPY_SRC),
    params: device.createBuffer({
      label: "orbit-params",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
  };
}

export function destroyOrbitBuffers(buffers: OrbitBuffers) {
  const { state, seed, scratch, status, params } = buffers;
  [state, seed, scratch, status, params].forEach((b) => b?.destroy());
}

/**
 * The bind group for either kernel. An "auto" layout drops bindings the entry
 * point never touches, so the shared kernel's group must not mention scratch.
 */
export function createOrbitBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  buffers: OrbitBuffers,
  samples: GPUBuffer
): GPUBindGroup {
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: buffers.state } },
    { binding: 1, resource: { buffer: buffers.seed } },
    { binding: 3, resource: { buffer: samples } },
    { binding: 4, resource: { buffer: buffers.status } },
    { binding: 5, resource: { buffer: buffers.params } },
  ];
  if (buffers.scratch) {
    entries.push({ binding: 2, resource: { buffer: buffers.scratch } });
  }
  return device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
}
