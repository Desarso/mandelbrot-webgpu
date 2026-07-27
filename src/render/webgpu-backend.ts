/** Adapts WebGpuRenderer to the RenderBackend interface. */

import { acquireGpu } from "../gpu/device";
import { WebGpuRenderer } from "./webgpu-renderer";
import type { BackendStats, DrawRequest, RenderBackend } from "./backend";

export class WebGpuBackend implements RenderBackend {
  readonly name = "webgpu" as const;
  /**
   * The limb profile grows with depth, so there is no fixed floor the way the
   * f32 path has one. This is simply where the largest profile (256 limbs,
   * ~2450 decimal digits) stops resolving a pixel.
   */
  readonly minSpan = 1e-2000;

  private renderer: WebGpuRenderer;
  adapterInfo: string;

  private constructor(renderer: WebGpuRenderer, adapterInfo: string) {
    this.renderer = renderer;
    this.adapterInfo = adapterInfo;
  }

  static async create(canvas: HTMLCanvasElement): Promise<WebGpuBackend> {
    const ctx = await acquireGpu();
    const renderer = new WebGpuRenderer(ctx, canvas);
    await renderer.init();
    return new WebGpuBackend(renderer, ctx.capabilities.adapterInfo);
  }

  async draw(request: DrawRequest): Promise<BackendStats> {
    const stats = await this.renderer.render(request);
    return {
      precision: `${stats.limbs} limbs / ${stats.decimalDigits} digits`,
      orbitLength: stats.orbitLength,
      orbitMs: stats.orbitMs,
      renderMs: stats.renderMs,
      skipRatio: stats.skipRatio,
      rebases: stats.rebases,
    };
  }

  reproject(request: DrawRequest): boolean {
    return this.renderer.reproject(request);
  }

  dispose() {
    // Buffers are owned by the renderer and released with the device.
  }
}
