import Decimal from "decimal.js";
import type { ColorSettings } from "../logic/colorSettings";

export interface DrawRequest {
  centerX: Decimal;
  centerY: Decimal;
  /** Complex units per device pixel. */
  unitsPerPixel: Decimal;
  width: number;
  height: number;
  maxIterations: number;
  colors: ColorSettings;
}

export interface BackendStats {
  /** Precision actually used, for display. */
  precision: string;
  orbitLength: number;
  orbitMs: number;
  renderMs: number;
  /** Fraction of per-pixel iterations skipped by linear approximation, 0..1. */
  skipRatio: number;
  /** Reference rebases this frame — the glitch-avoidance path. */
  rebases: number;
}

export interface RenderBackend {
  readonly name: "webgpu" | "webgl";
  /** Deepest span this backend renders correctly. */
  readonly minSpan: number;
  draw(request: DrawRequest): Promise<BackendStats>;
  dispose(): void;
}
