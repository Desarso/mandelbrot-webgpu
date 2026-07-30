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
  /**
   * True while the user is panning or zooming. Backends should avoid expensive
   * rebuilds (reference orbit, approximation table) and reuse what they have:
   * a slightly stale reference is still mathematically exact, just less
   * efficient, whereas rebuilding mid-gesture is what makes zoom stutter.
   */
  interacting?: boolean;
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
  /** Fraction of samples that used the whole iteration budget, 0..1. */
  cappedRatio: number;
}

export interface RenderBackend {
  readonly name: "webgpu" | "webgl";
  /** Deepest span this backend renders correctly. */
  readonly minSpan: number;
  draw(request: DrawRequest): Promise<BackendStats>;
  /**
   * Presents the last completed frame under a new view, without iterating
   * anything. A pan or a zoom moves the picture far more often than it changes
   * it: the pixels the user is looking at were nearly all computed already,
   * just at different screen positions. Returns false when there is no frame
   * to reuse, or when the new view has moved too far for the reuse to mean
   * anything, in which case the caller should draw normally.
   */
  reproject?(request: DrawRequest): boolean;
  /**
   * Asks a draw in flight to give up early. The view has moved, so most of
   * what it is still computing is for somewhere the user is no longer looking.
   */
  abort?(): void;
  dispose(): void;
}
