/**
 * View state, input handling and URL syncing. Pixel generation is delegated to
 * a RenderBackend, so the same pan/zoom behaviour drives either the WebGL or
 * the WebGPU path.
 */

import Decimal from "decimal.js";
import { Accessor, createEffect, createSignal, onCleanup, Setter } from "solid-js";
import { ColorSettings, encodeColors } from "./colorSettings";
import { decodeView, encodeView } from "./viewCode";
import { findNucleus, type Nucleus } from "../orbit/nucleus";
import type { BackendStats, DrawRequest, RenderBackend } from "../render/backend";
import { WebGlBackend } from "../render/webgl-backend";
import { WebGpuBackend } from "../render/webgpu-backend";

const DEFAULT_SPAN = "2.8";
const DEFAULT_CENTER_X = "-0.6";
const DEFAULT_CENTER_Y = "0";
const MAX_SPAN = 8;
const LEGACY_URL_PARAMS = ["shift", "scale", "cx", "cy", "span"];

/**
 * Fraction of full resolution rendered while panning or zooming — 16x fewer
 * pixels. Resolution is the only thing reduced during interaction: capping the
 * iteration count makes deep views collapse to a solid interior colour,
 * because at high zoom nearly every pixel needs thousands of iterations before
 * it escapes. A coarse image is useful; a wrong one is not.
 */
const INTERACTIVE_SCALE = 0.25;
const SETTLE_DELAY_MS = 160;

export interface RenderOptions {
  maxIterations: Accessor<number>;
  colors: Accessor<ColorSettings>;
}

export interface ViewInfo {
  centerX: string;
  centerY: string;
  zoom: number;
  preview: boolean;
  backend: "webgpu" | "webgl";
  precision: string;
  orbitLength: number;
  orbitMs: number;
  renderMs: number;
  skipRatio: number;
  rebases: number;
  atDepthLimit: boolean;
}

export class MandelbrotView {
  private canvas: HTMLCanvasElement;
  private backend: RenderBackend;
  private opts: RenderOptions;

  private centerX = new Decimal(DEFAULT_CENTER_X);
  private centerY = new Decimal(DEFAULT_CENTER_Y);
  private spanY = new Decimal(DEFAULT_SPAN);

  private dpr = 1;
  private pending = false;
  private rafHandle = 0;
  private timerHandle = 0;
  private drawing = false;
  private queued = false;
  private dragging = false;
  private lastPointer = { x: 0, y: 0 };
  private quality = 1;
  private settleTimer: number | undefined;

  readonly view: Accessor<ViewInfo>;
  private setView: Setter<ViewInfo>;

  private constructor(
    canvas: HTMLCanvasElement,
    backend: RenderBackend,
    opts: RenderOptions
  ) {
    this.canvas = canvas;
    this.backend = backend;
    this.opts = opts;

    const [view, setView] = createSignal<ViewInfo>({
      centerX: this.centerX.toString(),
      centerY: this.centerY.toString(),
      zoom: 1,
      preview: false,
      backend: backend.name,
      precision: "",
      orbitLength: 0,
      orbitMs: 0,
      renderMs: 0,
      skipRatio: 0,
      rebases: 0,
      atDepthLimit: false,
    });
    this.view = view;
    this.setView = setView;

    this.restoreFromUrl();
    this.attachEvents();

    createEffect(() => {
      opts.maxIterations();
      opts.colors();
      this.requestRender();
    });

    this.resize();
  }

  /** Prefers WebGPU; falls back to WebGL2 when it is unavailable. */
  static async create(
    canvas: HTMLCanvasElement,
    opts: RenderOptions
  ): Promise<MandelbrotView> {
    let backend: RenderBackend;
    try {
      backend = await WebGpuBackend.create(canvas);
    } catch (error) {
      console.info("[mandelbrot] WebGPU unavailable, using WebGL2:", error);
      backend = new WebGlBackend(canvas);
    }
    return new MandelbrotView(canvas, backend, opts);
  }

  /**
   * Jumps to a minibrot nucleus near the current centre.
   *
   * Hand-zooming past ~1e-30 nearly always lands in a smooth region where every
   * pixel escapes at the same iteration. Newton on f_p(c) = 0 finds the centre
   * of a nearby period-p minibrot instead, and the size estimate says how far
   * to zoom. Returns the nucleus, or null if Newton did not converge.
   */
  findMinibrot(): Nucleus | null {
    const digits = Math.max(
      60,
      Math.ceil(-Math.log10(this.unitsPerPixel().toNumber())) + 30
    );
    const nucleus = findNucleus(this.centerX, this.centerY, {
      maxPeriod: Math.max(2000, Math.round(this.opts.maxIterations())),
      digits,
    });
    if (!nucleus) return null;

    this.centerX = nucleus.centerX;
    this.centerY = nucleus.centerY;
    // Frame the minibrot with a little room around it.
    const span = nucleus.size.times(4);
    this.spanY = span.lessThan(this.minSpan())
      ? new Decimal(this.minSpan())
      : span.greaterThan(MAX_SPAN)
        ? new Decimal(MAX_SPAN)
        : span;
    this.requestRender();
    return nucleus;
  }

  /** Jumps to a fixed coordinate, e.g. a preset location. */
  goTo(centerX: string, centerY: string, span: string) {
    this.centerX = new Decimal(centerX);
    this.centerY = new Decimal(centerY);
    const wanted = new Decimal(span);
    const floor = new Decimal(this.minSpan());
    this.spanY = wanted.lessThan(floor)
      ? floor
      : wanted.greaterThan(MAX_SPAN)
        ? new Decimal(MAX_SPAN)
        : wanted;
    this.requestRender();
  }

  resetView() {
    this.centerX = new Decimal(DEFAULT_CENTER_X);
    this.centerY = new Decimal(DEFAULT_CENTER_Y);
    this.spanY = new Decimal(DEFAULT_SPAN);
    this.requestRender();
  }

  private minSpan(): number {
    return this.backend.minSpan;
  }

  private unitsPerPixel(): Decimal {
    return this.spanY.div(Math.max(this.canvas.height, 1));
  }

  private unitsPerCssPixel(): Decimal {
    return this.spanY.div(Math.max(this.canvas.clientHeight, 1));
  }

  /**
   * Coalesces redraw requests to one per frame.
   *
   * requestAnimationFrame never fires while the document is hidden, so a
   * pending callback would latch out every later request and rendering would
   * stall permanently — including after the tab becomes visible again, since
   * nothing would ask for a frame. Fall back to a timer when hidden so view
   * state still converges, and re-request on visibilitychange.
   */
  requestRender() {
    if (this.pending) return;
    this.pending = true;

    const run = () => {
      this.pending = false;
      this.rafHandle = 0;
      this.timerHandle = 0;
      void this.render();
    };

    if (document.hidden) {
      this.timerHandle = window.setTimeout(run, 16);
    } else {
      this.rafHandle = requestAnimationFrame(run);
    }
  }

  private onVisibilityChange = () => {
    if (!document.hidden) this.requestRender();
  };

  private beginInteraction() {
    if (this.quality !== INTERACTIVE_SCALE) {
      this.quality = INTERACTIVE_SCALE;
      this.applyBackingStore();
    }
    clearTimeout(this.settleTimer);
    this.settleTimer = window.setTimeout(() => this.endInteraction(), SETTLE_DELAY_MS);
  }

  private endInteraction() {
    clearTimeout(this.settleTimer);
    this.settleTimer = undefined;
    if (this.dragging) return;
    if (this.quality !== 1) {
      this.quality = 1;
      this.applyBackingStore();
    }
    this.requestRender();
  }

  private async render() {
    const { width, height } = this.canvas;
    if (width === 0 || height === 0) return;

    // The WebGPU path is async; never let two draws overlap.
    if (this.drawing) {
      this.queued = true;
      return;
    }
    this.drawing = true;

    const request: DrawRequest = {
      centerX: this.centerX,
      centerY: this.centerY,
      unitsPerPixel: this.unitsPerPixel(),
      width,
      height,
      maxIterations: Math.round(this.opts.maxIterations()),
      colors: this.opts.colors(),
      interacting: this.quality !== 1,
    };

    try {
      const stats = await this.backend.draw(request);
      this.publishView(stats);
    } catch (error) {
      console.error("[mandelbrot] draw failed:", error);
    } finally {
      this.drawing = false;
      if (this.queued) {
        this.queued = false;
        this.requestRender();
      }
    }
  }

  private publishView(stats: BackendStats) {
    const digits = Math.max(
      6,
      Math.ceil(-Math.log10(this.unitsPerPixel().toNumber())) + 2
    );
    this.setView({
      centerX: this.centerX.toSignificantDigits(digits).toString(),
      centerY: this.centerY.toSignificantDigits(digits).toString(),
      zoom: new Decimal(DEFAULT_SPAN).div(this.spanY).toNumber(),
      preview: this.quality !== 1,
      backend: this.backend.name,
      precision: stats.precision,
      orbitLength: stats.orbitLength,
      orbitMs: stats.orbitMs,
      renderMs: stats.renderMs,
      skipRatio: stats.skipRatio,
      rebases: stats.rebases,
      atDepthLimit: this.spanY.lessThanOrEqualTo(this.minSpan() * 1.001),
    });

    const url = new URL(window.location.href);
    for (const stale of LEGACY_URL_PARAMS) url.searchParams.delete(stale);
    url.searchParams.set(
      "v",
      encodeView(
        { centerX: this.centerX, centerY: this.centerY, span: this.spanY },
        this.unitsPerPixel()
      )
    );
    url.searchParams.set("c", encodeColors(this.opts.colors()));
    window.history.replaceState({}, "", decodeURIComponent(url.toString()));
  }

  private applyBackingStore(): boolean {
    const scale = this.dpr * this.quality;
    const width = Math.max(1, Math.round(this.canvas.clientWidth * scale));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * scale));
    if (width === this.canvas.width && height === this.canvas.height) return false;
    this.canvas.width = width;
    this.canvas.height = height;
    return true;
  }

  private resize = () => {
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (this.applyBackingStore()) this.requestRender();
  };

  private attachEvents() {
    const canvas = this.canvas;
    const observer = new ResizeObserver(this.resize);
    observer.observe(canvas);
    window.addEventListener("resize", this.resize);
    document.addEventListener("visibilitychange", this.onVisibilityChange);

    canvas.addEventListener("wheel", this.onWheel, { passive: false });
    canvas.addEventListener("pointerdown", this.onPointerDown);
    canvas.addEventListener("pointermove", this.onPointerMove);
    canvas.addEventListener("pointerup", this.onPointerUp);
    canvas.addEventListener("pointercancel", this.onPointerUp);

    onCleanup(() => {
      observer.disconnect();
      window.removeEventListener("resize", this.resize);
      document.removeEventListener("visibilitychange", this.onVisibilityChange);
      canvas.removeEventListener("wheel", this.onWheel);
      canvas.removeEventListener("pointerdown", this.onPointerDown);
      canvas.removeEventListener("pointermove", this.onPointerMove);
      canvas.removeEventListener("pointerup", this.onPointerUp);
      canvas.removeEventListener("pointercancel", this.onPointerUp);
      clearTimeout(this.settleTimer);
      if (this.rafHandle) cancelAnimationFrame(this.rafHandle);
      if (this.timerHandle) clearTimeout(this.timerHandle);
      this.backend.dispose();
    });
  }

  private onPointerDown = (event: PointerEvent) => {
    this.dragging = true;
    this.lastPointer = { x: event.clientX, y: event.clientY };
    this.canvas.setPointerCapture(event.pointerId);
    this.canvas.style.cursor = "grabbing";
    this.beginInteraction();
  };

  private onPointerMove = (event: PointerEvent) => {
    if (!this.dragging) return;
    const units = this.unitsPerCssPixel();
    const dx = event.clientX - this.lastPointer.x;
    const dy = event.clientY - this.lastPointer.y;
    this.lastPointer = { x: event.clientX, y: event.clientY };

    // Screen y grows downwards, the imaginary axis grows upwards.
    this.centerX = this.centerX.minus(units.times(dx));
    this.centerY = this.centerY.plus(units.times(dy));
    this.beginInteraction();
    this.requestRender();
  };

  private onPointerUp = (event: PointerEvent) => {
    this.dragging = false;
    if (this.canvas.hasPointerCapture(event.pointerId)) {
      this.canvas.releasePointerCapture(event.pointerId);
    }
    this.canvas.style.cursor = "grab";
    this.endInteraction();
  };

  private onWheel = (event: WheelEvent) => {
    event.preventDefault();

    const rect = this.canvas.getBoundingClientRect();
    const units = this.unitsPerCssPixel();
    const offsetX = event.clientX - rect.left - this.canvas.clientWidth / 2;
    const offsetY = this.canvas.clientHeight / 2 - (event.clientY - rect.top);

    const step = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY;
    const factor = Math.min(4, Math.max(0.25, Math.pow(1.0015, step)));

    let nextSpan = this.spanY.times(factor);
    const floor = this.minSpan();
    if (nextSpan.lessThan(floor)) nextSpan = new Decimal(floor);
    if (nextSpan.greaterThan(MAX_SPAN)) nextSpan = new Decimal(MAX_SPAN);

    const nextUnits = nextSpan.div(Math.max(this.canvas.clientHeight, 1));
    const delta = units.minus(nextUnits);
    this.centerX = this.centerX.plus(delta.times(offsetX));
    this.centerY = this.centerY.plus(delta.times(offsetY));
    this.spanY = nextSpan;

    this.beginInteraction();
    this.requestRender();
  };

  private restoreFromUrl() {
    const params = new URL(window.location.href).searchParams;

    const compact = params.get("v");
    if (compact) {
      const view = decodeView(compact);
      if (view) {
        this.centerX = view.centerX;
        this.centerY = view.centerY;
        this.spanY = view.span;
      }
      return;
    }

    try {
      const cx = params.get("cx");
      const cy = params.get("cy");
      const span = params.get("span");
      if (cx) this.centerX = new Decimal(cx);
      if (cy) this.centerY = new Decimal(cy);
      if (span) this.spanY = new Decimal(span);
    } catch {
      // Ignore malformed parameters and keep the default view.
    }
  }
}
