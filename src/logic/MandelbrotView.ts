/**
 * View state, input handling and URL syncing. Pixel generation is delegated to
 * a RenderBackend, so the same pan/zoom behaviour drives either the WebGL or
 * the WebGPU path.
 */

import Decimal from "decimal.js";
import { Accessor, createEffect, createSignal, onCleanup, Setter } from "solid-js";
import { ColorSettings, encodeColors } from "./colorSettings";
import { MAX_ITERATIONS, iterationsForSpan } from "./iterations";
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
  /** Iterations this depth is likely to need, for the auto-iteration toggle. */
  suggestedIterations: number;
  /** Set when the backend has stopped working, e.g. a lost GPU device. */
  error?: string;
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
  /** Every finger or button currently down, by pointer id. */
  private pointers = new Map<number, { x: number; y: number }>();
  /** Midpoint and separation of the last two-finger sample, when pinching. */
  private lastPinch: { x: number; y: number; distance: number } | null = null;
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
      suggestedIterations: 0,
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

  /** The current view, as the backend wants it. */
  private drawRequest(width: number, height: number): DrawRequest {
    return {
      centerX: this.centerX,
      centerY: this.centerY,
      unitsPerPixel: this.unitsPerPixel(),
      width,
      height,
      maxIterations: Math.round(this.opts.maxIterations()),
      colors: this.opts.colors(),
      interacting: this.quality !== 1,
    };
  }

  private async render() {
    const { width, height } = this.canvas;
    if (width === 0 || height === 0) return;

    // The WebGPU path is async; never let two draws overlap.
    if (this.drawing) {
      this.queued = true;
      // A frame is already being computed for a view the user has since moved
      // past, so the screen is showing the wrong place until it lands. Shifting
      // the last finished frame into position costs a single blit and keeps the
      // gesture tracking the pointer instead of lurching a frame behind, and
      // the stale frame is told to stop rather than finish work for nowhere.
      this.backend.abort?.();
      this.backend.reproject?.(this.drawRequest(width, height));
      return;
    }
    this.drawing = true;

    const request = this.drawRequest(width, height);

    try {
      const stats = await this.backend.draw(request);
      this.publishView(stats);
    } catch (error) {
      console.error("[mandelbrot] draw failed:", error);
      this.setView({
        ...this.view(),
        error: error instanceof Error ? error.message : String(error),
      });
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
      suggestedIterations: iterationsForSpan(this.spanY, MAX_ITERATIONS),
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

    // Safari raises its own non-standard pinch events and will zoom the page
    // with them even where touch-action has already suppressed the gesture.
    // The canvas does its own zooming, so refuse them.
    for (const name of ["gesturestart", "gesturechange", "gestureend"]) {
      canvas.addEventListener(name, this.preventGesture, { passive: false });
    }

    onCleanup(() => {
      observer.disconnect();
      window.removeEventListener("resize", this.resize);
      document.removeEventListener("visibilitychange", this.onVisibilityChange);
      canvas.removeEventListener("wheel", this.onWheel);
      canvas.removeEventListener("pointerdown", this.onPointerDown);
      canvas.removeEventListener("pointermove", this.onPointerMove);
      canvas.removeEventListener("pointerup", this.onPointerUp);
      canvas.removeEventListener("pointercancel", this.onPointerUp);
      for (const name of ["gesturestart", "gesturechange", "gestureend"]) {
        canvas.removeEventListener(name, this.preventGesture);
      }
      clearTimeout(this.settleTimer);
      if (this.rafHandle) cancelAnimationFrame(this.rafHandle);
      if (this.timerHandle) clearTimeout(this.timerHandle);
      this.backend.dispose();
    });
  }

  /** Midpoint and separation of the two active pointers, in client space. */
  private pinchState(): { x: number; y: number; distance: number } | null {
    if (this.pointers.size < 2) return null;
    const [a, b] = [...this.pointers.values()];
    return {
      x: (a.x + b.x) / 2,
      y: (a.y + b.y) / 2,
      distance: Math.hypot(a.x - b.x, a.y - b.y),
    };
  }

  private preventGesture = (event: Event) => event.preventDefault();

  private onPointerDown = (event: PointerEvent) => {
    this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });

    // A second finger converts the gesture from a pan into a pinch. Both
    // baselines are reset here so the transition contributes no movement of
    // its own.
    this.lastPointer = { x: event.clientX, y: event.clientY };
    this.lastPinch = this.pinchState();
    this.dragging = true;
    this.canvas.style.cursor = "grabbing";

    // Capture last, and defensively: it throws when the id does not match a
    // live pointer, and doing it first meant one throw left the gesture
    // half-initialised -- pointer recorded, pinch baseline never set -- so
    // every subsequent move saw two fingers and no baseline to measure
    // against, and pinch quietly did nothing.
    try {
      this.canvas.setPointerCapture(event.pointerId);
    } catch {
      // Capture is an optimisation: it keeps events coming when a finger
      // leaves the canvas. The gesture works without it.
    }

    this.beginInteraction();
  };

  private onPointerMove = (event: PointerEvent) => {
    if (!this.pointers.has(event.pointerId)) return;
    this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });

    const pinch = this.pinchState();
    if (pinch && this.lastPinch) {
      // Two fingers: scale by how much they separated, about their midpoint,
      // and pan by however far that midpoint travelled. Doing both in one step
      // is what makes the point under the fingers stay under the fingers.
      const previous = this.lastPinch;
      this.lastPinch = pinch;

      if (previous.distance > 0 && pinch.distance > 0) {
        this.zoomAbout(
          pinch.x,
          pinch.y,
          previous.distance / pinch.distance,
          pinch.x - previous.x,
          pinch.y - previous.y
        );
      }
      this.beginInteraction();
      this.requestRender();
      return;
    }

    if (!this.dragging || this.pointers.size !== 1) return;
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
    this.pointers.delete(event.pointerId);
    if (this.canvas.hasPointerCapture(event.pointerId)) {
      this.canvas.releasePointerCapture(event.pointerId);
    }

    // Lifting one finger of a pinch leaves the other one panning. Its baseline
    // is stale by however far the pinch moved, so re-seed it from where that
    // finger actually is; otherwise the view jumps by the accumulated
    // difference the moment it moves again.
    const remaining = [...this.pointers.entries()][0];
    if (remaining) {
      this.lastPointer = { x: remaining[1].x, y: remaining[1].y };
      this.lastPinch = this.pinchState();
      return;
    }

    this.dragging = false;
    this.lastPinch = null;
    this.canvas.style.cursor = "grab";
    this.endInteraction();
  };

  /**
   * Scales the view by `factor` while holding the complex point under
   * (`clientX`, `clientY`) fixed, then pans by a screen-space offset.
   *
   * Shared by the wheel and by pinch, because "zoom toward the cursor" and
   * "zoom between two fingers" are the same operation with a different anchor.
   */
  private zoomAbout(
    clientX: number,
    clientY: number,
    factor: number,
    panX = 0,
    panY = 0
  ) {
    const rect = this.canvas.getBoundingClientRect();
    const units = this.unitsPerCssPixel();
    const offsetX = clientX - rect.left - this.canvas.clientWidth / 2;
    const offsetY = this.canvas.clientHeight / 2 - (clientY - rect.top);

    let nextSpan = this.spanY.times(factor);
    const floor = this.minSpan();
    if (nextSpan.lessThan(floor)) nextSpan = new Decimal(floor);
    if (nextSpan.greaterThan(MAX_SPAN)) nextSpan = new Decimal(MAX_SPAN);

    const nextUnits = nextSpan.div(Math.max(this.canvas.clientHeight, 1));
    const delta = units.minus(nextUnits);
    this.centerX = this.centerX.plus(delta.times(offsetX));
    this.centerY = this.centerY.plus(delta.times(offsetY));
    this.spanY = nextSpan;

    if (panX !== 0 || panY !== 0) {
      // The pan happens at the new scale, which is where the fingers now are.
      this.centerX = this.centerX.minus(nextUnits.times(panX));
      this.centerY = this.centerY.plus(nextUnits.times(panY));
    }
  }

  private onWheel = (event: WheelEvent) => {
    event.preventDefault();
    const step = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY;
    const factor = Math.min(4, Math.max(0.25, Math.pow(1.0015, step)));
    this.zoomAbout(event.clientX, event.clientY, factor);
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
