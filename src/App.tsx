import {
  createEffect,
  createSignal,
  For,
  Index,
  onCleanup,
  onMount,
  Show,
  type Component,
} from "solid-js";
import { MandelbrotView, type ViewInfo } from "./logic/MandelbrotView";
import { MAX_ITERATIONS } from "./logic/iterations";
import { nextIterations, type Probe } from "./logic/autoIterations";

export { MAX_ITERATIONS };
import {
  ColorSettings,
  DEFAULT_COLORS,
  MAPPINGS,
  MAX_STOPS,
  PALETTE_CUSTOM,
  PRESETS,
  decodeColors,
} from "./logic/colorSettings";
import { LOCATIONS } from "./logic/locations";
import { Slider, Toggle } from "./ui/Slider";
import styles from "./App.module.css";
import { AVAILABLE, applyDocumentLanguage, language, languageName, setLanguage, t } from "./i18n";
import type { StringKey } from "./i18n/strings";

const PALETTES = [
  { id: 0, name: "Spectrum", swatch: styles.spectrum },
  { id: 1, name: "Ultra", swatch: styles.ultra },
  { id: 2, name: "Ember", swatch: styles.ember },
  { id: 3, name: "Ice", swatch: styles.ice },
  { id: 4, name: "Mono", swatch: styles.mono },
  { id: PALETTE_CUSTOM, name: "Custom", swatch: styles.custom },
];

const TABS = ["Colour", "Light", "Advanced", "Places"] as const;

/** Tab identity stays English; only what is drawn is translated. */
const TAB_LABELS: Record<(typeof TABS)[number], StringKey> = {
  Colour: "ui.tab.colour",
  Light: "ui.tab.light",
  Advanced: "ui.tab.advanced",
  Places: "ui.tab.places",
};

/**
 * How many palette presets the Colour tab shows. The rest are one tab away:
 * sixteen swatches is a wall, and the first few are the ones worth reaching
 * for without looking.
 */
const FEATURED_PRESETS = 6;

/** Below this the frame is not a representative sample of the view. */
const MIN_MEASURABLE_PIXELS = 64 * 64;

/**
 * Ceiling on the iteration slider. Deep views legitimately need tens of
 * thousands: the iteration count needed grows with zoom depth.
 */


/** The slider is logarithmic; linear would make everything below 20k unusable. */
function sliderToIterations(position: number): number {
  const value = Math.round(50 * Math.pow(MAX_ITERATIONS / 50, position / 1000));
  return Math.min(MAX_ITERATIONS, Math.max(50, value));
}

function iterationsToSlider(value: number): number {
  return Math.round((1000 * Math.log(value / 50)) / Math.log(MAX_ITERATIONS / 50));
}

function formatZoom(zoom: number): string {
  if (zoom < 1000) return `${zoom.toFixed(zoom < 10 ? 2 : 0)}x`;
  const exponent = Math.floor(Math.log10(zoom));
  return `${(zoom / 10 ** exponent).toFixed(1)}e${exponent}x`;
}

function initialColors(): ColorSettings {
  const code = new URL(window.location.href).searchParams.get("c");
  return (code && decodeColors(code)) || DEFAULT_COLORS;
}

function initialIterations(): number {
  const raw = new URL(window.location.href).searchParams.get("i");
  const value = Number.parseInt(raw ?? "", 10);
  return Number.isFinite(value) ? Math.min(MAX_ITERATIONS, Math.max(50, value)) : 500;
}

/** CSS gradient preview for a set of stops, looping back to the first. */
function gradientOf(stops: string[]): string {
  return `linear-gradient(90deg, ${[...stops, stops[0]].join(", ")})`;
}

const App: Component = () => {
  const [maxIterations, setMaxIterations] = createSignal(initialIterations());
  // An explicit ?i= is a deliberate choice, so honour it rather than
  // overriding it on the first frame.
  const [autoIterations, setAutoIterations] = createSignal(
    !new URL(window.location.href).searchParams.has("i")
  );
  const [colors, setColors] = createSignal<ColorSettings>(initialColors());
  const patch = (change: Partial<ColorSettings>) =>
    setColors((current) => ({ ...current, ...change }));

  const [panelOpen, setPanelOpen] = createSignal(true);
  const [tab, setTab] = createSignal<(typeof TABS)[number]>("Colour");
  const [error, setError] = createSignal<string | null>(null);
  createEffect(() => applyDocumentLanguage(language()));
  // Falling back to WebGL2 is silent otherwise, and the difference is large:
  // it runs out of precision around 1e-34 where WebGPU keeps going.
  const [webgpuNoticeDismissed, setWebgpuNoticeDismissed] = createSignal(
    localStorage.getItem("mandelbrot.webgpu-notice") === "dismissed"
  );
  // A coarse primary pointer is the closest thing to a reliable "this is a
  // phone or tablet" signal that does not involve sniffing the user agent.
  const isTouchDevice =
    typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches;
  // On by default: the iteration count is the one number that changes on its
  // own while auto is on, and it is worth seeing with the panel hidden.
  // A slider cannot express "exactly 250,000", and a number field is a poor
  // way to sweep a range. Both, and remember which was last used.
  const [numericInput, setNumericInput] = createSignal(
    localStorage.getItem("mandelbrot.iteration-input") === "number"
  );
  const toggleNumericInput = () => {
    const next = !numericInput();
    localStorage.setItem("mandelbrot.iteration-input", next ? "number" : "slider");
    setNumericInput(next);
  };
  const [showHud, setShowHud] = createSignal(
    localStorage.getItem("mandelbrot.hud") !== "off"
  );
  const toggleHud = (on: boolean) => {
    localStorage.setItem("mandelbrot.hud", on ? "on" : "off");
    setShowHud(on);
  };
  const [mobileNoticeDismissed, setMobileNoticeDismissed] = createSignal(
    localStorage.getItem("mandelbrot.mobile-notice") === "dismissed"
  );
  // Only one of the two, and the WebGPU one takes precedence: it explains a
  // hard limit on depth, where this one only explains slowness.
  const showMobileNotice = () =>
    isTouchDevice &&
    !mobileNoticeDismissed() &&
    view()?.backend === "webgpu";
  const [renderer, setRenderer] = createSignal<MandelbrotView | null>(null);
  const [copied, setCopied] = createSignal(false);
  const [searching, setSearching] = createSignal(false);
  const [found, setFound] = createSignal<string | null>(null);

  const view = (): ViewInfo | null => renderer()?.view() ?? null;
  const distanceMode = () => colors().mode === 1;

  let canvas!: HTMLCanvasElement;

  const syncIterations = (value: number) => {
    setMaxIterations(value);
    const url = new URL(window.location.href);
    // While auto is on the count is a function of the view, so writing it into
    // the URL would be noise -- and would switch auto off for whoever opens
    // the link.
    if (autoIterations()) url.searchParams.delete("i");
    else url.searchParams.set("i", String(value));
    window.history.replaceState({}, "", decodeURIComponent(url.toString()));
  };

  // Follow the zoom while auto is on. The guard matters: setting the count
  // re-renders, which republishes the view, which lands back here.
  // Probes taken for the view currently being measured, and what the last
  // completed search settled on.
  let search: { key: string; probes: Probe[]; done: boolean } | null = null;
  let settledIterations = 0;

  createEffect(() => {
    if (!autoIterations()) return;
    const info = view();
    if (!info) return;
    // Not mid-gesture. Changing the count invalidates the reference orbit, so
    // adjusting on every frame of a zoom rebuilt it on every frame of a zoom --
    // hundreds of milliseconds each. Wait for the view to settle.
    if (info.preview) return;

    const key = `${info.centerX}|${info.centerY}|${info.zoom}`;
    if (!search || search.key !== key) {
      // A new view. Start from the depth estimate, which is a decent opening
      // guess, and let the measurements correct it from there.
      search = { key, probes: [], done: false };
      // Open from whatever the last search settled on rather than the depth
      // estimate. Panning at a fixed zoom lands on a view with much the same
      // requirement, and restarting from a guess makes the count visibly jump
      // around while the search rediscovers the answer it already had.
      const opening = settledIterations || info.suggestedIterations;
      if (opening > 0 && opening !== maxIterations()) {
        syncIterations(opening);
        return;
      }
    }
    if (search.done) return;

    // The count this frame was rendered with, not the current setting. The
    // effect re-runs the moment the setting changes, which is a frame before
    // anything has been rendered at it; pairing the new count with the old
    // frame's measurement reads as "raising bought nothing" and walks the
    // search straight down to the floor.
    const measured = info.iterations;
    if (measured <= 0) return;
    // A frame too small to be representative tells us nothing. A collapsed
    // container renders one pixel, and if that pixel happens to be inside the
    // set the measurement reads as "100% capped at every budget", which is
    // true of that pixel and useless as evidence about the view.
    if (info.pixels < MIN_MEASURABLE_PIXELS) return;
    if (search.probes.some((p) => p.iterations === measured)) return;
    search.probes.push({ iterations: measured, capped: info.cappedRatio });

    const decision = nextIterations(search.probes, MAX_ITERATIONS);
    if (decision.action === "settle") {
      search.done = true;
      settledIterations = decision.iterations;
    }
    if (decision.iterations !== measured) syncIterations(decision.iterations);
  });

  // Auto is the default, so it is *not* in the URL -- an explicit ?i= is what
  // records the choice to override it. Clearing the parameter has to happen on
  // the toggle itself and not only when the count changes: switching auto on
  // for a view it already agrees with left the stale ?i= in place, and the
  // next reload came back in manual mode.
  createEffect(() => {
    const url = new URL(window.location.href);
    const has = url.searchParams.has("i");
    if (autoIterations() === !has) return;
    if (autoIterations()) url.searchParams.delete("i");
    else url.searchParams.set("i", String(maxIterations()));
    window.history.replaceState({}, "", decodeURIComponent(url.toString()));
  });

  const setStop = (index: number, value: string) => {
    const stops = [...colors().stops];
    stops[index] = value;
    patch({ stops });
  };

  const addStop = () => {
    const stops = colors().stops;
    if (stops.length >= MAX_STOPS) return;
    patch({ stops: [...stops, "#ffffff"] });
  };

  const removeStop = (index: number) => {
    const stops = colors().stops;
    if (stops.length <= 2) return;
    patch({ stops: stops.filter((_, i) => i !== index) });
  };

  // Newton on the nucleus equation. It is CPU-bound decimal arithmetic, so
  // yield a frame first to let the button's pressed state paint.
  const findMinibrot = () => {
    setSearching(true);
    setFound(null);
    setTimeout(() => {
      const nucleus = renderer()?.findMinibrot();
      if (nucleus) {
        // A period-p minibrot needs several multiples of p before its
        // surroundings resolve; at the default count everything reads interior.
        const needed = Math.min(MAX_ITERATIONS, nucleus.period * 8);
        if (needed > maxIterations()) syncIterations(needed);
      }
      setFound(
        nucleus
          ? `period ${nucleus.period} · size ${nucleus.size.toExponential(2)}`
          : "no nucleus found near here"
      );
      setSearching(false);
    }, 16);
  };

  const copyLink = async () => {
    await navigator.clipboard.writeText(window.location.href);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };

  onMount(() => {
    MandelbrotView.create(canvas, { maxIterations, colors })
      .then(setRenderer)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));

    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if (event.key === "h" || event.key === "H") setPanelOpen((open) => !open);
    };
    window.addEventListener("keydown", onKeyDown);
    onCleanup(() => window.removeEventListener("keydown", onKeyDown));
  });

  return (
    <div class={styles.app}>
      <canvas ref={canvas} class={styles.canvas} />

      <Show when={error()}>
        <div class={styles.error}>{error()}</div>
      </Show>

      <Show when={!panelOpen() && !error()}>
        <button class={styles.showButton} onClick={() => setPanelOpen(true)}>
          {t("ui.controls")}
        </button>
      </Show>

      <div
        class={`${styles.cornerLinks} ${
          panelOpen() && !error() ? styles.cornerLinksShifted : ""
        }`}
      >
        <a class={styles.howButton} href="/about.html">
          {t("ui.whatIsThis")}
        </a>
        <a class={styles.howButton} href="/tech.html">
          {t("ui.howItWorks")} →
        </a>
      </div>

      <Show when={showHud()}>
        <div class={styles.hud}>
          <Show when={autoIterations()}>
            <span class={styles.hudAuto}>{t("ui.auto")}</span>
          </Show>
          <span class={styles.hudValue}>{maxIterations().toLocaleString()}</span>
          <span class={styles.hudLabel}>{t("ui.maxIterations")}</span>
        </div>
      </Show>

      <Show when={view()?.atDepthLimit}>
        <div class={styles.notice}>
          <p>{t("ui.notice.depthLimit")}</p>
        </div>
      </Show>

      <Show when={showMobileNotice() && !view()?.atDepthLimit}>
        <div class={styles.notice}>
          <p>{t("ui.notice.mobile")}</p>
          <button
            class={styles.noticeClose}
            onClick={() => {
              localStorage.setItem("mandelbrot.mobile-notice", "dismissed");
              setMobileNoticeDismissed(true);
            }}
          >
            {t("ui.notice.dismiss")}
          </button>
        </div>
      </Show>

      <Show
        when={
          view()?.backend === "webgl" &&
          !webgpuNoticeDismissed() &&
          !view()?.atDepthLimit
        }
      >
        <div class={styles.notice}>
          <p>{t("ui.notice.noWebgpu")}</p>
          <p>{t("ui.notice.enableFlag")}</p>
          <button
            class={styles.noticeClose}
            onClick={() => {
              localStorage.setItem("mandelbrot.webgpu-notice", "dismissed");
              setWebgpuNoticeDismissed(true);
            }}
          >
            {t("ui.notice.dismiss")}
          </button>
        </div>
      </Show>

      <Show when={panelOpen() && !error()}>
        <aside class={styles.panel}>
          <div class={styles.panelHeader}>
            <span class={styles.title}>{t("ui.title")}</span>
            <button
              class={styles.iconButton}
              title="Hide controls (H)"
              aria-label="Hide controls"
              onClick={() => setPanelOpen(false)}
            >
              ×
            </button>
          </div>

          <section class={styles.group}>
            <div class={styles.field}>
              <label class={styles.fieldLabel} for="iterations">
                <span>{t("ui.maxIterations")}</span>
                <span class={styles.value}>
                  <button
                    class={`${styles.autoChip} ${
                      autoIterations() ? styles.autoChipOn : ""
                    }`}
                    title={t("ui.autoIterationsHint")}
                    aria-pressed={autoIterations()}
                    onClick={() => setAutoIterations((on) => !on)}
                  >
                    {t("ui.auto")}
                  </button>
                  <button
                    class={styles.autoChip}
                    title={numericInput() ? t("ui.useSlider") : t("ui.useNumber")}
                    onClick={toggleNumericInput}
                  >
                    {numericInput() ? "\u2194" : "123"}
                  </button>
                  <Show when={!numericInput()}>
                    {maxIterations().toLocaleString()}
                  </Show>
                </span>
              </label>
              <Show
                when={numericInput()}
                fallback={
                  <input
                    id="iterations"
                    class={styles.slider}
                    type="range"
                    min="0"
                    max="1000"
                    step="1"
                    value={iterationsToSlider(maxIterations())}
                    onInput={(e) => {
                      // Reaching for the control is a clear statement that the
                      // guess is not wanted, so stop overriding it.
                      setAutoIterations(false);
                      syncIterations(sliderToIterations(+e.currentTarget.value));
                    }}
                  />
                }
              >
                <input
                  id="iterations"
                  class={styles.numberInput}
                  type="number"
                  min="50"
                  step="100"
                  value={maxIterations()}
                  onChange={(e) => {
                    // Committed on change, not input: typing "1" on the way to
                    // "100000" should not trigger a render at one iteration.
                    const value = Math.round(Number(e.currentTarget.value));
                    if (!Number.isFinite(value) || value < 50) {
                      e.currentTarget.value = String(maxIterations());
                      return;
                    }
                    setAutoIterations(false);
                    syncIterations(value);
                  }}
                />
              </Show>
              <Toggle
                label={t("ui.showOnCanvas")}
                value={showHud()}
                onChange={toggleHud}
              />
            </div>
          </section>

          <div class={styles.tabs}>
            <For each={TABS}>
              {(name) => (
                <button
                  class={`${styles.tab} ${tab() === name ? styles.tabActive : ""}`}
                  onClick={() => setTab(name)}
                >
                  {t(TAB_LABELS[name])}
                </button>
              )}
            </For>
          </div>

          <Show when={tab() === "Colour"}>
            <section class={styles.group}>
              <div class={styles.field}>
                <div class={styles.palettes}>
                  <For each={PALETTES}>
                    {(entry) => (
                      <button
                        class={`${styles.swatch} ${entry.swatch} ${
                          colors().palette === entry.id ? styles.swatchActive : ""
                        }`}
                        title={entry.name}
                        onClick={() => patch({ palette: entry.id })}
                      >
                        {entry.name}
                      </button>
                    )}
                  </For>
                </div>
              </div>

              <Show
                when={distanceMode()}
                fallback={
                  <>
                    <Slider
                      label={t("ui.bandWidth")}
                      value={colors().cycle}
                      min={4}
                      max={400}
                      step={1}
                      onInput={(cycle) => patch({ cycle })}
                    />
                    <Slider
                      label={t("ui.shift")}
                      value={colors().offset}
                      min={0}
                      max={1}
                      step={0.005}
                      digits={2}
                      onInput={(offset) => patch({ offset })}
                    />
                  </>
                }
              >
                <Slider
                  label={t("ui.colourDensity")}
                  value={colors().colorDensity}
                  min={0.01}
                  max={2}
                  step={0.01}
                  digits={2}
                  onInput={(colorDensity) => patch({ colorDensity })}
                />
                <Slider
                  label={t("ui.colourPhase")}
                  value={colors().colorPhase}
                  min={0}
                  max={1}
                  step={0.005}
                  digits={3}
                  onInput={(colorPhase) => patch({ colorPhase })}
                />
              </Show>

            </section>

            <section class={styles.group}>
              <div class={styles.groupTitle}>{t("ui.palettePresets")}</div>
              <div class={styles.presets}>
                <For each={PRESETS.slice(0, FEATURED_PRESETS)}>
                  {(preset) => (
                    <button
                      class={styles.preset}
                      title={preset.name}
                      style={{ background: gradientOf(preset.stops) }}
                      onClick={() =>
                        patch({ stops: [...preset.stops], palette: PALETTE_CUSTOM })
                      }
                    >
                      <span>{preset.name}</span>
                    </button>
                  )}
                </For>
              </div>
            </section>

          </Show>

          <Show when={tab() === "Advanced"}>
            <section class={styles.group}>
              <div class={styles.groupTitle}>Colouring method</div>
              <div class={styles.segmented}>
                <button
                  class={`${styles.segment} ${!distanceMode() ? styles.segmentActive : ""}`}
                  onClick={() => patch({ mode: 0 })}
                  title="Classic escape-count bands"
                >
                  {t("ui.mode.iteration")}
                </button>
                <button
                  class={`${styles.segment} ${distanceMode() ? styles.segmentActive : ""}`}
                  onClick={() => patch({ mode: 1 })}
                  title="Analytic distance estimation — the flowing, embossed look"
                >
                  {t("ui.mode.distance")}
                </button>
              </div>
              <p class={styles.hint}>
                Distance estimation measures how far each pixel is from the set
                rather than counting iterations. It resolves the thin filaments
                that banding misses, and it is what the Light tab shades.
              </p>
            </section>

            <section class={styles.group}>
              <div class={styles.groupTitle}>{t("ui.bandShaping")}</div>
              <Toggle
                label={t("ui.smoothShading")}
                value={colors().smooth}
                onChange={(smooth) => patch({ smooth })}
              />
              <Toggle
                label={t("ui.mirrorBands")}
                value={colors().mirror}
                onChange={(mirror) => patch({ mirror })}
              />
              <div class={styles.field}>
                <div class={styles.fieldLabel}>
                  <span>{t("ui.iterationMapping")}</span>
                </div>
                <div class={styles.segmented}>
                  <For each={MAPPINGS}>
                    {(name, index) => (
                      <button
                        class={`${styles.segment} ${
                          colors().mapping === index() ? styles.segmentActive : ""
                        }`}
                        onClick={() => patch({ mapping: index() })}
                      >
                        {name}
                      </button>
                    )}
                  </For>
                </div>
              </div>
              <div class={`${styles.field} ${styles.toggleRow}`}>
                <span>{t("ui.insideColour")}</span>
                <input
                  class={styles.colorInput}
                  type="color"
                  aria-label="Inside colour"
                  value={colors().interior}
                  onInput={(e) => patch({ interior: e.currentTarget.value })}
                />
              </div>
            </section>

            <section class={styles.group}>
              <div class={styles.groupTitle}>{t("ui.customStops")}</div>
              <div class={styles.stops}>
                {/* Index, not For: For is keyed by item value, so editing a
                    stop replaces its DOM node and the open native colour
                    picker is destroyed mid-drag. Index keys by position. */}
                <Index each={colors().stops}>
                  {(stop, index) => (
                    <div class={styles.stopRow}>
                      <input
                        class={styles.colorInput}
                        type="color"
                        aria-label={`Stop ${index + 1}`}
                        value={stop()}
                        onInput={(e) => setStop(index, e.currentTarget.value)}
                      />
                      <button
                        class={styles.stopRemove}
                        title="Remove stop"
                        disabled={colors().stops.length <= 2}
                        onClick={() => removeStop(index)}
                      >
                        ×
                      </button>
                    </div>
                  )}
                </Index>
                <button
                  class={styles.stopAdd}
                  title="Add stop"
                  disabled={colors().stops.length >= MAX_STOPS}
                  onClick={addStop}
                >
                  +
                </button>
              </div>
              <div
                class={styles.gradientPreview}
                style={{ background: gradientOf(colors().stops) }}
              />
            </section>
          </Show>

          <Show when={tab() === "Advanced"}>
            <section class={styles.group}>
              <div class={styles.groupTitle}>{t("ui.language")}</div>
              <select
                class={styles.select}
                value={language()}
                onChange={(e) => setLanguage(e.currentTarget.value)}
              >
                <For each={AVAILABLE}>
                  {(tag) => <option value={tag}>{languageName(tag)}</option>}
                </For>
              </select>
            </section>

            <section class={styles.group}>
              <div class={styles.groupTitle}>More palettes</div>
              <div class={styles.presets}>
                <For each={PRESETS.slice(FEATURED_PRESETS)}>
                  {(preset) => (
                    <button
                      class={styles.preset}
                      title={preset.name}
                      style={{ background: gradientOf(preset.stops) }}
                      onClick={() =>
                        patch({ stops: [...preset.stops], palette: PALETTE_CUSTOM })
                      }
                    >
                      <span>{preset.name}</span>
                    </button>
                  )}
                </For>
              </div>
            </section>
          </Show>

          <Show when={tab() === "Light"}>
            <section class={styles.group}>
              <Show
                when={distanceMode()}
                fallback={
                  <p class={styles.hint}>
                    Slope lighting shades the distance field, so switch the
                    Colour tab to <strong>Distance</strong> to use it.
                  </p>
                }
              >
                <Toggle
                  label="Slope lighting"
                  value={colors().slopeLighting}
                  onChange={(slopeLighting) => patch({ slopeLighting })}
                />
                <Slider
                  label="Relief depth"
                  value={colors().slopeDepth}
                  min={0}
                  max={12}
                  step={0.1}
                  digits={1}
                  onInput={(slopeDepth) => patch({ slopeDepth })}
                />
                <Slider
                  label="Light angle"
                  value={colors().lightAngle}
                  min={0}
                  max={359}
                  step={1}
                  suffix="°"
                  onInput={(lightAngle) => patch({ lightAngle })}
                />
                <Slider
                  label="Light height"
                  value={colors().lightElevation}
                  min={0}
                  max={90}
                  step={1}
                  suffix="°"
                  onInput={(lightElevation) => patch({ lightElevation })}
                />
                <Slider
                  label="Ambient"
                  value={colors().ambientLight}
                  min={0}
                  max={1.5}
                  step={0.01}
                  digits={2}
                  onInput={(ambientLight) => patch({ ambientLight })}
                />
                <Slider
                  label="Diffuse"
                  value={colors().diffuseStrength}
                  min={0}
                  max={2}
                  step={0.01}
                  digits={2}
                  onInput={(diffuseStrength) => patch({ diffuseStrength })}
                />
                <Slider
                  label="Specular"
                  value={colors().specularStrength}
                  min={0}
                  max={1.5}
                  step={0.01}
                  digits={2}
                  onInput={(specularStrength) => patch({ specularStrength })}
                />
              </Show>
            </section>

            <section class={styles.group}>
              <div class={styles.groupTitle}>{t("ui.quality")}</div>
              <div class={styles.field}>
                <div class={styles.fieldLabel}>
                  <span>{t("ui.supersampling")}</span>
                  <span class={styles.value}>
                    {colors().supersample}×{colors().supersample}
                  </span>
                </div>
                <div class={styles.segmented}>
                  <For each={[1, 2, 3]}>
                    {(n) => (
                      <button
                        class={`${styles.segment} ${
                          colors().supersample === n ? styles.segmentActive : ""
                        }`}
                        onClick={() => patch({ supersample: n })}
                      >
                        {n === 1 ? "off" : `${n}×${n}`}
                      </button>
                    )}
                  </For>
                </div>
              </div>
              <Slider
                label={t("ui.gamma")}
                value={colors().gamma}
                min={1}
                max={3.2}
                step={0.05}
                digits={2}
                onInput={(gamma) => patch({ gamma })}
              />
              <p class={styles.hint}>{t("ui.hint.gamma")}</p>
            </section>
          </Show>

          <Show when={tab() === "Places"}>
            <section class={styles.group}>
              <div class={styles.locations}>
                <For each={LOCATIONS}>
                  {(place) => (
                    <button
                      class={styles.location}
                      onClick={() => {
                        renderer()?.goTo(place.centerX, place.centerY, place.span);
                        syncIterations(place.iterations);
                      }}
                    >
                      <span class={styles.locationName}>{place.name}</span>
                      <span class={styles.locationBlurb}>{place.blurb}</span>
                    </button>
                  )}
                </For>
              </div>
            </section>

            <section class={styles.group}>
              <button
                class={styles.button}
                disabled={searching() || !renderer()}
                onClick={findMinibrot}
                title="Newton-solve for a nearby minibrot centre and zoom to it"
              >
                {searching() ? "Searching…" : "Find minibrot here"}
              </button>
              <Show when={found()}>
                <p class={styles.hint}>{found()}</p>
              </Show>
            </section>
          </Show>

          <section class={styles.group}>
            <div class={styles.groupTitle}>{t("ui.view")}</div>
            <div class={styles.readout}>
              <span>{t("ui.zoom")}</span>
              <span>{view() ? formatZoom(view()!.zoom) : "—"}</span>
            </div>
            <div class={styles.readout}>
              <span>Re</span>
              <span>{view()?.centerX ?? "—"}</span>
            </div>
            <div class={styles.readout}>
              <span>Im</span>
              <span>{view()?.centerY ?? "—"}</span>
            </div>
            <div class={styles.readout}>
              <span>{t("ui.engine")}</span>
              <span>
                {view()
                  ? `${view()!.backend === "webgpu" ? "WebGPU" : "WebGL2"} · ${view()!.precision}`
                  : "starting…"}
              </span>
            </div>
            <Show when={(view()?.skipRatio ?? 0) > 0}>
              <div class={styles.readout}>
                <span>Approx</span>
                <span>{(view()!.skipRatio * 100).toFixed(1)}% skipped</span>
              </div>
            </Show>
            <div class={styles.readout}>
              <span>{t("ui.timing")}</span>
              <span>
                {view()
                  ? `orbit ${view()!.orbitMs.toFixed(0)}ms · draw ${view()!.renderMs.toFixed(0)}ms`
                  : "—"}
              </span>
            </div>
            <Show when={view()?.atDepthLimit}>
              <p class={styles.hint}>{t("ui.hint.depthLimit")}</p>
            </Show>
            <Show when={view()?.error}>
              <p class={styles.hint}>{t("ui.error.stopped")}</p>
            </Show>

            <div class={styles.buttonRow}>
              <button class={styles.button} onClick={() => renderer()?.resetView()}>
                {t("ui.reset")}
              </button>
              <button class={styles.button} onClick={copyLink}>
                {copied() ? t("ui.copied") : t("ui.copyLink")}
              </button>
            </div>
            <p class={styles.hint}>{t("ui.hint.drag")}</p>
          </section>
        </aside>
      </Show>
    </div>
  );
};

export default App;
