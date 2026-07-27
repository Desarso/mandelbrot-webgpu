import {
  createSignal,
  For,
  Index,
  onCleanup,
  onMount,
  Show,
  type Component,
} from "solid-js";
import { MandelbrotView, type ViewInfo } from "./logic/MandelbrotView";
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

const PALETTES = [
  { id: 0, name: "Spectrum", swatch: styles.spectrum },
  { id: 1, name: "Ultra", swatch: styles.ultra },
  { id: 2, name: "Ember", swatch: styles.ember },
  { id: 3, name: "Ice", swatch: styles.ice },
  { id: 4, name: "Mono", swatch: styles.mono },
  { id: PALETTE_CUSTOM, name: "Custom", swatch: styles.custom },
];

const TABS = ["Colour", "Light", "Places"] as const;

/**
 * Ceiling on the iteration slider. Deep views legitimately need tens of
 * thousands: the iteration count needed grows with zoom depth.
 */
export const MAX_ITERATIONS = 200000;

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
  const [colors, setColors] = createSignal<ColorSettings>(initialColors());
  const patch = (change: Partial<ColorSettings>) =>
    setColors((current) => ({ ...current, ...change }));

  const [panelOpen, setPanelOpen] = createSignal(true);
  const [tab, setTab] = createSignal<(typeof TABS)[number]>("Colour");
  const [error, setError] = createSignal<string | null>(null);
  // Falling back to WebGL2 is silent otherwise, and the difference is large:
  // it runs out of precision around 1e-34 where WebGPU keeps going.
  const [webgpuNoticeDismissed, setWebgpuNoticeDismissed] = createSignal(
    localStorage.getItem("mandelbrot.webgpu-notice") === "dismissed"
  );
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
    url.searchParams.set("i", String(value));
    window.history.replaceState({}, "", decodeURIComponent(url.toString()));
  };

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
          Controls
        </button>
      </Show>

      <a
        class={`${styles.howButton} ${panelOpen() && !error() ? styles.howButtonShifted : ""}`}
        href="/tech.html"
      >
        How it works →
      </a>

      <Show when={view()?.backend === "webgl" && !webgpuNoticeDismissed()}>
        <div class={styles.notice}>
          <strong>Running without WebGPU.</strong> This is the WebGL2 fallback,
          which runs out of precision around 10<sup>−34</sup>; the WebGPU engine
          has no fixed limit. If your browser supports WebGPU but it is not
          being used, open <code>chrome://flags</code> and set{" "}
          <code>Override software rendering list</code> to Enabled, then
          restart. <code>chrome://gpu</code> shows why it was refused.
          <br />
          <button
            class={styles.noticeClose}
            onClick={() => {
              localStorage.setItem("mandelbrot.webgpu-notice", "dismissed");
              setWebgpuNoticeDismissed(true);
            }}
          >
            Dismiss
          </button>
        </div>
      </Show>

      <Show when={panelOpen() && !error()}>
        <aside class={styles.panel}>
          <div class={styles.panelHeader}>
            <span class={styles.title}>Mandelbrot</span>
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
                <span>Max iterations</span>
                <span class={styles.value}>{maxIterations().toLocaleString()}</span>
              </label>
              <input
                id="iterations"
                class={styles.slider}
                type="range"
                min="0"
                max="1000"
                step="1"
                value={iterationsToSlider(maxIterations())}
                onInput={(e) => syncIterations(sliderToIterations(+e.currentTarget.value))}
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
                  {name}
                </button>
              )}
            </For>
          </div>

          <Show when={tab() === "Colour"}>
            <section class={styles.group}>
              <div class={styles.segmented}>
                <button
                  class={`${styles.segment} ${!distanceMode() ? styles.segmentActive : ""}`}
                  onClick={() => patch({ mode: 0 })}
                  title="Classic escape-count bands"
                >
                  Iteration
                </button>
                <button
                  class={`${styles.segment} ${distanceMode() ? styles.segmentActive : ""}`}
                  onClick={() => patch({ mode: 1 })}
                  title="Analytic distance estimation — the flowing, embossed look"
                >
                  Distance
                </button>
              </div>

              <div class={styles.field} style={{ "margin-top": "10px" }}>
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
                      label="Band width"
                      value={colors().cycle}
                      min={4}
                      max={400}
                      step={1}
                      onInput={(cycle) => patch({ cycle })}
                    />
                    <Slider
                      label="Shift"
                      value={colors().offset}
                      min={0}
                      max={1}
                      step={0.005}
                      digits={2}
                      onInput={(offset) => patch({ offset })}
                    />
                    <Toggle
                      label="Smooth shading"
                      value={colors().smooth}
                      onChange={(smooth) => patch({ smooth })}
                    />
                    <Toggle
                      label="Mirror bands"
                      value={colors().mirror}
                      onChange={(mirror) => patch({ mirror })}
                    />
                    <div class={styles.field}>
                      <div class={styles.fieldLabel}>
                        <span>Iteration mapping</span>
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
                  </>
                }
              >
                <Slider
                  label="Colour density"
                  value={colors().colorDensity}
                  min={0.01}
                  max={2}
                  step={0.01}
                  digits={2}
                  onInput={(colorDensity) => patch({ colorDensity })}
                />
                <Slider
                  label="Colour phase"
                  value={colors().colorPhase}
                  min={0}
                  max={1}
                  step={0.005}
                  digits={3}
                  onInput={(colorPhase) => patch({ colorPhase })}
                />
              </Show>

              <div class={`${styles.field} ${styles.toggleRow}`}>
                <span>Inside colour</span>
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
              <div class={styles.groupTitle}>Palette presets</div>
              <div class={styles.presets}>
                <For each={PRESETS}>
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

            <section class={styles.group}>
              <div class={styles.groupTitle}>Custom stops</div>
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
              <div class={styles.groupTitle}>Quality</div>
              <div class={styles.field}>
                <div class={styles.fieldLabel}>
                  <span>Supersampling</span>
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
                label="Gamma"
                value={colors().gamma}
                min={1}
                max={3.2}
                step={0.05}
                digits={2}
                onInput={(gamma) => patch({ gamma })}
              />
              <p class={styles.hint}>
                Palette mixing and lighting run in linear light; gamma is applied
                once at the end.
              </p>
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
            <div class={styles.groupTitle}>View</div>
            <div class={styles.readout}>
              <span>Zoom</span>
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
              <span>Engine</span>
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
              <span>Timing</span>
              <span>
                {view()
                  ? `orbit ${view()!.orbitMs.toFixed(0)}ms · draw ${view()!.renderMs.toFixed(0)}ms`
                  : "—"}
              </span>
            </div>
            <Show when={view()?.atDepthLimit}>
              <p class={styles.hint}>At this backend's depth limit.</p>
            </Show>
            <Show when={view()?.error}>
              <p class={styles.hint}>
                Rendering stopped: {view()!.error}. This usually means the GPU
                driver gave up on a frame that took too long — lower the
                iteration count and reload.
              </p>
            </Show>

            <div class={styles.buttonRow}>
              <button class={styles.button} onClick={() => renderer()?.resetView()}>
                Reset
              </button>
              <button class={styles.button} onClick={copyLink}>
                {copied() ? "Copied" : "Copy link"}
              </button>
            </div>
            <p class={styles.hint}>
              Drag to pan, scroll to zoom. <span class={styles.kbd}>H</span> hides
              the panel. The URL always holds the full state.
            </p>
          </section>
        </aside>
      </Show>
    </div>
  );
};

export default App;
