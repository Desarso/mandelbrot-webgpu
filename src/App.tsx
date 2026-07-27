import { createSignal, For, onCleanup, onMount, Show, type Component } from "solid-js";
import { MandelbrotView, type ViewInfo } from "./logic/MandelbrotView";
import {
  ColorSettings,
  DEFAULT_COLORS,
  decodeColors,
  MAPPINGS,
  MAX_STOPS,
  PALETTE_CUSTOM,
} from "./logic/colorSettings";
import styles from "./App.module.css";

const PALETTES = [
  { id: 0, name: "Spectrum", swatch: styles.spectrum },
  { id: 1, name: "Ultra", swatch: styles.ultra },
  { id: 2, name: "Ember", swatch: styles.ember },
  { id: 3, name: "Ice", swatch: styles.ice },
  { id: 4, name: "Mono", swatch: styles.mono },
  { id: PALETTE_CUSTOM, name: "Custom", swatch: styles.custom },
];

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
  return Number.isFinite(value) ? Math.min(4000, Math.max(50, value)) : 500;
}

const App: Component = () => {
  const [maxIterations, setMaxIterations] = createSignal(initialIterations());
  const [colors, setColors] = createSignal<ColorSettings>(initialColors());
  const patch = (change: Partial<ColorSettings>) =>
    setColors((current) => ({ ...current, ...change }));

  const [panelOpen, setPanelOpen] = createSignal(true);
  const [advanced, setAdvanced] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [renderer, setRenderer] = createSignal<MandelbrotView | null>(null);
  const [copied, setCopied] = createSignal(false);

  const view = (): ViewInfo | null => renderer()?.view() ?? null;

  let canvas!: HTMLCanvasElement;

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

  const copyLink = async () => {
    await navigator.clipboard.writeText(window.location.href);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };

  onMount(() => {
    // Backend selection is async (WebGPU adapter request), so the canvas stays
    // blank for a frame or two before the first draw.
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

  // Keep the iteration count in the shareable URL alongside view and colours.
  const syncIterations = (value: number) => {
    setMaxIterations(value);
    const url = new URL(window.location.href);
    url.searchParams.set("i", String(value));
    window.history.replaceState({}, "", decodeURIComponent(url.toString()));
  };

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
            <div class={styles.groupTitle}>Detail</div>
            <div class={styles.field}>
              <label class={styles.fieldLabel} for="iterations">
                <span>Max iterations</span>
                <span class={styles.value}>{maxIterations()}</span>
              </label>
              <input
                id="iterations"
                class={styles.slider}
                type="range"
                min="50"
                max="4000"
                step="10"
                value={maxIterations()}
                onInput={(e) => syncIterations(+e.currentTarget.value)}
              />
            </div>
          </section>

          <section class={styles.group}>
            <div class={styles.groupTitle}>Colour</div>

            <div class={styles.field}>
              <div class={styles.palettes}>
                <For each={PALETTES}>
                  {(entry) => (
                    <button
                      class={`${styles.swatch} ${entry.swatch} ${
                        colors().palette === entry.id ? styles.swatchActive : ""
                      }`}
                      title={entry.name}
                      aria-pressed={colors().palette === entry.id}
                      onClick={() => patch({ palette: entry.id })}
                    >
                      {entry.name}
                    </button>
                  )}
                </For>
              </div>
            </div>

            <div class={styles.field}>
              <label class={styles.fieldLabel} for="cycle">
                <span>Band width</span>
                <span class={styles.value}>{colors().cycle}</span>
              </label>
              <input
                id="cycle"
                class={styles.slider}
                type="range"
                min="4"
                max="400"
                step="1"
                value={colors().cycle}
                onInput={(e) => patch({ cycle: +e.currentTarget.value })}
              />
            </div>

            <div class={styles.field}>
              <label class={styles.fieldLabel} for="offset">
                <span>Shift</span>
                <span class={styles.value}>{colors().offset.toFixed(2)}</span>
              </label>
              <input
                id="offset"
                class={styles.slider}
                type="range"
                min="0"
                max="1"
                step="0.005"
                value={colors().offset}
                onInput={(e) => patch({ offset: +e.currentTarget.value })}
              />
            </div>

            <div class={`${styles.field} ${styles.toggleRow}`}>
              <span>Smooth shading</span>
              <button
                class={`${styles.switch} ${colors().smooth ? styles.switchOn : ""}`}
                role="switch"
                aria-checked={colors().smooth}
                aria-label="Smooth shading"
                onClick={() => patch({ smooth: !colors().smooth })}
              />
            </div>

            <button
              class={styles.disclosure}
              aria-expanded={advanced()}
              onClick={() => setAdvanced((open) => !open)}
            >
              <span>{advanced() ? "▾" : "▸"}</span> Advanced
            </button>
          </section>

          <Show when={advanced()}>
            <section class={styles.group}>
              <div class={styles.groupTitle}>Advanced colour</div>

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

              <div class={`${styles.field} ${styles.toggleRow}`}>
                <span>Mirror bands</span>
                <button
                  class={`${styles.switch} ${colors().mirror ? styles.switchOn : ""}`}
                  role="switch"
                  aria-checked={colors().mirror}
                  aria-label="Mirror bands"
                  onClick={() => patch({ mirror: !colors().mirror })}
                />
              </div>

              <div class={`${styles.field} ${styles.toggleRow}`}>
                <span>Interior</span>
                <input
                  class={styles.colorInput}
                  type="color"
                  aria-label="Interior colour"
                  value={colors().interior}
                  onInput={(e) => patch({ interior: e.currentTarget.value })}
                />
              </div>

              <div class={styles.field}>
                <div class={styles.fieldLabel}>
                  <span>Custom stops</span>
                  <span class={styles.value}>{colors().stops.length}</span>
                </div>
                <div class={styles.stops}>
                  <For each={colors().stops}>
                    {(stop, index) => (
                      <div class={styles.stopRow}>
                        <input
                          class={styles.colorInput}
                          type="color"
                          aria-label={`Stop ${index() + 1}`}
                          value={stop}
                          onInput={(e) => setStop(index(), e.currentTarget.value)}
                        />
                        <button
                          class={styles.stopRemove}
                          title="Remove stop"
                          aria-label={`Remove stop ${index() + 1}`}
                          disabled={colors().stops.length <= 2}
                          onClick={() => removeStop(index())}
                        >
                          ×
                        </button>
                      </div>
                    )}
                  </For>
                  <button
                    class={styles.stopAdd}
                    title="Add stop"
                    aria-label="Add stop"
                    disabled={colors().stops.length >= MAX_STOPS}
                    onClick={addStop}
                  >
                    +
                  </button>
                </div>
                <p class={styles.hint}>
                  Stops apply to the <strong>Custom</strong> palette and loop back
                  to the first colour.
                </p>
              </div>

              <button
                class={styles.button}
                onClick={() => setColors({ ...DEFAULT_COLORS })}
              >
                Reset colours
              </button>
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
            <div class={styles.readout}>
              <span>Timing</span>
              <span>
                {view()
                  ? `orbit ${view()!.orbitMs.toFixed(0)}ms · draw ${view()!.renderMs.toFixed(0)}ms`
                  : "—"}
              </span>
            </div>
            <Show when={view()?.atDepthLimit}>
              <p class={styles.hint}>
                At this backend's depth limit — zoom is clamped here.
              </p>
            </Show>
            <div class={styles.buttonRow}>
              <button class={styles.button} onClick={() => renderer()?.resetView()}>
                Reset view
              </button>
              <button class={styles.button} onClick={copyLink}>
                {copied() ? "Copied" : "Copy link"}
              </button>
            </div>
            <p class={styles.hint}>
              Drag to pan, scroll to zoom. Press <span class={styles.kbd}>H</span> to
              hide the controls. The URL always holds the current view and colours.
            </p>
          </section>
        </aside>
      </Show>
    </div>
  );
};

export default App;
