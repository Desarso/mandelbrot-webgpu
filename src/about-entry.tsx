/**
 * The "what is this" page.
 *
 * Deliberately separate from tech.html: that page is about how the renderer
 * works, this one is about why the thing it renders is worth looking at, and
 * mixing the two serves neither reader. Rendered from the translation strings
 * rather than written as static markup, so it exists in every language.
 */

import { For, Show, createEffect, createSignal, onMount } from "solid-js";
import { render } from "solid-js/web";
import {
  AVAILABLE,
  applyDocumentLanguage,
  language,
  languageName,
  setLanguage,
  t,
} from "./i18n";
import type { StringKey } from "./i18n/strings";

/** Heading key, then the paragraph keys under it. */
const SECTIONS: Array<{ heading: StringKey; paragraphs: StringKey[] }> = [
  {
    heading: "about.rule.heading",
    paragraphs: ["about.rule.p1", "about.rule.p2", "about.rule.p3"],
  },
  {
    heading: "about.surprise.heading",
    paragraphs: ["about.surprise.p1", "about.surprise.p2"],
  },
  {
    heading: "about.boundary.heading",
    paragraphs: ["about.boundary.p1", "about.boundary.p2"],
  },
  {
    heading: "about.copies.heading",
    paragraphs: ["about.copies.p1", "about.copies.p2"],
  },
  {
    heading: "about.atlas.heading",
    paragraphs: ["about.atlas.p1", "about.atlas.p2"],
  },
  {
    heading: "about.discovery.heading",
    paragraphs: [
      "about.discovery.p1",
      "about.discovery.p2",
      "about.discovery.p3",
    ],
  },
  {
    heading: "about.scale.heading",
    paragraphs: ["about.scale.p1", "about.scale.p2"],
  },
  {
    heading: "about.explore.heading",
    paragraphs: ["about.explore.p1"],
  },
];

function About() {
  const [ready, setReady] = createSignal(false);

  // The strings arrive asynchronously for every language but English, so the
  // first paint would otherwise be English text that flips a moment later.
  onMount(() => setReady(true));
  createEffect(() => applyDocumentLanguage(language()));
  createEffect(() => {
    if (ready()) document.title = `${t("about.title")} · Mandelbrot`;
  });

  return (
    <div class="wrap">
      <div class="topbar">
        <a class="back" href="/">
          ← {t("about.back")}
        </a>
        <label class="langPicker">
          <span class="srOnly">{t("ui.language")}</span>
          <select
            value={language()}
            onChange={(e) => setLanguage(e.currentTarget.value)}
          >
            <For each={AVAILABLE}>
              {(tag) => <option value={tag}>{languageName(tag)}</option>}
            </For>
          </select>
        </label>
      </div>

      <header>
        <h1>{t("about.title")}</h1>
        <p class="lede">{t("about.lede")}</p>
      </header>

      <For each={SECTIONS}>
        {(section) => (
          <section>
            <h2>{t(section.heading)}</h2>
            <For each={section.paragraphs}>{(key) => <p>{t(key)}</p>}</For>
          </section>
        )}
      </For>

      <div class="ctaRow">
        <a class="cta ctaPrimary" href="/">
          {t("about.cta.render")}
        </a>
        <a class="cta" href="/tech.html">
          {t("about.cta.tech")}
        </a>
      </div>

      <Show when={language() !== "en"}>
        <p class="machineNote">
          Translated automatically. The English original is at{" "}
          <a href="?lang=en">?lang=en</a>.
        </p>
      </Show>
    </div>
  );
}

render(() => <About />, document.getElementById("root")!);
