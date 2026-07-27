/**
 * Locale selection and lookup.
 *
 * English is compiled in; every other language is a JSON file loaded on demand,
 * so a visitor downloads exactly one translation rather than all eighty. A
 * missing key falls through to English, which is what makes it safe to add a
 * string and ship before the translations catch up.
 */

import { createResource, createSignal } from "solid-js";
import { STRINGS, type StringKey } from "./strings";

/** Vite resolves this at build time into one lazy chunk per locale. */
const LOCALE_MODULES = import.meta.glob<{ default: Record<string, string> }>(
  "./locales/*.json"
);

/** Language tags with a translation, derived from what is actually on disk. */
export const AVAILABLE: readonly string[] = [
  "en",
  ...Object.keys(LOCALE_MODULES)
    .map((path) => path.slice("./locales/".length, -".json".length))
    .sort(),
];

/**
 * Endonyms — each language named in itself, which is how language pickers are
 * supposed to read. Anything without an entry falls back to its tag.
 */
const NAMES: Record<string, string> = {
  en: "English", af: "Afrikaans", am: "አማርኛ", ar: "العربية", az: "Azərbaycan",
  be: "Беларуская", bg: "Български", bn: "বাংলা", bs: "Bosanski", ca: "Català",
  cs: "Čeština", cy: "Cymraeg", da: "Dansk", de: "Deutsch", el: "Ελληνικά",
  eo: "Esperanto", es: "Español", et: "Eesti", eu: "Euskara", fa: "فارسی",
  fi: "Suomi", tl: "Filipino", fr: "Français", ga: "Gaeilge", gl: "Galego",
  gu: "ગુજરાતી", ha: "Hausa", he: "עברית", hi: "हिन्दी", hr: "Hrvatski",
  hu: "Magyar", hy: "Հայերեն", id: "Indonesia", is: "Íslenska", it: "Italiano",
  ja: "日本語", jv: "Jawa", ka: "ქართული", kk: "Қазақша", km: "ខ្មែរ",
  kn: "ಕನ್ನಡ", ko: "한국어", ku: "Kurdî", ky: "Кыргызча", lo: "ລາວ",
  lt: "Lietuvių", lv: "Latviešu", mk: "Македонски", ml: "മലയാളം",
  mn: "Монгол", mr: "मराठी", ms: "Melayu", my: "မြန်မာ", ne: "नेपाली",
  nl: "Nederlands", no: "Norsk", pa: "ਪੰਜਾਬੀ", pl: "Polski", ps: "پښتو",
  pt: "Português", ro: "Română", ru: "Русский", si: "සිංහල", sk: "Slovenčina",
  sl: "Slovenščina", so: "Soomaali", sq: "Shqip", sr: "Српски", sv: "Svenska",
  sw: "Kiswahili", ta: "தமிழ்", te: "తెలుగు", tg: "Тоҷикӣ", th: "ไทย",
  tr: "Türkçe", uk: "Українська", ur: "اردو", uz: "Oʻzbek", vi: "Tiếng Việt",
  yi: "ייִדיש", "zh-CN": "简体中文", "zh-TW": "繁體中文", zu: "isiZulu",
};

export const languageName = (tag: string) => NAMES[tag] ?? tag;

/** Right-to-left scripts, so the page can set `dir` correctly. */
const RTL = new Set(["ar", "fa", "he", "ps", "ur", "yi"]);
export const isRtl = (tag: string) => RTL.has(tag);

const STORAGE_KEY = "mandelbrot.language";

/**
 * Matches a browser language tag against what we have.
 *
 * `navigator.languages` gives entries like "pt-BR" or "zh-Hant-TW". Chinese is
 * the one case where the region genuinely picks a different translation rather
 * than a dialect of the same one, so it is resolved on script first; everything
 * else falls back to its base tag.
 */
export function resolveLanguage(preferred: readonly string[]): string {
  for (const raw of preferred) {
    const tag = raw.toLowerCase();
    if (tag.startsWith("zh")) {
      const traditional =
        tag.includes("hant") || tag.includes("tw") || tag.includes("hk") || tag.includes("mo");
      return traditional ? "zh-TW" : "zh-CN";
    }
    const exact = AVAILABLE.find((a) => a.toLowerCase() === tag);
    if (exact) return exact;
    const base = tag.split("-")[0];
    const partial = AVAILABLE.find((a) => a.toLowerCase() === base);
    if (partial) return partial;
  }
  return "en";
}

/** An explicit choice wins over the browser's, and persists. */
function initialLanguage(): string {
  const url = new URL(window.location.href).searchParams.get("lang");
  if (url && AVAILABLE.includes(url)) return url;
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved && AVAILABLE.includes(saved)) return saved;
  return resolveLanguage(navigator.languages ?? [navigator.language ?? "en"]);
}

const [language, setLanguageSignal] = createSignal(initialLanguage());
export { language };

export function setLanguage(tag: string) {
  if (!AVAILABLE.includes(tag)) return;
  localStorage.setItem(STORAGE_KEY, tag);
  setLanguageSignal(tag);
}

const [messages] = createResource(language, async (tag) => {
  if (tag === "en") return {} as Record<string, string>;
  const load = LOCALE_MODULES[`./locales/${tag}.json`];
  if (!load) return {};
  try {
    return (await load()).default;
  } catch {
    // A locale that fails to load is not worth breaking the page over; every
    // lookup falls through to English anyway.
    return {};
  }
});

/** Applies `lang` and `dir` to the document, so the browser hyphenates and
 *  lays out text correctly and screen readers pick the right voice. */
export function applyDocumentLanguage(tag: string) {
  document.documentElement.lang = tag;
  document.documentElement.dir = isRtl(tag) ? "rtl" : "ltr";
}

/** Looks up a key in the active language, falling back to English. */
export function t(key: StringKey): string {
  return messages()?.[key] ?? STRINGS[key];
}
