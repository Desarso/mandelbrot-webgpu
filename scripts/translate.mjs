/**
 * Regenerates src/i18n/locales/*.json from the English strings.
 *
 *   node scripts/translate.mjs              # every language in LANGUAGES
 *   node scripts/translate.mjs es fr ja     # just these
 *   node scripts/translate.mjs --missing    # only keys not already translated
 *
 * Needs bulktranslatorgo on PATH:
 *   go install github.com/Desarso/BulkTranslator/BulkTranslatorGo@latest
 *
 * Output is committed, so a normal build and a normal clone never run this.
 */

import { spawn } from "node:child_process";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Runs the CLI with `input` on stdin and returns stdout.
 *
 * Not execFile: its async form silently ignores the `input` option -- that is
 * execFileSync only -- so the child sits waiting on a stdin that never closes.
 */
function run(args, input) {
  return new Promise((resolve, reject) => {
    const child = spawn("bulktranslatorgo", args, { stdio: ["pipe", "pipe", "pipe"] });
    let out = "";
    let err = "";
    child.stdout.on("data", (d) => (out += d));
    child.stderr.on("data", (d) => (err += d));
    child.on("error", reject);
    child.on("close", (code) =>
      code === 0 ? resolve(out) : reject(new Error(err.trim() || `exit ${code}`))
    );
    child.stdin.end(input);
  });
}
const here = dirname(fileURLToPath(import.meta.url));
const localesDir = join(here, "..", "src", "i18n", "locales");
const stringsFile = join(here, "..", "src", "i18n", "strings.ts");

/**
 * The languages shipped. Google Translate offers about 130; this is the subset
 * with enough speakers to be worth the bytes, and it is what the locale
 * detector matches against.
 */
const LANGUAGES = [
  "af", "am", "ar", "az", "be", "bg", "bn", "bs", "ca", "cs", "cy", "da", "de",
  "el", "eo", "es", "et", "eu", "fa", "fi", "tl", "fr", "ga", "gl", "gu",
  "ha", "he", "hi", "hr", "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "kk",
  "km", "kn", "ko", "ku", "ky", "lo", "lt", "lv", "mk", "ml", "mn", "mr", "ms",
  "my", "ne", "nl", "no", "pa", "pl", "ps", "pt", "ro", "ru", "si", "sk", "sl",
  "so", "sq", "sr", "sv", "sw", "ta", "te", "tg", "th", "tr", "uk", "ur", "uz",
  "vi", "yi", "zh-CN", "zh-TW", "zu",
];

/** How many languages to ask for in one invocation of the CLI. */
const LANGUAGE_CHUNK = 8;

/**
 * Pulls the key/value pairs out of strings.ts.
 *
 * Importing it would mean compiling TypeScript for a script that runs by hand
 * a few times a year, and the file is a plain object literal, so a regex over
 * the source is enough. Anything cleverer than a flat map of string literals
 * will be skipped, loudly.
 */
async function loadSource() {
  const text = await readFile(stringsFile, "utf8");
  const body = text.slice(text.indexOf("export const STRINGS = {"));
  const entries = [];
  const pattern = /"((?:[^"\\]|\\.)*)":\s*((?:"(?:[^"\\]|\\.)*"\s*)+),?\n/g;
  let match;
  while ((match = pattern.exec(body))) {
    // Values may be split over several adjacent literals for line length.
    const value = [...match[2].matchAll(/"((?:[^"\\]|\\.)*)"/g)]
      .map((m) => JSON.parse(`"${m[1]}"`))
      .join("");
    entries.push([JSON.parse(`"${match[1]}"`), value]);
  }
  if (entries.length === 0) throw new Error("no strings found in strings.ts");
  return Object.fromEntries(entries);
}

async function readLocale(language) {
  try {
    return JSON.parse(await readFile(join(localesDir, `${language}.json`), "utf8"));
  } catch {
    return {};
  }
}

/** Languages that do not put spaces between sentences. */
const UNSPACED = new Set(["zh-CN", "zh-TW", "ja", "th", "lo", "km", "my"]);

/**
 * Splits text into sentences.
 *
 * The endpoint behind the CLI returns only the *first* sentence of anything it
 * is given, silently -- "Pick a point on a flat plane. Square it, ..." came
 * back as "Elija un punto en un plano." and nothing else. So each sentence is
 * sent separately and the pieces are rejoined afterwards.
 *
 * The split needs a following capital to be a sentence end, which keeps
 * decimals like "about 1.506." from being cut in half.
 */
function splitSentences(text) {
  return text.split(/(?<=[.!?])\s+(?=[A-Z"'(])/u);
}

/**
 * Translates `texts` into several languages at once.
 *
 * Newlines are the record separator on stdin, so a string containing one would
 * silently split into two records and desynchronise the whole batch. None
 * currently do; this asserts it rather than trusting it.
 */
async function translate(texts, languages) {
  for (const text of texts) {
    if (text.includes("\n")) throw new Error(`string contains a newline: ${text}`);
  }

  // Flatten every sentence of every string into one request, remembering how
  // many sentences each string contributed so they can be put back together.
  const pieces = texts.map(splitSentences);
  const flat = pieces.flat();

  const stdout = await run(
    ["-from", "en", "-to", languages.join(","), "-format", "json", "-timeout", "180s"],
    flat.join("\n")
  );
  const parsed = JSON.parse(stdout);
  if (parsed.translations.length !== flat.length) {
    throw new Error(
      `asked for ${flat.length} sentences, got ${parsed.translations.length} back`
    );
  }
  const perSentence = parsed.translations.map((entry) =>
    // A single destination comes back shaped differently from several.
    languages.length === 1 ? { [languages[0]]: entry.translation } : entry.translations
  );

  let at = 0;
  return pieces.map((sentences) => {
    const slice = perSentence.slice(at, at + sentences.length);
    at += sentences.length;
    return Object.fromEntries(
      languages.map((language) => [
        language,
        slice
          .map((s) => s[language])
          .filter(Boolean)
          .join(UNSPACED.has(language) ? "" : " "),
      ])
    );
  });
}

async function main() {
  const args = process.argv.slice(2);
  const onlyMissing = args.includes("--missing");
  const requested = args.filter((a) => !a.startsWith("--"));
  const targets = requested.length > 0 ? requested : LANGUAGES;

  const source = await loadSource();
  await mkdir(localesDir, { recursive: true });
  const failed = [];

  for (let i = 0; i < targets.length; i += LANGUAGE_CHUNK) {
    const chunk = targets.slice(i, i + LANGUAGE_CHUNK);
    const existing = Object.fromEntries(
      await Promise.all(chunk.map(async (l) => [l, await readLocale(l)]))
    );

    // Only send keys that some language in this chunk is actually missing.
    const keys = Object.keys(source).filter((key) =>
      onlyMissing ? chunk.some((l) => existing[l][key] === undefined) : true
    );
    if (keys.length === 0) {
      console.log(`${chunk.join(",")}: already complete`);
      continue;
    }

    process.stdout.write(`${chunk.join(",")}: ${keys.length} strings… `);
    const results = await translate(keys.map((k) => source[k]), chunk);

    for (const language of chunk) {
      // The endpoint drops a whole language from a multi-destination request
      // now and then, without an error -- an entire chunk of eight came back
      // untranslated once. Falling back to English per key hides that
      // completely, so check before writing rather than after.
      const returned = keys.filter((key, index) => results[index][language]).length;
      const identical = keys.filter(
        (key, index) => results[index][language] === source[key]
      ).length;
      if (returned === 0 || identical > keys.length * 0.5) {
        failed.push(language);
        console.log(
          `\n  ${language}: dropped (${returned}/${keys.length} returned, ` +
            `${identical} identical to English) — not written`
        );
        continue;
      }

      const out = { ...existing[language] };
      keys.forEach((key, index) => {
        out[key] = results[index][language] ?? source[key];
      });
      // Written in source order so diffs stay readable.
      const ordered = {};
      for (const key of Object.keys(source)) if (out[key] !== undefined) ordered[key] = out[key];
      await writeFile(
        join(localesDir, `${language}.json`),
        JSON.stringify(ordered, null, 2) + "\n"
      );
    }
    console.log("done");
  }

  const written = (await readdir(localesDir)).filter((f) => f.endsWith(".json"));
  console.log(`\n${written.length} locales in src/i18n/locales`);

  if (failed.length > 0) {
    console.log(`\nre-run these: node scripts/translate.mjs ${failed.join(" ")}`);
    process.exit(1);
  }
}

main().catch((error) => {
  console.error(error.message ?? error);
  process.exit(1);
});
