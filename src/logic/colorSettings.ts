/** Maximum custom stops — must match MAX_STOPS in the fragment shader. */
export const MAX_STOPS = 8;

export const PALETTE_CUSTOM = 5;

export const MAPPINGS = ["linear", "sqrt", "log"] as const;
export type Mapping = (typeof MAPPINGS)[number];

export interface ColorSettings {
  palette: number;
  cycle: number;
  offset: number;
  smooth: boolean;
  mapping: number;
  mirror: boolean;
  interior: string;
  stops: string[];
}

export const DEFAULT_COLORS: ColorSettings = {
  palette: 1,
  cycle: 64,
  offset: 0,
  smooth: true,
  mapping: 0,
  mirror: false,
  interior: "#000000",
  stops: ["#08103a", "#2f6bcb", "#f2ffff", "#ffaa00", "#3a1400"],
};

const HEX = /^#[0-9a-f]{6}$/i;

export function hexToRgb(hex: string): [number, number, number] {
  const value = HEX.test(hex) ? hex : "#000000";
  return [
    parseInt(value.slice(1, 3), 16) / 255,
    parseInt(value.slice(3, 5), 16) / 255,
    parseInt(value.slice(5, 7), 16) / 255,
  ];
}

/**
 * Serialises to a dot-separated record, e.g. `5.64.0.1.2.1.000000.08103a2f6bcb`.
 * Stops are only written for the custom palette, since nothing else reads them.
 */
export function encodeColors(settings: ColorSettings): string {
  const fields = [
    settings.palette,
    Math.round(settings.cycle),
    Math.round(settings.offset * 1000),
    settings.smooth ? 1 : 0,
    settings.mapping,
    settings.mirror ? 1 : 0,
    settings.interior.slice(1),
  ];
  if (settings.palette === PALETTE_CUSTOM) {
    fields.push(settings.stops.map((stop) => stop.slice(1)).join(""));
  }
  return fields.join(".");
}

export function decodeColors(code: string): ColorSettings | null {
  const parts = code.split(".");
  if (parts.length < 7) return null;

  const number = (text: string, fallback: number) => {
    const value = Number.parseInt(text, 10);
    return Number.isFinite(value) ? value : fallback;
  };
  const color = (text: string, fallback: string) =>
    HEX.test(`#${text}`) ? `#${text}` : fallback;

  const packedStops = parts[7] ?? "";
  const stops: string[] = [];
  for (let i = 0; i + 6 <= packedStops.length && stops.length < MAX_STOPS; i += 6) {
    stops.push(color(packedStops.slice(i, i + 6), "#000000"));
  }

  return {
    palette: Math.min(5, Math.max(0, number(parts[0], DEFAULT_COLORS.palette))),
    cycle: Math.min(400, Math.max(4, number(parts[1], DEFAULT_COLORS.cycle))),
    offset: Math.min(1, Math.max(0, number(parts[2], 0) / 1000)),
    smooth: parts[3] !== "0",
    mapping: Math.min(2, Math.max(0, number(parts[4], 0))),
    mirror: parts[5] === "1",
    interior: color(parts[6], DEFAULT_COLORS.interior),
    stops: stops.length ? stops : DEFAULT_COLORS.stops,
  };
}
