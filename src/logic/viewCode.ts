import Decimal from "decimal.js";

/**
 * Compact URL encoding for the view state.
 *
 * Coordinates are rounded to the number of digits the current zoom can
 * actually resolve, then written as a base36 integer mantissa plus the number
 * of decimal places, e.g. `1sy46sy20zcoqgjmkg85h1s.35`. Base36 packs ~5.17
 * bits per character against the ~3.32 bits of a decimal digit, which is a
 * ~36% saving; LZ-based compression measures *longer* here because the digits
 * of a fractal coordinate are essentially incompressible.
 */

const SEPARATOR = "~";
const B36 = 36n;

export interface ViewCode {
  centerX: Decimal;
  centerY: Decimal;
  span: Decimal;
}

function encodeDecimal(value: Decimal, significantDigits: number): string {
  const fixed = value.toSignificantDigits(significantDigits).toFixed();
  const negative = fixed.startsWith("-");
  const body = negative ? fixed.slice(1) : fixed;
  const [whole, fraction = ""] = body.split(".");
  const mantissa = BigInt(whole + fraction);
  return `${negative ? "-" : ""}${mantissa.toString(36)}.${fraction.length}`;
}

function decodeDecimal(code: string): Decimal {
  const [mantissaCode, scaleCode] = code.split(".");
  const negative = mantissaCode.startsWith("-");
  const digits = negative ? mantissaCode.slice(1) : mantissaCode;
  if (!digits || !/^[0-9a-z]+$/.test(digits)) {
    throw new Error(`Invalid mantissa: ${mantissaCode}`);
  }

  let mantissa = 0n;
  for (const character of digits) {
    mantissa = mantissa * B36 + BigInt(parseInt(character, 36));
  }

  const scale = Number.parseInt(scaleCode ?? "0", 10) || 0;
  const padded = mantissa.toString().padStart(scale + 1, "0");
  const split = padded.length - scale;
  const text = scale
    ? `${padded.slice(0, split)}.${padded.slice(split)}`
    : padded;
  return new Decimal(negative ? `-${text}` : text);
}

/**
 * @param unitsPerPixel drives how many digits are worth keeping — anything
 * finer than a pixel is invisible.
 */
export function encodeView(view: ViewCode, unitsPerPixel: Decimal): string {
  const resolvable = Math.max(0, -Math.log10(unitsPerPixel.toNumber()));
  const significantDigits = Math.max(6, Math.ceil(resolvable) + 4);
  return [
    encodeDecimal(view.centerX, significantDigits),
    encodeDecimal(view.centerY, significantDigits),
    view.span.toSignificantDigits(4).toExponential(),
  ].join(SEPARATOR);
}

export function decodeView(code: string): ViewCode | null {
  const parts = code.split(SEPARATOR);
  if (parts.length !== 3) return null;
  try {
    return {
      centerX: decodeDecimal(parts[0]),
      centerY: decodeDecimal(parts[1]),
      span: new Decimal(parts[2]),
    };
  } catch {
    return null;
  }
}
