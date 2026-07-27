/**
 * Well-known places in the Mandelbrot set.
 *
 * Coordinates are given with enough digits for the listed span; the renderer
 * pads with zeros beyond that, which is exact for the point as written.
 * Iteration counts are what the location actually needs — several of these are
 * unrecognisable at the default 500.
 */

export interface Location {
  name: string;
  blurb: string;
  centerX: string;
  centerY: string;
  span: string;
  iterations: number;
}

export const LOCATIONS: Location[] = [
  {
    name: "Home",
    blurb: "The whole set",
    centerX: "-0.6",
    centerY: "0",
    span: "2.8",
    iterations: 500,
  },
  {
    name: "Seahorse Valley",
    blurb: "The notch between the cardioid and the period-2 bulb",
    centerX: "-0.743643887037158704752191506114774",
    centerY: "0.131825904205311970493132056385139",
    span: "1.2e-5",
    iterations: 3000,
  },
  {
    name: "Elephant Valley",
    blurb: "Trunk-like spirals on the cardioid's right flank",
    centerX: "0.28692299709",
    centerY: "0.014286480847",
    span: "6e-6",
    iterations: 2000,
  },
  {
    name: "Triple Spiral",
    blurb: "Three-armed vortex in the upper filaments",
    centerX: "-0.088",
    centerY: "0.654",
    span: "0.02",
    iterations: 1500,
  },
  {
    name: "Scepter Valley",
    blurb: "Along the western needle",
    // Sits above a period-108 nucleus found with findNucleus(); the old
    // coordinates were off the needle entirely and escaped after 41
    // iterations, so the whole frame came out one flat colour.
    centerX: "-1.74920463345901130344710267272610509993390621278893868444415",
    centerY: "1.04228658044466761986238520891015273896290502200064496392051e-16",
    span: "3e-11",
    iterations: 6000,
  },
  {
    name: "Quad Spiral",
    blurb: "Four-fold symmetry deep in the valley",
    centerX: "-0.7436447860",
    centerY: "0.1318252536",
    span: "4e-9",
    iterations: 5000,
  },
  {
    name: "Deep Spiral",
    blurb: "2.8e8x — spirals of spirals",
    centerX: "-0.600705755160234496572763605385",
    centerY: "0.441239870241679552586993134940",
    span: "1e-8",
    iterations: 2000,
  },
  {
    name: "Minibrot 1215",
    blurb: "A period-1215 island at 1e-42, found by Newton",
    centerX: "-0.600705755160234496572768007106812410353017771",
    centerY: "0.4412398702416795525869921575573731614595608",
    span: "6e-42",
    iterations: 20000,
  },
  {
    name: "Needle Tip",
    blurb: "The far west end of the antenna",
    centerX: "-1.9999999",
    centerY: "0",
    span: "1e-6",
    iterations: 4000,
  },
  {
    name: "Period-3 Bulb",
    blurb: "The nucleus at -1.7548776662",
    centerX: "-1.7548776662466927600495088",
    centerY: "0",
    span: "0.02",
    iterations: 1000,
  },
];
