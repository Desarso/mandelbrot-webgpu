/**
 * Every user-visible string, in English.
 *
 * This is the translation source. `scripts/translate.mjs` feeds the values
 * through BulkTranslatorGo and writes one JSON file per language into
 * `locales/`, keyed identically. Keys are flat and stable so a translation can
 * be regenerated for a single string without disturbing the rest.
 *
 * Keep values as whole sentences. Machine translation needs the grammar of a
 * complete clause to get word order and agreement right, and a string built by
 * concatenating fragments will be wrong in most languages even when each
 * fragment translates correctly on its own.
 */

export const STRINGS = {
  // ------------------------------------------------------------------ about
  "about.title": "What the Mandelbrot set is",
  "about.lede":
    "One line of arithmetic, repeated. What comes out has detail at every scale, without end, and nothing in the rule hints that it should. This page is about why that is strange.",

  "about.rule.heading": "One rule, repeated",
  "about.rule.p1":
    "Pick a point on a flat plane. Square it, add the point you started with, and you have a new point. Square that, add the original again, and keep going forever.",
  "about.rule.p2":
    "Two things can happen. The points may wander off and never come back, racing away to infinity. Or they may stay near home no matter how long you keep squaring. The Mandelbrot set is simply the collection of starting points that stay.",
  "about.rule.p3":
    "That is the whole definition. The black region in every image on this site is the set itself; the colours outside it record how quickly each escaping point gave up and fled.",

  "about.surprise.heading": "Nobody designed the shape",
  "about.surprise.p1":
    "The rule contains no circles, no spirals, no seahorses, no lightning. It is a multiplication and an addition. Everything you can see was already implied by those two operations, waiting for someone to plot it.",
  "about.surprise.p2":
    "This is the part that unsettles people who meet it properly. The shape was not invented and it cannot be adjusted. Change the rule and you get a different object, not a better-looking version of this one. Every detail is forced.",

  "about.boundary.heading": "The edge never resolves",
  "about.boundary.p1":
    "Most shapes get simpler as you look closer. A circle magnified enough is indistinguishable from a straight line. The edge of the Mandelbrot set does the opposite: every magnification uncovers structure that was not visible before, and it does so without end.",
  "about.boundary.p2":
    "The inside is ordinary enough to measure — its area is about 1.506. The boundary is not. In 1998 Mitsuhiro Shishikura proved it has fractal dimension 2, which is a precise way of saying that a curve with no thickness manages to be as crowded as a filled-in surface.",

  "about.copies.heading": "Copies of the whole, but never quite",
  "about.copies.p1":
    "Zoom into the boundary anywhere and eventually a small black island appears, unmistakably the same shape you started from. These are usually called minibrots, and there are infinitely many of them, at every scale, all the way down.",
  "about.copies.p2":
    "They are not exact copies. Each one is distorted, and each is wrapped in scenery — spirals, filaments, chains of smaller islands — that belongs to that location and no other. The set repeats its theme without ever repeating itself.",

  "about.atlas.heading": "It is an index of other worlds",
  "about.atlas.p1":
    "Run the same rule but hold the added point fixed, and you get a different object called a Julia set. There is one for every point on the plane, and they range from clean connected loops to disconnected dust.",
  "about.atlas.p2":
    "The Mandelbrot set turns out to be exactly the map of which is which. A point inside it corresponds to a Julia set in one piece; a point outside corresponds to one that has crumbled. It is a single picture that catalogues an infinite family of others, and nobody put that property there on purpose either.",

  "about.discovery.heading": "It was found, not made",
  "about.discovery.p1":
    "The first known picture was printed by Robert Brooks and Peter Matelski in 1978. Benoit Mandelbrot, working at IBM, produced images in 1980 and recognised what he was looking at; the coarse printouts of the time were so speckled that he initially took some of the detail for dust on the hardware.",
  "about.discovery.p2":
    "The mathematics behind it is older still, going back to Pierre Fatou and Gaston Julia around 1918, who studied these iterations decades before there was any machine that could draw the results. They were reasoning about a shape none of them would ever see.",
  "about.discovery.p3":
    "Anyone, anywhere, computing the same rule gets the same object down to the last filament. In that sense it was not created in 1980. It was found there.",

  "about.scale.heading": "How far down this goes",
  "about.scale.p1":
    "One of the places listed here sits at a magnification of about 10⁴¹. If the whole set were drawn at that scale on a screen a metre wide, the screen would have to be hundreds of trillions of times wider than the observable universe to hold it. The arithmetic underneath carries around 2,400 decimal digits, so the renderer does not run out until roughly 10²⁴⁰⁰ — a number with no physical comparison left to make.",
  "about.scale.p2":
    "None of that detail is stored anywhere. It is computed on demand, from the same one-line rule, every time you look. There is no underlying image being magnified — there is only arithmetic, and it goes on forever.",

  "about.explore.heading": "Go and look",
  "about.explore.p1":
    "Drag to move, scroll to zoom. The Places tab has a set of destinations worth seeing, including a deep island that sits forty-two decimal places down. Everything you find is in the address bar, so any view can be shared as a link.",
  "about.cta.render": "Open the renderer",
  "about.cta.tech": "How it is computed",
  "about.back": "back to the renderer",

  // --------------------------------------------------------------------- ui
  "ui.title": "Mandelbrot",
  "ui.maxIterations": "Max iterations",
  "ui.auto": "Auto",
  "ui.autoIterationsHint":
    "Choose the iteration count from the zoom level. Moving the slider turns this off.",
  "ui.showOnCanvas": "Show count on canvas",
  "ui.useSlider": "Switch to a slider",
  "ui.useNumber": "Type an exact number",
  "ui.tab.colour": "Colour",
  "ui.tab.light": "Light",
  "ui.tab.advanced": "Advanced",
  "ui.tab.places": "Places",
  "ui.mode.iteration": "Iteration",
  "ui.mode.distance": "Distance",
  "ui.bandWidth": "Band width",
  "ui.shift": "Shift",
  "ui.palettePresets": "Palette presets",
  "ui.bandShaping": "Band shaping",
  "ui.smoothShading": "Smooth shading",
  "ui.mirrorBands": "Mirror bands",
  "ui.iterationMapping": "Iteration mapping",
  "ui.insideColour": "Inside colour",
  "ui.customStops": "Custom stops",
  "ui.colourDensity": "Colour density",
  "ui.colourPhase": "Colour phase",
  "ui.quality": "Quality",
  "ui.supersampling": "Supersampling",
  "ui.gamma": "Gamma",
  "ui.view": "View",
  "ui.zoom": "Zoom",
  "ui.engine": "Engine",
  "ui.timing": "Timing",
  "ui.reset": "Reset",
  "ui.copyLink": "Copy link",
  "ui.copied": "Copied",
  "ui.controls": "Controls",
  "ui.howItWorks": "How it works",
  "ui.whatIsThis": "What is this?",
  "ui.language": "Language",
  "ui.hint.drag":
    "Drag to pan, scroll to zoom. The URL always holds the full state.",
  "ui.hint.depthLimit": "At this backend's depth limit.",
  "ui.hint.gamma":
    "Palette mixing and lighting run in linear light; gamma is applied once at the end.",
  "ui.error.stopped":
    "Rendering stopped. This usually means the graphics driver gave up on a frame that took too long. Lower the iteration count and reload the page.",
  "ui.notice.noWebgpu":
    "This browser is not using WebGPU, so the renderer has fallen back to WebGL2, which runs out of precision at far shallower zooms.",
  "ui.notice.enableFlag":
    "If your browser supports WebGPU but is not using it, open chrome://flags, set Override software rendering list to Enabled, and restart the browser. The page chrome://gpu explains why it was refused.",
  "ui.notice.mobile":
    "On a phone or tablet the renderer works, but deep zooms are far slower than on a desktop GPU, and very deep views may take many seconds a frame. Lowering the iteration count helps.",
  "ui.notice.depthLimit":
    "This is as deep as the current engine goes. The WebGL2 fallback runs out of precision here; a browser with WebGPU enabled goes very much further.",
  "ui.notice.dismiss": "Dismiss",
} as const;

export type StringKey = keyof typeof STRINGS;
