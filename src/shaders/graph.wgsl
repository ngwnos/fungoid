// The timeline's graph (gpu/graph.ts): the garden's measures over its
// history, overlapping. `bin` gathers the samples (history.ts Samples) into
// one column a pixel: each measure's least, most and mean there, how damp it
// was (from the weather, world.wgsl), how dark, the checkpoints and years in
// it. `fs` draws from the columns: nights, the seasons along the foot,
// checkpoints and years, the measures (filled or as lines, each scaled to
// the most it has been, the scale following it gradually: `ease`; with the
// spread of the samples in a column behind),
// the playhead and the pointer, and below it all the bar the playhead runs
// along.

struct Graph {
  // the canvas and the chart above the bar, in pixels
  w: f32,
  h: f32,
  chart: f32,
  dpr: f32,
  // steps a pixel column, and the columns (from `pad` in)
  perPx: f32,
  cols: u32,
  pad: f32,
  count: u32,
  head: u32,
  first: u32,
  // the playhead and the pointer (x in pixels; < 0: none)
  at: f32,
  hover: f32,
  cps: u32,
  // measures not shown (bits), the one brought forward (-1: none)
  hidden: u32,
  focus: i32,
  // seconds since the last drawing (< 0: the scales go straight to theirs)
  dt: f32,
  // where on the planet the nights, the damp and the seasons are of
  look: vec4f,
  rgb: array<vec4f, 6>,
}

struct Row {
  step: u32,
  v: array<f32, 7>,
}

@group(0) @binding(0) var<uniform> g: Graph;
@group(0) @binding(1) var<storage, read> rows: array<Row>;
@group(0) @binding(2) var<storage, read> cps: array<u32>;
@group(0) @binding(3) var<storage, read_write> binsOut: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> peaksOut: array<atomic<u32>, 8>;
@group(0) @binding(5) var<storage, read> bins: array<vec4f>;
// each measure's scale as drawn (its log, and how fast that changes)
@group(0) @binding(7) var<storage, read_write> scalesOut: array<vec2f, 8>;
@group(0) @binding(8) var<storage, read> scales: array<vec2f, 8>;

// the measures: mycelium, mushrooms, damp, food stored, colonies, slime
// mould; and a column's own (night, checkpoints, year, a year begun)
const NS = 6u;
const DAMP = 2u;
const COLONIES = 4u;
const OWN = 6u;
const COL = 8u;
// (a checkpoint just after something was done by hand)
const BY_HAND = 0x80000000u;

// the field of a sample each measure is (damp is from the weather)
fn field(s: u32) -> u32 {
  switch s {
    case 0u: { return 1u; }
    case 1u: { return 2u; }
    case 3u: { return 3u; }
    case 4u: { return 4u; }
    default: { return 5u; }
  }
}

// the first sample at or after step s
fn rowFrom(s: f32) -> u32 {
  var lo = 0u;
  var hi = g.count;
  while (lo < hi) {
    let m = (lo + hi) / 2u;
    if (f32(rows[m].step) < s) { lo = m + 1u; } else { hi = m; }
  }
  return lo;
}

fn cpFrom(s: f32) -> u32 {
  var lo = 0u;
  var hi = g.cps;
  while (lo < hi) {
    let m = (lo + hi) / 2u;
    if (f32(cps[m] & ~BY_HAND) < s) { lo = m + 1u; } else { hi = m; }
  }
  return lo;
}

@compute @workgroup_size(64)
fn bin(@builtin(global_invocation_id) id: vec3u) {
  let c = id.x;
  if (c >= g.cols) { return; }
  let s0 = f32(c) * g.perPx;
  let s1 = s0 + g.perPx;
  let mid = s0 + 0.5 * g.perPx;
  let o = c * COL;
  var lo: array<f32, 6>;
  var hi: array<f32, 6>;
  var sum: array<f32, 6>;
  var n = 0.0;
  let i0 = rowFrom(s0);
  let i1 = rowFrom(s1);
  if (g.count > 0u && i1 > i0) {
    for (var s = 0u; s < NS; s++) {
      lo[s] = 3.4e38;
      hi[s] = -3.4e38;
    }
    // (no more than a thousand or so samples a column)
    let stride = max(1u, (i1 - i0) / 1024u);
    for (var i = i0; i < i1; i += stride) {
      for (var s = 0u; s < NS; s++) {
        if (s == DAMP) { continue; }
        let v = rows[i].v[field(s)];
        lo[s] = min(lo[s], v);
        hi[s] = max(hi[s], v);
        sum[s] += v;
      }
      n += 1.0;
    }
    for (var s = 0u; s < NS; s++) { sum[s] /= n; }
  } else if (g.count > 0u && i0 > 0u) {
    // between two samples, along the line from one to the next (past the
    // last, as it was)
    let a = rows[i0 - 1u];
    let b = rows[min(i0, g.count - 1u)];
    let t = select(0.0, clamp((mid - f32(a.step)) / max(1.0, f32(b.step) - f32(a.step)), 0.0, 1.0), i0 < g.count);
    for (var s = 0u; s < NS; s++) {
      if (s == DAMP) { continue; }
      let v = mix(a.v[field(s)], b.v[field(s)], t);
      lo[s] = v;
      hi[s] = v;
      sum[s] = v;
    }
    n = 1.0;
  }
  for (var s = 0u; s < NS; s++) {
    if (s == DAMP) { continue; }
    binsOut[o + s] = vec4f(lo[s], hi[s], sum[s], select(0.0, 1.0, n > 0.0));
    if (n > 0.0) { atomicMax(&peaksOut[s], bitcast<u32>(max(hi[s], 0.0))); }
  }
  // how damp, and how dark, across the column
  var dl = 1.0;
  var dh = 0.0;
  var dm = 0.0;
  var night = 0.0;
  for (var k = 0u; k < 16u; k++) {
    let st = u32(s0 + (f32(k) + 0.5) / 16.0 * g.perPx);
    let w = climate(g.look.xyz, st);
    dl = min(dl, w);
    dh = max(dh, w);
    dm += w / 16.0;
    night += nightness(g.look.xyz, st) / 16.0;
  }
  binsOut[o + DAMP] = vec4f(dl, dh, dm, 1.0);
  // the checkpoints in it: 1 the garden's own, 2 one after something done
  // by hand
  var cp = 0.0;
  let j0 = cpFrom(s0);
  for (var j = j0; j < g.cps && j < j0 + 64u; j++) {
    if (f32(cps[j] & ~BY_HAND) >= s1) { break; }
    cp = max(cp, select(1.0, 2.0, (cps[j] & BY_HAND) != 0u));
  }
  let year = select(0.0, 1.0, s0 > 0.0 && floor(s1 / f32(YEAR)) > floor(s0 / f32(YEAR)));
  // (the seasons the other way about in the south)
  let season = fract(yearPhase(u32(mid)) + select(0.0, 0.5, g.look.y < 0.0));
  binsOut[o + OWN] = vec4f(night, cp, season, year);
}

// Each measure's scale follows the most it has been (with HEADROOM above),
// gradually: critically damped towards it (EASE, per second), no faster than
// SLEW (in its log, per second), and faster up (where what is drawn would be
// cut off at the top) than down. Until there is any of a measure, its scale
// goes straight to the first there is.
const EASE = 10.0;
const SLEW_UP = 5.0;
const SLEW_DOWN = 1.5;
const HEADROOM = 1.1;

@compute @workgroup_size(8)
fn ease(@builtin(local_invocation_index) s: u32) {
  if (s >= NS) { return; }
  let goal = log(max(bitcast<f32>(atomicLoad(&peaksOut[s])) * HEADROOM, 1.0));
  let st = scalesOut[s];
  if (g.dt < 0.0 || st.x <= 0.0) {
    scalesOut[s] = vec2f(goal, 0.0);
    return;
  }
  let dt = min(g.dt, 0.1);
  let a = EASE * EASE * (goal - st.x) - 2.0 * EASE * st.y;
  let v = clamp(st.y + a * dt, -SLEW_DOWN, SLEW_UP);
  scalesOut[s] = vec2f(st.x + v * dt, v);
}

// ------------------------------------------------------------------ draw ---

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

// premultiplied: `rgb` at `a` over what is there
fn over(d: vec4f, rgb: vec3f, a: f32) -> vec4f {
  let k = clamp(a, 0.0, 1.0);
  return vec4f(rgb * k + d.rgb * (1.0 - k), k + d.a * (1.0 - k));
}

fn seg(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let ab = b - a;
  let t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
  return length(p - a - ab * t);
}

const CREAM = vec3f(1.0, 0.957, 0.894);
const ACCENT = vec3f(0.910, 0.710, 0.416);
const BLUE = vec3f(0.561, 0.718, 0.910);
const INK = vec3f(0.071, 0.063, 0.055);
const NIGHT = vec3f(0.02, 0.035, 0.11);

// the seasons around the year, each at its middle (autumn first)
fn seasonRGB(t: f32) -> vec3f {
  let x = fract(t - 0.125) * 4.0;
  let k = u32(x) % 4u;
  let f = smoothstep(0.0, 1.0, fract(x));
  var c = array<vec3f, 4>(
    vec3f(0.588, 0.384, 0.204),
    vec3f(0.361, 0.424, 0.510),
    vec3f(0.408, 0.573, 0.282),
    vec3f(0.745, 0.627, 0.306),
  );
  return mix(c[k], c[(k + 1u) % 4u], f);
}

fn peak(s: u32) -> f32 {
  return select(exp(scales[s].x), 1.0, s == DAMP);
}

fn colAt(c: i32, s: u32) -> vec4f {
  if (c < 0 || c >= i32(g.cols)) { return vec4f(0.0); }
  return bins[u32(c) * COL + s];
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let d = g.dpr;
  let p = pos.xy;
  let lx = p.x - g.pad;
  let c = i32(floor(lx));
  let inside = lx >= 0.0 && c < i32(g.cols);
  let top = 7.0 * d;
  let strip = 3.0 * d;
  let base = g.chart - strip - 1.0 * d;
  var o = vec4f(0.0);
  if (p.y < g.chart) {
    if (inside) {
      let own = colAt(c, OWN);
      // nights, where a day is a few pixels wide or more
      let fade = smoothstep(3.0, 10.0, f32(SOL) / g.perPx / d);
      o = over(o, NIGHT, 0.6 * own.x * fade);
      if (p.y >= g.chart - strip) { o = over(o, seasonRGB(own.z), 0.9); }
      else {
        if (own.w > 0.5) { o = over(o, CREAM, 0.13); }
        if (own.y > 1.5) { o = over(o, BLUE, 0.5); }
        else if (own.y > 0.5) { o = over(o, CREAM, 0.06); }
      }
    }
    // the measures, back to front: damp, the lines, then mycelium and
    // mushrooms
    var order = array<u32, 6>(2u, 3u, 4u, 5u, 0u, 1u);
    for (var k = 0u; k < NS; k++) {
      let s = order[k];
      if (((g.hidden >> s) & 1u) != 0u || !inside) { continue; }
      let pk = peak(s);
      let span = base - top;
      let b0 = colAt(c - 1, s);
      let b1 = colAt(c, s);
      let b2 = colAt(c + 1, s);
      if (b1.w == 0.0) { continue; }
      let y0 = base - clamp(b0.z / pk, 0.0, 1.0) * span;
      let y1 = base - clamp(b1.z / pk, 0.0, 1.0) * span;
      let y2 = base - clamp(b2.z / pk, 0.0, 1.0) * span;
      let cx = f32(c) + 0.5;
      let p0 = vec2f(cx - 1.0, y0);
      let p1 = vec2f(cx, y1);
      let p2 = vec2f(cx + 1.0, y2);
      let q = vec2f(lx, p.y);
      var dist = 1e9;
      if (b0.w > 0.0) { dist = min(dist, seg(q, p0, p1)); }
      if (b2.w > 0.0) { dist = min(dist, seg(q, p1, p2)); }
      if (b0.w == 0.0 && b2.w == 0.0) { dist = abs(p.y - y1); }
      let dim = select(1.0, 0.16, g.focus >= 0 && u32(g.focus) != s) * select(1.0, 0.45, s == DAMP && g.focus != i32(s));
      let rgb = g.rgb[s].rgb;
      let filled = s <= DAMP;
      // the spread of the samples in the column
      let yHi = base - clamp(b1.y / pk, 0.0, 1.0) * span;
      let yLo = base - clamp(b1.x / pk, 0.0, 1.0) * span;
      if (yLo - yHi > 1.5 * d && p.y >= yHi && p.y <= yLo) { o = over(o, rgb, 0.13 * dim); }
      if (filled) {
        // under the line, brightest at it
        let yl = select(mix(y1, y2, lx - cx), mix(y0, y1, lx - cx + 1.0), lx < cx);
        let cover = clamp(p.y - yl + 0.5, 0.0, 1.0) * step(p.y, base);
        let t = clamp((p.y - yl) / (span * 0.75), 0.0, 1.0);
        o = over(o, rgb, cover * mix(0.22, 0.02, sqrt(t)) * dim);
      }
      // the line, with a soft glow (colonies dashed)
      let w = select(1.35, 1.0, s == DAMP) * d;
      var a = 1.0 - smoothstep(w * 0.5 - 0.5, w * 0.5 + 0.5, dist);
      var glow = 0.2 * exp(-dist / (2.2 * d));
      if (s == COLONIES) {
        let on = step(fract(lx / (7.0 * d)), 0.58);
        a *= on;
        glow *= on;
      }
      o = over(o, rgb, glow * dim);
      o = over(o, rgb, a * select(0.95, 0.7, s == DAMP) * dim);
    }
    // the pointer, and the playhead
    if (g.hover >= 0.0) { o = over(o, CREAM, 0.3 * (1.0 - smoothstep(0.0, 1.0 * d, abs(p.x - g.hover)))); }
    let ad = abs(p.x - g.at);
    o = over(o, ACCENT, 0.25 * exp(-ad / (2.5 * d)));
    o = over(o, ACCENT, 1.0 - smoothstep(0.5 * d, 0.5 * d + 1.0, ad));
    // where they read each measure, as dots
    for (var e = 0u; e < 2u; e++) {
      let x = select(g.at, g.hover, e == 1u);
      if (x < 0.0 || abs(p.x - x) > 8.0 * d) { continue; }
      let cc = clamp(i32(floor(x - g.pad)), 0, i32(g.cols) - 1);
      for (var s = 0u; s < NS; s++) {
        if (((g.hidden >> s) & 1u) != 0u) { continue; }
        let b = colAt(cc, s);
        if (b.w == 0.0) { continue; }
        let y = base - clamp(b.z / peak(s), 0.0, 1.0) * (base - top);
        let r = length(p - vec2f(x, y));
        let dim = select(1.0, 0.2, g.focus >= 0 && u32(g.focus) != s);
        o = over(o, INK, (1.0 - smoothstep(4.0 * d - 0.5, 4.0 * d + 0.5, r)) * 0.85 * dim);
        o = over(o, g.rgb[s].rgb, (1.0 - smoothstep(2.6 * d - 0.5, 2.6 * d + 0.5, r)) * dim);
      }
    }
  } else {
    // the bar: played up to the playhead
    let ty = g.chart + (g.h - g.chart) * 0.5;
    let bx = clamp(p.x, g.pad, g.pad + f32(g.cols));
    let bd = length(vec2f(p.x - bx, p.y - ty)) - 1.5 * d;
    let bar = 1.0 - smoothstep(-0.5, 0.5, bd);
    o = over(o, CREAM, 0.1 * bar);
    if (p.x < g.at) { o = over(o, ACCENT, 0.5 * bar); }
    let r = length(p - vec2f(g.at, ty));
    o = over(o, ACCENT, 0.22 * (1.0 - smoothstep(8.0 * d - 0.5, 8.0 * d + 0.5, r)));
    o = over(o, ACCENT, 1.0 - smoothstep(5.0 * d - 0.5, 5.0 * d + 0.5, r));
  }
  // what has gone from the store, hatched over
  if (inside && f32(c) * g.perPx < f32(g.first)) {
    let hatch = step(fract((p.x + p.y) / (6.0 * d)), 0.5);
    o = over(o, INK, mix(0.3, 0.6, hatch));
  }
  return o;
}
