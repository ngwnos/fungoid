// ---- fruit.wgsl: fruit bodies ----
// A fruit body is an organ: a stem (a curved tube from the substrate) and a
// head. The head is one sheet of flesh swept around its axis along a profile
// of PSEG arcs from the crown outward, each turning by its own angle: they
// develop as the head grows (sim.wgsl, growFruits), so a closed button opens
// into a dome, a bell, a flat cap, a funnel or a cup, and the margin rolls
// in, flares out, or splits into rays. Gills, teeth and spore masses are
// carved into or added to that sheet, and its surface
// carries a field that developed while the head expanded (sim.wgsl, develop):
// it bends the sheet into waves and folds, raises bumps or sinks pits in the
// outer face, or tears it into a lattice. The simulation ages the organ and
// rewrites the shape fields every frame; rendering only evaluates them.

const FR_ALIVE: u32 = 1u;
const FR_TIPHEAD: u32 = 2u;        // grew on the end of a filament
const FR_ROT: u32 = 4u;            // dying
// head development: slot in flags bits 16-27 (NO_DEV: none) of a 64 x 64
// field, kept in an atlas of PAT_COLS columns (PAT_SLOTS, PAT_RES: dims.ts)
const NO_DEV: u32 = 0xfffu;
fn devSlot(f: Fruit) -> u32 { return (f.i.z >> 16u) & 0xfffu; }
// A blinking fruit body's brightness, sharp at the start of each blink
// (pa.z: how far through its blink; < 0: it shines steadily).
fn fruitFlash(f: Fruit) -> f32 {
  if (f.pa.z < 0.0) { return 1.0; }
  return 0.04 + 0.96 * pow(0.5 + 0.5 * cos(TAU * f.pa.z), 12.0);
}
// Tissue ids for fruit bodies carry what shading needs from the hit:
// bit 31 fruit, 0-11 fruit index, 12-14 part, 15-22 position along the head
// profile (or the stem), 23-28 angle around the head, 29-30 wart/pit mask.
const FRUIT_BIT: u32 = 0x80000000u;

// parts, for colouring
const RG_TOP: u32 = 0u;     // upper face of the head (inside a cup)
const RG_UNDER: u32 = 1u;   // lower face (the spore-bearing face of a dome)
const RG_GILL: u32 = 2u;
const RG_STEM: u32 = 3u;
const RG_RING: u32 = 4u;
const RG_VOLVA: u32 = 5u;
const RG_GLEBA: u32 = 6u;
const RG_TOOTH: u32 = 7u;

struct Fruit {
  base: vec4f,   // base on the substrate; w: size factor
  norm: vec4f,   // substrate normal; w: step it emerges
  dir: vec4f,    // stem direction (tropisms act on it); w: maturity
  apex: vec4f,   // stem top; w: top radius
  ctrl: vec4f,   // stem curve control point; w: base radius
  head: vec4f,   // start of the head profile; w: head radius R
  axis: vec4f,   // head axis; w: the turning A the head has developed (< 0: a cup; the margin turns further)
  frame: vec4f,  // reference direction across the head; w: elongation
  prof: vec4f,   // cone, flesh thickness, umbo, margin turning
  marg: vec4f,   // feature wavelength, displacement per unit of field (cm), ray count, margin fraction
  feat: vec4f,   // development grid size, tearing, relief (0) ↔ bending (1), bumps (+) ↔ pits (-)
  feat2: vec4f,  // development time so far, gill count, gill depth, teeth length
  ring: vec4f,   // ring position on the stem, ring flare, skirt, bulb
  misc: vec4f,   // spore mass, eggs, volva radius, lateral
  life: vec4f,   // decay, growth (cm per step), margin kept, crown opened
  pa: vec4f,     // profile arc length, largest head radius reached, how far through its blink (< 0: none), how far closed for the day
  pb: vec4f,     // a mycoparasite on it: organism + 1 (0: none), how far it has crusted it over, its colour (8-bit rgb as an integer), unused
  hc: vec4f,     // head bounding cylinder: radius, half height
  hb: vec4f,     // head bounding sphere (its centre is the cylinder's)
  sb: vec4f,     // stem bounding sphere
  bound: vec4f,  // whole bounding sphere (w = 0: not drawn)
  i: vec4u,      // organism, birth step, flags, seed
  seg0: vec4f,   // turning of the profile's arcs as it has developed, crown first
  seg1: vec4f,
}

struct FruitHit {
  d: f32,
  region: u32,
  s: f32,      // along the head profile: 0 centre .. 1 margin
  phi: f32,    // around the head axis
  st: f32,     // along the stem: 0 base .. 1 top
  mark: f32,   // how far the developed surface stands out (packed), for colouring
  w: f32,      // the developed field here (heads; 0 elsewhere)
  exact: bool, // false: a bound, not a distance (never smooth-blend those)
}

// ------------------------------------------------------------ profile ---

// atan2 that holds up on the axis (atan2(y, 0) came back as 0 here).
fn ang(y: f32, x: f32) -> f32 {
  if (abs(x) < 1e-7) { return select(-0.5 * PI, 0.5 * PI, y >= 0.0); }
  return atan2(y, x);
}

// An arc from s0 heading at angle th0, turning clockwise with curvature k
// (anticlockwise if negative) for length len.
fn arcPoint(s0: vec2f, th0: f32, k: f32, len: f32, t: f32) -> vec2f {
  let tg = vec2f(cos(th0), sin(th0));
  if (abs(k) * len < 1e-3) { return s0 + tg * len * t; }
  let rn = vec2f(tg.y, -tg.x);
  let c = s0 + rn / k;
  let a0 = th0 + sign(k) * 0.5 * PI;
  let a = a0 - k * len * t;
  return c + abs(1.0 / k) * vec2f(cos(a), sin(a));
}

// (distance, position along 0..1, signed distance to the right of travel)
fn arcDist(q: vec2f, s0: vec2f, th0: f32, k: f32, len: f32) -> vec3f {
  let tg = vec2f(cos(th0), sin(th0));
  let rn = vec2f(tg.y, -tg.x);
  if (abs(k) * len < 1e-3) {
    let v = q - s0;
    let h = clamp(dot(v, tg) / len, 0.0, 1.0);
    return vec3f(length(v - tg * h * len), h, dot(v, rn));
  }
  let rho = 1.0 / k;
  let c = s0 + rn * rho;
  let v = q - c;
  let a0 = th0 + sign(k) * 0.5 * PI;
  let sw = -k * len;
  let am = a0 + sw * 0.5;
  var dl = ang(v.y, v.x) - am;
  dl -= TAU * round(dl / TAU);
  let lv = length(v);
  let side = (abs(rho) - lv) * sign(k);
  if (abs(dl) <= abs(sw) * 0.5) {
    return vec3f(abs(lv - abs(rho)), 0.5 + dl / sw, side);
  }
  let e = c + abs(rho) * vec2f(cos(a0 + sw), sin(a0 + sw));
  let d0 = length(q - s0);
  let d1 = length(q - e);
  return select(vec3f(d1, 1.0, side), vec3f(d0, 0.0, side), d0 < d1);
}

fn angIn(a: f32, a0: f32, sw: f32) -> bool {
  var d = a - a0;
  if (sw >= 0.0) {
    d -= TAU * floor(d / TAU);
    return d <= sw;
  }
  d -= TAU * ceil(d / TAU);
  return d >= sw;
}

// (min y, max y, max r) of an arc
fn arcBox(s0: vec2f, th0: f32, k: f32, len: f32) -> vec3f {
  let e = arcPoint(s0, th0, k, len, 1.0);
  var b = vec3f(min(s0.y, e.y), max(s0.y, e.y), max(s0.x, e.x));
  if (abs(k) * len >= 1e-3) {
    let tg = vec2f(cos(th0), sin(th0));
    let c = s0 + vec2f(tg.y, -tg.x) / k;
    let a0 = th0 + sign(k) * 0.5 * PI;
    let sw = -k * len;
    let rr = abs(1.0 / k);
    if (angIn(PI * 0.5, a0, sw)) { b.y = max(b.y, c.y + rr); }
    if (angIn(-PI * 0.5, a0, sw)) { b.x = min(b.x, c.y - rr); }
    if (angIn(0.0, a0, sw)) { b.z = max(b.z, c.x + rr); }
  }
  return b;
}

// The profile: PSEG arcs of equal length from the crown (on the axis,
// heading outward) to the margin. Built once per pixel from the developed
// turning of each arc plus the margin's own (curl, rays), scaled to arc
// length S.
const PSEG: u32 = 8u;

struct Profile {
  S: f32,
  seg: array<vec4f, 8>,  // start (xy), heading there, curvature
  end: vec2f,            // the margin
}

fn segTurn(f: Fruit, i: u32) -> f32 {
  var t = 0.0;
  if (i < 4u) { t = f.seg0[i]; } else { t = f.seg1[i - 4u]; }
  // the outer mf of the profile turns further by prof.w
  let mf = f.marg.w;
  if (mf > 0.0) {
    let a = f32(i) / f32(PSEG);
    let b = a + 1.0 / f32(PSEG);
    t += f.prof.w * max(b - max(a, 1.0 - mf), 0.0) / mf;
  }
  return t;
}

fn profileScaled(f: Fruit, S: f32) -> Profile {
  var pr: Profile;
  pr.S = S;
  let L = S / f32(PSEG);
  var q = vec2f(0.0);
  var h = 0.0;
  for (var i = 0u; i < PSEG; i++) {
    let th = segTurn(f, i);
    let k = th / max(L, 1e-6);
    pr.seg[i] = vec4f(q, h, k);
    q = arcPoint(q, h, k, L, 1.0);
    h -= th;
  }
  pr.end = q;
  return pr;
}

fn profileOf(f: Fruit) -> Profile { return profileScaled(f, f.pa.x); }

fn segEnd(pr: ptr<function, Profile>, i: u32) -> vec2f {
  if (i + 1u < PSEG) { return (*pr).seg[i + 1u].xy; }
  return (*pr).end;
}

fn chordDist(q: vec2f, a: vec2f, b: vec2f) -> f32 {
  let ba = b - a;
  let h = clamp(dot(q - a, ba) / max(dot(ba, ba), 1e-12), 0.0, 1.0);
  return length(q - a - ba * h);
}

// (distance, position 0..1 along the whole profile, side (+: right of travel,
// the lower face), position on the margin (its outer mf) or -1). The arc
// with the nearest chord first, then any other whose chord less its sagitta
// could be nearer.
fn profDist(q: vec2f, pr: ptr<function, Profile>, mf: f32) -> vec4f {
  let L = (*pr).S / f32(PSEG);
  var lb: array<f32, 8>;
  var bi = 0u;
  var bl = 1e9;
  for (var i = 0u; i < PSEG; i++) {
    let sg = (*pr).seg[i];
    lb[i] = chordDist(q, sg.xy, segEnd(pr, i)) - abs(sg.w) * L * L * 0.125;
    if (lb[i] < bl) {
      bl = lb[i];
      bi = i;
    }
  }
  let s0 = (*pr).seg[bi];
  var best = arcDist(q, s0.xy, s0.z, s0.w, L);
  var bs = bi;
  for (var i = 0u; i < PSEG; i++) {
    if (i == bi || lb[i] >= best.x) { continue; }
    let sg = (*pr).seg[i];
    let c = arcDist(q, sg.xy, sg.z, sg.w, L);
    if (c.x < best.x) {
      best = c;
      bs = i;
    }
  }
  let s = (f32(bs) + best.y) / f32(PSEG);
  let onMargin = mf > 0.0 && s >= 1.0 - mf;
  return vec4f(best.x, s, best.z, select(-1.0, (s - 1.0 + mf) / max(mf, 1e-4), onMargin));
}

// (min y, max y, max r) of the profile
fn profBox(pr: ptr<function, Profile>) -> vec3f {
  let L = (*pr).S / f32(PSEG);
  var b = vec3f(1e9, -1e9, 0.0);
  for (var i = 0u; i < PSEG; i++) {
    let sg = (*pr).seg[i];
    let m = arcBox(sg.xy, sg.z, sg.w, L);
    b = vec3f(min(b.x, m.x), max(b.y, m.y), max(b.z, m.z));
  }
  return b;
}

// The head's lowest point as drawn (its profile stretched by el and sheared
// into a cone), below its centre: the profile's lowest point and its widest
// are not the same point.
fn profLow(pr: ptr<function, Profile>, el: f32, cone: f32, R: f32) -> f32 {
  let L = (*pr).S / f32(PSEG);
  var lo = 1e9;
  for (var i = 0u; i < PSEG; i++) {
    let sg = (*pr).seg[i];
    for (var j = 0u; j <= 4u; j++) {
      let q = arcPoint(sg.xy, sg.z, sg.w, L, f32(j) / 4.0);
      lo = min(lo, q.y * el - cone * min(q.x, R * 1.2));
    }
  }
  return lo;
}

// Height of the profile where it first reaches radius rr (outward from the
// crown), between arc ends; past its widest, the margin's.
fn profYAt(pr: ptr<function, Profile>, rr: f32) -> f32 {
  for (var i = 0u; i < PSEG; i++) {
    let a = (*pr).seg[i].xy;
    let b = segEnd(pr, i);
    if ((rr - a.x) * (rr - b.x) <= 0.0) { return mix(a.y, b.y, clamp((rr - a.x) / (b.x - a.x + 1e-7), 0.0, 1.0)); }
  }
  return (*pr).end.y;
}

// ----------------------------------------------------------- features ---

// Cellular pattern on the head, periodic around the axis.
// (nearest, second nearest, id)
fn fvor(p: vec2f, nu: f32, seed: u32) -> vec3f {
  let c = floor(p);
  var f1 = 9.0;
  var f2 = 9.0;
  var id = 0u;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let cc = c + vec2f(f32(i), f32(j));
      let cx = cc.x - nu * floor(cc.x / nu);
      let h = hash3(u32(cx), bitcast<u32>(i32(cc.y)), seed);
      let o = vec2f(u01(h), u01(pcg(h))) * 0.8 + 0.1;
      let d = length(cc + o - p);
      if (d < f1) {
        f2 = f1;
        f1 = d;
        id = h;
      } else if (d < f2) {
        f2 = d;
      }
    }
  }
  return vec3f(f1, f2, f32(id & 0xffffu));
}

// IQ's quadratic Bezier distance: (distance, t)
fn sdBezier(pos: vec3f, A: vec3f, B: vec3f, C: vec3f) -> vec2f {
  let a = B - A;
  let b = A - 2.0 * B + C;
  if (dot(b, b) < 1e-8) {
    let ba = C - A;
    let h = clamp(dot(pos - A, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
    return vec2f(length(pos - A - ba * h), h);
  }
  let c = a * 2.0;
  let d = A - pos;
  let kk = 1.0 / dot(b, b);
  let kx = kk * dot(a, b);
  let ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
  let kz = kk * dot(d, a);
  let p = ky - kx * kx;
  let p3 = p * p * p;
  let q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
  var h = q * q + 4.0 * p3;
  var res: vec2f;
  if (h >= 0.0) {
    h = sqrt(h);
    let x = (vec2f(h, -h) - q) * 0.5;
    let uv = sign(x) * pow(abs(x), vec2f(1.0 / 3.0));
    let t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
    let w = d + (c + b * t) * t;
    res = vec2f(dot(w, w), t);
  } else {
    let z = sqrt(-p);
    let v = acos(clamp(q / (p * z * 2.0), -1.0, 1.0)) / 3.0;
    let m = cos(v);
    let n = sin(v) * 1.732050808;
    let t = clamp(vec3f(m + m, -n - m, n - m) * z - kx, vec3f(0.0), vec3f(1.0));
    let wx = d + (c + b * t.x) * t.x;
    let wy = d + (c + b * t.y) * t.y;
    res = select(vec2f(dot(wy, wy), t.y), vec2f(dot(wx, wx), t.x), dot(wx, wx) < dot(wy, wy));
  }
  return vec2f(sqrt(res.x), res.y);
}

fn bezPoint(A: vec3f, B: vec3f, C: vec3f, t: f32) -> vec3f {
  return mix(mix(A, B, t), mix(B, C, t), t);
}

fn sdSegment(p: vec3f, a: vec3f, b: vec3f) -> f32 {
  let ba = b - a;
  let h = clamp(dot(p - a, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
  return length(p - a - ba * h);
}

// ---------------------------------------------------------------- head ---

// Where the spore mass is: 0 a coating on the outer face (or none), 1 a ball
// inside a closed head, 2 eggs in a cup, 3 a ball on top of a star.
fn glebaMode(f: Fruit) -> u32 {
  if (f.misc.x <= 0.0) { return 0u; }
  if (f.axis.w < 0.0) { return select(0u, 2u, f.axis.w < -0.4); }
  // closed heads keep it inside, unless their surface sank into pits (then it
  // lines them) without tearing open
  let pitted = devSlot(f) < PAT_SLOTS && f.feat.w < 0.0 && f.feat.y < 0.3;
  if (f.axis.w >= 1.8 && !pitted) { return 1u; }
  if (f.marg.z > 0.5 && f.misc.x >= 0.5) { return 3u; }
  return 0u;
}

fn seedAngle(f: Fruit, k: u32) -> f32 { return u01(hash2(f.i.w, k)) * TAU; }

fn headEval(f: Fruit, pr: ptr<function, Profile>, p: vec3f, detail: bool) -> FruitHit {
  var o: FruitHit;
  o.region = RG_TOP;
  let R = f.head.w;
  let Y = f.axis.xyz;
  let X = f.frame.xyz;
  let Z = cross(Y, X);
  let lp = p - f.head.xyz;
  let y0 = dot(lp, Y);
  let lx = dot(lp, X);
  let lz = dot(lp, Z);
  let r = length(vec2f(lx, lz));
  let A = f.axis.w;
  let cup = A < 0.0;
  let el = f.frame.w;
  let cone = f.prof.x;
  let th = f.prof.y;
  let rays = f.marg.z;
  // the developed field (sim.wgsl, develop): its wavelength, and how far a
  // unit of it moves the sheet (bend) or the outer face (relief)
  let dev = devSlot(f) < PAT_SLOTS && f.feat2.x > 0.0;
  let lam = f.marg.x;
  let amp = select(0.0, f.marg.y, dev);
  let bend = f.feat.z;
  // the angle around the axis is only needed by features that vary with it
  let angular = (detail && (f.feat2.y > 0.5 || f.feat2.w > 0.0)) || rays > 0.5 || dev || f.life.z < 1.0;
  var phi = 0.0;
  if (angular) { phi = ang(lz, lx); }
  o.phi = phi;

  // warps: conical shear, elongation
  var y = y0;
  y += cone * min(r, R * 1.2);
  y /= el;
  // a bent sheet is steeper than its profile
  let slope = 7.5 * amp / max(lam, 1e-4);
  let lip = min(el, 1.0) / (1.0 + cone) / sqrt(1.0 + slope * slope);

  let pd = profDist(vec2f(r, y), pr, f.marg.w);
  let S = (*pr).S;
  let s = pd.y;
  o.s = s;
  let ht = th * 0.5 * (1.0 - 0.6 * s);

  // away from the sheet, its distance less everything that can stand off
  // that face is enough: the developed field either way, bumps and the umbo
  // above, gills and teeth below, the spore mass
  let lower = pd.z > 0.0;
  let extra = select(f.prof.z * 2.0, f.feat2.z + f.feat2.w, lower) + amp * 1.8;
  let reach = ht + extra + R * 0.05 + f.misc.x * R;
  let far = (pd.x - reach) * lip;
  if (far > 0.02 + 0.05 * R) {
    o.d = far;
    return o;
  }
  o.exact = true;

  // the field bends the sheet (both faces and what hangs from them move
  // together; + bulges outward), or its features stand up from the outer
  // face (bumps, ridges) or sink into it (pits) while the rest stays put
  var wv = 0.0;
  if (dev) { wv = devAt(f, s, phi); }
  let pol = select(-1.0, 1.0, f.feat.w >= 0.0);
  let side = pd.z + wv * amp * bend;  // + : the lower face (right of travel outward)
  let top = max(ht + pol * max(wv * pol, 0.0) * amp * (1.0 - bend), -0.6 * ht);
  o.w = wv;
  o.mark = smoothstep(0.3, 0.9, wv * pol);
  // the upper face is left of travel
  var d = max(-side - top, side - ht);
  if (s > 0.97) { d = max(d, pd.x - ht); }
  // tearing: the sheet gives way where the field sinks furthest, leaving a
  // lattice of what stood up
  let tear = f.feat.y;
  if (dev && tear > 0.0) { d = max(d, (mix(-1.7, 0.45, tear) - wv) * lam / TAU); }

  // rays: split the margin arc into arms
  if (rays > 0.5 && pd.w >= 0.0) {
    var a = phi * rays / TAU + u01(hash2(f.i.w, 3u));
    a = fract(a) - 0.5;
    let half = 0.42 * (1.0 - 0.55 * pd.w);
    d = max(d, (abs(a) - half) * TAU / rays * max(r, 0.03 * R));
  }
  // decay: the margin dissolves or crumbles away unevenly; a crown opens
  let gone = 1.0 - clamp(f.life.z, 0.0, 1.0);
  let sd = f32(f.i.w & 255u);
  let kept = f.life.z - gone * (0.07 * sin(phi * 5.0 + sd) + 0.05 * sin(phi * 11.0 + sd * 1.7));
  d = max(d, (s - kept) * S);
  if (f.life.w > 0.0) { d = max(d, (f.life.w - s) * S); }

  // gills hang from the lower face (the inside of a dome, the outside of a
  // funnel)
  let gills = f.feat2.y;
  if (detail && gills > 0.5 && s > 0.02) {
    let sg = side;
    let sStem = clamp(f.apex.w / max(R, 1e-3), 0.04, 0.6);
    let gd = f.feat2.z * smoothstep(sStem, sStem + 0.12, s) * smoothstep(1.0, 0.82, s);
    let pa = phi * gills / TAU + u01(hash2(f.i.w, 5u));
    let idx = round(pa);
    let plate = abs(r * sin((pa - idx) * TAU / gills)) - max(0.006, R * 0.012);
    var g = max(plate, max(-sg, sg - ht - gd));
    // short gills between the long ones start partway out
    let sMin = select(sStem, max(sStem, 0.45), (i32(idx) & 1) == 1);
    g = max(g, (sMin - s) * S);
    g = max(g, (s - min(0.97, kept)) * S);
    if (g < d) {
      d = g;
      o.region = RG_GILL;
    }
  }
  if (o.region == RG_TOP && side > 0.0) { o.region = RG_UNDER; }

  // teeth: spines hanging straight down from the lower face, on a polar grid
  // over the head so a spine is found however far below the cap it reaches
  let teeth = f.feat2.w;
  if (detail && teeth > 0.0 && !cup && y0 < ht * 2.0) {
    let cs = clamp(R * 0.11, 0.025, 0.14);
    let j = floor(r / cs);
    let rj = (j + 0.5) * cs;
    if (rj < R * 0.97) {
      let nphi = max(3.0, floor(TAU * rj / cs));
      let ph = ang(lz, lx);
      let pk = round(ph / TAU * nphi - 0.5) + 0.5;
      let dphi = ph - pk * TAU / nphi;
      let hk = hash3(u32(j), u32(pk + 64.0), f.i.w);
      let len = teeth * (0.55 + 0.45 * u01(hk)) * smoothstep(0.0, 0.3, rj / R);
      let ay = profYAt(pr, rj) * el - cone * rj - th * 0.5;
      let dr = r - rj;
      let dt = rj * sin(dphi);
      let dy = y0 - ay;
      let rb = cs * 0.38;
      var dtooth: f32;
      if (dy > 0.0) {
        dtooth = length(vec3f(dr, dt, dy)) - rb;
      } else if (-dy > len) {
        dtooth = length(vec3f(dr, dt, dy + len));
      } else {
        dtooth = (length(vec2f(dr, dt)) - rb * pow(1.0 + dy / len, 1.3)) * 0.85;
      }
      if (dtooth < d) {
        d = dtooth;
        o.region = RG_TOOTH;
      }
    }
  }

  d *= lip;

  // umbo: a boss on the crown
  let umbo = f.prof.z;
  if (umbo > 0.0 && !cup) {
    let du = length(vec3f(lx, y0 - (ht - umbo * 0.45), lz)) - umbo;
    d = smin(d, du, umbo * 0.6);
  }

  // spore mass: a ball inside closed heads, eggs in cups, a ball on top of
  // star-like heads
  let gleba = f.misc.x;
  let gm = glebaMode(f);
  if (gm != 0u) {
    var dg = 1e9;
    // the head's mean curvature
    let km = abs(A) / max(S, 1e-4);
    if (gm == 1u) {
      let rho = 1.0 / max(km, 1e-4);
      let cy = -rho * el;
      dg = (length(vec3f(lx, (y0 - cy) / el, lz)) - gleba * 0.82 * rho) * min(el, 1.0);
    } else if (gm == 2u) {
      let n = f.misc.y;
      let er = gleba * select(0.42, 0.2, n > 1.5) * R;
      var ec = vec3f(0.0, ht + er, 0.0);
      if (n > 1.5) {
        let pa = phi * n / TAU;
        let k = round(pa);
        let a = k * TAU / n;
        let re = 0.36 * R;
        ec = vec3f(re * cos(a), ht + er + 0.5 * km * re * re * el, re * sin(a));
      }
      dg = length(vec3f(lx, y0, lz) - ec) - er;
    } else {
      let gr = gleba * 0.5 * R;
      dg = length(vec3f(lx, y0 - ht - gr * 0.85, lz)) - gr;
    }
    if (dg < d) {
      d = dg;
      o.region = RG_GLEBA;
    }
  }

  // shelves: nothing behind the attachment
  if (f.misc.w > 0.5) { d = max(d, -dot(p - f.apex.xyz, X)); }
  o.d = d;
  return o;
}

// ---------------------------------------------------------------- stem ---

fn stemEval(f: Fruit, p: vec3f, detail: bool) -> FruitHit {
  var o: FruitHit;
  o.region = RG_STEM;
  let A = f.base.xyz;
  let B = f.ctrl.xyz;
  let C = f.apex.xyz;
  // cheap bound first: the straight segment, widened by the curve's bow
  let reach = max(f.ctrl.w * (1.0 + f.ring.w), f.apex.w) + f.ring.y + f.misc.z + 0.5 * length(B - (A + C) * 0.5);
  let far = sdSegment(p, A, C) - reach;
  if (far > 0.02) {
    o.d = far;
    return o;
  }
  o.exact = true;
  let bz = sdBezier(p, A, B, C);
  let t = bz.y;
  o.st = t;
  let rb = f.ctrl.w;
  let rt = f.apex.w;
  var rad = mix(rb, rt, t) + f.ring.w * rb * (1.0 - smoothstep(0.0, 0.3, t));
  if (detail) { rad *= 1.0 + 0.03 * sin(t * 7.0 + seedAngle(f, 7u)); }
  // a flat top that meets the underside of the head instead of poking through
  var d = max((bz.x - rad) * 0.9, dot(p - C, normalize(C - B + vec3f(0.0, 1e-5, 0.0))));

  let L = length(C - B) + length(B - A);
  // ring, or a long netted skirt hanging below the head
  let flare = f.ring.y;
  if (flare > 0.0) {
    let skirt = f.ring.z;
    let tr = f.ring.x;
    let sk = 0.03 + 0.75 * skirt;
    let x = clamp((tr - t) / sk, 0.0, 1.0);
    let rr = mix(rb, rt, t) + flare * (1.0 - (1.0 - x) * (1.0 - x));
    let rth = 0.006 + 0.05 * rt;
    var dr = max(abs(bz.x - rr) - rth, max(t - tr, tr - sk - t) * L);
    if (detail && skirt > 0.3) {
      let v = p - bezPoint(A, B, C, t);
      let X = f.frame.xyz;
      let Z = cross(f.axis.xyz, X);
      let ang = ang(dot(v, Z), dot(v, X));
      let cell = max(0.05, (rr + rt) * 0.55);
      let nu = max(4.0, round(TAU * rr / cell));
      let vv = fvor(vec2f(ang / TAU * nu, t * L / cell), nu, f.i.w + 11u);
      let edge = (vv.y - vv.x) * 0.5;
      dr = max(dr, (edge - mix(0.3, 0.1, smoothstep(0.3, 0.9, skirt))) * cell);
    }
    if (dr < d) {
      d = dr;
      o.region = RG_RING;
    }
  }
  // volva: the torn sac the fruit body hatched from
  let vr = f.misc.z;
  if (vr > 0.0) {
    let n = f.norm.xyz;
    let q = p - (A + n * vr * 0.3);
    let yv = dot(q, n);
    let rv = length(q - n * yv);
    let h = vr * (0.35 + 0.08 * sin(ang(dot(q, f.frame.xyz), dot(q, cross(n, f.frame.xyz))) * 5.0 + 1.0));
    let w = sqrt(max(vr * vr - h * h, 0.0));
    var dv: f32;
    if (h * rv < w * yv) { dv = length(vec2f(rv, yv) - vec2f(w, h)); }
    else { dv = abs(length(vec2f(rv, yv)) - vr); }
    dv -= max(0.008, vr * 0.07);
    if (dv < d) {
      d = dv;
      o.region = RG_VOLVA;
    }
  }
  o.d = d;
  return o;
}

// ---------------------------------------------------------------- whole ---

fn fruitEval(f: Fruit, pr: ptr<function, Profile>, p: vec3f, detail: bool) -> FruitHit {
  // far from a part, its bounding sphere is distance enough
  let dh = length(p - f.hb.xyz) - f.hb.w;
  let ds = length(p - f.sb.xyz) - f.sb.w;
  let near = 0.02 + f.head.w * 0.1;
  var h: FruitHit;
  h.d = dh;
  h.region = RG_TOP;
  if (dh < near) { h = headEval(f, pr, p, detail); }
  var st: FruitHit;
  st.d = ds;
  st.region = RG_STEM;
  if (ds < near && f.sb.w > 0.0) { st = stemEval(f, p, detail); }
  let k = max(f.apex.w * 0.7, 0.004);
  var o = h;
  if (st.d < h.d) {
    o.region = st.region;
    o.mark = st.mark;
  }
  o.st = st.st;
  o.d = select(min(h.d, st.d), smin(h.d, st.d, k), h.exact && st.exact);
  return o;
}

fn fruitDist(f: Fruit, pr: ptr<function, Profile>, p: vec3f, detail: bool) -> f32 { return fruitEval(f, pr, p, detail).d; }

// One part alone (0 head, 1 stem), for the raster pass: each part has its
// own quad and the depth test joins them.
fn partEval(f: Fruit, pr: ptr<function, Profile>, p: vec3f, detail: bool, part: u32) -> FruitHit {
  if (part == 0u) {
    let dh = length(p - f.hb.xyz) - f.hb.w;
    if (dh > 0.02 + f.head.w * 0.1) {
      var o: FruitHit;
      o.d = dh;
      return o;
    }
    return headEval(f, pr, p, detail);
  }
  return stemEval(f, p, detail);
}

fn partNormal(f: Fruit, pr: ptr<function, Profile>, p: vec3f, e: f32, detail: bool, part: u32) -> vec3f {
  let k = vec2f(1.0, -1.0);
  return normalize(k.xyy * partEval(f, pr, p + k.xyy * e, detail, part).d + k.yyx * partEval(f, pr, p + k.yyx * e, detail, part).d +
                   k.yxy * partEval(f, pr, p + k.yxy * e, detail, part).d + k.xxx * partEval(f, pr, p + k.xxx * e, detail, part).d);
}

fn isStemPart(rg: u32) -> bool { return rg == RG_STEM || rg == RG_RING || rg == RG_VOLVA; }

fn packFruitHit(idx: u32, h: FruitHit) -> u32 {
  let sv = select(h.s, h.st, isStemPart(h.region));
  let sq = u32(clamp(sv, 0.0, 1.0) * 255.0 + 0.5);
  let pq = u32(fract(h.phi / TAU) * 63.0 + 0.5) & 63u;
  let mq = u32(clamp(h.mark, 0.0, 1.0) * 3.0 + 0.5);
  return FRUIT_BIT | idx | (h.region << 12u) | (sq << 15u) | (pq << 23u) | (mq << 29u);
}

fn unpackFruitHit(id: u32) -> FruitHit {
  var h: FruitHit;
  h.region = (id >> 12u) & 7u;
  let sv = f32((id >> 15u) & 255u) / 255.0;
  h.s = sv;
  h.st = sv;
  h.phi = f32((id >> 23u) & 63u) / 63.0 * TAU;
  h.mark = f32((id >> 29u) & 3u) / 3.0;
  h.exact = true;
  return h;
}

fn fruitNormal(f: Fruit, pr: ptr<function, Profile>, p: vec3f, e: f32, detail: bool) -> vec3f {
  let k = vec2f(1.0, -1.0);
  return normalize(k.xyy * fruitDist(f, pr, p + k.xyy * e, detail) + k.yyx * fruitDist(f, pr, p + k.yyx * e, detail) +
                   k.yxy * fruitDist(f, pr, p + k.yxy * e, detail) + k.xxx * fruitDist(f, pr, p + k.xxx * e, detail));
}
