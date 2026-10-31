// ---- shade.wgsl: deterministic renderer ----
// One march per pixel through the baked world volume, within the sphere the
// planet's air reaches (SKY_R); space beyond. Fungal tissue comes from the
// raster pass (tissue.wgsl); light comes from volumes baked once (sun
// visibility, sky openness, one bounce) times a per-frame tissue light field.
// No random rays, so there is no noise to denoise: temporal AA only. The
// ground's patterns are of where on the planet a point is (its direction at
// the ground's radius), so they run straight up and have no seams.

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> scene: Scene;
@group(0) @binding(2) var<storage, read> pheno: array<f32>;
@group(0) @binding(3) var<storage, read> cells: array<Cell>;
@group(0) @binding(4) var vol: texture_3d<f32>;
@group(0) @binding(5) var aux: texture_3d<f32>;
@group(0) @binding(6) var ground: texture_2d<f32>;
@group(0) @binding(7) var tissueGeo: texture_2d<f32>;
@group(0) @binding(8) var linSamp: sampler;
@group(0) @binding(9) var outRad: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var outGeo: texture_storage_2d<rgba32float, write>;
@group(0) @binding(11) var outAlb: texture_storage_2d<rgba16float, write>;
@group(0) @binding(12) var<storage, read_write> ids: array<u32>;
@group(0) @binding(13) var tissueId: texture_2d<u32>;
@group(0) @binding(14) var<storage, read> dens4: array<u32>;   // the sim's grid; cell counts first
@group(0) @binding(15) var light: texture_3d<f32>;
@group(0) @binding(16) var bounce: texture_3d<f32>;
@group(0) @binding(17) var tissueLight: texture_3d<f32>;
@group(0) @binding(18) var<storage, read> counters: array<u32>;
@group(0) @binding(19) var<storage, read> fruits: array<Fruit>;
@group(0) @binding(20) var fieldTex: texture_3d<f32>;  // airborne spores, settled spores, food, how damp
@group(0) @binding(21) var devTex: texture_2d<f32>;   // developed heads (fruitdev.wgsl)
@group(0) @binding(22) var devSamp: sampler;
@group(0) @binding(23) var slimeTex: texture_2d<f32>;  // slime mould: plasmodium, slime left behind, fruiting
@group(0) @binding(24) var glowTex: texture_3d<f32>;   // light cast by what glows (glow.wgsl)
// the sun's light for two places of the sun (bake_sun.wgsl bakeSunVis): now
// is between them, scene.day.x of the way to the second
@group(0) @binding(25) var sunVis0: texture_3d<f32>;
@group(0) @binding(26) var sunVis1: texture_3d<f32>;

// below this fraction of its height a moss cushion is effectively solid
const MOSS_CORE: f32 = 0.5;
const SIG_MOSS: f32 = 30.0;
// the planet's air: thin, thinning with height (HAZE per cm at the ground,
// fading over HAZE_H), a little blue
const HAZE: f32 = 0.012;
const HAZE_H: f32 = 1.6;
const SPORE_DUST: f32 = 0.6;   // scattering per unit of airborne spores, per cm
// what glows: faint by day, bright in the dark (the eye opens up at night);
// the light it casts, per unit of the glow light field
const GLOW_EMIT: f32 = 0.1;
const GLOW_LIGHT: f32 = 2.0;

// ------------------------------------------------------------- sampling ---

fn volAt(p: vec3f) -> vec4f { return textureSampleLevel(vol, linSamp, volUVW(p), 0.0); }
fn auxAt(p: vec3f) -> vec4f { return textureSampleLevel(aux, linSamp, volUVW(p), 0.0); }

// Baked layer thickness; very thin layers are treated as bare surface.
fn mossThick(a: vec4f) -> f32 { return select(0.0, a.w, a.w > 0.02); }

// Upper bound of the density for a layer of this thickness.
fn mossMajorant(thick: f32) -> f32 { return SIG_MOSS * clamp(thick * 2.2, 0.15, 1.0); }

// where on the ground p is: its direction, at the ground's radius
fn onGround(p: vec3f) -> vec3f { return upAt(p) * PLANET_R; }
// noise along the ground at `a` a cm, and slower, `b`, along up
fn upNoise(p: vec3f, a: f32, b: f32, seed: u32) -> f32 {
  let r = length(p) * b;
  return gnoise3(onGround(p) * a + vec3f(r, r * 1.31, r * 0.77), seed);
}

fn mossDensity(p: vec3f, h01: f32, thick: f32, cheap: bool) -> f32 {
  let moss = thick * 2.2;
  let hh = clamp((h01 - MOSS_CORE) / (1.0 - MOSS_CORE), 0.0, 1.0);
  let prof = 1.0 - hh * hh;
  var s = 0.55;
  if (!cheap) {
    let cell = cellular3(onGround(p) / 0.12, 51u, 0.9);
    let shoot = 1.0 - smoothstep(0.1, 0.38 + 0.25 * (1.0 - h01), cell.f1);
    let leaf = 0.5 + 0.5 * upNoise(p, 20.0, 7.0, 52u);
    s = 0.12 + 0.88 * max(shoot, leaf * 0.65);
  }
  return SIG_MOSS * clamp(moss * 1.4, 0.0, 1.0) * prof * s;
}

// A few kinds of moss in lobed patches that mingle where they meet: bright
// cushion moss, dark feather moss, pale silvery-green cushions and a brown
// one. Each has a smooth field of its own, and where one rises above the
// others its moss takes over, softly, so no patch has a hard edge.
fn mossKinds(g: vec3f) -> vec4f {
  let q = g + vec3f(fbm3(g * 0.22, 63u, 2u), fbm3(g * 0.22 + 7.0, 64u, 2u), fbm3(g * 0.22 + 13.0, 65u, 2u)) * 3.0;
  var s = vec4f(fbm3(q * 0.28, 70u, 2u) + 0.12, fbm3(q * 0.28, 71u, 2u), fbm3(q * 0.28, 72u, 2u) - 0.04, fbm3(q * 0.28, 73u, 2u) - 0.1);
  s += vec4f(gnoise3(g * 3.0, 74u), gnoise3(g * 3.0, 75u), gnoise3(g * 3.0, 76u), gnoise3(g * 3.0, 77u)) * 0.05;
  let e = exp((s - max(max(s.x, s.y), max(s.z, s.w))) * 40.0);
  return e / (e.x + e.y + e.z + e.w);
}

// What colours moss at p: its kinds' shares (mossKinds), and the patterns
// over it, all of where on the ground it is (across a cushion they hardly
// change: worked out once where a ray enters it)
struct MossLook {
  w: vec4f,
  v: f32,      // the tips' brightness
  caps: bool,  // haircap capsules on the tallest growth
  rust: f32,
  dry: f32,    // browned in the dry of summer, or where the ground was dried
}

fn mossLook(p: vec3f) -> MossLook {
  let g = onGround(p);
  var m: MossLook;
  m.w = mossKinds(g);
  m.v = fbm3(g * 0.6, 61u, 2u);
  m.caps = m.w.x > 0.7 && cellular3(g / 0.5, 66u, 0.9).f1 < 0.12 && fbm3(g * 0.3, 67u, 2u) > 0.15;
  m.rust = smoothstep(0.35, 0.65, fbm3(g * 0.22, 62u, 3u));
  m.dry = smoothstep(0.55, 0.0, fieldsAt(p).w);
  return m;
}

fn mossAlbedo(p: vec3f, h01: f32) -> vec3f { return mossColour(mossLook(p), h01); }

// (h01: how far up the cushion)
fn mossColour(m: MossLook, h01: f32) -> vec3f {
  let w = m.w;
  var tip = w.x * vec3f(0.36, 0.56, 0.09) + w.y * vec3f(0.2, 0.36, 0.07) + w.z * vec3f(0.56, 0.66, 0.24) + w.w * vec3f(0.44, 0.4, 0.1);
  let deep = w.x * vec3f(0.12, 0.2, 0.05) + w.y * vec3f(0.06, 0.12, 0.03) + w.z * vec3f(0.2, 0.3, 0.08) + w.w * vec3f(0.18, 0.15, 0.06);
  tip *= 0.85 + 0.3 * smoothstep(-0.4, 0.4, m.v);
  var c = mix(deep, tip, smoothstep(0.15, 0.85, h01));
  if (m.caps && h01 > 0.75) { c = vec3f(0.55, 0.2, 0.07); }
  c = mix(c, vec3f(0.48, 0.32, 0.12), m.rust * 0.45);
  let grey = dot(c, vec3f(0.3, 0.55, 0.15));
  return mix(c, grey * vec3f(1.35, 1.1, 0.55), m.dry * 0.75);
}

// ---------------------------------------------------------------- tissue ---
// Primary visibility of fungal tissue comes from the raster pass
// (tissue.wgsl). Secondary rays only see it as a soft absorber.

// Trilinear 4 mm cell density (the nearest-voxel value shows as tiles).
fn densSmooth(p: vec3f) -> f32 {
  let g = (p - BOX_MIN) / F4 - 0.5;
  let c = vec3i(floor(g));
  let f = g - floor(g);
  var acc = 0.0;
  for (var k = 0; k < 8; k++) {
    let o = vec3i(k & 1, (k >> 1) & 1, k >> 2);
    let q = c + o;
    let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1) * select(1.0 - f.z, f.z, o.z == 1);
    acc += select(0.0, f32(dens4[f4Index(q)]), inF4(q)) * w;
  }
  return acc;
}

// Where to read the 4 mm fields for a ground point: pushed about along the
// surface, so patches of eaten soil and mycelium have wandering edges, not
// the voxels' straight ones.
fn fieldWarp(p: vec3f, n: vec3f) -> vec3f {
  let q = p * 0.9;
  var w = vec3f(gnoise3(q, 141u), gnoise3(q, 142u), gnoise3(q, 143u)) * 0.3;
  let r = q * 2.7;
  w += vec3f(gnoise3(r, 144u), gnoise3(r, 145u), gnoise3(r, 146u)) * 0.12;
  return p + w - n * dot(w, n);
}

fn tissueDensity(p: vec3f) -> f32 {
  let c = f4Coord(p);
  return select(0.0, f32(dens4[f4Index(c)]), inF4(c));
}

// ------------------------------------------------------------- lighting ---

// sun (x) and sky (y) visibility
fn lightAt(p: vec3f) -> vec4f {
  let uvw = volUVW(p);
  let a = textureSampleLevel(sunVis0, linSamp, uvw, 0.0).x;
  let b = textureSampleLevel(sunVis1, linSamp, uvw, 0.0).x;
  return vec4f(mix(a, b, scene.day.x), textureSampleLevel(light, linSamp, uvw, 0.0).y, 0.0, 0.0);
}
// (baked as if the sun stood overhead: as much as the day here is)
fn bounceAt(p: vec3f) -> vec3f {
  let el = dot(upAt(p), scene.sunDir.xyz);
  return textureSampleLevel(bounce, linSamp, volUVW(p), 0.0).rgb * dayHere(p) * (0.3 + 0.7 * max(el, 0.0));
}
// The sun's light at p: reddening as it sinks, gone below the horizon.
fn sunAt(p: vec3f) -> vec3f {
  let el = dot(upAt(p), scene.sunDir.xyz);
  let k = vec3f(0.05, 0.12, 0.28);
  return scene.sunCol.rgb * exp(k - k / max(el + 0.08, 0.04)) * smoothstep(-0.1, 0.02, el);
}
// (light from within: its own colour, a little darker where the flesh is)
fn glowTone(alb: vec3f) -> f32 { return 0.35 + 0.65 * sqrt(dot(alb, vec3f(0.3, 0.55, 0.15))); }
fn glowAt(p: vec3f) -> vec3f { return textureSampleLevel(glowTex, linSamp, glowUVW(p), 0.0).rgb * GLOW_LIGHT; }
fn tissueLightAt(p: vec3f) -> vec2f { return textureSampleLevel(tissueLight, linSamp, volUVW(p), 0.0).rg; }
fn fieldsAt(p: vec3f) -> vec4f { return textureSampleLevel(fieldTex, linSamp, volUVW(p), 0.0); }

// Colour of the spores at p: of the organisms that shed them most in the
// voxels around, blended as their amounts are (so prints have no square
// edges where two kinds meet).
fn sporeColour(p: vec3f) -> vec3f {
  let g = (p - BOX_MIN) / F4 - 0.5;
  let c = vec3i(floor(g));
  let f = g - floor(g);
  var acc = vec3f(0.0);
  for (var k = 0; k < 8; k++) {
    let o = vec3i(k & 1, (k >> 1) & 1, k >> 2);
    let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1) * select(1.0 - f.z, f.z, o.z == 1);
    let id = u32(textureLoad(tissueLight, clamp(c + o, vec3i(0), vec3i(F4_RES) - 1), 0).b + 0.5);
    var col = vec3f(0.55, 0.45, 0.35);
    if (id > 0u && id <= MAX_ORGS) { col = vec3f(PG(id - 1u, G_SPORE_R), PG(id - 1u, G_SPORE_G), PG(id - 1u, G_SPORE_B)); }
    acc += col * w;
  }
  return acc;
}

// Drifting spores: a march through the 3.2 cm blocks that hold any, so clean
// air costs nothing. (scattered light, transmittance)
fn sporeDust(ro: vec3f, rd: vec3f, ta: f32, tb: f32, jit: f32) -> vec4f {
  let bs = F4 * 8.0;
  let L = scene.sunDir.xyz;
  let phase = hgPhase(dot(rd, L), 0.6);
  let inv = 1.0 / select(rd, vec3f(1e-6), abs(rd) < vec3f(1e-6));
  let stp = vec3i(sign(rd));
  let p0 = ro + rd * ta;
  var cell = vec3i(floor((p0 - BOX_MIN) / bs));
  var tMax = ta + (BOX_MIN + (vec3f(cell) + max(sign(rd), vec3f(0.0))) * bs - p0) * inv;
  let tDelta = abs(bs * inv);
  var col = vec3f(0.0);
  var T = 1.0;
  var t = ta;
  var samples = 0;
  for (var k = 0; k < 32; k++) {
    if (t >= tb || samples >= 64 || T < 0.02) { break; }
    let te = min(min(min(tMax.x, tMax.y), tMax.z), tb);
    if (all(cell >= vec3i(0)) && all(cell < BLOCKS) && dens4[GRID_BLOCKS + blockIndex(cell)] > 3u) {
      let tint = mix(sporeColour(ro + rd * (t + te) * 0.5), vec3f(1.0), 0.3);
      let dt = 0.12;
      var tt = t + dt * jit;
      loop {
        if (tt >= te || samples >= 64) { break; }
        let ps = ro + rd * tt;
        let s = fieldsAt(ps).x;
        if (s > 0.002) {
          let sigma = SPORE_DUST * s;
          let lit = lightAt(ps).x * tissueLightAt(ps).x;
          col += T * sigma * dt * tint * (lit * phase * sunAt(ps) + skyIrradiance(upAt(ps), ps) * 0.3 + glowAt(ps));
          T *= exp(-sigma * dt);
        }
        tt += dt;
        samples++;
      }
    }
    t = te;
    if (tMax.x < tMax.y && tMax.x < tMax.z) {
      cell.x += stp.x;
      tMax.x += tDelta.x;
    } else if (tMax.y < tMax.z) {
      cell.y += stp.y;
      tMax.y += tDelta.y;
    } else {
      cell.z += stp.z;
      tMax.z += tDelta.z;
    }
  }
  return vec4f(col, T);
}

// What the living garden does to the ground: spore prints where spores
// settle, dark zone lines where colonies fought, wood bleached where it has
// been eaten.
fn groundLife(m: Mat, p: vec3f, wood: bool) -> Mat {
  var o = m;
  let below = fieldsAt(p - o.n * 0.15);
  if (wood) {
    let rot = clamp(1.0 - below.z / 3.0, 0.0, 1.0);
    o.alb = mix(o.alb, vec3f(0.62, 0.56, 0.46) * (0.85 + 0.3 * gnoise3(p * 3.0, 91u)), rot * 0.55);
    o.rough = mix(o.rough, 1.0, rot * 0.5);
  }
  let scar = textureSampleLevel(tissueLight, linSamp, volUVW(p - o.n * 0.15), 0.0).a;
  o.alb *= 1.0 - 0.65 * smoothstep(0.05, 0.6, scar) * (0.7 + 0.3 * gnoise3(p * 9.0, 92u));
  let bed = fieldsAt(p + o.n * 0.12).y;
  o.alb = mix(o.alb, sporeColour(p + o.n * 0.12), clamp(bed * 0.5, 0.0, 0.45));
  return o;
}

// A slime mould over the top of the ground: a glossy, translucent yellow
// plasmodium, thick in its veins (it dulls as it turns to fruit), and the
// glaze of slime it leaves where it has been.
fn slimeAt(p: vec3f) -> vec4f { return textureSampleLevel(slimeTex, linSamp, slimeUV(p), 0.0); }
// how far from the centre the ground is, in p's direction
fn groundR(p: vec3f) -> f32 { return PLANET_R + textureSampleLevel(ground, linSamp, groundUV(upAt(p)), 0.0).x; }
fn onTop(p: vec3f, below: f32) -> bool { return length(p) > groundR(p) - below; }
fn slimeOver(m: Mat, p: vec3f) -> Mat {
  let s = slimeAt(p);
  if (s.x + s.y < 0.02 || !onTop(p, 0.6)) { return m; }
  var o = m;
  let glaze = smoothstep(0.05, 0.6, s.y) * 0.6;
  o.alb *= 1.0 - 0.2 * glaze;
  o.rough = mix(o.rough, 0.3, glaze);
  let h = 1.0 - exp(-s.x * 0.3);
  let body = smoothstep(0.12, 0.45, h);
  if (body > 0.0) {
    // veins are tubes a millimetre or so high over the ground's own shape
    // (not the bumps of the moss or soil under them): their sides turn away
    let e = 0.05;
    let b = basisOf(upAt(p));
    let hx = exp(-slimeAt(p - b[0] * e).x * 0.3) - exp(-slimeAt(p + b[0] * e).x * 0.3);
    let hz = exp(-slimeAt(p - b[1] * e).x * 0.3) - exp(-slimeAt(p + b[1] * e).x * 0.3);
    let tube = normalize(worldNormal(p) - (b[0] * hx + b[1] * hz) * (0.12 / (2.0 * e)));
    let streams = 0.9 + 0.2 * gnoise3(p * 5.0, 131u);
    let yellow = mix(vec3f(0.96, 0.74, 0.04), vec3f(0.78, 0.62, 0.3), s.z) * streams;
    o.alb = mix(o.alb, yellow, body);
    o.n = normalize(mix(o.n, tube, body));
    o.rough = mix(o.rough, 0.18, body);
    o.trans = mix(o.trans, 0.55, body);
    o.sheen = mix(o.sheen, 0.25, body);
  }
  return o;
}

// Sun and sky visibility just off a surface.
fn visibility(p: vec3f, n: vec3f) -> vec2f { return visibilityOff(p, n, 0.14); }

fn visibilityOff(p: vec3f, n: vec3f, off: f32) -> vec2f {
  let q = p + n * off;
  let l = lightAt(q);
  let t = tissueLightAt(q);
  return vec2f(l.x * t.x, l.y * t.y);
}

// How much day it is at p, by how high the sun stands there (0 night .. 1).
fn dayHere(p: vec3f) -> f32 { return smoothstep(-0.12, 0.22, dot(upAt(p), scene.sunDir.xyz)); }

// Sky light reaching a surface at p facing n: blue from above (warm low
// down as the sun sets or rises), dim and warm from below; by night, the
// faint cool of starlight.
const ZEN_DUSK: vec3f = vec3f(0.24, 0.27, 0.56);
const HOR_DUSK: vec3f = vec3f(1.0, 0.56, 0.3);
fn skyIrradiance(n: vec3f, p: vec3f) -> vec3f {
  let up = upAt(p);
  let el = dot(up, scene.sunDir.xyz);
  let dusk = exp(-(el / 0.2) * (el / 0.2));
  let above = mix(mix(scene.skyHor.rgb, HOR_DUSK, dusk * 0.8), mix(scene.skyZen.rgb, ZEN_DUSK, dusk * 0.6), 0.55);
  let day = mix(vec3f(0.08, 0.065, 0.05), above, smoothstep(-0.6, 0.9, dot(n, up)));
  return mix(vec3f(0.0035, 0.0045, 0.008), day, dayHere(p));
}

fn shade(p: vec3f, n: vec3f, v: vec3f, m: Mat, vis: vec2f, bnc: vec3f, backVis: f32) -> vec3f {
  let L = scene.sunDir.xyz;
  let sunE = sunAt(p);
  let ndl = dot(n, L);
  let nv = max(dot(n, v), 0.0);
  var c = m.alb * sunE * (max(ndl, 0.0) / PI) * vis.x;
  c += m.alb * skyIrradiance(n, p) * vis.y;
  c += m.alb * bnc;
  c += sunE * ggx(n, v, L, m.rough) * max(ndl, 0.0) * vis.x;
  c += m.alb * sunE * m.sheen * pow(1.0 - nv, 3.0) * 0.12 * vis.x;
  if (m.trans > 0.0) {
    c += m.alb * m.alb * m.trans * max(-ndl, 0.0) * sunE * backVis * (1.6 / PI);
  }
  return c;
}

// ------------------------------------------------------------ materials ---

// Space, looking out along d: stars, and the sun with its glare.
fn space(d: vec3f) -> vec3f {
  var c = vec3f(0.0012, 0.0014, 0.002);
  // a star in some of the cells of a grid over the sky, of its own warmth
  let g = d * 220.0;
  let cell = vec3i(floor(g));
  for (var k = 0; k < 8; k++) {
    let o = cell + vec3i(k & 1, (k >> 1) & 1, k >> 2) - vec3i(i32(fract(g.x) < 0.5), i32(fract(g.y) < 0.5), i32(fract(g.z) < 0.5));
    let h = hashCell3(o, 811u);
    if (u01(h) > 0.03) { continue; }
    let sp = (vec3f(o) + 0.5 + (vec3f(u01(pcg(h)), u01(pcg(h + 1u)), u01(pcg(h + 2u))) - 0.5) * 0.8) / 220.0;
    let a = length(normalize(sp) - d) * 220.0;
    let b = pow(u01(pcg(h + 3u)), 6.0) * 0.6 + 0.01;
    let warm = u01(pcg(h + 4u));
    let tint = mix(vec3f(0.75, 0.85, 1.0), vec3f(1.0, 0.82, 0.6), warm);
    c += tint * b * exp(-a * a * 30.0);
  }
  let mu = max(dot(d, scene.sunDir.xyz), 0.0);
  let disk = smoothstep(0.99986, 0.99992, mu);
  return c + scene.sunCol.rgb * (disk * 60.0 + 0.02 * pow(mu, 200.0) + 0.004 * pow(mu, 12.0));
}

struct Mat {
  alb: vec3f,
  rough: f32,
  n: vec3f,
  trans: f32,
  sheen: f32,
}

fn worldNormal(p: vec3f) -> vec3f {
  let e = 0.12;
  let k = vec2f(1.0, -1.0);
  return normalize(
    k.xyy * volAt(p + k.xyy * e).x + k.yyx * volAt(p + k.yyx * e).x +
    k.yxy * volAt(p + k.yxy * e).x + k.xxx * volAt(p + k.xxx * e).x);
}

fn bump(n: vec3f, g: vec3f, k: f32) -> vec3f { return normalize(n - (g - n * dot(g, n)) * k); }

fn noiseGrad(p: vec3f, f: f32, seed: u32) -> vec3f {
  let e = 0.5 / f;
  let c = gnoise3(p * f, seed);
  return vec3f(gnoise3((p + vec3f(e, 0.0, 0.0)) * f, seed) - c,
               gnoise3((p + vec3f(0.0, e, 0.0)) * f, seed) - c,
               gnoise3((p + vec3f(0.0, 0.0, e)) * f, seed) - c) / e;
}

// Decaying leaf litter: one leaf per cell, rotated and tinted per cell, lying
// flat over the ground where it fell (g: where on the ground, onGround).
fn litter(g: vec3f, s: f32, seed: u32) -> vec4f {
  let cell = cellular3(g / s, seed, 0.85);
  let ang = u01(cell.id) * TAU;
  let lp3 = -cell.off * s;
  let fb = basisOf(upAt(g - lp3));
  let lp = vec2f(dot(lp3, fb[0]), dot(lp3, fb[1]));
  let u = cos(ang) * lp.x + sin(ang) * lp.y;
  let v = -sin(ang) * lp.x + cos(ang) * lp.y;
  let len = s * mix(0.42, 0.7, u01(pcg(cell.id)));
  let x = u / len + 0.5;
  if (x < 0.0 || x > 1.0) { return vec4f(0.0); }
  let w = len * 0.42 * pow(sin(PI * x), 0.8);
  if (abs(v) > w) { return vec4f(0.0); }
  let k = u01(pcg(cell.id + 3u));
  var col = vec3f(0.27, 0.15, 0.065);
  col = mix(col, vec3f(0.4, 0.27, 0.1), smoothstep(0.3, 0.6, k));
  col = mix(col, vec3f(0.12, 0.075, 0.045), smoothstep(0.7, 0.95, k));
  let vein = 1.0 - smoothstep(0.0, 0.02 * len, abs(v)) * 0.5 - 0.3 * (1.0 - smoothstep(0.0, 0.12, abs(fract(x * 7.0 + abs(v) / len * 2.5) - 0.5)));
  col *= 0.75 + 0.25 * vein;
  // holes where it has rotted through
  let rot = cellular3(g * 3.0 / s, seed + 7u, 1.0);
  if (rot.f1 < 0.18 * k) { return vec4f(0.0); }
  return vec4f(col * (0.85 + 0.3 * u01(pcg(cell.id + 5u))), 1.0);
}

fn soilMat(p: vec3f, n: vec3f, fine: bool) -> Mat {
  var m: Mat;
  m.n = n;
  m.rough = 0.9;
  let depth = groundR(p) - length(p);
  let g = onGround(p);
  let seed = 101u;
  // the substrate where it shows (cut faces, deep down): gravel, charcoal,
  // soil, in layers by depth
  if (depth > 1.2 || abs(dot(n, upAt(p))) < 0.5) {
    if (depth > 4.1 + gnoise3(g * 0.8, 103u) * 0.2) {
      let c = cellular3(p / 0.55, 104u, 0.9);
      let tone = u01(c.id);
      var col = mix(vec3f(0.32, 0.3, 0.27), vec3f(0.52, 0.47, 0.4), tone);
      col = mix(vec3f(0.06, 0.05, 0.045), col, smoothstep(0.02, 0.18, c.f2 - c.f1));
      m.alb = col;
      m.rough = 0.5;
    } else if (depth > 3.4 + gnoise3(g * 1.0, 105u) * 0.15) {
      m.alb = vec3f(0.03, 0.028, 0.026) * (0.8 + 0.4 * gnoise3(p * 4.0, 106u));
      m.rough = 0.35;
    } else {
      var col = vec3f(0.07, 0.048, 0.032) * (0.8 + 0.35 * fbm3(p * 1.5, 107u, 3u));
      let perl = cellular3(p / 0.35, 108u, 1.0);
      col = mix(col, vec3f(0.78, 0.76, 0.72), (1.0 - smoothstep(0.08, 0.13, perl.f1)) * step(0.85, u01(perl.id)));
      m.alb = col;
    }
    return m;
  }
  var col = vec3f(0.06, 0.043, 0.03) * (0.75 + 0.5 * (0.5 + 0.5 * fbm3(g * 0.6, seed, 3u)));
  if (!fine) {
    m.alb = col * 1.4;
    return m;
  }
  let crumb = cellular3(g / 0.22, seed + 1u, 0.9);
  col *= mix(1.3, 0.85, smoothstep(0.0, 0.5, crumb.f1));
  m.n = bump(n, noiseGrad(p, 3.0, seed + 2u), 0.05);
  var lf = litter(g, 1.5, seed + 10u);
  if (lf.w == 0.0) { lf = litter(g + 11.3, 1.1, seed + 20u); }
  col = mix(col, lf.rgb, lf.w * smoothstep(-0.3, 0.3, fbm3(g * 0.14, seed + 30u, 3u)));
  m.alb = col;
  return m;
}

fn rockMat(p: vec3f, n: vec3f, typ: f32, fine: bool) -> Mat {
  var m: Mat;
  m.rough = 0.75;
  let seed = 201u + u32(typ * 7.0);
  var base = vec3f(0.34, 0.33, 0.31);
  if (typ > 0.33) { base = vec3f(0.46, 0.36, 0.26); }
  if (typ > 0.66) { base = vec3f(0.17, 0.175, 0.19); }
  var col = base * (0.8 + 0.35 * fbm3(p * 0.5, seed, 3u));
  m.n = n;
  if (!fine) {
    m.alb = col;
    return m;
  }
  let gr = cellular3(p * 5.0, seed + 1u, 1.0);
  col *= mix(0.7, 1.15, u01(gr.id)) * mix(1.0, 0.85, smoothstep(0.1, 0.0, gr.f2 - gr.f1));
  // crustose lichens on the upper faces
  let li = cellular3(p / 0.7, seed + 2u, 0.9);
  let lichen = (1.0 - smoothstep(0.25, 0.45, li.f1)) * step(0.55, u01(li.id)) * smoothstep(0.2, 0.7, dot(n, upAt(p)));
  let lcol = select(vec3f(0.6, 0.64, 0.52), vec3f(0.72, 0.42, 0.1), u01(pcg(li.id)) > 0.85);
  col = mix(col, lcol * (0.85 + 0.2 * gnoise3(p * 9.0, seed + 3u)), lichen);
  m.alb = col;
  m.n = bump(n, noiseGrad(p, 2.5, seed + 4u), 0.12);
  return m;
}

fn woodMat(p: vec3f, n: vec3f, typ: f32, tan: vec3f, fine: bool) -> Mat {
  var m: Mat;
  let t = normalize(tan + vec3f(1e-4));
  let along = dot(p, t);
  let across = p - t * along;
  let seed = 301u;
  let ridges = gnoise3(across * 4.0 + t * along * 0.35, seed);
  let grain = gnoise3(across * 14.0 + t * along * 1.2, seed + 1u);
  var col: vec3f;
  if (typ > 0.5) {
    // weathered driftwood
    col = vec3f(0.44, 0.41, 0.37) * (0.85 + 0.15 * grain) * mix(0.55, 1.0, smoothstep(-0.6, 0.0, ridges));
    m.rough = 0.8;
  } else {
    col = vec3f(0.16, 0.11, 0.075) * (0.8 + 0.25 * grain) * mix(0.35, 1.0, smoothstep(-0.5, 0.2, ridges));
    m.rough = 0.85;
  }
  m.n = n;
  if (fine) {
    let li = cellular3(across * 1.3 + t * along * 0.4, seed + 2u, 0.9);
    col = mix(col, vec3f(0.55, 0.6, 0.5), (1.0 - smoothstep(0.2, 0.4, li.f1)) * step(0.7, u01(li.id)) * 0.8);
    m.n = bump(n, noiseGrad(across * 1.0 + t * along * 0.3, 3.0, seed + 3u), 0.18);
  }
  m.alb = col;
  return m;
}

fn ggx(n: vec3f, v: vec3f, l: vec3f, rough: f32) -> f32 {
  let h = normalize(v + l);
  let nh = max(dot(n, h), 0.0);
  let nv = max(dot(n, v), 1e-3);
  let nl = max(dot(n, l), 0.0);
  let a = max(rough * rough, 0.02);
  let a2 = a * a;
  let dd = nh * nh * (a2 - 1.0) + 1.0;
  let Dg = a2 / (PI * dd * dd);
  let k = a * 0.5;
  let G = (nl / (nl * (1.0 - k) + k)) * (nv / (nv * (1.0 - k) + k));
  let F = 0.04 + 0.96 * pow(1.0 - max(dot(h, v), 0.0), 5.0);
  return Dg * G * F / max(4.0 * nl * nv, 1e-3);
}


// --------------------------------------------------------------- fruits ---

// Soft shadow of a fruit body on itself (the light field is too coarse for
// a cap shading its own stem).
fn fruitShadow(f: Fruit, p: vec3f, L: vec3f) -> f32 {
  var pr = profileOf(f);
  var t = 0.01;
  var res = 1.0;
  for (var k = 0; k < 14; k++) {
    let d = fruitDist(f, &pr, p + L * t, false);
    res = min(res, 10.0 * d / t);
    t += clamp(d, 0.02, 0.4);
    if (res < 0.02 || t > f.bound.w * 2.0) { break; }
  }
  return clamp(res, 0.0, 1.0);
}

fn fruitAO(f: Fruit, p: vec3f, n: vec3f) -> f32 {
  let sc = clamp(f.head.w * 0.3, 0.015, 0.4);
  var pr = profileOf(f);
  var occ = 0.0;
  var w = 1.0;
  for (var k = 1; k <= 3; k++) {
    let h = sc * f32(k) / 3.0;
    occ += (h - fruitDist(f, &pr, p + n * h, k == 1)) * w;
    w *= 0.6;
  }
  return clamp(1.0 - 2.2 * occ / sc, 0.12, 1.0);
}

// Colour by part, pattern and age.
fn fruitMat(f: Fruit, h: FruitHit, p: vec3f, n: vec3f) -> Mat {
  let org = f.i.x;
  let flesh = vec3f(PG(org, G_FLESH_R), PG(org, G_FLESH_G), PG(org, G_FLESH_B));
  let pig = vec3f(PG(org, G_PIG_R), PG(org, G_PIG_G), PG(org, G_PIG_B));
  let spore = vec3f(PG(org, G_SPORE_R), PG(org, G_SPORE_G), PG(org, G_SPORE_B));
  let mat = f.dir.w;
  let dcy = f.life.x;
  let cup = f.axis.w < 0.0;
  let s = h.s;
  let seed = f.i.w;
  var m: Mat;
  m.n = n;
  m.rough = 0.55;
  m.sheen = 0.45;
  m.trans = 0.35;
  let ripe = smoothstep(0.4, 1.0, mat);
  let hym = mix(flesh * 0.93, spore, 0.1 + ripe * ripe * 0.75);
  var c = flesh;
  let rg = h.region;
  if (rg == RG_TOP || rg == RG_UNDER) {
    // the upper face carries the pigment (the inside of a cup too)
    if (rg == RG_TOP) {
      c = pig;
      let fib = 0.5 + 0.5 * gnoise2(vec2f(h.phi * 38.0, s * 4.0), seed);
      c *= 1.0 - PG(org, G_STREAK) * 0.4 * fib;
      c = mix(c * (1.0 - 0.3 * PG(org, G_STREAK)), c, smoothstep(0.0, 0.35, s));
      let zones = PG(org, G_ZONES);
      if (zones > 0.0) {
        let band = 0.5 + 0.5 * sin(s * f.head.w * 26.0 + 2.0 * gnoise2(vec2f(h.phi * 3.0, s * 6.0), seed + 3u));
        c = mix(c, c * mix(0.4, 1.4, band), zones * 0.85);
        c = mix(c, flesh, smoothstep(0.88, 1.0, s) * zones * 0.6);
      }
      if (h.mark > 0.0 || h.w != 0.0) {
        let k = clamp(h.mark, 0.0, 1.0);
        if (f.feat.w < 0.0) {
          // pits darken; the ridges between them catch the light
          c = mix(c, c * 0.45, k);
          c = mix(c, c * 1.4, smoothstep(0.1, 0.7, h.w) * 0.7);
        } else if (PG(org, G_VOLVA) > 0.2) {
          // warts left by the veil the fruit body hatched from
          c = mix(c, flesh * 1.04, k);
        } else {
          // scales of torn cuticle over paler flesh, solid at the crown
          let ground = mix(mix(flesh, c, 0.3), c, smoothstep(0.35, 0.0, s));
          c = mix(ground, c * 0.85, k);
        }
      }
      // spore slime over the crown
      let gl = PG(org, G_GLEBA);
      if (gl > 0.0 && !cup && glebaMode(f) == 0u) {
        let k = smoothstep(gl * 0.9, gl * 0.9 - 0.12, s);
        c = mix(c, spore * 0.5, k);
        m.rough = mix(m.rough, 0.22, k);
      }
    } else if (cup) {
      c = mix(flesh, pig, 0.3) * 0.95;
    } else {
      c = hym;
      m.rough = 0.75;
    }
  } else if (rg == RG_GILL) {
    c = hym * (0.9 + 0.1 * gnoise2(vec2f(h.phi * 90.0, s * 2.0), seed));
    m.rough = 0.8;
  } else if (rg == RG_STEM) {
    c = mix(flesh, pig, PG(org, G_STEM_PIG));
    c *= 0.88 + 0.12 * gnoise2(vec2f(h.phi * 20.0, h.st * 30.0), seed + 9u);
    c *= mix(0.8, 1.0, smoothstep(0.0, 0.25, h.st));
  } else if (rg == RG_RING) {
    c = flesh * 1.03;
    m.trans = 0.6;
  } else if (rg == RG_VOLVA) {
    c = mix(flesh, vec3f(0.88, 0.86, 0.82), 0.6);
    m.rough = 0.8;
  } else if (rg == RG_GLEBA) {
    if (cup) {
      c = mix(spore, flesh, 0.45);
      m.rough = 0.4;
    } else {
      c = spore * 0.6;
      m.rough = 0.25;
    }
  } else if (rg == RG_TOOTH) {
    c = mix(flesh, pig, 0.25);
    m.trans = 0.55;
  }

  // a mycoparasite's crust spreads in blotches over the gills and underside,
  // then everything: dull, dense and pimpled
  if (f.pb.x > 0.5) {
    let q = u32(f.pb.z);
    let crust = vec3f(f32(q & 255u), f32((q >> 8u) & 255u), f32(q >> 16u)) / 255.0;
    let lp = p - f.base.xyz;
    let nz = fbm3(lp * 6.0, seed + 41u, 3u);
    let early = rg == RG_GILL || rg == RG_UNDER || rg == RG_TOOTH;
    let start = select(0.3, 0.0, early);
    let k = smoothstep(start, start + 0.35, f.pb.y + 0.3 * nz);
    // (a pale fringe of mycelium where it is still advancing)
    let fringe = smoothstep(0.0, 0.35, k) * (1.0 - smoothstep(0.35, 0.8, k));
    c = mix(c, crust * (0.8 + 0.2 * nz), k);
    c = mix(c, vec3f(0.86, 0.84, 0.78), fringe * 0.6);
    m.rough = mix(m.rough, 0.85, k);
    m.sheen = mix(m.sheen, 0.15, k);
    m.trans = mix(m.trans, 0.1, k);
    m.n = bump(m.n, noiseGrad(lp, 22.0, seed + 43u), 0.006 * k);
  }

  let jelly = PG(org, G_JELLY);
  m.trans = mix(m.trans, 0.95, jelly);
  m.rough = mix(m.rough, 0.15, jelly);
  c *= 1.0 + 0.2 * jelly;

  let mode = u32(PG(org, G_DECAY) + 0.5);
  // withering: the colour drains to a dull brown
  let dull = dot(c, vec3f(0.3, 0.59, 0.11)) * vec3f(0.95, 0.68, 0.42);
  if (mode == 0u) { c = mix(c, dull, smoothstep(0.0, 0.5, dcy)); }
  if (mode == 1u) {
    let ink = smoothstep(f.life.z - 0.4, f.life.z, s) * min(1.0, dcy * 3.0);
    let k = select(ink, max(ink, ripe * 0.6), rg == RG_GILL);
    c = mix(c, vec3f(0.015), k);
    m.rough = mix(m.rough, 0.2, k);
  }
  if (mode == 2u) { c = mix(c, spore * 0.7, dcy); }
  // dead brackets darken where they stand; everything ends dark and wet,
  // blackest where it is crumbling away
  if (mode == 3u) { c = mix(c, dull * 0.45, smoothstep(0.0, 0.6, dcy)); }
  let rotted = max(smoothstep(0.25, 1.0, dcy), smoothstep(f.life.z - 0.35, f.life.z, s) * smoothstep(0.0, 0.2, 1.0 - f.life.z));
  c = mix(c, vec3f(0.035, 0.026, 0.018), rotted);
  m.rough = mix(m.rough, 0.5, rotted);
  m.sheen *= 1.0 - rotted;
  m.alb = c;
  return m;
}

// ----------------------------------------------------------------- main ---

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let W = u32(view.right.w);
  let H = u32(view.up.w);
  if (gid.x >= W || gid.y >= H) { return; }

  let ray = viewRay(view, vec2f(gid.xy) + 0.5);
  var ro = ray.o;
  let rd = ray.d;
  let v = -rd;
  let pix = view.fwd.w;
  let eps = max(0.0025, pix * 0.4);
  let fine = pix < 0.07 && scene.den.w > 0.5;
  let jit = u01(hash3(gid.x, gid.y, view.misc.x));
  let L = scene.sunDir.xyz;

  let tg = textureLoad(tissueGeo, gid.xy, 0);
  let tid = textureLoad(tissueId, gid.xy, 0).x;
  let tissueT = select(1e9, tg.w, tid > 0u);

  var col = vec3f(0.0);
  var T = vec3f(1.0);
  var gAlb = vec3f(1.0);
  var gN = vec3f(0.0, 1.0, 0.0);
  var gDepth = 1e6;
  var gId = 0u;
  var gKind = 0.0;   // 0 scenery, 1 settled tissue, 2 tissue that is new or moving
  var gSet = false;
  var travelled = 0.0;

  // what lies behind it all, looking out (an orthographic view looks out
  // over a wide field all the same)
  var out = rd;
  if (!perspective(view)) {
    let sx = f32(gid.x) + 0.5 - 0.5 * view.right.w;
    let sy = f32(gid.y) + 0.5 - 0.5 * view.up.w;
    out = normalize(view.fwd.xyz * view.up.w * 1.2 + view.right.xyz * sx - view.up.xyz * sy);
  }

  for (var seg = 0; seg < 4; seg++) {
    let tmax = tissueT - travelled;
    let bi = sphereHit(ro, rd, SKY_R);
    if (bi.y < max(bi.x, 0.0)) {
      col += T * space(out);
      break;
    }

    // march the planet's air
    let t0 = max(bi.x, 0.0);
    let t1 = bi.y;
    var t = t0;
    var kind = 0u;          // 0 left the air, 1 surface, 2 tissue, 3 moss core, 4 opaque moss
    var mossEntry = -1.0;
    // the moss's look and the light on it (where the ray enters it)
    var look: MossLook;
    var mossLit = vec3f(0.0);
    var inFuzz = false;
    for (var i = 0; i < 260; i++) {
      if (t > t1) { break; }
      if (t >= tmax) {
        kind = 2u;
        t = tmax;
        break;
      }
      let p = ro + rd * t;
      let vv = volAt(p);
      var thk = 0.0;
      if (vv.x < 1.0) { thk = mossThick(auxAt(p)); }
      if (thk > 0.02 && vv.x < thk && vv.x > thk * MOSS_CORE + eps) {
        // fuzzy top of the moss: composite front to back (a slime mould
        // crawling over it hides it)
        if (mossEntry < 0.0) {
          if (slimeAt(p).x > 0.4 && onTop(p, 1.0)) {
            kind = 3u;
            break;
          }
          mossEntry = t;
          look = mossLook(p);
          let up = upAt(p);
          let l = lightAt(p + up * 0.04);
          let tl = tissueLightAt(p);
          mossLit = sunAt(p) * l.x * tl.x * 0.3 + skyIrradiance(up, p) * l.y * tl.y + bounceAt(p) + glowAt(p);
          t += 0.025 * jit;
          inFuzz = true;
          continue;
        }
        let h01 = clamp(vv.x / max(thk, 1e-3), 0.0, 1.0);
        let dt = 0.025;
        let a = 1.0 - exp(-mossDensity(p, h01, thk, !fine) * dt);
        col += T * a * mossColour(look, h01) * mossLit;
        T *= 1.0 - a;
        if (max(T.x, max(T.y, T.z)) < 0.03) {
          kind = 4u;
          break;
        }
        t += min(dt, tmax - t + 1e-3);
        continue;
      }
      if (thk > 0.02 && vv.x <= thk * MOSS_CORE + eps) {
        kind = 3u;
        break;
      }
      if (vv.x < eps) {
        kind = 1u;
        break;
      }
      var d = vv.x;
      if (thk > 0.02) { d = vv.x - thk; }
      t += max(min(d * 0.85, tmax - t + 1e-3), eps * 0.5);
    }

    // the air: a few fixed samples of sunlight along the visible stretch
    let tEnd = select(t, min(t1, tmax), kind == 0u);
    let segLen = max(tEnd - t0, 0.0);
    if (segLen > 0.0) {
      var hz = vec3f(0.0);
      var thick = 0.0;
      for (var k = 0; k < 4; k++) {
        let ps = ro + rd * (t0 + (f32(k) + jit) / 4.0 * segLen);
        let dens = exp(-max(length(ps) - PLANET_R, 0.0) / HAZE_H);
        hz += lightAt(ps).x * tissueLightAt(ps).x * dens * sunAt(ps);
        thick += dens;
      }
      let depth = HAZE * segLen * thick / 4.0;
      col += T * depth * (hz / max(thick, 1e-4)) * hgPhase(dot(rd, L), 0.45) * vec3f(0.62, 0.8, 1.15);
      T *= exp(-depth);
      let dust = sporeDust(ro, rd, t0, tEnd, jit);
      col += T * dust.rgb;
      T *= dust.a;
    }

    if (mossEntry >= 0.0 && !gSet) {
      let pe = ro + rd * mossEntry;
      gN = worldNormal(pe);
      gAlb = mossAlbedo(pe, 0.8);
      gDepth = mossEntry + travelled;
      gSet = true;
    }

    if (kind == 0u) {
      // out of the air again, into space
      let adv = t1 + 0.02;
      ro += rd * adv;
      travelled += adv;
      continue;
    }
    if (kind == 4u) { break; }

    let p = ro + rd * t;
    var m: Mat;
    var sunMul = 1.0;
    var skyMul = 1.0;
    var visOff = 0.14;
    // (what glows is not lit by its own light: it shines out)
    var selfGlow = 0.0;
    if (kind == 2u && (tid & FRUIT_BIT) != 0u) {
      let fi = tid & 0xfffu;
      let fr = fruits[fi];
      let org = fr.i.x;
      var h = unpackFruitHit(tid);
      // a developed head: the field exactly here (the packed mark is coarse)
      if (devSlot(fr) < PAT_SLOTS && fr.feat2.x > 0.0 && !isStemPart(h.region)) {
        var pr = profileOf(fr);
        let hx = headEval(fr, &pr, p, true);
        h.w = hx.w;
        h.mark = hx.mark;
      }
      m = fruitMat(fr, h, p, tg.xyz);
      // fruit bodies a few pixels across get their shading from the light field
      let ppuHere = pxPerUnit(view, dot(p - view.eye.xyz, view.fwd.xyz));
      let px = fr.head.w * ppuHere;
      var ao = 1.0;
      if (px > 6.0) { ao = fruitAO(fr, p, m.n); }
      sunMul = mix(1.0, ao, 0.5);
      if (px > 10.0) { sunMul *= fruitShadow(fr, p + m.n * 0.004, L); }
      skyMul = ao;
      visOff = 0.25;
      selfGlow = PG(org, G_BODY_GLOW) * smoothstep(0.3, 1.0, fr.dir.w) * fruitFlash(fr);
      gId = org + 1u;
      if (!gSet) {
        // (with the garden paused nothing is new or moving)
        let perFrame = max(scene.sim.y, 1.0);
        let young = f32(counters[5] - fr.i.y) / perFrame < 2.0;
        let moving = fr.life.y * ppuHere > 0.15;
        gKind = select(1.0, 2.0, scene.sim.y > 0.0 && (young || moving));
      }
      let glowK = select(select(0.35, 0.15, h.region == RG_STEM), 1.0, h.region == RG_GILL || h.region == RG_UNDER);
      col += T * glowTone(m.alb) * selfGlow * glowK * glowColour(org) * GLOW_EMIT;
    } else if (kind == 2u) {
      // fungal tissue from the raster pass; each cell varies a little
      let ci = tid - 1u;
      let cell = cells[ci];
      let org = cell.i.y;
      let ph = phaseOf(cell.i.z);
      let hv = u01(pcg(bitcast<u32>(cell.q.w) * 7u + 3u));
      // (fruiting tissue at the end of its life dulls to a dry brown)
      let wi = wither(org, ph, f32(counters[5] - cell.i.w));
      m.alb = mix(cellColor(org, ph) * (0.84 + 0.3 * hv), vec3f(0.17, 0.12, 0.08) * (0.8 + 0.4 * hv), wi * 0.85);
      m.rough = mix(0.5 + 0.35 * hv, 0.95, wi);
      m.trans = select(0.3, 0.6, cell.p.w < 0.03) * (1.0 - 0.7 * wi);
      m.sheen = 0.5 * (1.0 - wi);
      m.n = tg.xyz;
      gId = org + 1u;
      if (!gSet) {
        let perFrame = max(scene.sim.y, 1.0);
        let frames = f32(counters[5] - cell.i.w) / perFrame;
        let px = length(cell.p.xyz - cell.q.xyz) * pxPerUnit(view, dot(cell.p.xyz - view.eye.xyz, view.fwd.xyz)) * perFrame;
        gKind = select(1.0, 2.0, scene.sim.y > 0.0 && (frames < 2.0 || px > 0.2));
      }
      let light = cellLight(org, ph, cell.i.z, f32(counters[5] - cell.i.w));
      selfGlow = dot(light, vec3f(1.0 / 3.0));
      col += T * glowTone(m.alb) * light * 0.5 * GLOW_EMIT;
    } else if (kind == 3u) {
      let n = worldNormal(p);
      var c = mossAlbedo(p, 0.35) * 0.85;
      c *= 0.8 + 0.35 * upNoise(p, 14.0, 5.0, 67u);
      m.alb = c;
      m.n = bump(n, noiseGrad(p, 9.0, 68u), 0.25);
      m.rough = 0.95;
      m.sheen = 0.8;
      m.trans = 0.0;
      m = slimeOver(m, p);
    } else {
      let vv = volAt(p);
      let n = worldNormal(p);
      let pw = fieldWarp(p, n);
      if (vv.y > 0.5) {
        m = rockMat(p, n, vv.w, true);
      } else if (vv.z > 0.5) {
        m = woodMat(p, n, vv.w, auxAt(p).xyz, true);
      } else {
        m = soilMat(p, n, true);
        // rich soil is dark and eaten-out soil pales; wet soil darkens
        let fs = fieldsAt(pw - n * 0.15);
        m.alb *= mix(1.35, 0.7, smoothstep(0.1, 2.0, fs.z)) * mix(1.15, 0.75, fs.w);
      }
      m = groundLife(m, pw, vv.z > 0.5 && vv.y <= 0.5);
      // the floor under moss is dark, old growth
      m.alb = mix(m.alb, vec3f(0.07, 0.07, 0.03), smoothstep(0.03, 0.2, auxAt(p).w) * 0.7);
      // mycelium just under the surface shows as a fine white bloom
      let my = densSmooth(pw - n * 0.1);
      let fib = smoothstep(0.1, 0.6, 0.5 + 0.5 * gnoise3(p * vec3f(24.0, 24.0, 24.0), 71u));
      m.alb = mix(m.alb, vec3f(0.86, 0.85, 0.8), clamp(my / 12.0, 0.0, 0.7) * fib);
      m = slimeOver(m, p);
    }
    var vis = visibilityOff(p, m.n, visOff);
    vis *= vec2f(sunMul, skyMul);
    let backVis = select(0.0, visibilityOff(p, -m.n, visOff).x, m.trans > 0.0);
    let glowIn = glowAt(p + m.n * 0.2) * mix(1.0, 0.15, smoothstep(0.0, 0.5, selfGlow));
    col += T * shade(p, m.n, v, m, vis, bounceAt(p + m.n * 0.14) * skyMul + glowIn, backVis);
    if (!gSet) {
      gAlb = m.alb;
      gN = m.n;
      gDepth = t + travelled;
      gSet = true;
    }
    break;
  }

  textureStore(outRad, gid.xy, vec4f(min(finite3(col), vec3f(60.0)), 1.0));
  if (bad(gN.x) || bad(gN.y) || bad(gN.z)) { gN = vec3f(0.0, 1.0, 0.0); }
  textureStore(outGeo, gid.xy, vec4f(gN, gDepth));
  textureStore(outAlb, gid.xy, vec4f(gAlb, gKind));
  ids[gid.y * W + gid.x] = gId;
}
