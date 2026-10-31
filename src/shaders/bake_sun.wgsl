// ---- bake_sun.wgsl: lighting volumes for the world ----
// Lighting that would otherwise need rays is baked into half-resolution
// volumes over the domain. The scenery does not move, so some of it is baked
// once:
//   light.g  sky openness (cosine-weighted, within a few cm, about up)
//   bounce   one bounce of light off nearby soil, moss, rock and wood, as if
//            the sun stood overhead (the renderer scales it by the day)
// The sun does move: its light (soil, rock and wood block it, moss dims it)
// is baked for one place of the sun at a time (bakeSunVis), the renderer
// blending between two of them.

struct SunU {
  dir: vec4f,      // towards the sun
  col: vec4f,      // sun irradiance
}

@group(0) @binding(0) var<uniform> sun: SunU;
@group(0) @binding(1) var vol: texture_3d<f32>;
@group(0) @binding(2) var aux: texture_3d<f32>;
@group(0) @binding(3) var linSamp: sampler;
@group(0) @binding(4) var outLight: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var light: texture_3d<f32>;
@group(0) @binding(6) var outBounce: texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var outVis: texture_storage_3d<rgba8unorm, write>;

const DIRS: u32 = 16u;

fn cellPos(gid: vec3u) -> vec3f { return BOX_MIN + (vec3f(gid) + 0.5) / vec3f(LIGHT_RES) * boxSize(); }

fn mossThickAt(a: vec4f) -> f32 { return select(0.0, a.w, a.w > 0.02); }

// Transmittance from p along d for up to maxT (moss attenuates, solids block).
// Returns (transmittance, hit distance or -1).
fn march(p: vec3f, d: vec3f, maxT: f32) -> vec2f {
  var T = 1.0;
  var t = 0.02;
  for (var i = 0; i < 160; i++) {
    if (t > maxT) { break; }
    let q = p + d * t;
    if (leftAir(q)) {
      T *= exitTint(q);
      break;
    }
    let uvw = volUVW(q);
    let v = textureSampleLevel(vol, linSamp, uvw, 0.0);
    let thk = mossThickAt(textureSampleLevel(aux, linSamp, uvw, 0.0));
    if (v.x < thk * 0.5) { return vec2f(0.0, t); }
    var step = max(v.x - thk, 0.04) * 0.9;
    if (v.x < thk) {
      let h = clamp((v.x / max(thk, 1e-3) - 0.5) / 0.5, 0.0, 1.0);
      step = 0.04;
      T *= exp(-16.0 * clamp(thk * 3.0, 0.0, 1.0) * (1.0 - h * h) * step);
    }
    if (T < 0.01) { return vec2f(0.0, t); }
    t += step;
  }
  return vec2f(T, -1.0);
}

// Fibonacci directions over the hemisphere above, cosine-weighted: in a frame
// with up along y (world: overHere).
fn hemiDir(i: u32) -> vec3f {
  let u = (f32(i) + 0.5) / f32(DIRS);
  let r = sqrt(u);
  let a = f32(i) * 2.39996323;
  return vec3f(r * cos(a), sqrt(max(1.0 - u, 0.0)), r * sin(a));
}

// a direction in a frame with up along y, as it is at p
fn overHere(p: vec3f, h: vec3f) -> vec3f {
  let b = basisOf(upAt(p));
  return b[0] * h.x + b[2] * h.y + b[1] * h.z;
}

@compute @workgroup_size(4, 4, 4)
fn bakeLight(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= LIGHT_RES)) { return; }
  let p = cellPos(gid);
  var sky = 0.0;
  for (var i = 0u; i < DIRS; i++) {
    sky += march(p, overHere(p, hemiDir(i)), 6.0).x;
  }
  textureStore(outLight, gid, vec4f(0.0, sky / f32(DIRS), 0.0, 1.0));
}

// The sun's light, with the sun at sun.dir: none where the world itself
// stands in the way (sunBlocked: cheaply), else a march towards it.
@compute @workgroup_size(4, 4, 4)
fn bakeSunVis(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= LIGHT_RES)) { return; }
  let p = cellPos(gid);
  let L = sun.dir.xyz;
  var v = 0.0;
  if (!sunBlocked(p, L)) { v = march(p, L, 60.0).x; }
  textureStore(outVis, gid, vec4f(v, 0.0, 0.0, 1.0));
}

// Rough surface colour by material, for bounce light.
fn surfaceAlbedo(q: vec3f) -> vec3f {
  let uvw = volUVW(q);
  let v = textureSampleLevel(vol, linSamp, uvw, 0.0);
  let thk = mossThickAt(textureSampleLevel(aux, linSamp, uvw, 0.0));
  var c = vec3f(0.07, 0.05, 0.035);
  if (v.y > 0.5) { c = vec3f(0.3, 0.29, 0.27); }
  if (v.z > 0.5) { c = vec3f(0.3, 0.25, 0.2); }
  if (thk > 0.05) { c = vec3f(0.12, 0.22, 0.05); }
  return c;
}

@compute @workgroup_size(4, 4, 4)
fn bakeBounce(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= LIGHT_RES)) { return; }
  let p = cellPos(gid);
  var acc = vec3f(0.0);
  // gather over the whole sphere: light comes back up from the ground too
  for (var i = 0u; i < DIRS * 2u; i++) {
    var h = hemiDir(i % DIRS);
    if (i >= DIRS) { h.y = -h.y; }
    let d = overHere(p, h);
    let m = march(p, d, 5.0);
    if (m.y > 0.0) {
      let q = p + d * m.y;
      let sv = textureSampleLevel(light, linSamp, volUVW(q - d * 0.15), 0.0).y;
      acc += surfaceAlbedo(q) * sun.col.rgb * sv * 0.5;
    }
  }
  textureStore(outBounce, gid, vec4f(acc / f32(DIRS * 2u), 1.0));
}
