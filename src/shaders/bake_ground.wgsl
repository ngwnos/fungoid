// ---- bake_ground.wgsl: the ground map, and objects placed on the ground ----
// The ground map holds the ground's height above PLANET_R over the sphere,
// unfolded (world.wgsl groundUV), and how steep it is there.

@group(0) @binding(0) var<uniform> world: WorldDesc;
@group(0) @binding(1) var<storage, read_write> wabs: array<vec4f>;
@group(0) @binding(2) var groundOut: texture_storage_2d<rgba16float, write>;

// the ground's height above PLANET_R in direction `dir`
fn terrainHeight(dir: vec3f) -> f32 {
  let tr = world.terrain;
  let q = dir * PLANET_R;
  var h = fbm3(dir * 1.3, tr.seeds.x, 3u) * tr.amp.x;
  h += fbm3(q * 0.11, tr.seeds.y, 4u) * tr.amp.y;
  h += fbm3(q * 0.48, tr.seeds.z, 3u) * tr.amp.z;
  for (var i = 0u; i < world.counts.z; i++) {
    let b = tr.bumps[i];
    let d = q - b.xyz;
    h += tr.heights[i / 4u][i % 4u] * exp(-dot(d, d) / (b.w * b.w));
  }
  return h;
}
fn groundRadius(dir: vec3f) -> f32 { return PLANET_R + terrainHeight(dir); }

// How far from the centre rock k's top is in direction `dir` (or the
// ground, if it is not there).
fn rockTop(k: u32, dir: vec3f) -> f32 {
  let rk = world.rocks[k];
  let c = wabs[k].xyz;
  let ground = groundRadius(dir);
  var r = length(c) + max(rk.r.x, max(rk.r.y, rk.r.z)) * 1.4;
  for (var s = 0; s < 64; s++) {
    let d = rockDist(dir * r, rk, c);
    if (d < 0.01 || r <= ground) { break; }
    r -= max(d, 0.02);
  }
  return max(r, ground);
}

// Rocks and roots are placed relative to the ground; resolve them once.
fn placeObjects() {
  for (var i = 0u; i < world.counts.x; i++) {
    let rk = world.rocks[i];
    let dir = normalize(rk.a.xyz);
    let c = dir * (groundRadius(dir) + rk.o.z * rk.r.y);
    wabs[i] = vec4f(c, max(rk.r.x, max(rk.r.y, rk.r.z)) * 1.6);
  }
  for (var i = 0u; i < world.counts.y; i++) {
    let rt = world.roots[i];
    var c = array<vec4f, 4>(rt.p0, rt.p1, rt.p2, rt.p3);
    for (var k = 0u; k < 4u; k++) {
      let r = length(c[k].xyz);
      let dir = c[k].xyz / r;
      c[k] = vec4f(dir * (groundRadius(dir) + r - PLANET_R), c[k].w);
    }
    // a branch resting on a rock: raise or lower it onto the rock's top
    if (rt.info.z > 0.5) {
      let t = rt.info.w;
      let at = bez4(c[0], c[1], c[2], c[3], t);
      let up = normalize(at.xyz);
      let lift = rockTop(u32(rt.info.z) - 1u, up) + at.w * 0.6 - length(at.xyz);
      if (t < 0.5) { c[1] += vec4f(up * lift / (3.0 * (1.0 - t) * (1.0 - t) * t), 0.0); }
      else { c[2] += vec4f(up * lift / (3.0 * (1.0 - t) * t * t), 0.0); }
    }
    var lo = vec3f(1e9);
    var hi = vec3f(-1e9);
    for (var k = 0u; k < 4u; k++) {
      let a = c[k];
      wabs[MAX_ROCKS + i * 6u + k] = a;
      lo = min(lo, a.xyz - vec3f(a.w + 2.0));
      hi = max(hi, a.xyz + vec3f(a.w + 2.0));
    }
    wabs[MAX_ROCKS + i * 6u + 4u] = vec4f(lo, 0.0);
    wabs[MAX_ROCKS + i * 6u + 5u] = vec4f(hi, 0.0);
  }
}

// The map: height, and how much a distance off the ground shrinks along the
// ground's normal (1/sqrt(1 + slope^2)).
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x == 0u && gid.y == 0u) { placeObjects(); }
  if (gid.x >= GROUND_RES.x || gid.y >= GROUND_RES.y) { return; }
  let dir = octDecode((vec2f(gid.xy) + 0.5) / vec2f(GROUND_RES) * 2.0 - 1.0);
  let h = terrainHeight(dir);
  let b = basisOf(dir);
  let e = 0.1 / PLANET_R;
  let hx = (terrainHeight(normalize(dir + b[0] * e)) - terrainHeight(normalize(dir - b[0] * e))) / 0.2;
  let hz = (terrainHeight(normalize(dir + b[1] * e)) - terrainHeight(normalize(dir - b[1] * e))) / 0.2;
  textureStore(groundOut, gid.xy, vec4f(h, inverseSqrt(1.0 + hx * hx + hz * hz), 0.0, 0.0));
}
