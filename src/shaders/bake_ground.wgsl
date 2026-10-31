// ---- bake_ground.wgsl: the ground map, and objects placed on the ground ----
// The ground map holds the ground's height over its reference (the shape's
// terrainHeight, at each key: groundUV), and how steep it is there.

@group(0) @binding(0) var<uniform> world: WorldDesc;
@group(0) @binding(1) var<storage, read_write> wabs: array<vec4f>;
@group(0) @binding(2) var groundOut: texture_storage_2d<rgba16float, write>;

// The level of rock k's top at key `dir` (or the ground's, if it is not
// there).
fn rockTop(k: u32, dir: vec3f) -> f32 {
  let rk = world.rocks[k];
  let c = wabs[k].xyz;
  let ground = groundRadius(dir);
  var r = levelOf(c) + max(rk.r.x, max(rk.r.y, rk.r.z)) * 1.4;
  for (var s = 0; s < 64; s++) {
    let d = rockDist(atLevel(dir, r), rk, c);
    if (d < 0.01 || r <= ground) { break; }
    r -= max(d, 0.02);
  }
  return max(r, ground);
}

// Rocks and roots are placed relative to the ground; resolve them once.
fn placeObjects() {
  for (var i = 0u; i < world.counts.x; i++) {
    let rk = world.rocks[i];
    let dir = restKey(rk.a.xyz);
    let c = atLevel(dir, groundRadius(dir) + rk.o.z * rk.r.y);
    wabs[i] = vec4f(c, max(rk.r.x, max(rk.r.y, rk.r.z)) * 1.6);
  }
  for (var i = 0u; i < world.counts.y; i++) {
    let rt = world.roots[i];
    var c = array<vec4f, 4>(rt.p0, rt.p1, rt.p2, rt.p3);
    for (var k = 0u; k < 4u; k++) { c[k] = rootPoint(c[k]); }
    // a branch resting on a rock: raise or lower it onto the rock's top
    if (rt.info.z > 0.5) {
      let t = rt.info.w;
      let at = bez4(c[0], c[1], c[2], c[3], t);
      let up = restUp(at.xyz);
      let lift = rockTop(u32(rt.info.z) - 1u, restKey(at.xyz)) + at.w * 0.6 - levelOf(at.xyz);
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
  let dir = groundTexelKey(gid.xy);
  let h = terrainHeight(dir);
  let b = keyFrame(dir);
  let e = groundSlopeStep();
  let hx = (terrainHeight(groundAlong(dir, b[0], e)) - terrainHeight(groundAlong(dir, b[0], -e))) / 0.2;
  let hz = (terrainHeight(groundAlong(dir, b[1], e)) - terrainHeight(groundAlong(dir, b[1], -e))) / 0.2;
  textureStore(groundOut, gid.xy, vec4f(h, inverseSqrt(1.0 + hx * hx + hz * hz), 0.0, 0.0));
}
