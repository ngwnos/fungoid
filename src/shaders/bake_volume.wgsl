// ---- bake_volume.wgsl: soil + rocks + roots -> distance volume ----
// rgba16f: distance (cm), rock weight, wood weight, object type.

@group(0) @binding(0) var<uniform> world: WorldDesc;
@group(0) @binding(1) var<storage, read> wabs: array<vec4f>;
@group(0) @binding(2) var ground: texture_2d<f32>;
@group(0) @binding(3) var linSamp: sampler;
@group(0) @binding(4) var volOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= VOL_RES)) { return; }
  let p = BOX_MIN + (vec3f(gid) + 0.5) / vec3f(VOL_RES) * boxSize();

  let dSoil = soilDist(p);

  var dWood = 1e9;
  var woodType = 0.0;
  for (var i = 0u; i < world.counts.y; i++) {
    let r = rootDist(p, i);
    if (r.d < dWood) {
      dWood = r.d;
      woodType = world.roots[i].info.y;
    }
  }
  var dRock = 1e9;
  var rockType = 0.0;
  for (var i = 0u; i < world.counts.x; i++) {
    let d = rockDist(p, world.rocks[i], wabs[i].xyz);
    if (d < dRock) {
      dRock = d;
      rockType = world.rocks[i].r.w;
    }
  }

  var d = smin(dSoil, dWood, 0.9);
  d = smin(d, dRock, 0.35);
  let wRock = smoothstep(0.2, -0.2, dRock - min(dSoil, dWood));
  let wWood = (1.0 - wRock) * smoothstep(0.2, -0.2, dWood - dSoil);
  let typ = select(woodType, rockType, wRock > 0.5);
  textureStore(volOut, gid, vec4f(clamp(d, -8.0, 40.0), wRock, wWood, typ));
}
