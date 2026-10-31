// ---- bake_aux.wgsl: wood grain direction and moss coverage ----
// rgba8snorm: tangent xyz, moss layer thickness (cm). Moss settles on upward
// faces, in damp low ground and on rocks and wood, and avoids overhangs; it
// grows in cushions. Its patterns are of where on the planet it is (the
// same all the way up).

@group(0) @binding(0) var<uniform> world: WorldDesc;
@group(0) @binding(1) var<storage, read> wabs: array<vec4f>;
@group(0) @binding(2) var vol: texture_3d<f32>;
@group(0) @binding(3) var auxOut: texture_storage_3d<rgba8snorm, write>;

fn vd(c: vec3i) -> f32 {
  return textureLoad(vol, clamp(c, vec3i(0), vec3i(VOL_RES) - 1), 0).x;
}
fn vdAt(p: vec3f) -> f32 { return vd(vec3i(floor(volUVW(p) * vec3f(VOL_RES)))); }

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= VOL_RES)) { return; }
  let c = vec3i(gid);
  let v = textureLoad(vol, c, 0);
  let d = v.x;
  if (abs(d) > 1.5) {
    textureStore(auxOut, gid, vec4f(0.0));
    return;
  }
  let p = BOX_MIN + (vec3f(gid) + 0.5) / vec3f(VOL_RES) * boxSize();
  let g = vec3f(
    vd(c + vec3i(1, 0, 0)) - vd(c - vec3i(1, 0, 0)),
    vd(c + vec3i(0, 1, 0)) - vd(c - vec3i(0, 1, 0)),
    vd(c + vec3i(0, 0, 1)) - vd(c - vec3i(0, 0, 1)));
  let up = upAt(p);
  let n = normalize(g + up * 1e-5);

  var tan = vec3f(0.0);
  if (v.z > 0.05) {
    var best = 1e9;
    for (var i = 0u; i < world.counts.y; i++) {
      let r = rootDist(p, i);
      if (r.d < best) {
        best = r.d;
        tan = r.tan;
      }
    }
  }

  // open sky above? (overhangs stay bare)
  let above = vdAt(p + up * 1.5);
  let open = smoothstep(0.5, 2.5, above);
  let facing = smoothstep(0.15, 0.7, dot(n, up));
  // (low ground is damp)
  let damp = mossDamp(p);
  let q = groundCoord(p, up);
  let patches = fbm3(q * 0.17, 31u, 4u) + 0.12 * fbm3(q * 1.0, 32u, 2u);
  let host = v.y * 0.4 + v.z * 0.5;
  var moss = facing * open * smoothstep(0.02, 0.4, patches + damp * 0.25 + host);
  // (none on the walls, if there are any)
  if (FLAT) { moss *= mossWall(p); }
  let cushion = 0.4 + 0.6 * smoothstep(-0.45, 0.45, fbm3(q * 1.1, 41u, 3u));
  let thick = clamp(moss, 0.0, 1.0) * cushion * 0.45;
  textureStore(auxOut, gid, vec4f(tan, thick));
}
