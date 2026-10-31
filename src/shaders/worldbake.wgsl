// ---- worldbake.wgsl: object distances for the volume bakes ----
// Requires `world: WorldDesc` and `wabs: array<vec4f>` (absolute placements).

fn rootCtrl(i: u32, k: u32) -> vec4f { return wabs[MAX_ROCKS + i * 6u + k]; }

// Distance to root i, plus the curve tangent at the nearest point.
struct RootHit {
  d: f32,
  tan: vec3f,
}

fn rootDist(p: vec3f, i: u32) -> RootHit {
  var o: RootHit;
  o.d = 1e9;
  let lo = wabs[MAX_ROCKS + i * 6u + 4u].xyz;
  let hi = wabs[MAX_ROCKS + i * 6u + 5u].xyz;
  let outside = max(max(lo - p, p - hi), vec3f(0.0));
  if (any(outside > vec3f(0.0))) {
    // the box is a curve radius + 2 around the curve, so out here the root is
    // at least 2 away: beyond the soil blend (0.9, bake_volume.wgsl) near
    // the ground, as it is just inside the box (nearer, the ground would
    // bulge on one side of the box's faces and not the other)
    o.d = length(outside) + 2.0;
    return o;
  }
  let a = rootCtrl(i, 0u);
  let b = rootCtrl(i, 1u);
  let c = rootCtrl(i, 2u);
  let d = rootCtrl(i, 3u);
  var prev = bez4(a, b, c, d, 0.0);
  for (var s = 1u; s <= ROOT_SEG; s++) {
    let cur = bez4(a, b, c, d, f32(s) / f32(ROOT_SEG));
    let dd = sdRoundCone(p, prev.xyz, cur.xyz, prev.w, cur.w);
    if (dd < o.d) {
      o.d = dd;
      o.tan = cur.xyz - prev.xyz;
    }
    prev = cur;
  }
  let rt = world.roots[i];
  let rr = max(a.w, d.w);
  if (o.d < rr) {
    // knots, bulges and bark relief
    o.d += fbm3(p * (0.9 / rr), u32(rt.info.x), 3u) * 0.22 * rr;
  }
  o.tan = normalize(o.tan + vec3f(1e-6));
  return o;
}
