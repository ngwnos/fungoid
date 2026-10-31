// ---- tissue.wgsl: rasterise cells as tube impostors ----
// Each live cell is a capsule to its parent, drawn as a screen-aligned quad
// with an exact ray-capsule test per pixel. The path tracer starts from this
// for primary visibility of fungal tissue, so cost no longer grows with how
// much tissue fills the screen.

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
@group(0) @binding(2) var<storage, read> pheno: array<f32>;
@group(0) @binding(3) var<storage, read> counters: array<u32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(flat) cell: u32,
  @location(1) @interpolate(flat) a: vec3f,
  @location(2) @interpolate(flat) b: vec4f,   // parent end, radius
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
  let c = cells[ii];
  if ((c.i.z & F_DEAD) != 0u) { return o; }
  let a = c.p.xyz;
  // (slight overlap so neighbouring fibres knit together; withering, thinner)
  // (firing, it swells a little)
  let r = c.p.w * 1.15 * (1.0 - 0.8 * wither(c.i.y, phaseOf(c.i.z), f32(counters[5] - c.i.w)))
        * (1.0 + 0.6 * f32(exciteOf(c.i.z)) / 255.0);
  var b = a;
  if (c.i.x != NO_PARENT) { b = cells[c.i.x].p.xyz; }
  // (in perspective, what is not well in front of the lens is not drawn)
  let za = dot(a - view.eye.xyz, view.fwd.xyz);
  let zb = dot(b - view.eye.xyz, view.fwd.xyz);
  if (perspective(view) && min(za, zb) < 0.3) { return o; }
  let pa = project(view, a);
  let pb = project(view, b);
  let rp = r * pxPerUnit(view, min(za, zb)) + 1.0;
  let mn = min(pa.xy, pb.xy) - rp;
  let mx = max(pa.xy, pb.xy) + rp;
  let corner = vec2f(f32(vi & 1u), f32(vi >> 1u));
  let s = mix(mn, mx, corner);
  o.pos = vec4f(s.x / view.right.w * 2.0 - 1.0, 1.0 - s.y / view.up.w * 2.0, 0.5, 1.0);
  o.cell = ii;
  o.a = a;
  o.b = vec4f(b, r);
  return o;
}

// Inigo Quilez's capsule intersection, open at pb; -1 on a miss.
fn capIntersect(ro: vec3f, rd: vec3f, pa: vec3f, pb: vec3f, r: f32) -> f32 {
  let ba = pb - pa;
  let oa = ro - pa;
  let baba = dot(ba, ba);
  if (baba < 1e-10) {
    let bb = dot(rd, oa);
    let h = bb * bb - (dot(oa, oa) - r * r);
    return select(-1.0, -bb - sqrt(max(h, 0.0)), h > 0.0);
  }
  let bard = dot(ba, rd);
  let baoa = dot(ba, oa);
  let rdoa = dot(rd, oa);
  let oaoa = dot(oa, oa);
  let a = baba - bard * bard;
  let b = baba * rdoa - baoa * bard;
  let c = baba * oaoa - baoa * baoa - r * r * baba;
  let h = b * b - a * c;
  if (h >= 0.0) {
    let t = (-b - sqrt(h)) / a;
    let y = baoa + t * bard;
    if (y > 0.0 && y < baba) { return t; }
    // (only its own end's cap: the end it shares with its parent is the
    // parent's own, so no two cells draw the same sphere, and which shows
    // never depends on the order they are drawn in)
    if (y > 0.0) { return -1.0; }
    let oc = oa;
    let bb = dot(rd, oc);
    let cc = dot(oc, oc) - r * r;
    let hh = bb * bb - cc;
    if (hh > 0.0) { return -bb - sqrt(hh); }
  }
  return -1.0;
}

struct FOut {
  @builtin(frag_depth) depth: f32,
  @location(0) geo: vec4f,   // normal, distance along the view ray
  @location(1) id: u32,      // cell + 1
}

@fragment
fn fs(in: VOut) -> FOut {
  let ray = viewRay(view, in.pos.xy);
  let rd = ray.d;
  // (tested from just short of the capsule: from the eye, metres off, the
  // test has too little precision left for a capsule this thin)
  let t0 = max(dot(in.a - ray.o, rd) - length(in.b.xyz - in.a) - in.b.w - 0.01, 0.0);
  let ro = ray.o + rd * t0;
  let tc = capIntersect(ro, rd, in.a, in.b.xyz, in.b.w);
  if (tc < 0.0) { discard; }
  let t = t0 + tc;
  let p = ro + rd * tc;
  let ba = in.b.xyz - in.a;
  let pa = p - in.a;
  let hh = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-10), 0.0, 1.0);
  var o: FOut;
  o.depth = rasterDepth(t);
  o.geo = vec4f(normalize(pa - ba * hh), t);
  o.id = in.cell + 1u;
  return o;
}
