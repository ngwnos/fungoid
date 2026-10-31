// ---- fruitraster.wgsl: fruit bodies into the tissue G-buffer ----
// Two quads per fruit, one over the head's bounding sphere and one over the
// stem; each pixel marches the fruit's distance field through that part's
// bounding sphere. Heads a few pixels wide skip their fine detail.

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> fruits: array<Fruit>;
@group(0) @binding(2) var devTex: texture_2d<f32>;
@group(0) @binding(3) var devSamp: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(flat) idx: u32,
  @location(1) @interpolate(flat) part: u32,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
  let fi = ii >> 1u;
  let part = ii & 1u;
  let f = fruits[fi];
  if ((f.i.z & FR_ALIVE) == 0u || f.bound.w <= 0.0) { return o; }
  // (in perspective, what is not well in front of the lens is not drawn)
  let sph = select(f.hb, f.sb, part == 1u);
  let zs = dot(sph.xyz - view.eye.xyz, view.fwd.xyz);
  if (perspective(view) && zs - sph.w < 0.3) { return o; }
  let ppu = pxPerUnit(view, zs - sph.w);
  var mn: vec2f;
  var mx: vec2f;
  if (part == 0u) {
    // screen box of the head's bounding cylinder
    let Y = f.axis.xyz * f.hc.y;
    let X = f.frame.xyz;
    let Z = cross(f.axis.xyz, X);
    mn = vec2f(1e9);
    mx = vec2f(-1e9);
    for (var k = 0u; k < 8u; k++) {
      let sx = select(-1.0, 1.0, (k & 1u) != 0u);
      let sy = select(-1.0, 1.0, (k & 2u) != 0u);
      let sz = select(-1.0, 1.0, (k & 4u) != 0u);
      let q = project(view, f.hb.xyz + X * (sx * f.hc.x) + Y * sy + Z * (sz * f.hc.x)).xy;
      mn = min(mn, q);
      mx = max(mx, q);
    }
    mn -= 1.0;
    mx += 1.0;
  } else {
    if (f.sb.w <= 0.0) { return o; }
    // the stem's control points bound its curve
    let a = project(view, f.base.xyz).xy;
    let b = project(view, f.ctrl.xyz).xy;
    let c = project(view, f.apex.xyz).xy;
    let rp = (max(f.ctrl.w * (1.0 + f.ring.w), f.apex.w) + f.ring.y + f.misc.z) * ppu + 1.0;
    mn = min(a, min(b, c)) - rp;
    mx = max(a, max(b, c)) + rp;
  }
  let corner = vec2f(f32(vi & 1u), f32(vi >> 1u));
  let s = mix(mn, mx, corner);
  o.pos = vec4f(s.x / view.right.w * 2.0 - 1.0, 1.0 - s.y / view.up.w * 2.0, 0.5, 1.0);
  o.idx = fi;
  o.part = part;
  return o;
}

struct FOut {
  @builtin(frag_depth) depth: f32,
  @location(0) geo: vec4f,
  @location(1) id: u32,
}

@fragment
fn fs(in: VOut) -> FOut {
  let f = fruits[in.idx];
  let ray = viewRay(view, in.pos.xy);
  let rd = ray.d;
  let sph = select(f.hb, f.sb, in.part == 1u);
  // (marched from just short of its bound, as tissue.wgsl tests cells: from
  // the eye, metres off, a small one's bound is lost to rounding)
  let t0 = max(dot(sph.xyz - ray.o, rd) - sph.w - 0.01, 0.0);
  let ro = ray.o + rd * t0;
  let oc = ro - sph.xyz;
  let b = dot(oc, rd);
  let h = b * b - (dot(oc, oc) - sph.w * sph.w);
  if (h < 0.0) { discard; }
  let sq = sqrt(h);
  var t1 = -b + sq;
  var t = -b - sq;
  if (in.part == 0u) {
    // the head's bounding cylinder is much tighter than its sphere
    let Y = f.axis.xyz;
    let X = f.frame.xyz;
    let Z = cross(Y, X);
    let o = vec3f(dot(oc, X), dot(oc, Y), dot(oc, Z));
    let dv = vec3f(dot(rd, X), dot(rd, Y), dot(rd, Z));
    let cr = f.hc.x;
    let ch = f.hc.y;
    // slab along the axis
    let iy = 1.0 / select(dv.y, 1e-6, abs(dv.y) < 1e-6);
    let ya = (-ch - o.y) * iy;
    let yb = (ch - o.y) * iy;
    t = max(t, min(ya, yb));
    t1 = min(t1, max(ya, yb));
    // infinite cylinder around it
    let a = dot(dv.xz, dv.xz);
    if (a > 1e-8) {
      let bb = dot(o.xz, dv.xz);
      let hh = bb * bb - a * (dot(o.xz, o.xz) - cr * cr);
      if (hh < 0.0) { discard; }
      let s2 = sqrt(hh);
      t = max(t, (-bb - s2) / a);
      t1 = min(t1, (-bb + s2) / a);
    } else if (dot(o.xz, o.xz) > cr * cr) {
      discard;
    }
    if (t > t1) { discard; }
  }
  let ppu = pxPerUnit(view, dot(sph.xyz - view.eye.xyz, view.fwd.xyz));
  let eps = max(0.0006, 0.25 / ppu);
  let detail = f.head.w * ppu > 10.0;
  var hit = false;
  var info: FruitHit;
  var pr = profileOf(f);
  // sculpted heads take shorter steps
  let steps = select(64, 112, in.part == 0u && devSlot(f) < PAT_SLOTS);
  for (var k = 0; k < steps; k++) {
    info = partEval(f, &pr, ro + rd * t, detail, in.part);
    if (info.d < eps) {
      hit = true;
      break;
    }
    t += info.d * 0.9;
    if (t > t1) { break; }
  }
  if (!hit) { discard; }
  let p = ro + rd * t;
  var o: FOut;
  o.depth = rasterDepth(t0 + t);
  o.geo = vec4f(partNormal(f, &pr, p, max(0.0008, 0.35 / ppu), detail, in.part), t0 + t);
  o.id = packFruitHit(in.idx, info);
  return o;
}
