// ---- glow.wgsl: the light that glowing tissue and fruit bodies cast ----
// Whenever the garden has grown, what glows is added up on an 8 mm grid over
// the box (fixed point, so the sum does not depend on the order threads add
// in), then blurred a few centimetres each way into a light field that the
// renderer adds to the light on every surface: bioluminescence lights the
// moss, soil and tissue around it. (Rock and wood do not block it.)

@group(0) @binding(0) var<storage, read> pheno: array<f32>;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
@group(0) @binding(2) var<storage, read> counters: array<u32>;
@group(0) @binding(3) var<storage, read> fruits: array<Fruit>;
@group(0) @binding(4) var<storage, read_write> acc: array<atomic<u32>>;   // rgb per voxel
@group(0) @binding(5) var<storage, read_write> blur: array<vec4f>;        // two passes' worth
@group(0) @binding(6) var glowOut: texture_storage_3d<rgba16float, write>;

const GLOW_FP: f32 = 65536.0;

// (fruit.wgsl asks for the developed surface; the light does not need it)
fn devAt(f: Fruit, s: f32, phi: f32) -> f32 { return 0.0; }
const TAPS: i32 = 4;

fn glowVoxel(p: vec3f) -> u32 {
  let g = clamp(vec3i(floor((p - BOX_MIN) / GLOW_CELL)), vec3i(0), vec3i(GLOW_RES) - 1);
  return (u32(g.z) * GLOW_RES.y + u32(g.y)) * GLOW_RES.x + u32(g.x);
}

fn addGlow(p: vec3f, e: vec3f) {
  let v = glowVoxel(p) * 3u;
  for (var k = 0u; k < 3u; k++) {
    let q = u32(e[k] * GLOW_FP);
    if (q > 0u) { atomicAdd(&acc[v + k], q); }
  }
}

// Glowing cells: as bright as they are and as much surface as they have.
@compute @workgroup_size(256)
fn glowCells(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= counters[4]) { return; }
  let c = cells[i];
  if ((c.i.z & F_DEAD) != 0u) { return; }
  let org = c.i.y;
  let e = cellLight(org, phaseOf(c.i.z), c.i.z, f32(counters[5] - c.i.w));
  if (e.x + e.y + e.z <= 0.0) { return; }
  addGlow(c.p.xyz, e * c.p.w * CELL_LEN * 3.0);
}

// Glowing fruit bodies: from the head, by its size, once it has matured.
@compute @workgroup_size(64)
fn glowFruits(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= MAX_FRUITS) { return; }
  let f = fruits[i];
  if ((f.i.z & FR_ALIVE) == 0u || f.bound.w <= 0.0) { return; }
  let org = f.i.x;
  let g = PG(org, G_BODY_GLOW);
  if (g <= 0.0) { return; }
  let R = f.head.w;
  addGlow(f.hb.xyz, glowColour(org) * g * smoothstep(0.3, 1.0, f.dir.w) * fruitFlash(f) * R * R * 3.0);
}

fn gauss(k: i32) -> f32 { return exp(-f32(k * k) / (2.0 * 1.6 * 1.6)) / 4.01; }

fn at(g: vec3i) -> u32 { return (u32(g.z) * GLOW_RES.y + u32(g.y)) * GLOW_RES.x + u32(g.x); }

// The blur, one axis a pass: the sums along x, then y (emptying the sums for
// the next time), then z into the light field.
@compute @workgroup_size(4, 4, 4)
fn glowBlurX(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= GLOW_RES)) { return; }
  var s = vec3f(0.0);
  for (var k = -TAPS; k <= TAPS; k++) {
    let x = i32(gid.x) + k;
    if (x < 0 || x >= i32(GLOW_RES.x)) { continue; }
    let v = at(vec3i(x, i32(gid.y), i32(gid.z))) * 3u;
    s += vec3f(f32(atomicLoad(&acc[v])), f32(atomicLoad(&acc[v + 1u])), f32(atomicLoad(&acc[v + 2u]))) * gauss(k);
  }
  blur[at(vec3i(gid))] = vec4f(s / GLOW_FP, 0.0);
}

@compute @workgroup_size(4, 4, 4)
fn glowBlurY(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= GLOW_RES)) { return; }
  var s = vec3f(0.0);
  for (var k = -TAPS; k <= TAPS; k++) {
    let y = i32(gid.y) + k;
    if (y < 0 || y >= i32(GLOW_RES.y)) { continue; }
    s += blur[at(vec3i(i32(gid.x), y, i32(gid.z)))].rgb * gauss(k);
  }
  let v = at(vec3i(gid));
  blur[GLOW_N + v] = vec4f(s, 0.0);
  for (var k = 0u; k < 3u; k++) { atomicStore(&acc[v * 3u + k], 0u); }
}

@compute @workgroup_size(4, 4, 4)
fn glowBlurZ(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= GLOW_RES)) { return; }
  var s = vec3f(0.0);
  for (var k = -TAPS; k <= TAPS; k++) {
    let z = i32(gid.z) + k;
    if (z < 0 || z >= i32(GLOW_RES.z)) { continue; }
    s += blur[GLOW_N + at(vec3i(i32(gid.x), i32(gid.y), z))].rgb * gauss(k);
  }
  textureStore(glowOut, gid, vec4f(s, 1.0));
}
