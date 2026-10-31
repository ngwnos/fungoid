// ---- common.wgsl: shared structs, hashing, noise, sampling ----

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

struct View {
  eye: vec4f,    // xyz eye (orthographic: screen centre, far back); w: pixels per world unit (perspective: at the focal plane)
  right: vec4f,  // w: width px
  up: vec4f,     // w: height px
  fwd: vec4f,    // w: world size of one pixel
  jitter: vec4f, // xy: subpixel offset (px), with the shift of the frame's point on the lens; z: focal length (px), 0: orthographic
  misc: vec4u,   // x: frame, y: body count
}

struct Scene {
  sunDir: vec4f,  // toward the sun; w: angular radius (rad)
  sunCol: vec4f,  // irradiance
  skyZen: vec4f,
  skyHor: vec4f,
  params: vec4f,  // exposure, time, debug view, max bounces
  sel: vec4u,     // selected id+1, second id+1, frame, unused
  den: vec4f,     // history floor while the camera moves, aperture, focus, fine moss
  sim: vec4f,     // history floor while growing, steps per frame, tool overlay (0 none, 1 moisture, 2 food), 1: average every frame (nothing changes)
  cursor: vec4f,  // the brush on the ground: where (xyz), radius (0: none)
  day: vec4f,     // the sun now over the sun the light was baked with (rgb), night (0 day .. 1 night)
}

fn perspective(v: View) -> bool { return v.jitter.z > 0.0; }

// Pixels per world unit at depth z along the view direction.
fn pxPerUnit(v: View, z: f32) -> f32 { return select(v.eye.w, v.jitter.z / max(z, 1e-3), perspective(v)); }

// p -> (px, py, distance along its view ray).
fn project(v: View, p: vec3f) -> vec3f {
  let d = p - v.eye.xyz;
  let z = dot(d, v.fwd.xyz);
  let s = pxPerUnit(v, z);
  return vec3f(
    dot(d, v.right.xyz) * s + 0.5 * v.right.w + v.jitter.x,
    -dot(d, v.up.xyz) * s + 0.5 * v.up.w + v.jitter.y,
    select(z, length(d), perspective(v)));
}

struct Ray {
  o: vec3f,
  d: vec3f,
}

// The view ray through screen position px (pixels, jittered as the frame is).
fn viewRay(v: View, px: vec2f) -> Ray {
  let sx = px.x - v.jitter.x - 0.5 * v.right.w;
  let sy = px.y - v.jitter.y - 0.5 * v.up.w;
  if (perspective(v)) { return Ray(v.eye.xyz, normalize(v.fwd.xyz * v.jitter.z + v.right.xyz * sx - v.up.xyz * sy)); }
  return Ray(v.eye.xyz + (v.right.xyz * sx - v.up.xyz * sy) / v.eye.w, v.fwd.xyz);
}

// Depth-buffer value for a hit at distance t along the view ray.
fn rasterDepth(t: f32) -> f32 { return clamp(t / 4000.0, 0.0, 1.0); }

// ---- hashing ----
fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn hash2(a: u32, b: u32) -> u32 { return pcg(a ^ pcg(b + 0x9e3779b9u)); }
fn hash3(a: u32, b: u32, c: u32) -> u32 { return pcg(a ^ pcg(b ^ pcg(c + 0x85ebca6bu))); }
fn u01(h: u32) -> f32 { return f32(h >> 8u) * (1.0 / 16777216.0); }
fn hashCell2(c: vec2i, seed: u32) -> u32 { return hash3(bitcast<u32>(c.x), bitcast<u32>(c.y), seed); }
fn hashCell3(c: vec3i, seed: u32) -> u32 { return hash3(bitcast<u32>(c.x), bitcast<u32>(c.y), bitcast<u32>(c.z) ^ (seed * 0x27d4eb2du)); }

fn grad2(h: u32) -> vec2f { return vec2f(f32(h & 0xffffu), f32(h >> 16u)) * (2.0 / 65535.0) - 1.0; }
fn grad3(h: u32) -> vec3f {
  return vec3f(f32(h & 0x3ffu), f32((h >> 10u) & 0x3ffu), f32((h >> 20u) & 0x3ffu)) * (2.0 / 1023.0) - 1.0;
}

// ---- gradient noise, roughly [-1, 1] ----
fn gnoise2(p: vec2f, seed: u32) -> f32 {
  let i = vec2i(floor(p));
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let a = dot(grad2(hashCell2(i, seed)), f);
  let b = dot(grad2(hashCell2(i + vec2i(1, 0), seed)), f - vec2f(1.0, 0.0));
  let c = dot(grad2(hashCell2(i + vec2i(0, 1), seed)), f - vec2f(0.0, 1.0));
  let d = dot(grad2(hashCell2(i + vec2i(1, 1), seed)), f - vec2f(1.0, 1.0));
  return 1.6 * mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn gnoise3(p: vec3f, seed: u32) -> f32 {
  let i = vec3i(floor(p));
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  var acc = array<f32, 8>();
  for (var k = 0u; k < 8u; k++) {
    let o = vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32(k >> 2u));
    acc[k] = dot(grad3(hashCell3(i + o, seed)), f - vec3f(o));
  }
  let x0 = mix(mix(acc[0], acc[1], u.x), mix(acc[2], acc[3], u.x), u.y);
  let x1 = mix(mix(acc[4], acc[5], u.x), mix(acc[6], acc[7], u.x), u.y);
  return 1.4 * mix(x0, x1, u.z);
}

fn fbm2(p: vec2f, seed: u32, oct: u32) -> f32 {
  var acc = 0.0;
  var amp = 0.5;
  var q = p;
  for (var i = 0u; i < oct; i++) {
    acc += amp * gnoise2(q, seed + i * 101u);
    q = mat2x2f(1.6, 1.2, -1.2, 1.6) * q;
    amp *= 0.5;
  }
  return acc;
}

fn fbm3(p: vec3f, seed: u32, oct: u32) -> f32 {
  var acc = 0.0;
  var amp = 0.5;
  var q = p;
  for (var i = 0u; i < oct; i++) {
    acc += amp * gnoise3(q, seed + i * 101u);
    q = q * 2.03 + vec3f(1.7, 9.2, 3.1);
    amp *= 0.5;
  }
  return acc;
}

struct Voronoi {
  f1: f32,
  f2: f32,
  id: u32,
  off: vec2f,   // from p to the nearest feature (cell units)
}

fn cellular(p: vec2f, seed: u32, jit: f32) -> Voronoi {
  let i = vec2i(floor(p));
  let f = fract(p);
  var c: Voronoi;
  c.f1 = 9.0;
  c.f2 = 9.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let o = vec2i(x, y);
      let h = hashCell2(i + o, seed);
      let fp = vec2f(o) + 0.5 + (vec2f(u01(h), u01(pcg(h))) - 0.5) * jit;
      let d = fp - f;
      let dd = dot(d, d);
      if (dd < c.f1) {
        c.f2 = c.f1;
        c.f1 = dd;
        c.id = h;
        c.off = d;
      } else if (dd < c.f2) {
        c.f2 = dd;
      }
    }
  }
  c.f1 = sqrt(c.f1);
  c.f2 = sqrt(c.f2);
  return c;
}

// The same in 3D (over the planet: see world.wgsl, a point on the ground's
// sphere, so its cells run straight up).
struct Voronoi3 {
  f1: f32,
  f2: f32,
  id: u32,
  off: vec3f,   // from p to the nearest feature (cell units)
}

fn cellVisit(c: ptr<function, Voronoi3>, i: vec3i, o: vec3i, f: vec3f, seed: u32, jit: f32) {
  let h = hashCell3(i + o, seed);
  let fp = vec3f(o) + 0.5 + (vec3f(u01(h), u01(pcg(h)), u01(pcg(h ^ 0x68e31da4u))) - 0.5) * jit;
  let d = fp - f;
  let dd = dot(d, d);
  if (dd < (*c).f1) {
    (*c).f2 = (*c).f1;
    (*c).f1 = dd;
    (*c).id = h;
    (*c).off = d;
  } else if (dd < (*c).f2) {
    (*c).f2 = dd;
  }
}

// (searching the eight cells nearest p)
fn cellular3(p: vec3f, seed: u32, jit: f32) -> Voronoi3 {
  let i = vec3i(floor(p - 0.5));
  let f = p - vec3f(i);
  var c: Voronoi3;
  c.f1 = 9.0;
  c.f2 = 9.0;
  for (var k = 0; k < 8; k++) { cellVisit(&c, i, vec3i(k & 1, (k >> 1) & 1, k >> 2), f, seed, jit); }
  c.f1 = sqrt(c.f1);
  c.f2 = sqrt(c.f2);
  return c;
}

// ---- SDF helpers ----
fn smin(a: f32, b: f32, k: f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}
fn smax(a: f32, b: f32, k: f32) -> f32 { return -smin(-a, -b, k); }

fn sdBox(p: vec3f, b: vec3f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Inigo Quilez's round cone between a (radius r1) and b (radius r2).
fn sdRoundCone(p: vec3f, a: vec3f, b: vec3f, r1: f32, r2: f32) -> f32 {
  let ba = b - a;
  let l2 = max(dot(ba, ba), 1e-8);
  let rr = r1 - r2;
  let a2 = l2 - rr * rr;
  let il2 = 1.0 / l2;
  let pa = p - a;
  let y = dot(pa, ba);
  let z = y - l2;
  let xv = pa * l2 - ba * y;
  let x2 = dot(xv, xv);
  let y2 = y * y * l2;
  let z2 = z * z * l2;
  let k = sign(rr) * rr * rr * x2;
  if (sign(z) * a2 * z2 > k) { return sqrt(x2 + z2) * il2 - r2; }
  if (sign(y) * a2 * y2 < k) { return sqrt(x2 + y2) * il2 - r1; }
  return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

fn bad(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) >= 0x7f800000u; }
// Replace NaN / inf with zero so one bad sample cannot poison the history.
fn finite3(c: vec3f) -> vec3f {
  return select(c, vec3f(0.0), bad(c.x) || bad(c.y) || bad(c.z));
}

fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

// Subtractive pigment model: flesh reflectance filtered by pigment absorbance.
fn pigment(flesh: vec3f, mel: f32, car: f32, red: f32, blue: f32) -> vec3f {
  let a = mel * vec3f(0.30, 0.55, 0.95)
        + car * vec3f(0.0, 0.2, 1.25)
        + red * vec3f(0.05, 1.25, 0.8)
        + blue * vec3f(1.15, 0.5, 0.05);
  return flesh * exp(-a);
}

// ---- random numbers & sampling ----
var<private> rngState: u32;

fn rngInit(px: vec2u, frame: u32) { rngState = hash3(px.x, px.y, frame); }
fn rnd() -> f32 {
  rngState = rngState * 747796405u + 2891336453u;
  let w = ((rngState >> ((rngState >> 28u) + 4u)) ^ rngState) * 277803737u;
  return f32((w >> 22u) ^ w) * (1.0 / 4294967296.0);
}

fn basisOf(n: vec3f) -> mat3x3f {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;
  let t = vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
  let bt = vec3f(b, s + n.y * n.y * a, -n.y);
  return mat3x3f(t, bt, n);
}

fn cosineSample(n: vec3f) -> vec3f {
  let u = rnd();
  let v = rnd();
  let r = sqrt(u);
  let phi = TAU * v;
  return normalize(basisOf(n) * vec3f(r * cos(phi), r * sin(phi), sqrt(max(1.0 - u, 0.0))));
}

fn coneSample(dir: vec3f, cosMax: f32) -> vec3f {
  let u = rnd();
  let v = rnd();
  let ct = 1.0 - u * (1.0 - cosMax);
  let st = sqrt(max(1.0 - ct * ct, 0.0));
  let phi = TAU * v;
  return normalize(basisOf(dir) * vec3f(st * cos(phi), st * sin(phi), ct));
}

fn hgSample(dir: vec3f, g: f32) -> vec3f {
  let u = rnd();
  let v = rnd();
  var ct: f32;
  if (abs(g) < 1e-3) {
    ct = 1.0 - 2.0 * u;
  } else {
    let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
    ct = (1.0 + g * g - s * s) / (2.0 * g);
  }
  let st = sqrt(max(1.0 - ct * ct, 0.0));
  let phi = TAU * v;
  return normalize(basisOf(dir) * vec3f(st * cos(phi), st * sin(phi), ct));
}

fn hgPhase(ct: f32, g: f32) -> f32 {
  let d = 1.0 + g * g - 2.0 * g * ct;
  return (1.0 - g * g) / (4.0 * PI * d * sqrt(d));
}
