// ---- fruitdev.wgsl: reading a head's developed field ----
// Requires `devTex` (the atlas the simulation writes) and `devSamp` (linear).
// The field covers a disk: the head profile from the crown (s = 0) out to the
// margin (s = 1), and the angle around the axis.

const DEV_ATLAS: vec2f = vec2f(f32(PAT_COLS * PAT_RES), f32((PAT_SLOTS / PAT_COLS) * PAT_RES));

// Every caller sees the same surface (shadows and occlusion included), so
// there is one filter: a cubic B-spline from four bilinear taps, smooth
// enough for normals.
fn devAt(f: Fruit, s: f32, phi: f32) -> f32 {
  let slot = devSlot(f);
  let N = f.feat.x;
  let q = vec2f(N * 0.5) + vec2f(cos(phi), sin(phi)) * min(s, 1.0) * (N * 0.5 - 1.0);
  let o = vec2f(f32(slot % PAT_COLS), f32(slot / PAT_COLS)) * f32(PAT_RES);
  let lo = vec2f(0.5);
  let hi = vec2f(N - 0.5);
  let tc = q - 0.5;
  let i = floor(tc);
  let t = tc - i;
  let t2 = t * t;
  let t3 = t2 * t;
  let w0 = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0;
  let w1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0;
  let w2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0;
  let w3 = t3 / 6.0;
  let g0 = w0 + w1;
  let g1 = w2 + w3;
  let h0 = clamp(i - 0.5 + w1 / g0, lo, hi);
  let h1 = clamp(i + 1.5 + w3 / g1, lo, hi);
  let a = textureSampleLevel(devTex, devSamp, (o + vec2f(h0.x, h0.y)) / DEV_ATLAS, 0.0).r;
  let b = textureSampleLevel(devTex, devSamp, (o + vec2f(h1.x, h0.y)) / DEV_ATLAS, 0.0).r;
  let c = textureSampleLevel(devTex, devSamp, (o + vec2f(h0.x, h1.y)) / DEV_ATLAS, 0.0).r;
  let d = textureSampleLevel(devTex, devSamp, (o + vec2f(h1.x, h1.y)) / DEV_ATLAS, 0.0).r;
  return g0.y * (g0.x * a + g1.x * b) + g1.y * (g0.x * c + g1.x * d);
}
