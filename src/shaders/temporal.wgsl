// ---- temporal.wgsl: temporal antialiasing ----
// Shading is deterministic, so history only has to smooth subpixel jitter:
// reproject, clamp to the current neighbourhood, blend. Most fungal fibres
// are thinner than a pixel and only resolve as an average over many jittered
// frames, so settled tissue keeps a long history even while the garden grows.

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> prev: View;
@group(0) @binding(2) var<uniform> scene: Scene;
@group(0) @binding(3) var curRad: texture_2d<f32>;
@group(0) @binding(4) var curGeo: texture_2d<f32>;
@group(0) @binding(5) var prevGeo: texture_2d<f32>;
@group(0) @binding(6) var prevCol: texture_2d<f32>;
@group(0) @binding(7) var prevMom: texture_2d<f32>;
@group(0) @binding(8) var outCol: texture_storage_2d<rgba32float, write>;
@group(0) @binding(9) var outMom: texture_storage_2d<rgba32float, write>;
@group(0) @binding(10) var curAlb: texture_2d<f32>;   // w: 0 scenery, 1 settled tissue, 2 new or moving tissue

fn toYCoCg(c: vec3f) -> vec3f {
  return vec3f(dot(c, vec3f(0.25, 0.5, 0.25)), dot(c, vec3f(0.5, 0.0, -0.5)), dot(c, vec3f(-0.25, 0.5, -0.25)));
}
fn fromYCoCg(c: vec3f) -> vec3f { return vec3f(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }

// Thin fibres cover a pixel in some jittered frames and miss it in others, so
// the surface stored for one history texel flips between fibre and ground.
// History is valid when anything around it last frame was at this depth.
fn sameSurface(q: vec2i, z: f32, sky: bool, dims: vec2u) -> bool {
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let s = clamp(q + vec2i(x, y), vec2i(0), vec2i(dims) - 1);
      let w = textureLoad(prevGeo, s, 0).w;
      if (select(abs(w - z) < 0.3, w > 1e5, sky)) { return true; }
    }
  }
  return false;
}

fn catmullRom(t: f32) -> vec4f {
  return vec4f(t * (-0.5 + t * (1.0 - 0.5 * t)),
               1.0 + t * t * (-2.5 + 1.5 * t),
               t * (0.5 + t * (2.0 - 1.5 * t)),
               t * t * (-0.5 + 0.5 * t));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = vec2u(u32(view.right.w), u32(view.up.w));
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let p = vec2i(gid.xy);
  let c = textureLoad(curRad, p, 0).rgb;
  let g = textureLoad(curGeo, p, 0);

  // reproject pixel centres without jitter, so a still camera reads history
  // texel-for-texel instead of resampling (and blurring) it every frame.
  // The nearest depth around the pixel carries a fibre's history even in
  // frames where this pixel's sample missed it and hit the ground behind.
  var z = g.w;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let q = clamp(p + vec2i(x, y), vec2i(0), vec2i(dims) - 1);
      z = min(z, textureLoad(curGeo, q, 0).w);
    }
  }
  let ray = viewRay(view, vec2f(gid.xy) + 0.5 + view.jitter.xy);
  let world = ray.o + ray.d * z;
  var pp = project(prev, world) - vec3f(prev.jitter.xy, 0.0);
  // nothing changes from frame to frame (a held frame of a recording, drawn
  // many times through different points of the lens): history is the plain
  // average of what this pixel saw
  let average = scene.sim.w > 0.5 && scene.den.x == 0.0;
  if (average) { pp = vec3f(vec2f(gid.xy) + 0.5, pp.z); }

  var sumW = 0.0;
  var hc = vec3f(0.0);
  var hl = 0.0;
  let base = vec2i(floor(pp.xy - 0.5));
  let f = fract(pp.xy - 0.5);
  let sky = g.w > 1e5;
  // a still camera reads history texel-for-texel, so nothing can be disoccluded
  let still = scene.den.x == 0.0;
  var valid = 0;
  for (var k = 0; k < 4; k++) {
    let o = vec2i(k & 1, k >> 1);
    let q = base + o;
    if (any(q < vec2i(0)) || any(q >= vec2i(dims))) { continue; }
    if (!still && !sameSurface(q, pp.z, sky, dims)) { continue; }
    let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
    hc += textureLoad(prevCol, q, 0).rgb * w;
    hl += textureLoad(prevMom, q, 0).x * w;
    sumW += w;
    valid++;
  }
  // bilinear history softens a little more every frame the camera moves;
  // Catmull-Rom keeps it sharp where all four texels are the same surface
  if (valid == 4 && !still) {
    let wx = catmullRom(f.x);
    let wy = catmullRom(f.y);
    var cr = vec3f(0.0);
    for (var y = 0; y < 4; y++) {
      for (var x = 0; x < 4; x++) {
        let q = clamp(base + vec2i(x - 1, y - 1), vec2i(0), vec2i(dims) - 1);
        cr += textureLoad(prevCol, q, 0).rgb * wx[x] * wy[y];
      }
    }
    hc = max(cr, vec3f(0.0)) * sumW;
  }

  var len = 1.0;
  var col = c;
  if (sumW > 0.05 && !bad(hc.x + hc.y + hc.z)) {
    hc /= sumW;
    len = min(hl / sumW + 1.0, 256.0);
    let kind = textureLoad(curAlb, p, 0).w;
    // new or moving tissue restarts the average instead of fading in
    if (kind > 1.5) { len = min(len, 2.0); }
    // keep history within what this frame's neighbourhood allows
    var m1 = vec3f(0.0);
    var m2 = vec3f(0.0);
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let q = clamp(p + vec2i(x, y), vec2i(0), vec2i(dims) - 1);
        let s = toYCoCg(textureLoad(curRad, q, 0).rgb);
        m1 += s;
        m2 += s * s;
      }
    }
    let mean = m1 / 9.0;
    let sd = sqrt(max(m2 / 9.0 - mean * mean, vec3f(0.0)));
    let k = select(1.25, 2.5, still);
    if (!average) { hc = fromYCoCg(clamp(toYCoCg(hc), mean - sd * k, mean + sd * k)); }
    // scenery under growing mycelium brightens, so it keeps a short history;
    // settled tissue does not change
    let simFloor = select(scene.sim.x, 0.0, kind > 0.5);
    let a = max(1.0 / len, max(scene.den.x, simFloor));
    col = mix(hc, c, a);
    // (what the history now counts for: a fresh start counts one frame)
    len = 1.0 / a;
  }
  textureStore(outCol, gid.xy, vec4f(col, 1.0));
  textureStore(outMom, gid.xy, vec4f(len, 0.0, 0.0, 0.0));
}
