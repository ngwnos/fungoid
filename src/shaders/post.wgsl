// ---- post.wgsl: remodulate and apply depth of field ----
// Gather-based bokeh: each neighbour contributes if its circle of confusion
// reaches this pixel. Blurred background never spreads over a sharper
// foreground; blurred foreground does spread over the background.

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> scene: Scene;
@group(0) @binding(2) var illum: texture_2d<f32>;
@group(0) @binding(3) var geo: texture_2d<f32>;
@group(0) @binding(4) var outCol: texture_storage_2d<rgba16float, write>;

const TAPS: i32 = 24;
const MAX_COC: f32 = 12.0;

fn colorAt(q: vec2i) -> vec3f { return textureLoad(illum, q, 0).rgb; }

fn coc(d: f32) -> f32 {
  // (space, behind it all, stays sharp: its stars are points)
  if (d > 1e5) { return 0.0; }
  let a = scene.den.y;      // aperture (orthographic: blur per cm of defocus; perspective: the lens's radius)
  let f = scene.den.z;      // focus depth
  // through a lens, frames averaged over its points do the blurring; this
  // only closes the gaps between them (a share of the blur, view.jitter.w)
  if (perspective(view)) { return min(view.jitter.w * a * view.jitter.z * abs(1.0 / f - 1.0 / max(d, 0.1)), MAX_COC); }
  return min(a * abs(min(d, f + 400.0) - f) * view.eye.w, MAX_COC);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = vec2i(i32(view.right.w), i32(view.up.w));
  let p = vec2i(gid.xy);
  if (p.x >= dims.x || p.y >= dims.y) { return; }
  let c0 = colorAt(p);
  let d0 = textureLoad(geo, p, 0).w;
  let coc0 = coc(d0);
  if (scene.den.y <= 0.0 || (perspective(view) && view.jitter.w <= 0.0)) {
    textureStore(outCol, p, vec4f(c0, 1.0));
    return;
  }
  // per-pixel rotation of a golden-angle spiral (stable from frame to frame)
  let rot = fract(52.9829189 * fract(dot(vec2f(p), vec2f(0.06711056, 0.00583715)))) * TAU;
  var sum = c0;
  var wsum = 1.0;
  for (var i = 1; i <= TAPS; i++) {
    let r = sqrt(f32(i) / f32(TAPS)) * MAX_COC;
    let a = f32(i) * 2.39996323 + rot;
    let q = p + vec2i(round(vec2f(cos(a), sin(a)) * r));
    if (any(q < vec2i(0)) || any(q >= dims)) { continue; }
    let dq = textureLoad(geo, q, 0).w;
    var cq = coc(dq);
    if (dq > d0) { cq = min(cq, coc0); }
    let w = clamp(cq - r + 1.0, 0.0, 1.0) / max(cq * cq, 1.0);
    sum += colorAt(q) * w;
    wsum += w;
  }
  let w0 = 1.0 / max(coc0 * coc0, 1.0);
  let col = (sum - c0 + c0 * w0) / (wsum - 1.0 + w0);
  textureStore(outCol, p, vec4f(col, 1.0));
}
