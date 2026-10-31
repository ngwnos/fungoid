// ---- downsample.wgsl: a supersampled frame down to the video's size ----
// Each output pixel is the average of the f x f pixels it covers, in linear
// light (the frame is display-encoded).

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;

fn toLinear(c: vec3f) -> vec3f { return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045)); }
fn toDisplay(c: vec3f) -> vec3f { return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308)); }

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let d = textureDimensions(dst);
  if (any(gid.xy >= d)) { return; }
  let f = textureDimensions(src).x / d.x;
  var acc = vec3f(0.0);
  for (var y = 0u; y < f; y++) {
    for (var x = 0u; x < f; x++) {
      acc += toLinear(textureLoad(src, gid.xy * f + vec2u(x, y), 0).rgb);
    }
  }
  textureStore(dst, gid.xy, vec4f(toDisplay(acc / f32(f * f)), 1.0));
}
