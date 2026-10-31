// ---- present.wgsl: remodulate, expose, tone map, selection outline ----

@group(0) @binding(0) var illum: texture_2d<f32>;
@group(0) @binding(1) var alb: texture_2d<f32>;
@group(0) @binding(2) var raw: texture_2d<f32>;
@group(0) @binding(3) var geo: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> ids: array<u32>;
@group(0) @binding(5) var<uniform> scene: Scene;
@group(0) @binding(6) var<uniform> view: View;
@group(0) @binding(7) var fieldTex: texture_3d<f32>;   // airborne spores, settled spores, food, how damp
@group(0) @binding(8) var linSamp: sampler;

struct VSOut {
  @builtin(position) pos: vec4f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
  let y = f32(vi & 2u) * 2.0 - 1.0;
  var o: VSOut;
  o.pos = vec4f(x, y, 0.0, 1.0);
  return o;
}

fn agxContrast(x: vec3f) -> vec3f {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn agx(c: vec3f) -> vec3f {
  let m = mat3x3f(
    0.842479062253094, 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772, 0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  let minEv = -12.47393;
  let maxEv = 4.026069;
  var v = m * max(c, vec3f(1e-10));
  v = clamp(log2(v), vec3f(minEv), vec3f(maxEv));
  v = (v - minEv) / (maxEv - minEv);
  return agxContrast(v);
}

fn look(c: vec3f) -> vec3f {
  let l = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  return pow(max(mix(vec3f(l), c, 1.1), vec3f(0.0)), vec3f(1.04));
}

fn idAt(p: vec2i, W: i32, H: i32) -> u32 {
  let q = clamp(p, vec2i(0), vec2i(W - 1, H - 1));
  return ids[u32(q.y * W + q.x)];
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let p = vec2i(in.pos.xy);
  let dims = vec2i(textureDimensions(illum));
  let debug = u32(scene.params.z);
  let a = textureLoad(alb, p, 0).rgb;
  var c = textureLoad(illum, p, 0).rgb;
  if (debug == 1u) { c = a; }
  if (debug == 2u) { return vec4f(textureLoad(geo, p, 0).xyz * 0.5 + 0.5, 1.0); }
  if (debug == 3u) { c = textureLoad(raw, p, 0).rgb; }
  c *= scene.params.x;
  var o = look(agx(c));

  let id = idAt(p, dims.x, dims.y);
  if (scene.sel.x != 0u || scene.sel.y != 0u) {
    var ea = 0.0;
    var eb = 0.0;
    for (var k = 0; k < 4; k++) {
      let d = select(select(vec2i(0, 1), vec2i(0, -1), k == 2), select(vec2i(1, 0), vec2i(-1, 0), k == 0), k < 2);
      let n = idAt(p + d, dims.x, dims.y);
      if ((id == scene.sel.x) != (n == scene.sel.x)) { ea = 1.0; }
      if ((id == scene.sel.y) != (n == scene.sel.y)) { eb = 1.0; }
    }
    if (scene.sel.x != 0u) { o = mix(o, vec3f(1.0, 0.82, 0.5), ea * 0.35); }
    if (scene.sel.y != 0u) { o = mix(o, vec3f(0.55, 0.78, 1.0), eb * 0.35); }
  }

  // tools: what the brush works on, shown over the ground, and the brush
  let mode = u32(scene.sim.z);
  if (mode != 0u || scene.cursor.w > 0.0) {
    let g = textureLoad(geo, p, 0);
    if (g.w < 1e5) {
      let ppu = view.eye.w;
      let ray = viewRay(view, in.pos.xy + view.jitter.xy);
      let ws = ray.o + ray.d * g.w;
      let q = ws - g.xyz * 0.15;
      if (mode != 0u && id == 0u && all(q > BOX_MIN) && all(q < BOX_MAX)) {
        // moisture: dry orange .. wet blue; food: none grey-violet .. rich amber
        // (untinted in between)
        let f = textureSampleLevel(fieldTex, linSamp, volUVW(q), 0.0);
        var x = f.w * 2.0 - 1.0;
        var tint = select(vec3f(1.0, 0.55, 0.15), vec3f(0.2, 0.5, 1.0), x > 0.0);
        if (mode == 2u) {
          x = clamp(f.z - 1.0, -1.0, 1.0);
          tint = select(vec3f(0.45, 0.42, 0.6), vec3f(1.0, 0.65, 0.1), x > 0.0);
        }
        let lum = dot(o, vec3f(0.3, 0.59, 0.11));
        o = mix(o, tint * (0.25 + 1.2 * lum), 0.15 + 0.55 * abs(x));
      }
      if (scene.cursor.w > 0.0) {
        let ring = 1.0 - smoothstep(0.0, 1.5 / ppu, abs(length(ws - scene.cursor.xyz) - scene.cursor.w));
        o = mix(o, vec3f(1.0), ring * 0.8);
      }
    }
  }

  let uv = in.pos.xy / vec2f(dims);
  let q = uv - 0.5;
  o *= 1.0 - 0.28 * smoothstep(0.2, 0.9, dot(q, q) * 2.2);
  let gr = u01(hash3(u32(in.pos.x), u32(in.pos.y), scene.sel.z)) - 0.5;
  o += gr * 0.014;
  return vec4f(o, 1.0);
}
