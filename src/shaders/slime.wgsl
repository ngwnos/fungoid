// ---- slime.wgsl: the slime mould's grids (simulation and shading) ----
// A slime mould crawls over the ground as a swarm of agents, each laying a
// trail the others follow (Physarum). What is seen is drawn from its trail
// grid, which covers the planet.

// (SLIME_RES, SLIME_N: dims.ts)
// its sporangia are fruit bodies with a program of their own, in the last
// organism slot
const SLIME_ORG: u32 = 63u;

// The grid unfolds the planet the way the ground map does (world.wgsl
// octEncode, y the pole), but so that every cell is the same area of
// ground: a direction to [-1, 1]^2, and back.
fn slimeEncode(n: vec3f) -> vec2f {
  let r = length(n.xz) * inverseSqrt(1.0 + abs(n.y));
  let a = atan2(abs(n.z), abs(n.x)) * (2.0 / PI);
  var e = vec2f(r * (1.0 - a), r * a);
  if (n.y < 0.0) { e = 1.0 - e.yx; }
  return e * select(vec2f(-1.0), vec2f(1.0), n.xz >= vec2f(0.0));
}
fn slimeDecode(e: vec2f) -> vec3f {
  var a = abs(e);
  let south = a.x + a.y > 1.0;
  if (south) { a = 1.0 - a.yx; }
  let r = a.x + a.y;
  let phi = select(0.0, a.y / r, r > 0.0) * (PI / 2.0);
  let xz = vec2f(cos(phi), sin(phi)) * r * sqrt(max(2.0 - r * r, 0.0)) * select(vec2f(-1.0), vec2f(1.0), e >= vec2f(0.0));
  return vec3f(xz.x, select(1.0 - r * r, r * r - 1.0, south), xz.y);
}
fn slimeUV(p: vec3f) -> vec2f { return slimeEncode(upAt(p)) * 0.5 + 0.5; }
