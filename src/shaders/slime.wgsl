// ---- slime.wgsl: the slime mould's grids (simulation and shading) ----
// A slime mould crawls over the ground as a swarm of agents, each laying a
// trail the others follow (Physarum). What is seen is drawn from its trail
// grid, which covers the ground (the shape's slimeEncode/slimeDecode).

// (SLIME_RES, SLIME_N: dims.ts)
// its sporangia are fruit bodies with a program of their own, in the last
// organism slot
const SLIME_ORG: u32 = 63u;

fn slimeUV(p: vec3f) -> vec2f { return slimeEncode(keyOf(p)) * 0.5 + 0.5; }
