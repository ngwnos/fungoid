// ---- shape_planet.wgsl: the world's shape, a small planet ----
// A ball of ground about PLANET_R from the centre, floating in space; up is
// away from the centre. Everything else (the simulation, the bakes, the
// light, the weather) asks the shape about where things are through these;
// shape_box.wgsl answers for the terrarium.
//
// In sections (gpu/shaders.ts): pure goes everywhere; ground where the
// ground map is bound; bake into the ground bake; sim into the simulation;
// shade into the renderer.

//== pure

// up, here
fn upAt(p: vec3f) -> vec3f { return p / max(length(p), 1e-4); }
// Where things stand: a point's key (where it is over the ground: a
// direction), its level (how far out: its distance from the centre), and
// the point of a key at a level.
fn keyOf(p: vec3f) -> vec3f { return upAt(p); }
fn levelOf(p: vec3f) -> f32 { return length(p); }
fn atLevel(k: vec3f, l: f32) -> vec3f { return k * l; }
// where on the ground p is (up: upAt(p)): its direction, at the ground's
// radius (patterns over the ground are of this)
fn groundCoord(p: vec3f, up: vec3f) -> vec3f { return up * PLANET_R; }
// (the ground is round: some planet expressions stay inline at their call
// sites, the flat terrarium's under this, so the planet's arithmetic is
// exactly what it was)
const FLAT: bool = false;
// where in the ground map (bake_ground.wgsl) a key is
fn groundUV(dir: vec3f) -> vec2f { return octEncode(dir) * 0.5 + 0.5; }
// a frame along the ground at key k (its third axis up)
fn keyFrame(k: vec3f) -> mat3x3f { return basisOf(k); }

// within the shell life keeps to; keeping to it
fn inLife(p: vec3f) -> bool {
  let r2 = dot(p, p);
  return r2 > LIFE_IN * LIFE_IN && r2 < LIFE_OUT * LIFE_OUT;
}
fn clampLife(np: vec3f) -> vec3f {
  let rn = length(np);
  return np * (clamp(rn, LIFE_IN, LIFE_OUT) / max(rn, 1e-4));
}
// (beyond the shell life keeps to and a voxel or two more, nothing in the
// fields ever changes; or past its outside)
fn fieldsIdle(p: vec3f) -> bool {
  let r2 = dot(p, p);
  return r2 < (LIFE_IN - 2.0 * F4) * (LIFE_IN - 2.0 * F4) || r2 > (LIFE_OUT + 2.0 * F4) * (LIFE_OUT + 2.0 * F4);
}
fn pastLife(q: vec3f) -> bool { return dot(q, q) > LIFE_OUT * LIFE_OUT; }
// where something dropped from high above p (falling along `down`) starts
fn dropStart(p: vec3f, down: vec3f) -> vec3f { return -down * LIFE_OUT; }

// Winds round the planet's axis, in bands: one way near the equator, the
// other a way off it, and again nearer the poles.
fn wind(p: vec3f) -> vec3f {
  let up = upAt(p);
  let east = cross(vec3f(0.0, 1.0, 0.0), up);
  return east * 0.008 * cos(3.0 * asin(clamp(up.y, -1.0, 1.0)));
}

// The air: where a ray enters and leaves it (enter > leave: it misses), and
// whether a point has left it (and what light keeps going through there).
fn airHit(ro: vec3f, rd: vec3f) -> vec2f { return sphereHit(ro, rd, SKY_R); }
fn inAir(p: vec3f) -> bool { return dot(p, p) < SKY_R * SKY_R; }
fn leftAir(q: vec3f) -> bool { return dot(q, q) > SKY_R * SKY_R; }
fn exitTint(q: vec3f) -> f32 { return 1.0; }
// whether the sun towards L is hidden from p by the planet itself (cheaply:
// the sphere well inside its ground)
fn sunBlocked(p: vec3f, L: vec3f) -> bool {
  let core = PLANET_R - 3.0;
  let hit = sphereHit(p, L, core);
  return !(dot(p, p) > core * core && !(hit.x <= hit.y && hit.y > 0.0));
}

// Moss: low ground is damp; there are no walls to keep it off
fn mossDamp(p: vec3f) -> f32 { return smoothstep(1.5, -1.5, length(p) - PLANET_R); }
fn mossWall(p: vec3f) -> f32 { return 1.0; }

// The slime mould's grid unfolds the planet the way the ground map does
// (octEncode, y the pole), but so that every cell is the same area of
// ground: a key to [-1, 1]^2, and back.
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
// A cell next to this one (dx, dz at most 1 apart), over the fold at the
// grid's edges: the halves of each edge meet each other, mirrored.
fn slimeNext(c: vec2i) -> vec2i {
  let r = i32(SLIME_RES.x);
  var q = c;
  if (q.x < 0) { q = vec2i(-q.x - 1, r - 1 - q.y); }
  else if (q.x >= r) { q = vec2i(2 * r - 1 - q.x, r - 1 - q.y); }
  if (q.y < 0) { q = vec2i(r - 1 - q.x, -q.y - 1); }
  else if (q.y >= r) { q = vec2i(r - 1 - q.x, 2 * r - 1 - q.y); }
  return q;
}
// on from key n along a, s of the way (s: radians)
fn slimeAhead(n: vec3f, a: vec3f, s: f32) -> vec3f { return normalize(n + a * s); }
// how far apart two keys are (radians), and which way from n is toward m
fn slimeApart(n: vec3f, m: vec3f) -> f32 { return acos(clamp(dot(n, m), -1.0, 1.0)); }
fn slimeToward(n: vec3f, m: vec3f) -> vec3f { return m; }
// (no walls: it goes anywhere)
fn slimeInside(m: vec3f) -> bool { return true; }
fn slimeClampIn(m: vec3f) -> vec3f { return m; }
// its middle, from its agents' keys summed (as SL_FIX fixed point), n of them
fn slimeMiddle(sum: vec3f, n: u32) -> vec3f { return normalize(sum); }
// where a slime mould set down by hand hatches (renderer.addSlime)
fn slimeAtKey(e: vec2f) -> vec3f { return octDecode(e); }
const SL_FIX: f32 = 4096.0;        // the fixed point its agents' keys are summed in
const COL_CM: f32 = 0.46;          // a column's width (about: 4 pi PLANET_R^2 / COLS)
const SLIME_R0: f32 = PLANET_R;    // cm per unit of key near the ground

//== ground

// how far from the centre the ground is, in p's direction
fn groundR(p: vec3f) -> f32 { return PLANET_R + textureSampleLevel(ground, linSamp, groundUV(upAt(p)), 0.0).x; }
// how far p is above the ground (along up; < 0 under it)
fn soilDist(p: vec3f) -> f32 {
  // (the planet's heart is all soil)
  let r = length(p);
  var dSoil = -8.0;
  if (r > 1.0) {
    let g = textureSampleLevel(ground, linSamp, groundUV(p / r), 0.0);
    dSoil = (r - PLANET_R - g.x) * g.y;
  }
  return dSoil;
}
// the slime mould's cm per radian at key n
fn slimeRadius(n: vec3f) -> f32 { return groundR(n); }

//== bake

// the ground's height above PLANET_R in direction `dir`
fn terrainHeight(dir: vec3f) -> f32 {
  let tr = world.terrain;
  let q = dir * PLANET_R;
  var h = fbm3(dir * 1.3, tr.seeds.x, 3u) * tr.amp.x;
  h += fbm3(q * 0.11, tr.seeds.y, 4u) * tr.amp.y;
  h += fbm3(q * 0.48, tr.seeds.z, 3u) * tr.amp.z;
  for (var i = 0u; i < world.counts.z; i++) {
    let b = tr.bumps[i];
    let d = q - b.xyz;
    h += tr.heights[i / 4u][i % 4u] * exp(-dot(d, d) / (b.w * b.w));
  }
  return h;
}
fn groundRadius(dir: vec3f) -> f32 { return PLANET_R + terrainHeight(dir); }
// a rock's key, from where it was set (Rock.a)
fn restKey(a: vec3f) -> vec3f { return normalize(a); }
// a root's control point (where, as world.ts sets it: PLANET_R plus its
// height above the ground, in its direction) on the ground as it is
fn rootPoint(c: vec4f) -> vec4f {
  let r = length(c.xyz);
  let dir = c.xyz / r;
  return vec4f(dir * (groundRadius(dir) + r - PLANET_R), c.w);
}
// up where a branch rests on a rock (its point there)
fn restUp(at: vec3f) -> vec3f { return normalize(at); }
// the ground map's texel: its key, and the step along the ground (in the
// key's frame) its slope is measured over
fn groundTexelKey(gid: vec2u) -> vec3f { return octDecode((vec2f(gid) + 0.5) / vec2f(GROUND_RES) * 2.0 - 1.0); }
fn groundSlopeStep() -> f32 { return 0.1 / PLANET_R; }
fn groundAlong(k: vec3f, a: vec3f, s: f32) -> vec3f { return normalize(k + a * s); }

//== sim

// where on the planet p is, for the calendar and the weather (world.wgsl):
// here
fn geo(p: vec3f) -> vec3f { return p; }
// a direction in the planet's frame, in the world's (the same)
fn toWorld(v: vec3f) -> vec3f { return v; }
fn sunWorld(step: u32) -> vec3f { return sunDir(step); }

//== shade

// the planet's air: thin, thinning with height (HAZE per cm at the ground,
// fading over HAZE_H), a little blue
const HAZE: f32 = 0.012;
const HAZE_H: f32 = 1.6;
fn hazeDensity(ps: vec3f) -> f32 { return exp(-max(length(ps) - PLANET_R, 0.0) / HAZE_H); }
// (soil seen from the side: only where it is cut deep)
fn cutFace(p: vec3f) -> bool { return false; }

// What a ray that misses the air sees: space.
fn backdrop(ro: vec3f, rd: vec3f, out: vec3f, v: vec3f) -> Backdrop {
  var b: Backdrop;
  b.col = space(out);
  return b;
}
// no glass
const HAS_PANES: bool = false;
fn crossPane(q: vec3f, rd: vec3f, col: ptr<function, vec3f>, T: ptr<function, vec3f>) -> bool { return false; }
fn onGlass(q: vec3f) -> bool { return false; }
fn glassVis(n: vec3f) -> vec2f { return vec2f(0.0); }
