// ---- world.wgsl: the world (pure functions and constants) ----
// Units are centimetres. The garden lives in the domain BOX_MIN..BOX_MAX,
// and the static scenery in it is baked into a 3D distance-field volume.
// Its shape (a small planet, or a terrarium: where up is, where the ground
// is) is shape_*.wgsl's. The sun, the calendar and the weather are those of
// a planet whatever the shape: a terrarium sits somewhere on one (geo).

const ROOT_SEG: u32 = 20u;
const MAX_ROCKS: u32 = 16u;
const MAX_ROOTS: u32 = 12u;
const MAX_BUMPS: u32 = 8u;

// Rock: a faceted, lumpy ellipsoid sitting in the soil (in a frame of its
// own: rockLocal).
struct Rock {
  a: vec4f,   // where: a point PLANET_R from the centre (the ground is found under it), seed
  r: vec4f,   // radii, type
  o: vec4f,   // yaw (about up), tilt, height offset (fraction of ry, relative to the ground), unused
  n0: vec4f,  // fracture planes: normal, offset (fraction of min radius)
  n1: vec4f,
  n2: vec4f,
}

// Root or branch: a cubic bezier tube. A control point lies PLANET_R plus its
// height above the ground from the centre, in its direction.
struct Root {
  p0: vec4f,  // where, radius
  p1: vec4f,
  p2: vec4f,
  p3: vec4f,
  info: vec4f, // seed, type, rock it rests on + 1 (0: none), where along it rests
}

// The ground (world.ts makes all of this from a seed): the planet raised and
// lowered by noise at three scales, and mounds and hollows.
struct Terrain {
  amp: vec4f,              // planet-wide, middling and fine noise amplitude (cm), unused
  seeds: vec4u,            // noise seeds
  bumps: array<vec4f, 8>,  // where (a point PLANET_R from the centre), radius
  heights: array<vec4f, 2>, // each bump's height (< 0: a hollow), four a vec4
}

struct WorldDesc {
  counts: vec4u,  // rocks, roots, bumps
  terrain: Terrain,
  rocks: array<Rock, 16>,
  roots: array<Root, 12>,
  lie: vec4f,     // a terrarium's ground: the way it rises (xz), from and to along it
}

// The calendar. A day is SOL steps from one midnight to the next at
// longitude 0 (0.5 its noon; a garden starts in the morning there); a year,
// YEAR steps round the sun from the northern autumn's equinox: then winter,
// spring and summer, the other way about in the south. The planet spins
// about world y, tilted TILT to its orbit: the sun circles the axis once a
// day, standing higher in the north's summer and lower in its winter, so
// night and how long the days are depend on where on the planet one is.
// The simulation and the renderer both follow it (from the step counter).
const SOL: u32 = 4800u;
const YEAR: u32 = 96000u;
const TILT: f32 = 0.41;
fn yearPhase(step: u32) -> f32 { return f32(step % YEAR) / f32(YEAR); }
fn dayPhase(step: u32) -> f32 { return f32((step + SOL * 3u / 10u) % SOL) / f32(SOL); }
// towards the sun
fn sunDir(step: u32) -> vec3f {
  let sd = -sin(TILT) * sin(TAU * yearPhase(step));
  let cd = sqrt(1.0 - sd * sd);
  let a = TAU * (dayPhase(step) - 0.5);
  return vec3f(cd * cos(a), sd, cd * sin(a));
}
// where the sun stands at its noon over p (the season's highest)
fn noonSun(p: vec3f, step: u32) -> vec3f {
  let sd = -sin(TILT) * sin(TAU * yearPhase(step));
  let cd = sqrt(1.0 - sd * sd);
  let h = normalize(vec3f(p.x, 0.0, p.z) + vec3f(1e-5, 0.0, 0.0));
  return vec3f(h.x * cd, sd, h.z * cd);
}
// up at a point of the planet (its frame: these take points geo gives)
fn geoUp(p: vec3f) -> vec3f { return p / max(length(p), 1e-4); }
// how high the sun stands at p (the sine of its elevation)
fn sunElev(p: vec3f, step: u32) -> f32 { return dot(geoUp(p), sunDir(step)); }
// 1 in the dark of night .. 0 by day (as the renderer lights it); and the
// hour after sunset (the sun gone down in the west)
fn nightness(p: vec3f, step: u32) -> f32 { return 1.0 - smoothstep(-0.12, 0.22, sunElev(p, step)); }
fn dusk(p: vec3f, step: u32) -> f32 {
  let s = sunDir(step);
  let up = geoUp(p);
  let el = dot(up, s);
  let west = dot(s, cross(up, vec3f(0.0, 1.0, 0.0))) > 0.0;
  return select(0.0, smoothstep(0.04, 0.0, el) * smoothstep(-0.4, -0.33, el), west);
}

// The weather, of where on the planet one is. The seasons there (from a
// hemisphere's autumn: the other way about in the south), the more marked
// the nearer the poles: autumn and winter are soaking, summer drier (the
// planet never quite dries out), damp all year near the equator.
fn seasonPhase(p: vec3f, step: u32) -> f32 { return fract(yearPhase(step) + select(0.0, 0.5, p.y < 0.0)); }
fn seasonal(p: vec3f) -> f32 { return smoothstep(0.05, 0.6, abs(geoUp(p).y)); }
// 0.57 in high summer .. 1 in winter (0.82 near the equator)
fn seasonWet(p: vec3f, step: u32) -> f32 {
  return 0.82 + 0.25 * seasonal(p) * cos(TAU * (seasonPhase(p, step) - 0.25));
}
// Rain comes in storms carried round the planet by the winds: STORMS about
// at any time, each living STORM_LIFE steps from where it formed, growing
// and then waning as it drifts east or west along its latitude, stronger in
// its latitude's wet season. Where one has passed the ground stays wet,
// drying over the days after.
const STORMS: u32 = 6u;
const STORM_LIFE: u32 = 2400u;
fn rainWet(p: vec3f, step: u32) -> f32 {
  let up = geoUp(p);
  let lat = asin(clamp(up.y, -1.0, 1.0));
  let lon = atan2(up.z, up.x);
  var r = 0.0;
  for (var n = 0u; n < STORMS; n++) {
    let t = step + n * (STORM_LIFE / STORMS);
    let g = t / STORM_LIFE;
    // (this one, and the last: its wet lingers)
    for (var k = 0u; k < 2u && k <= g; k++) {
      let gen = g - k;
      let age = f32(t - gen * STORM_LIFE);
      let h = hash2(gen * STORMS + n, 0x5707u);
      let sLat = asin(u01(h) * 1.8 - 0.9);
      let lon0 = u01(pcg(h)) * TAU;
      let w = mix(3.0, 6.0, u01(pcg(h + 1u))) / PLANET_R;
      let spin = mix(0.5, 1.2, u01(pcg(h + 2u))) * select(-1.0, 1.0, u01(pcg(h + 3u)) < 0.5) / f32(STORM_LIFE);
      let hS = vec3f(cos(sLat), sin(sLat), 0.0);
      let strength = mix(0.35, 0.8, u01(pcg(h + 4u))) * mix(0.45, 1.0, (seasonWet(hS, step) - 0.57) / 0.5);
      // when it came by here (along its latitude), and how near
      let ahead = fract((lon - lon0) / TAU * sign(spin) + 1.0) * TAU;
      let when = ahead / abs(spin);
      let gone = min(age, f32(STORM_LIFE));
      var dLon = 0.0;
      var at = when;
      if (when > gone) {
        // (not yet here: how near its front is, or its start)
        let back = TAU - ahead;
        dLon = min(ahead - gone * abs(spin), back);
        at = select(gone, 0.0, back < ahead - gone * abs(spin));
      }
      let d = (lat - sLat) * (lat - sLat) + dLon * dLon * cos(lat) * cos(lat);
      let life = sin(PI * clamp(at / f32(STORM_LIFE), 0.0, 1.0));
      r += strength * life * exp(-d / (w * w)) * exp(-max(age - at, 0.0) / 900.0);
    }
  }
  return r;
}
// 1 wet .. 0 dry
fn climate(p: vec3f, step: u32) -> f32 { return clamp(seasonWet(p, step) + rainWet(p, step), 0.0, 1.0); }
// leaves falling: in autumn where there are seasons, a little all year round
// where there are none
fn leafFall(p: vec3f, step: u32) -> f32 {
  let a = seasonPhase(p, step);
  let k = seasonal(p);
  return k * smoothstep(0.0, 0.04, a) * smoothstep(0.28, 0.1, a) + (1.0 - k) * 0.22;
}

fn boxSize() -> vec3f { return BOX_MAX - BOX_MIN; }
fn boxCenter() -> vec3f { return (BOX_MAX + BOX_MIN) * 0.5; }
fn volUVW(p: vec3f) -> vec3f { return (p - BOX_MIN) / boxSize(); }

// A direction as a point of the square [-1, 1]^2 and back: the sphere
// unfolded flat (octahedrally, about the axis), for maps over the planet.
fn octEncode(n: vec3f) -> vec2f {
  let p = n.xz / (abs(n.x) + abs(n.y) + abs(n.z));
  if (n.y >= 0.0) { return p; }
  return (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0));
}
fn octDecode(e: vec2f) -> vec3f {
  var n = vec3f(e.x, 1.0 - abs(e.x) - abs(e.y), e.y);
  let t = max(-n.y, 0.0);
  n.x += select(t, -t, n.x >= 0.0);
  n.z += select(t, -t, n.z >= 0.0);
  return normalize(n);
}
// Where a ray enters and leaves a sphere about the centre (enter > leave:
// it misses).
fn sphereHit(ro: vec3f, rd: vec3f, r: f32) -> vec2f {
  let b = dot(ro, rd);
  let h = b * b - dot(ro, ro) + r * r;
  if (h < 0.0) { return vec2f(1e9, -1e9); }
  let s = sqrt(h);
  return vec2f(-b - s, -b + s);
}

fn rotY(p: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

fn rotX(p: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

// p in rock frame: up along y, from the frame at the rock's middle (one frame
// for the whole rock, whatever its turn: it has a yaw of its own)
fn rockLocal(p: vec3f, c: vec3f) -> vec3f {
  let b = basisOf(upAt(c));
  let d = p - c;
  return vec3f(dot(d, b[0]), dot(d, b[2]), dot(d, b[1]));
}

fn rockDist(p: vec3f, rk: Rock, c: vec3f) -> f32 {
  let r = rk.r.xyz;
  let rmax = max(r.x, max(r.y, r.z));
  let dl = length(p - c);
  // far away: a lower bound, never within the soil blend (bake_volume.wgsl)
  if (dl > rmax * 2.0 + 0.6) { return dl - rmax * 1.25; }
  let q = rotX(rotY(rockLocal(p, c), rk.o.x), rk.o.y);
  let k0 = length(q / r);
  let k1 = length(q / (r * r));
  var d = k0 * (k0 - 1.0) / max(k1, 1e-5);
  let rmin = min(r.x, min(r.y, r.z));
  let e = rmin * 0.1;
  d = smax(d, dot(q, rk.n0.xyz) - rk.n0.w * rmin, e);
  d = smax(d, dot(q, rk.n1.xyz) - rk.n1.w * rmin, e);
  d = smax(d, dot(q, rk.n2.xyz) - rk.n2.w * rmin, e);
  let seed = u32(rk.a.w);
  d += fbm3(q * (1.2 / rmin), seed, 4u) * 0.17 * rmin;
  return d;
}

fn bez4(a: vec4f, b: vec4f, c: vec4f, d: vec4f, t: f32) -> vec4f {
  let it = 1.0 - t;
  return a * (it * it * it) + b * (3.0 * it * it * t) + c * (3.0 * it * t * t) + d * (t * t * t);
}
