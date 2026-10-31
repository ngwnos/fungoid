// ---- shape_box.wgsl: the world's shape, a terrarium ----
// A glass box of ground (BOX_MIN..BOX_MAX, open at the top) on a wooden
// table; up is +y. It sits somewhere on a planet like the other world, at a
// latitude of its own (the simulation's site): its sun and weather are that
// planet's there, turned into the box's frame (toWorld). shape_planet.wgsl
// says what each of these is for.

//== pure

const TABLE_Y: f32 = -6.4;
const GLASS_TINT: vec3f = vec3f(0.92, 0.965, 0.94);

fn upAt(p: vec3f) -> vec3f { return vec3f(0.0, 1.0, 0.0); }
// a point's key is where it is over the floor; its level, its height
fn keyOf(p: vec3f) -> vec3f { return vec3f(p.x, 0.0, p.z); }
fn levelOf(p: vec3f) -> f32 { return p.y; }
fn atLevel(k: vec3f, l: f32) -> vec3f { return vec3f(k.x, l, k.z); }
// (off the noise lattice's planes)
fn groundCoord(p: vec3f, up: vec3f) -> vec3f { return vec3f(p.x, 0.37, p.z); }
fn groundUV(k: vec3f) -> vec2f { return (k.xz - BOX_MIN.xz) / boxSize().xz; }
fn keyFrame(k: vec3f) -> mat3x3f { return basisOf(vec3f(0.0, 1.0, 0.0)); }
// (the ground is flat: the planet's inline expressions give way to these)
const FLAT: bool = true;

fn insideBox(p: vec3f) -> bool { return all(p > BOX_MIN) && all(p < BOX_MAX); }
fn boxInterval(ro: vec3f, rd: vec3f) -> vec2f {
  let inv = 1.0 / rd;
  let t0 = (BOX_MIN - ro) * inv;
  let t1 = (BOX_MAX - ro) * inv;
  let tn = min(t0, t1);
  let tf = max(t0, t1);
  return vec2f(max(max(tn.x, tn.y), tn.z), min(min(tf.x, tf.y), tf.z));
}
fn boxFaceNormal(p: vec3f) -> vec3f {
  let q = (p - boxCenter()) / (boxSize() * 0.5);
  let a = abs(q);
  if (a.x > a.y && a.x > a.z) { return vec3f(sign(q.x), 0.0, 0.0); }
  if (a.y > a.z) { return vec3f(0.0, sign(q.y), 0.0); }
  return vec3f(0.0, 0.0, sign(q.z));
}

// life keeps inside the glass
fn inLife(p: vec3f) -> bool { return all(p > BOX_MIN + 0.15) && all(p < BOX_MAX - 0.15); }
fn clampLife(np: vec3f) -> vec3f { return clamp(np, BOX_MIN + 0.05, BOX_MAX - 0.05); }
fn fieldsIdle(p: vec3f) -> bool { return false; }
fn pastLife(q: vec3f) -> bool { return false; }
fn dropStart(p: vec3f, down: vec3f) -> vec3f { return vec3f(p.x, BOX_MAX.y - 0.5, p.z); }

// The air turns over in one slow roll across the box (up the warm side, over
// and down), with a little drift along it.
fn wind(p: vec3f) -> vec3f {
  let u = (p - BOX_MIN) / boxSize();
  let a = 0.06;
  return vec3f(a * PI / boxSize().y * sin(PI * u.x) * cos(PI * u.y),
               -a * PI / boxSize().x * cos(PI * u.x) * sin(PI * u.y),
               0.004 * sin(TAU * u.x + 1.3));
}

// The air is the box's inside; light leaving through a pane is dimmed by
// the glass, through the open top not at all.
fn airHit(ro: vec3f, rd: vec3f) -> vec2f { return boxInterval(ro, rd); }
fn inAir(p: vec3f) -> bool { return insideBox(p); }
fn leftAir(q: vec3f) -> bool { return !insideBox(q); }
fn exitTint(q: vec3f) -> f32 { return select(0.88, 1.0, q.y >= BOX_MAX.y); }
// below the horizon, the table and the ground beyond hide the sun
fn sunBlocked(p: vec3f, L: vec3f) -> bool { return L.y <= 0.0; }

// Moss: low ground is damp; none on the glass
fn mossDamp(p: vec3f) -> f32 { return smoothstep(4.5, -0.5, p.y); }
fn mossWall(p: vec3f) -> f32 { return smoothstep(-0.1, -0.6, sdBox(p - boxCenter(), boxSize() * 0.5)); }

// The slime mould's grid is the box's floor, 1 mm a cell; it keeps off the
// glass and turns back from it.
fn slimeEncode(n: vec3f) -> vec2f { return (n.xz - BOX_MIN.xz) / boxSize().xz * 2.0 - 1.0; }
fn slimeDecode(e: vec2f) -> vec3f {
  let xz = BOX_MIN.xz + (e * 0.5 + 0.5) * boxSize().xz;
  return vec3f(xz.x, 0.0, xz.y);
}
fn slimeNext(c: vec2i) -> vec2i { return clamp(c, vec2i(0), vec2i(SLIME_RES) - 1); }
fn slimeAhead(n: vec3f, a: vec3f, s: f32) -> vec3f { return n + a * s; }
fn slimeApart(n: vec3f, m: vec3f) -> f32 { return length(n - m); }
fn slimeToward(n: vec3f, m: vec3f) -> vec3f { return m - n; }
fn slimeInside(m: vec3f) -> bool { return all(m.xz > BOX_MIN.xz + 0.3) && all(m.xz < BOX_MAX.xz - 0.3); }
fn slimeClampIn(m: vec3f) -> vec3f { return vec3f(clamp(m.xz, BOX_MIN.xz + 0.3, BOX_MAX.xz - 0.3), 0.0).xzy; }
fn slimeMiddle(sum: vec3f, n: u32) -> vec3f { return sum / (f32(n) * SL_FIX); }
fn slimeAtKey(e: vec2f) -> vec3f { return vec3f(e.x, 0.0, e.y); }
const SL_FIX: f32 = 1024.0;
const COL_CM: f32 = 0.4;
const SLIME_R0: f32 = 1.0;

//== ground

// the ground's height under p
fn groundR(p: vec3f) -> f32 { return textureSampleLevel(ground, linSamp, groundUV(keyOf(p)), 0.0).x; }
// how far p is above the ground; the glass walls cut the soil
fn soilDist(p: vec3f) -> f32 {
  let g = textureSampleLevel(ground, linSamp, groundUV(keyOf(p)), 0.0);
  return max((p.y - g.x) * g.y, sdBox(p - boxCenter(), boxSize() * 0.5));
}
fn slimeRadius(n: vec3f) -> f32 { return 1.0; }

//== bake

// The ground: a rise toward the back (world.lie: the direction, from and to
// along it), noise at two scales, and mounds and hollows.
fn terrainHeight(k: vec3f) -> f32 {
  let tr = world.terrain;
  let xz = k.xz;
  var h = tr.amp.x + tr.amp.y * smoothstep(world.lie.z, world.lie.w, dot(xz, world.lie.xy));
  h += fbm2(xz * 0.11, tr.seeds.x, 4u) * tr.amp.z;
  h += fbm2(xz * 0.48, tr.seeds.y, 3u) * tr.amp.w;
  for (var i = 0u; i < world.counts.z; i++) {
    let b = tr.bumps[i];
    let d = xz - b.xz;
    h += tr.heights[i / 4u][i % 4u] * exp(-dot(d, d) / (b.w * b.w));
  }
  return h;
}
fn groundRadius(dir: vec3f) -> f32 { return terrainHeight(dir); }
fn restKey(a: vec3f) -> vec3f { return vec3f(a.x, 0.0, a.z); }
// (a root's point: where, and its height above the ground there)
fn rootPoint(c: vec4f) -> vec4f { return vec4f(c.x, groundRadius(restKey(c.xyz)) + c.y, c.z, c.w); }
fn restUp(at: vec3f) -> vec3f { return vec3f(0.0, 1.0, 0.0); }
fn groundTexelKey(gid: vec2u) -> vec3f {
  let xz = BOX_MIN.xz + (vec2f(gid) + 0.5) / vec2f(GROUND_RES) * boxSize().xz;
  return vec3f(xz.x, 0.0, xz.y);
}
fn groundSlopeStep() -> f32 { return 0.1; }
fn groundAlong(k: vec3f, a: vec3f, s: f32) -> vec3f { return k + a * s; }

//== sim

// where on the planet the terrarium is (the site: up there, in the planet's
// frame), and a direction in the planet's frame in the box's (sim.siteX,
// site, siteZ: the box's axes there)
fn geo(p: vec3f) -> vec3f { return sim.site.xyz; }
fn toWorld(v: vec3f) -> vec3f { return vec3f(dot(sim.siteX.xyz, v), dot(sim.site.xyz, v), dot(sim.siteZ.xyz, v)); }
fn sunWorld(step: u32) -> vec3f { return toWorld(sunDir(step)); }

//== shade

const HAZE: f32 = 0.006;
const HAZE_H: f32 = 1.6;
fn hazeDensity(ps: vec3f) -> f32 { return 1.0; }
const HAS_PANES: bool = true;
// (the soil as seen through the glass: right against it, whatever the normal)
fn cutFace(p: vec3f) -> bool { return any(p.xz < BOX_MIN.xz + 0.08) || any(p.xz > BOX_MAX.xz - 0.08); }

// The sky over the box (by the sun there: blue by day, warm low down as it
// sets or rises, stars at night), and below the horizon the room it stands
// in, as lit as the day is.
fn boxSky(d: vec3f) -> vec3f {
  let el = scene.sunDir.y;
  let day = smoothstep(-0.12, 0.22, el);
  let dusk = exp(-(el / 0.2) * (el / 0.2));
  let hor = mix(scene.skyHor.rgb, HOR_DUSK, dusk * 0.8);
  let zen = mix(scene.skyZen.rgb, ZEN_DUSK, dusk * 0.6);
  var c = mix(hor, zen, pow(clamp(d.y, 0.0, 1.0), 0.45)) * day;
  c += space(d) * (1.0 - day);
  if (d.y < 0.0) { c = mix(hor * 0.6 * day, vec3f(0.07, 0.06, 0.05) * max(day, 0.03), smoothstep(0.0, -0.25, d.y)); }
  let mu = max(dot(d, scene.sunDir.xyz), 0.0);
  return c + scene.sunCol.rgb * 0.015 * pow(mu, 12.0) * day;
}

// the table's wood, under the box
fn tableMat(p: vec3f) -> Mat {
  var m: Mat;
  let q = vec2f(p.x * 0.97 + p.z * 0.25, p.z * 0.97 - p.x * 0.25);
  let warp = fbm2(q * vec2f(0.02, 0.15), 401u, 4u) * 6.0;
  let rings = 0.5 + 0.5 * sin((q.y + warp) * 2.2);
  let fine = gnoise2(vec2f(q.x * 0.05, q.y * 9.0), 402u);
  m.alb = mix(vec3f(0.11, 0.065, 0.035), vec3f(0.2, 0.12, 0.065), rings * 0.7 + 0.15 * fine);
  m.rough = 0.55;
  m.n = vec3f(0.0, 1.0, 0.0);
  return m;
}
// The box's shadow on the table: the sun's light where the sun's ray from the
// table enters the box.
fn tableSun(p: vec3f) -> f32 {
  let L = scene.sunDir.xyz;
  let bi = boxInterval(p, L);
  if (bi.y < max(bi.x, 0.0)) { return 1.0; }
  return lightAt(p + L * (max(bi.x, 0.0) + 0.05)).x * 0.9;
}

// What a ray that misses the air sees: the table, or the sky and the room.
fn backdrop(ro: vec3f, rd: vec3f, out: vec3f, v: vec3f) -> Backdrop {
  var b: Backdrop;
  if (rd.y < 0.0) {
    let t = (TABLE_Y - ro.y) / rd.y;
    let p = ro + rd * t;
    var m = tableMat(p);
    m.alb *= exp(-length(p.xz) / 64.0);
    let near = smoothstep(2.0, 10.0, max(abs(p.x) - BOX_MAX.x, abs(p.z) - BOX_MAX.z));
    b.col = shade(p, m.n, v, m, vec2f(tableSun(p), mix(0.55, 1.0, near)), vec3f(0.0), 0.0);
    b.hit = true;
    b.alb = m.alb;
    b.n = m.n;
    b.t = t;
  } else {
    b.col = boxSky(out);
  }
  return b;
}

// A ray crossing a pane at q (its sides and floor are glass, its top open):
// the sky it reflects, and the light it lets through, tinted. False: it is
// not glass there.
fn crossPane(q: vec3f, rd: vec3f, col: ptr<function, vec3f>, T: ptr<function, vec3f>) -> bool {
  let n = boxFaceNormal(q);
  if (n.y >= 0.5) { return false; }
  let ci = abs(dot(rd, n));
  let F = 0.04 + 0.96 * pow(1.0 - ci, 5.0);
  let R = 2.0 * F / (1.0 + F);
  *col += *T * R * boxSky(reflect(rd, -n));
  *T *= (1.0 - R) * GLASS_TINT;
  return true;
}
// Faces against the glass look out into the room: lit by the sun if they
// face it, and by the room's light.
fn onGlass(q: vec3f) -> bool { return any(q < BOX_MIN + 0.05) || any(q > BOX_MAX - 0.05); }
fn glassVis(n: vec3f) -> vec2f { return vec2f(select(0.0, 0.9, dot(n, scene.sunDir.xyz) > 0.0), 0.55); }
