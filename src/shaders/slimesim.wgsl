// ---- slimesim.wgsl: the slime mould (after sim.wgsl) ----
// A plasmodium is a swarm of agents on the ground. Each senses the trail
// ahead, ahead-left and ahead-right, turns toward the strongest, moves on
// and lays trail. Food in the ground adds to the trail where it lies, and
// light and dry ground wear it away, so the swarm streams toward food along
// veins that thicken where many travel. It eats as it goes (what lies in the
// ground, settled spores, mycelium near the surface, fruit bodies), and its
// food sets how many agents it has: it grows while fed and shrinks while
// starving. Starving, or as the ground dries, it gathers into sporangia,
// whose spores wait in the ground for a wet season to hatch.
//
// Per update (every SLIME_EVERY steps): slimeHatch, slimeMove, slimeTrail,
// slimeColumns, slimePop. These have a bind group of their own (renderer.ts);
// what the other passes share with them is in the grid buffer.

// where each is (its direction, slime.wgsl slimeEncode), its heading (in the
// ground's frame there: basisOf), its age + 1 (0: none)
@group(0) @binding(17) var<storage, read_write> slimeAgents: array<vec4f>;
// the trail and where the plasmodium is (two copies each, by update
// parity), the slime it leaves behind, and per column what draws and what
// repels it
@group(0) @binding(18) var<storage, read_write> slimeMap: array<f32>;
@group(0) @binding(19) var slimeOut: texture_storage_2d<rgba16float, write>;

const SLIME_MOST: f32 = 12000.0;   // agents one grows to at most
const SL_UNIT: f32 = 0.05;         // food per agent
const SL_UPKEEP: f32 = 0.00002;    // food per agent per step
const SL_STEP: f32 = 0.06;         // cm an agent moves per update
const SL_SENSE: f32 = 0.9;         // cm ahead it senses
const SL_ANGLE: f32 = 0.6;         // radians to either side it senses
const SL_TURN: f32 = 0.45;         // radians it turns
const SL_FIX: f32 = 4096.0;        // the fixed point its agents' directions are summed in
const COLS: u32 = SLIME_COLS;
const COL_CM: f32 = 0.46;          // a column's width (about: 4 pi PLANET_R^2 / COLS)

// In the grid buffer (GRID_SLIME), at the end: a header, then per column,
// then per cell
const SL_FOOD: u32 = 0u;           // its food (POOL_FP, i32 bits)
const SL_COUNT: u32 = 1u;          // live agents, two slots by update parity
const SL_SPAWN: u32 = 3u;          // agents still to set down where it hatches (i32 bits)
const SL_MODE: u32 = 4u;           // 0 none, 1 feeding, 2 fruiting
const SL_AT: u32 = 5u;             // where it hatches (a direction, f32 bits)
const SL_REQ: u32 = 8u;            // the last hatching the player asked for
const SL_AGE: u32 = 9u;            // steps since it hatched, or began to fruit
const SL_TICK: u32 = 10u;          // updates so far (the parity of the trail and counts)
const SL_SUM: u32 = 11u;           // where its agents are (sums of directions: SL_FIX, i32 bits), two by parity
const SL_MID: u32 = 17u;           // its middle (a direction, f32 bits)
const SL_REACH: u32 = 20u;         // how far it reaches from its middle (cm x 1000)
const SL_COLS: u32 = 32u;          // agents per column this update
const SL_DENS: u32 = SL_COLS + COLS;   // agents per column last update
const SL_BANK: u32 = SL_DENS + COLS;   // spores waiting per column (x 1000)
const SL_DEP: u32 = SL_BANK + COLS;    // agents per cell this update
// slimeMap
const ST_BODY: u32 = 2u * SLIME_N;
const ST_SHEATH: u32 = 4u * SLIME_N;
const ST_DRAW: u32 = 5u * SLIME_N;
const ST_REPEL: u32 = 5u * SLIME_N + COLS;

fn sl(k: u32) -> u32 { return GRID_SLIME + k; }
fn slimeUpdate() -> u32 { return atomicLoad(&grid[sl(SL_TICK)]); }
// the cell and the column a place on the grid (slimeEncode) is in, and the
// column a point is over
fn slimeCellOf(e: vec2f) -> u32 {
  let c = clamp(vec2i(floor((e * 0.5 + 0.5) * vec2f(SLIME_RES))), vec2i(0), vec2i(SLIME_RES) - 1);
  return u32(c.y) * SLIME_RES.x + u32(c.x);
}
fn slimeColOf(e: vec2f) -> u32 {
  let c = clamp(vec2i(floor((e * 0.5 + 0.5) * f32(SLIME_COL_RES))), vec2i(0), vec2i(i32(SLIME_COL_RES) - 1));
  return u32(c.y) * SLIME_COL_RES + u32(c.x);
}
fn slimeCol(p: vec3f) -> u32 { return slimeColOf(slimeEncode(upAt(p))); }
// the direction of column c's middle
fn slimeColDir(c: u32) -> vec3f {
  return slimeDecode((vec2f(f32(c % SLIME_COL_RES), f32(c / SLIME_COL_RES)) + 0.5) / f32(SLIME_COL_RES) * 2.0 - 1.0);
}
// agents over this point's column last update, its food, what it is doing
fn slimeAt(p: vec3f) -> u32 { return atomicLoad(&grid[sl(SL_DENS + slimeCol(p))]); }
fn slimeFood() -> f32 { return f32(bitcast<i32>(atomicLoad(&grid[sl(SL_FOOD)]))) / POOL_FP; }
fn slimeFeed(food: f32) { atomicAdd(&grid[sl(SL_FOOD)], bitcast<u32>(i32(food * POOL_FP))); }
fn slimeMode() -> u32 { return atomicLoad(&grid[sl(SL_MODE)]); }
// the voxel of soil just under the ground in direction n
fn slimeSoil(n: vec3f) -> u32 { return f4Index(f4Coord(n * (groundR(n) - 0.12))); }

// a direction kept in the header (f32 bits)
fn slimeDirAt(k: u32) -> vec3f {
  return vec3f(bitcast<f32>(atomicLoad(&grid[sl(k)])), bitcast<f32>(atomicLoad(&grid[sl(k + 1u)])), bitcast<f32>(atomicLoad(&grid[sl(k + 2u)])));
}
fn slimeDirTo(k: u32, n: vec3f) {
  for (var j = 0u; j < 3u; j++) { atomicStore(&grid[sl(k + j)], bitcast<u32>(n[j])); }
}
fn slimeMid() -> vec3f { return slimeDirAt(SL_MID); }

// A heading at n (an angle in basisOf(n)) as a direction along the ground,
// and a direction along the ground at n as a heading there.
fn slimeHeading(n: vec3f, h: f32) -> vec3f {
  let b = basisOf(n);
  return b[0] * cos(h) + b[1] * sin(h);
}
fn slimeAngle(n: vec3f, t: vec3f) -> f32 {
  let b = basisOf(n);
  return atan2(dot(t, b[1]), dot(t, b[0]));
}

fn hatchAt(n: vec3f, count: i32) {
  if (slimeMode() == 0u) { slimeDirTo(SL_MID, n); }
  slimeDirTo(SL_AT, n);
  atomicStore(&grid[sl(SL_SPAWN)], bitcast<u32>(count));
  atomicStore(&grid[sl(SL_MODE)], 1u);
  atomicStore(&grid[sl(SL_AGE)], 0u);
  slimeFeed(f32(count) * SL_UNIT * 1.5);
}

// Where it hatches, and when it turns from feeding to fruiting and from
// fruiting to nothing.
@compute @workgroup_size(1)
fn slimeHatch() {
  stepNo = atomicLoad(&counters[5]);
  let k = atomicAdd(&grid[sl(SL_TICK)], 1u) + 1u;
  let n = atomicLoad(&grid[sl(SL_COUNT + ((k + 1u) & 1u))]);
  // last update's agents set down are down; and their upkeep (paid here,
  // in one thread, so slimePop's threads all see the same store)
  atomicStore(&grid[sl(SL_SPAWN)], 0u);
  slimeFeed(-f32(n) * SL_UPKEEP * f32(SLIME_EVERY));
  let mode = slimeMode();
  let age = atomicAdd(&grid[sl(SL_AGE)], SLIME_EVERY) + SLIME_EVERY;
  // its middle, and how far it reaches from it (it is one body: a
  // plasmodium), from where its agents were last update
  let ps = SL_SUM + ((k + 1u) & 1u) * 3u;
  var sum = vec3f(0.0);
  for (var j = 0u; j < 3u; j++) { sum[j] = f32(bitcast<i32>(atomicExchange(&grid[sl(ps + j)], 0u))); }
  if (n > 0u && dot(sum, sum) > 0.0) { slimeDirTo(SL_MID, normalize(sum)); }
  atomicStore(&grid[sl(SL_REACH)], u32((1.0 + 0.045 * sqrt(f32(n))) * 1000.0));
  // how wet the ground is under its middle
  let wet = wetAt(slimeSoil(slimeMid()));
  // set down by the player
  if (sim.slimeAt.z > 0.0 && u32(sim.slimeAt.w) != atomicLoad(&grid[sl(SL_REQ)])) {
    atomicStore(&grid[sl(SL_REQ)], u32(sim.slimeAt.w));
    hatchAt(octDecode(sim.slimeAt.xy), i32(sim.slimeAt.z));
    return;
  }
  if (mode == 1u) {
    if (age > 400u && n < 20u) {
      atomicStore(&grid[sl(SL_MODE)], 0u);
    } else if ((age > 1500u && (slimeFood() < f32(n) * SL_UNIT * 0.4 || wet < 0.3)) || age > 8000u) {
      // starving, or the ground drying: it fruits
      atomicStore(&grid[sl(SL_MODE)], 2u);
      atomicStore(&grid[sl(SL_AGE)], 0u);
    }
  } else if (mode == 2u) {
    if (n == 0u && age > 50u) { atomicStore(&grid[sl(SL_MODE)], 0u); }
  } else if (k % 64u == 0u) {
    // spores in the ground hatch where most lie, when it is wet there
    var best = 0u;
    var at = 0u;
    for (var c = 0u; c < COLS; c++) {
      let b = atomicLoad(&grid[sl(SL_BANK + c)]);
      if (b > best) {
        best = b;
        at = c;
      }
    }
    rinit(k, 0x51a7u);
    let there = slimeColDir(at);
    if (best > 2000u && wetAt(slimeSoil(there)) > 0.75 && rr() < 0.01) {
      atomicStore(&grid[sl(SL_BANK + at)], 0u);
      hatchAt(there, i32(clamp(best / 4u, 600u, 4000u)));
    }
  }
}

// What an agent at n senses `d` cm on along t: the trail.
fn slimeSense(n: vec3f, t: vec3f, d: f32, rd: u32) -> f32 {
  return slimeMap[rd * SLIME_N + slimeCellOf(slimeEncode(normalize(n + t * d)))];
}

@compute @workgroup_size(64)
fn slimeMove(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= SLIME_MAX) { return; }
  let a = slimeAgents[i];
  // (one that died last update leaves its slot empty)
  if (a.w <= 0.0) {
    if (a.w < 0.0) { slimeAgents[i].w = 0.0; }
    return;
  }
  stepNo = atomicLoad(&counters[5]);
  let rd = slimeUpdate() & 1u;
  rinit(i, stepNo ^ 0x5113u);
  let n = slimeDecode(a.xy);
  // (radians round the planet per cm, at the ground here)
  let k = 1.0 / groundR(n);
  var h = a.z;
  let ahead = slimeSense(n, slimeHeading(n, h), SL_SENSE * k, rd);
  let left = slimeSense(n, slimeHeading(n, h - SL_ANGLE), SL_SENSE * k, rd);
  let right = slimeSense(n, slimeHeading(n, h + SL_ANGLE), SL_SENSE * k, rd);
  // (ahead is best: keep on; ahead is worst: either way)
  if (ahead < left && ahead < right) {
    h += select(SL_TURN, -SL_TURN, rr() < 0.5);
  } else if (left > ahead || right > ahead) {
    h += select(SL_TURN, -SL_TURN, left > right);
  }
  h += (rr() - 0.5) * 0.15;
  // (one body: an agent straying beyond its reach turns back toward its
  // middle)
  let mid = slimeMid();
  let away = acos(clamp(dot(n, mid), -1.0, 1.0)) / k;
  let reach = f32(atomicLoad(&grid[sl(SL_REACH)])) / 1000.0;
  if (away > reach) {
    let back = slimeAngle(n, mid);
    h += atan2(sin(back - h), cos(back - h)) * clamp(2.0 * (away / reach - 1.0), 0.0, 1.0);
  }
  // fruiting, it slows and gathers
  let speed = select(SL_STEP, SL_STEP * 0.2, slimeMode() == 2u);
  // on along the ground, its heading carried with it
  let t = slimeHeading(n, h);
  let m = normalize(n + t * speed * k);
  let e = slimeEncode(m);
  slimeAgents[i] = vec4f(e, slimeAngle(m, t), a.w + 1.0);
  atomicAdd(&grid[sl(SL_DEP + slimeCellOf(e))], 1u);
  atomicAdd(&grid[sl(SL_COLS + slimeColOf(e))], 1u);
  atomicAdd(&grid[sl(SL_COUNT + rd)], 1u);
  let q = vec3i(round(m * SL_FIX));
  for (var j = 0u; j < 3u; j++) { atomicAdd(&grid[sl(SL_SUM + rd * 3u + j)], bitcast<u32>(q[j])); }
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

// The trail spreads and fades, food adds to it where it lies, light and dry
// ground wear it away; and what is seen: the plasmodium where its agents
// are thick, the slime it leaves behind.
@compute @workgroup_size(16, 16)
fn slimeTrail(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid.xy >= SLIME_RES)) { return; }
  stepNo = atomicLoad(&counters[5]);
  let rd = slimeUpdate() & 1u;
  let x = i32(gid.x);
  let z = i32(gid.y);
  let i = gid.y * SLIME_RES.x + gid.x;
  var sum = 0.0;
  var bsum = 0.0;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dx = -1; dx <= 1; dx++) {
      let q = slimeNext(vec2i(x + dx, z + dz));
      let j = u32(q.y) * SLIME_RES.x + u32(q.x);
      sum += slimeMap[rd * SLIME_N + j];
      bsum += slimeMap[ST_BODY + rd * SLIME_N + j];
    }
  }
  let here = f32(atomicExchange(&grid[sl(SL_DEP + i)], 0u));
  let col = (gid.y / 4u) * SLIME_COL_RES + gid.x / 4u;
  let draw = slimeMap[ST_DRAW + col];
  let repel = slimeMap[ST_REPEL + col];
  let t = (sum / 9.0 * 0.9 + here + draw * 0.05) * (1.0 - repel);
  slimeMap[(rd ^ 1u) * SLIME_N + i] = t;
  // (the plasmodium: where agents are, smoothed into veins)
  let body = bsum / 9.0 * 0.75 + here * 0.6;
  slimeMap[ST_BODY + (rd ^ 1u) * SLIME_N + i] = body;
  // (the slime: left where the plasmodium was thick, not where one agent
  // wandered)
  let sheath = max(slimeMap[ST_SHEATH + i] * 0.9995, smoothstep(0.4, 2.0, body));
  slimeMap[ST_SHEATH + i] = sheath;
  textureStore(slimeOut, gid.xy, vec4f(body, sheath, select(0.0, 1.0, slimeMode() == 2u), 0.0));
}

// The plasmodium and its slime shown again as they are (after a checkpoint
// is restored): what the last trail update showed.
@compute @workgroup_size(16, 16)
fn slimePaint(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid.xy >= SLIME_RES)) { return; }
  let rd = slimeUpdate() & 1u;
  let i = gid.y * SLIME_RES.x + gid.x;
  let body = slimeMap[ST_BODY + (rd ^ 1u) * SLIME_N + i];
  textureStore(slimeOut, gid.xy, vec4f(body, slimeMap[ST_SHEATH + i], select(0.0, 1.0, slimeMode() == 2u), 0.0));
}

// Per column: what lies here draws it (and what is light or dry puts it
// off), it eats what lies in the ground, and as it fruits sporangia rise
// where it has gathered.
@compute @workgroup_size(64)
fn slimeColumns(@builtin(global_invocation_id) gid: vec3u) {
  let c = gid.x;
  if (c >= COLS) { return; }
  stepNo = atomicLoad(&counters[5]);
  let n = atomicExchange(&grid[sl(SL_COLS + c)], 0u);
  atomicStore(&grid[sl(SL_DENS + c)], n);
  let b = atomicLoad(&grid[sl(SL_BANK + c)]);
  if (b > 0u) { atomicStore(&grid[sl(SL_BANK + c)], b - max(b / 20000u, 1u)); }
  let dir = slimeColDir(c);
  let top = groundR(dir);
  let v = f4Index(f4Coord(dir * (top - 0.12)));
  let va = f4Index(f4Coord(dir * (top + 0.25)));
  let f = fieldA[v];
  let nut = max(f.y - f32(atomicLoad(&grid[gi(GF_EATEN, v)])) / 65536.0, 0.0);
  let spores = fieldA[F4N + va].y;
  let tissue = f32(atomicLoad(&grid[gi(GF_DENS, v)]));
  let bodies = f32(atomicLoad(&grid[gi(GF_OCC, va)]));
  slimeMap[ST_DRAW + c] = nut + f.w * 2.0 + spores * 3.0 + tissue * 0.05 + min(bodies, 4.0) * 0.3;
  slimeMap[ST_REPEL + c] = clamp(0.25 * fieldA[va].z + 0.5 * (1.0 - wetAt(v)), 0.0, 0.6);
  let mode = slimeMode();
  if (n > 0u && mode == 1u) {
    // (grazing: what an agent finds runs out where many have been)
    let take = min(nut * 0.05, f32(n) * 0.0001 * f32(SLIME_EVERY));
    if (take > 0.0) {
      atomicAdd(&grid[gi(GF_EATEN, v)], u32(take * 65536.0));
      slimeFeed(take);
    }
  }
  let age = atomicLoad(&grid[sl(SL_AGE)]);
  if (mode == 2u && n >= 4u && age > 60u && age < 1200u) {
    rinit(c, stepNo ^ 0x5b0u);
    if (rr() < 0.002 * min(f32(n) / 8.0, 1.0)) {
      let tb = basisOf(dir);
      let q = dir * (top + 0.2) + (tb[0] * (rr() - 0.5) + tb[1] * (rr() - 0.5)) * COL_CM;
      let s = toSurface(q);
      spawnFruit(SLIME_ORG, s, worldN(s), rr() * 40.0, 0u, 0);
    }
  }
}

// Agents are born from others while there is food for more (or set down
// where it hatches), and die while it starves, as it fruits, and where the
// ground has dried.
@compute @workgroup_size(64)
fn slimePop(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= SLIME_MAX) { return; }
  stepNo = atomicLoad(&counters[5]);
  let rd = slimeUpdate() & 1u;
  let n = f32(atomicLoad(&grid[sl(SL_COUNT + rd)]));
  if (i == 0u) { atomicStore(&grid[sl(SL_COUNT + (rd ^ 1u))], 0u); }
  let mode = slimeMode();
  let want = clamp(slimeFood() / SL_UNIT, 0.0, SLIME_MOST);
  rinit(i, stepNo ^ 0x9e5u);
  let a = slimeAgents[i];
  if (a.w > 0.0) {
    var die = select(0.0, (n - want) / max(n, 1.0) * 0.05, n > want);
    if (mode == 2u) { die = max(die, 0.004); }
    if (mode == 0u) { die = 1.0; }
    die += 0.01 * smoothstep(0.35, 0.1, wetAt(slimeSoil(slimeDecode(a.xy))));
    // (it draws back from ground it has grazed bare, so it moves on)
    die += 0.003 * smoothstep(0.3, 0.05, slimeMap[ST_DRAW + slimeColOf(a.xy)]);
    // (dying: still a parent for this update's buds, till the next move)
    if (rr() < die) { slimeAgents[i].w = -1.0; }
    return;
  }
  if (mode != 1u) { return; }
  // agents set down where it hatches: the empty slots among the first ones
  // (so which slots does not depend on the order threads run in); new ones,
  // set down or budded (w 0.5), are no parents till the next update
  let spawn = bitcast<i32>(atomicLoad(&grid[sl(SL_SPAWN)]));
  if (spawn > 0 && i32(i) < spawn + i32(n)) {
    let at = slimeDirAt(SL_AT);
    let r = sqrt(rr()) * 0.8;
    let t = rr() * TAU;
    let m = normalize(at + slimeHeading(at, t) * (r / groundR(at)));
    slimeAgents[i] = vec4f(slimeEncode(m), rr() * TAU, 0.5);
    return;
  }
  // a bud finds a parent among 8 tries, as often as it did when there was
  // room for 65536 agents (it grows from small no faster)
  let hit = (1.0 - pow(1.0 - n / 65536.0, 8.0)) / max(1.0 - pow(1.0 - n / f32(SLIME_MAX), 8.0), 1e-6);
  if (n < want && rr() < (want - n) / (f32(SLIME_MAX) - n + 1.0) * hit) {
    for (var k = 0u; k < 8u; k++) {
      let b = slimeAgents[hash3(i, stepNo, k) % SLIME_MAX];
      if (b.w >= 1.0 || b.w < 0.0) {
        let pn = slimeDecode(b.xy);
        let pb = basisOf(pn);
        let m = normalize(pn + (pb[0] * (rr() - 0.5) + pb[1] * (rr() - 0.5)) * (0.05 / PLANET_R));
        let h = slimeAngle(m, slimeHeading(pn, b.z + (rr() - 0.5) * 2.0));
        slimeAgents[i] = vec4f(slimeEncode(m), h, 0.5);
        break;
      }
    }
  }
}
