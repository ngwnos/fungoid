// ---- sim.wgsl: the living garden ----
// Entry points (one module, separate pipelines). Per step:
//   clearFields  - zero the per-step accumulators; organisms pay upkeep for
//                  their mycelium (orgStep, in its first workgroup)
//   deposit      - cells -> density, signal and territory (4 mm)
//   signal       - the signal diffuses
//   fields       - (every FIELD_EVERY steps) litter rots into food, spores
//                  drift and settle, light is refreshed every few steps
//   grow         - cells: feeding, starving, fighting, physics, growth,
//                  branching, the phase program and fruiting
//   reap         - dead cells hand their slots back
//   pack...      - (every PACK_EVERY steps) cells move down into freed slots
//   finalize     - next dispatch size
// Per frame (after the steps):
//   listFruits   - live fruit bodies, for neighbour tests
//   growFruits   - age fruit bodies, keep them clear of each other and of
//                  rock and wood, release spores
//   clearOcc, depositFruits - fruit-body occupancy
//   develop      - growing heads' surfaces develop (folds, bumps, pits...)
//   germinate    - settled spores start new colonies (events for the CPU)
//   shadeTissue  - light through tissue and the fields the renderer reads
// And: inoculate (new organisms), resetFields (fresh soil).

struct SimU {
  nInoc: u32,
  pad0: u32,
  steps: u32,              // simulation steps in this frame
  pad1: u32,
  sun: vec4f,              // towards the sun
  inoc: array<vec4f, 128>, // pairs: (x, y, z, organism), (mode, 0, 0, 0)
  brush: vec4f,            // the player's brush: where (xyz), radius
  brushOp: vec4f,          // what it does (1 water, 2 dry, 3 feed, 4 deplete), strength
  slimeAt: vec4f,          // a slime mould set down: where (octEncode), agents, request (slimesim.wgsl)
}

@group(0) @binding(0) var<uniform> sim: SimU;
@group(0) @binding(1) var<storage, read> pheno: array<f32>;
@group(0) @binding(2) var<storage, read_write> cells: array<Cell>;
// 0 cell slots used, 1 live fruit bodies listed, 2 germination events,
// 3 free cell slots, 4 cells to dispatch, 5 step, 6 fruit slot cursor
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;
// per organism: food, mycelium cells, step of last fruiting, live fruit bodies
@group(0) @binding(4) var<storage, read_write> pools: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> grid: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> fieldA: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> fieldB: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> fruits: array<Fruit>;
@group(0) @binding(9) var<storage, read_write> events: array<vec4f>;  // germination: position, parents
@group(0) @binding(10) var vol: texture_3d<f32>;
@group(0) @binding(11) var ground: texture_2d<f32>;
@group(0) @binding(12) var linSamp: sampler;
@group(0) @binding(13) var tissueSun: texture_storage_3d<rgba16float, write>;
@group(0) @binding(14) var fieldTex: texture_storage_3d<rgba16float, write>;
// head development: PAT_SLOTS fields of PAT_RES x PAT_RES, and the atlas the
// renderer samples (fruitdev.wgsl)
@group(0) @binding(15) var<storage, read_write> devField: array<f32>;
@group(0) @binding(16) var devOut: texture_storage_2d<rgba16float, write>;
// moss (bake_aux.wgsl): w its thickness
@group(0) @binding(20) var aux: texture_3d<f32>;
fn mossAt(p: vec3f) -> f32 {
  let a = textureSampleLevel(aux, linSamp, volUVW(p), 0.0).w;
  return select(0.0, a, a > 0.02);
}

const ORG_CELLS: i32 = 30000;     // per-organism ceiling on mycelium
const GARDEN_CELLS: u32 = 124000u; // the whole garden's (the box holds only so much)
const SET_AGE: u32 = 40u;         // steps after which grown tissue sets
const TEMPO_AERIAL: f32 = 0.4;    // fruiting tissue grows slower than mycelium
const TEMPO_VEG: f32 = 0.7;
const BRANCH_DECAY: f32 = 0.55;   // each split makes further splits on that line less likely
const VEG_CROWD: f32 = 6.0;       // cells per 4 mm voxel where mycelium stops
const ORG_FRUITS: i32 = 14;       // fruit bodies alive at once per organism
// an organism's fruit body count: the living ones, plus ROT_COUNT for each
// dying one (those still stand, but no longer hold back new ones)
const ROT_COUNT: i32 = 65536;
const BLINK_REACH: f32 = 6.0;      // cm: how far a blinking fruit body sees others blink
const UPKEEP: f32 = 0.00008;      // food per mycelium cell per step
const LITTER: f32 = 0.05;         // food a dead cell leaves behind
const RESERVE: f32 = 0.5;         // food a colony can store per mycelium cell
const HYPHA_LIFE: f32 = 36000.0;  // steps a hypha lives (about a year and a half, give or take)
const SETTLE: f32 = 0.003;        // how fast spores fall (cm per step)
// (MAX_EVENTS germinations are reported between readbacks: enough that none
// is turned away, which would be depends on thread order)
const NO_OWNER: u32 = 0xffffffffu;
const FEED_EVERY: u32 = 4u;       // steps between a hypha's feeding (staggered)
// In the grid buffer: per 4-voxel block, the last step any cell was in it;
// per 8-voxel block, the last step the signal was alive in it (above
// SIG_ALIVE, or emitted). The per-step grid passes skip blocks with nothing
// to do (GRID_ACT over SUBS, GRID_SIGACT over BLOCKS).
fn subIndex(s: vec3i) -> u32 { return u32((s.z * SUBS.y + s.y) * SUBS.x + s.x); }
const SIG_ALIVE: f32 = 0.01;
// In the grid buffer, per organism: its hyphae a mycoparasite touched last
// step, the strongest that did (parasitism << 6 | organism), and the
// infection its network carries: how far through it (f32 bits, 0..1), and
// the parasite (organism + 1, 0: none). (GRID_HOSTS)
fn hostLoad(org: u32) -> f32 { return bitcast<f32>(atomicLoad(&grid[GRID_HOSTS + org * 4u + 2u])); }
fn hostParasite(org: u32) -> u32 { return atomicLoad(&grid[GRID_HOSTS + org * 4u + 3u]); }
// Packing the cells (packCount..packFix): cells in each block of 256 slots
// (then cells before it), then L (cells), cells below slot L, holes below L.
// (GRID_PACK)
const PACK_BLOCKS: u32 = MAX_CELLS / 256u;
const PK_L: u32 = GRID_PACK + PACK_BLOCKS;
const PK_BELOW: u32 = PK_L + 1u;
const PK_HOLES: u32 = PK_L + 2u;
// What a step's passes read of the colonies, as each step begins (orgStep,
// after upkeep): per organism its store, its hyphae, when it last fruited,
// its fruit bodies; the garden's cells; and fruiting claims, this step's (the
// strongest wins) and the last step's winner. Reading these rather than the
// live counts other threads are changing, and keying every cell's random
// rolls to a number of its own rather than to its slot, a garden repeats
// exactly however the GPU orders its threads.
const SNAP_CELLS: u32 = GRID_SNAP + 4u * MAX_ORGS;
const SNAP_CLAIM: u32 = SNAP_CELLS + 4u;
const SNAP_WON: u32 = SNAP_CLAIM + MAX_ORGS;
// what its hyphae spent growing, this step and last
const SNAP_SPEND: u32 = SNAP_WON + MAX_ORGS;
const SNAP_SPENT: u32 = SNAP_SPEND + MAX_ORGS;
// What each live fruit body's neighbours read of it in a frame, as the frame
// began (listFruits; they may rewrite it meanwhile): by its place in the list,
// its head's middle (f32 bits), its head's radius, its birth step, its seed
// (GRID_FNB); then (GRID_SIGNEW) the blocks the signal pass found alive this
// step, merged at the next
fn snapFood(org: u32) -> f32 { return f32(bitcast<i32>(atomicLoad(&grid[GRID_SNAP + org * 4u]))) / POOL_FP; }
fn snapHyphae(org: u32) -> i32 { return bitcast<i32>(atomicLoad(&grid[GRID_SNAP + org * 4u + 1u])); }
fn snapFruited(org: u32) -> i32 { return bitcast<i32>(atomicLoad(&grid[GRID_SNAP + org * 4u + 2u])); }
fn snapFruits(org: u32) -> i32 { return bitcast<i32>(atomicLoad(&grid[GRID_SNAP + org * 4u + 3u])) & 0xffff; }
fn snapOrg(org: u32) {
  for (var k = 0u; k < 4u; k++) { atomicStore(&grid[GRID_SNAP + org * 4u + k], bitcast<u32>(atomicLoad(&pools[org * 4u + k]))); }
  let won = atomicExchange(&grid[SNAP_CLAIM + org], 0u);
  atomicStore(&grid[SNAP_WON + org], won);
  // (a claim won: it fruits this step, and this step's cells see it so)
  if (won != 0u) { atomicStore(&grid[GRID_SNAP + org * 4u + 2u], stepNo); }
  atomicStore(&grid[SNAP_SPENT + org], atomicExchange(&grid[SNAP_SPEND + org], 0u));
}

var<private> rs: u32;
var<private> stepNo: u32;
fn rinit(a: u32, b: u32) { rs = hash3(a, b, 0x51ed270bu); }
// A cell's own number (Cell.q.w), and its children's: from its own, the
// step and which child (1 grown on, 2 a branch, 3 another phase, 4 a knot)
fn cellId(c: Cell) -> u32 { return u32(c.q.w * 16777216.0); }
fn childSeed(id: u32, kind: u32) -> f32 { return u01(hash3(id, stepNo, kind)); }
// (a claim to fruit: the strongest this step fruits next step)
fn claimKey(id: u32, s: u32) -> u32 { return hash3(id, s, 0xf4u) | 1u; }
fn rr() -> f32 {
  rs = rs * 747796405u + 2891336453u;
  let w = ((rs >> ((rs >> 28u) + 4u)) ^ rs) * 277803737u;
  return f32((w >> 22u) ^ w) * (1.0 / 4294967296.0);
}
fn rdir() -> vec3f {
  let z = rr() * 2.0 - 1.0;
  let a = rr() * TAU;
  let s = sqrt(max(1.0 - z * z, 0.0));
  return vec3f(s * cos(a), z, s * sin(a));
}

fn world(p: vec3f) -> vec4f { return textureSampleLevel(vol, linSamp, volUVW(p), 0.0); }
// how far from the centre the ground is, in p's direction; how high p is
// above it
fn groundR(p: vec3f) -> f32 { return PLANET_R + textureSampleLevel(ground, linSamp, groundUV(upAt(p)), 0.0).x; }
fn aboveGround(p: vec3f) -> f32 { return length(p) - groundR(p); }
// (v less its part along up)
fn level(v: vec3f, up: vec3f) -> vec3f { return v - up * dot(v, up); }
// within the shell life keeps to
fn inLife(p: vec3f) -> bool {
  let r2 = dot(p, p);
  return r2 > LIFE_IN * LIFE_IN && r2 < LIFE_OUT * LIFE_OUT;
}
fn worldN(p: vec3f) -> vec3f {
  let e = 0.15;
  let k = vec2f(1.0, -1.0);
  return normalize(k.xyy * world(p + k.xyy * e).x + k.yyx * world(p + k.yyx * e).x +
                   k.yxy * world(p + k.yxy * e).x + k.xxx * world(p + k.xxx * e).x + vec3f(0.0, 1e-6, 0.0));
}

fn gridAt(field: u32, c: vec3i) -> u32 { return select(0u, atomicLoad(&grid[gi(field, f4Index(c))]), inF4(c)); }
fn densAt(c: vec3i) -> f32 { return f32(gridAt(GF_DENS, c)); }
fn occAt(p: vec3f) -> u32 { return gridAt(GF_OCC, f4Coord(p)); }
fn fieldAt(c: vec3i) -> vec4f { return fieldA[f4Index(c)]; }
// How wet it is in voxel v: the season, plus what has been watered or dried
// here (the field's x, which relaxes back over a season)
// how damp it is in voxel v (as the slow fields last left it: the weather
// there, and what the player watered or dried)
fn wetAt(v: u32) -> f32 { return fieldA[v].x; }
// the signal: written to slot (step + 1) & 1 by this step's signal pass
fn sigSlot(v: u32, slot: u32) -> f32 { return bitcast<f32>(atomicLoad(&grid[gi(GF_SIGNAL + slot, v)])); }
fn sigAt(c: vec3i) -> f32 { return sigSlot(f4Index(c), (stepNo + 1u) & 1u); }

// Central-difference gradients on the 4 mm grids, normalised to unit-ish.
fn densGrad(p: vec3f) -> vec4f {
  let c = f4Coord(p);
  let g = vec3f(densAt(c + vec3i(1, 0, 0)) - densAt(c - vec3i(1, 0, 0)),
                densAt(c + vec3i(0, 1, 0)) - densAt(c - vec3i(0, 1, 0)),
                densAt(c + vec3i(0, 0, 1)) - densAt(c - vec3i(0, 0, 1)));
  return vec4f(g, densAt(c));
}
fn sigGrad(p: vec3f) -> vec3f {
  let c = f4Coord(p);
  return vec3f(sigAt(c + vec3i(1, 0, 0)) - sigAt(c - vec3i(1, 0, 0)),
               sigAt(c + vec3i(0, 1, 0)) - sigAt(c - vec3i(0, 1, 0)),
               sigAt(c + vec3i(0, 0, 1)) - sigAt(c - vec3i(0, 0, 1)));
}
fn fieldGrad(p: vec3f, ch: u32) -> vec3f {
  let c = f4Coord(p);
  return vec3f(fieldAt(c + vec3i(1, 0, 0))[ch] - fieldAt(c - vec3i(1, 0, 0))[ch],
               fieldAt(c + vec3i(0, 1, 0))[ch] - fieldAt(c - vec3i(0, 1, 0))[ch],
               fieldAt(c + vec3i(0, 0, 1))[ch] - fieldAt(c - vec3i(0, 0, 1))[ch]);
}
fn unit(v: vec3f) -> vec3f {
  let l = length(v);
  return select(vec3f(0.0), v / l, l > 1e-6);
}

// Toward the ground other living colonies hold, two and four voxels out.
fn hostward(p: vec3f, org: u32) -> vec3f {
  let c = f4Coord(p);
  var d = vec3f(0.0);
  for (var k = 0u; k < 12u; k++) {
    var o = vec3i(0);
    o[(k / 2u) % 3u] = select(-2, 2, (k & 1u) == 1u) * select(1, 2, k >= 6u);
    let q = c + o;
    if (!inF4(q)) { continue; }
    let ow = atomicLoad(&grid[gi(GF_OWNER, f4Index(q))]);
    if (ow != NO_OWNER && (ow & 63u) != org && PG(ow & 63u, G_ALIVE) > 0.5) { d += vec3f(o); }
  }
  return unit(d);
}

fn dropLitter(p: vec3f, amount: f32) {
  let c = f4Coord(p);
  if (inF4(c)) { atomicAdd(&grid[gi(GF_LITTER_IN, f4Index(c))], u32(amount * 256.0)); }
}

// ------------------------------------------------------------ organisms ---

// How damp it is where a colony was set down (its voxel: GRID_HOME).
fn homeWet(org: u32) -> f32 { return fieldA[min(atomicLoad(&grid[GRID_HOME + org]), F4N - 1u)].x; }

// Mycelium costs food to keep alive; a colony that has eaten out its ground
// goes hungry and starts to die back.
fn orgStep(org: u32) {
  let h = GRID_HOSTS + org * 4u;
  let touch = atomicExchange(&grid[h], 0u);
  let pv = atomicExchange(&grid[h + 1u], 0u);
  if (PG(org, G_ALIVE) < 0.5) {
    atomicStore(&grid[h + 2u], 0u);
    atomicStore(&grid[h + 3u], 0u);
    snapOrg(org);
    return;
  }
  let n = max(atomicLoad(&pools[org * 4u + 1u]), 0);
  // a mycoparasite that touches the colony's hyphae gets into its network
  // (or a stronger one takes over), and spreads through it as fast as it can
  // against the colony's defences, which also clear it (a well-defended
  // colony rids itself of it; the dry season sets it back); it is gone with
  // its organism
  var load = hostLoad(org);
  var par = hostParasite(org);
  if (touch > 0u && pv != 0u && (par == 0u || par == (pv & 63u) + 1u || load < 0.05)) {
    if (par != (pv & 63u) + 1u) { load = 0.0; }
    par = (pv & 63u) + 1u;
    load = min(load + f32(touch) / f32(max(n, 60)) * 0.01, 1.0);
  }
  if (par > 0u) {
    let p = par - 1u;
    let grip = PG(p, G_PARASITE) * (1.0 - 0.8 * PG(org, G_COMBAT));
    load += load * (1.0 - load) * 0.0015 * grip * mix(0.3, 1.0, homeWet(org)) - load * 0.0003 * (0.5 + PG(org, G_COMBAT));
    if ((load < 1e-4 && touch == 0u) || PG(p, G_ALIVE) < 0.5 || PG(p, G_PARASITE) < 0.1) {
      load = 0.0;
      par = 0u;
    }
  }
  atomicStore(&grid[h + 2u], bitcast<u32>(load));
  atomicStore(&grid[h + 3u], par);
  // weapons (combat) and enzymes (absorption) cost a colony to keep up, and
  // so does moving food across a large network; through the dry of summer
  // it lies nearly dormant
  let rate = UPKEEP * (0.55 + 0.6 * PG(org, G_COMBAT) + 0.15 * PG(org, G_ABSORB)) * (1.0 + f32(n) / 10000.0)
           * mix(0.12, 1.0, homeWet(org));
  let cost = i32(f32(n) * rate * POOL_FP);
  let food = atomicSub(&pools[org * 4u], cost) - cost;
  // debt is capped so a starving colony recovers once it finds food again
  let debt = -i32(30.0 * POOL_FP);
  if (food < debt) { atomicStore(&pools[org * 4u], debt); }
  // a full store holds no more (its hyphae all fed last step on what it held
  // as the step began, so it may have run over)
  let full = i32(RESERVE * f32(max(n, 60)) * POOL_FP);
  if (food > full) { atomicStore(&pools[org * 4u], full); }
  snapOrg(org);
}

// ------------------------------------------------------------- fields ---

// The voxel thread t of a 64-thread workgroup over 8-voxel block b looks
// after in its 4-voxel quarter q (0..7).
// the blocks life can reach (dims.ts SHELL)
@group(0) @binding(21) var<uniform> shell: array<vec4u, SHELL_VEC4>;
fn shellBlock(k: u32) -> vec3u {
  let w = shell[k / 4u][k % 4u];
  return vec3u(w & 255u, (w >> 8u) & 255u, w >> 16u);
}

fn blockVoxel(b: vec3u, q: u32, t: u32) -> vec3i {
  return vec3i(b * 8u + vec3u(q & 1u, (q >> 1u) & 1u, q >> 2u) * 4u + vec3u(t & 3u, (t >> 2u) & 3u, t >> 4u));
}

// (only where there were cells last step: elsewhere they are clear already;
// a workgroup a block life can reach)
@compute @workgroup_size(64)
fn clearFields(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  stepNo = atomicLoad(&counters[5]);
  if (wg.x == 0u && t < MAX_ORGS) { orgStep(t); }
  if (wg.x == 0u && t == 0u) { atomicStore(&grid[SNAP_CELLS], atomicLoad(&counters[0]) - atomicLoad(&counters[3])); }
  let b = shellBlock(wg.x);
  if (t == 0u) {
    let k = blockIndex(vec3i(b));
    let m = atomicExchange(&grid[GRID_SIGNEW + k], 0u);
    if (m > 0u) { atomicMax(&grid[GRID_SIGACT + k], m); }
  }
  for (var q = 0u; q < 8u; q++) {
    let c = blockVoxel(b, q, t);
    if (atomicLoad(&grid[GRID_ACT + subIndex(c / 4)]) + 1u < stepNo || !inF4(c)) { continue; }
    let i = f4Index(c);
    atomicStore(&grid[gi(GF_DENS, i)], 0u);
    atomicStore(&grid[gi(GF_EXCITE, i)], 0u);
    atomicStore(&grid[gi(GF_OWNER, i)], NO_OWNER);
    atomicStore(&grid[gi(GF_PARA, i)], 0u);
    // what was eaten here, and fights lost here, last step
    let ate = atomicExchange(&grid[gi(GF_EATEN_NEW, i)], 0u);
    if (ate > 0u) { atomicAdd(&grid[gi(GF_EATEN, i)], ate); }
    let lost = atomicExchange(&grid[gi(GF_KILL_NEW, i)], 0u);
    if (lost > 0u) { atomicMax(&grid[gi(GF_KILL, i)], lost); }
  }
}

// Marks grid word a with this step.
fn markStep(a: u32) {
  if (atomicLoad(&grid[a]) != stepNo) { atomicMax(&grid[a], stepNo); }
}

@compute @workgroup_size(256)
fn deposit(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= atomicLoad(&counters[4])) { return; }
  var c = cells[i];
  if ((c.i.z & F_DEAD) != 0u) { return; }
  // where it moved to last step (grow held it aside)
  if ((c.i.z & F_NEWPOS) != 0u) {
    let np = c.q.xyz;
    c.q = vec4f(c.p.xyz, c.q.w);
    c.p = vec4f(np, c.p.w);
    c.i.z &= ~F_NEWPOS;
    cells[i] = c;
  }
  let k = f4Coord(c.p.xyz);
  if (!inF4(k)) { return; }
  stepNo = atomicLoad(&counters[5]);
  markStep(GRID_ACT + subIndex(k / 4));
  let v = f4Index(k);
  atomicAdd(&grid[gi(GF_DENS, v)], 1u);
  // firing tissue stirs the tissue here and next to here (excite)
  if (exciteOf(c.i.z) > 96u) {
    for (var n = 0u; n < 7u; n++) {
      var o = vec3i(0);
      if (n > 0u) { o[(n - 1u) / 2u] = select(-1, 1, (n & 1u) == 0u); }
      let q = k + o;
      if (!inF4(q)) { continue; }
      atomicAdd(&grid[gi(GF_EXCITE, f4Index(q))], select(1u, 2u, n == 0u));
      if (n > 0u) { markStep(GRID_ACT + subIndex(q / 4)); }
    }
  }
  let ph = phaseOf(c.i.z);
  // the colony that got here first holds the ground
  if (ph == 0u) { atomicMin(&grid[gi(GF_OWNER, v)], (c.i.w << 6u) | c.i.y); }
  // mycoparasites mark where they are, for the fruit bodies they can reach
  let para = PG(c.i.y, G_PARASITE);
  if (ph == 0u && para > 0.1) { atomicMax(&grid[gi(GF_PARA, v)], (u32(para * 255.0) << 6u) | c.i.y); }
  let emit = PQ(c.i.y, ph, Q_EMIT);
  if (emit > 0.0) {
    atomicAdd(&grid[gi(GF_SIG, v)], u32(emit * 256.0));
    markStep(GRID_SIGACT + blockIndex(k / 8));
  }
}

// Winds round the planet's axis, in bands: one way near the equator, the
// other a way off it, and again nearer the poles.
fn airFlow(p: vec3f) -> vec3f {
  let up = upAt(p);
  let east = cross(vec3f(0.0, 1.0, 0.0), up);
  return east * 0.008 * cos(3.0 * asin(clamp(up.y, -1.0, 1.0)));
}

fn airAt(c: vec3i) -> f32 { return select(0.0, fieldA[F4N + f4Index(c)].x, inF4(c)); }

fn airSample(p: vec3f) -> f32 {
  let g = (p - BOX_MIN) / F4 - 0.5;
  let c = vec3i(floor(g));
  let f = g - floor(g);
  let x00 = mix(airAt(c), airAt(c + vec3i(1, 0, 0)), f.x);
  let x10 = mix(airAt(c + vec3i(0, 1, 0)), airAt(c + vec3i(1, 1, 0)), f.x);
  let x01 = mix(airAt(c + vec3i(0, 0, 1)), airAt(c + vec3i(1, 0, 1)), f.x);
  let x11 = mix(airAt(c + vec3i(0, 1, 1)), airAt(c + vec3i(1, 1, 1)), f.x);
  return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

// Spore parents: up to two organisms (+1), seven bits each.
fn mergeParents(a: u32, b: u32) -> u32 {
  var o = a;
  for (var k = 0u; k < 2u; k++) {
    let q = (b >> (7u * k)) & 127u;
    if (q == 0u) { continue; }
    let p1 = o & 127u;
    let p2 = (o >> 7u) & 127u;
    if (p1 == 0u) { o = q; }
    else if (p1 != q && p2 == 0u) { o = p1 | (q << 7u); }
  }
  return o;
}

// The signal released this step diffuses and fades (only in and next to
// blocks where it is alive: elsewhere it has faded to nothing).
@compute @workgroup_size(64)
fn signal(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  stepNo = atomicLoad(&counters[5]);
  let wb = shellBlock(wg.x);
  let b = vec3i(wb);
  var live = 0u;
  for (var n = 0u; n < 7u; n++) {
    var o = vec3i(0);
    if (n > 0u) { o[(n - 1u) / 2u] = select(-1, 1, (n & 1u) == 0u); }
    let q = b + o;
    if (all(q >= vec3i(0)) && all(q < BLOCKS)) { live = max(live, atomicLoad(&grid[GRID_SIGACT + blockIndex(q)])); }
  }
  if (live + 4u < stepNo) { return; }
  let rd = stepNo & 1u;
  let hi = vec3i(F4_RES) - 1;
  for (var q = 0u; q < 8u; q++) {
    let c = blockVoxel(wb, q, t);
    if (!inF4(c)) { continue; }
    let i = f4Index(c);
    let nsum = sigSlot(f4Index(min(c + vec3i(1, 0, 0), hi)), rd) + sigSlot(f4Index(max(c - vec3i(1, 0, 0), vec3i(0))), rd)
             + sigSlot(f4Index(min(c + vec3i(0, 1, 0), hi)), rd) + sigSlot(f4Index(max(c - vec3i(0, 1, 0), vec3i(0))), rd)
             + sigSlot(f4Index(min(c + vec3i(0, 0, 1), hi)), rd) + sigSlot(f4Index(max(c - vec3i(0, 0, 1), vec3i(0))), rd);
    let emit = f32(atomicExchange(&grid[gi(GF_SIG, i)], 0u)) / 256.0;
    let v = (sigSlot(i, rd) * 0.4 + nsum * 0.1) * 0.985 + emit * 0.05;
    atomicStore(&grid[gi(GF_SIGNAL + (rd ^ 1u), i)], bitcast<u32>(v));
    // (held aside till the next step: what the blocks around read in this
    // pass does not depend on which of them ran first)
    if (v > SIG_ALIVE) { atomicMax(&grid[GRID_SIGNEW + blockIndex(b)], stepNo); }
  }
}

// Light, now and then: the sun's where it is up, through soil, rock and
// wood, tissue and fruit bodies up to 6 cm towards it, and the open sky's
// by day; none at night. Into the fields the next fields pass reads.
@compute @workgroup_size(4, 4, 4)
fn light(@builtin(global_invocation_id) gid: vec3u) {
  stepNo = atomicLoad(&counters[5]);
  if (any(gid >= F4_RES)) { return; }
  let p = BOX_MIN + (vec3f(gid) + 0.5) * F4;
  let v = f4Index(vec3i(gid));
  if (!inLife(p)) {
    fieldA[v].z = 0.0;
    return;
  }
  let L = sunDir(stepNo);
  let el = dot(upAt(p), L);
  var sun = 0.0;
  if (el > -0.05) {
    var tr = 1.0;
    for (var k = 1; k <= 15; k++) {
      let c = f4Coord(p + L * (f32(k) * F4));
      if (!inF4(c)) { break; }
      let w = f4Index(c);
      if ((atomicLoad(&grid[gi(GF_WORLD, w)]) & W_SOLID) != 0u) { tr *= 0.05; }
      tr *= exp(-f32(atomicLoad(&grid[gi(GF_DENS, w)]) + atomicLoad(&grid[gi(GF_OCC, w)])) * 0.03);
      if (tr < 0.01) { break; }
    }
    sun = tr * smoothstep(-0.05, 0.25, el);
  }
  fieldA[v].z = clamp(0.75 * sun + 0.25 * openAt(v) * smoothstep(-0.12, 0.22, el), 0.0, 1.0);
}

// How open to the sky a voxel is (0..1: GF_WORLD's bits 8-15, set on reset).
fn openAt(v: u32) -> f32 { return f32((atomicLoad(&grid[gi(GF_WORLD, v)]) >> 8u) & 255u) / 255.0; }

// How much of the sky over p is clear of soil, rock and wood within 4 cm:
// straight up and five ways around it, up counting double.
fn openSky(p: vec3f) -> f32 {
  let up = upAt(p);
  let b = basisOf(up);
  var open = 0.0;
  for (var k = 0; k < 6; k++) {
    var d = up;
    if (k > 0) {
      let a = f32(k) * 1.2566;
      d = normalize(up * 0.64 + (b[0] * cos(a) + b[1] * sin(a)) * 0.77);
    }
    var t = 0.3;
    var clear = 1.0;
    for (var s = 0; s < 24 && t < 4.0; s++) {
      let w = world(p + d * t).x;
      if (w < 0.05) {
        clear = 0.0;
        break;
      }
      t += max(w, 0.1);
    }
    open += clear * select(1.0, 2.0, k == 0);
  }
  return open / 7.0;
}

// The slow fields, every FIELD_EVERY steps: litter rots into food, spores
// ride the air, fall, and settle on surfaces.
@compute @workgroup_size(4, 4, 4)
fn fields(@builtin(global_invocation_id) gid: vec3u) {
  stepNo = atomicLoad(&counters[5]);
  if (any(gid >= F4_RES)) { return; }
  let c = vec3i(gid);
  let i = f4Index(c);
  let p = BOX_MIN + (vec3f(c) + 0.5) * F4;
  // (beyond the shell life keeps to and a voxel or two more, nothing ever
  // changes: the two field buffers hold the same there)
  let r2 = dot(p, p);
  if (r2 < (LIFE_IN - 2.0 * F4) * (LIFE_IN - 2.0 * F4) || r2 > (LIFE_OUT + 2.0 * F4) * (LIFE_OUT + 2.0 * F4)) { return; }
  let dt = f32(FIELD_EVERY);
  var f = fieldA[i];
  var s = fieldA[F4N + i];
  // how damp it is: the weather here, and what was watered or dried (which
  // drifts back to the weather's over about a season)
  s.w *= exp(-dt / 6000.0);
  f.x = clamp(climate(p, stepNo) + s.w, 0.0, 1.0);

  // litter rots into food. The soil gets food back from above all year,
  // more where it is fertile, and the fertile ground drifts across the
  // terrarium over the years; the leaves of autumn add more, deep in some
  // places and thin in others, somewhere else each year. Wood very slowly
  // softens (more of it becomes food) as long as it lies there.
  let wf = atomicLoad(&grid[gi(GF_WORLD, i)]);
  let lit = f.w + f32(atomicExchange(&grid[gi(GF_LITTER_IN, i)], 0u)) / 256.0;
  let rot = lit * 0.002 * dt;
  f.w = lit - rot;
  // a slime mould crawling over the surface here eats litter and spores
  var sd = 0.0;
  if ((wf & W_SHORE) != 0u) { sd = f32(slimeAt(p)); }
  let slimed = sd > 0.0 && abs(aboveGround(p)) < 0.6;
  let bite = select(0.0, min(sd * 0.0001 * dt, 0.0125 * dt), slimed);
  if (bite > 0.0) {
    slimeFeed(f.w * bite);
    f.w -= f.w * bite;
  }
  f.y = max(f.y - f32(atomicExchange(&grid[gi(GF_EATEN, i)], 0u)) / 65536.0, 0.0) + rot;
  if ((wf & W_TOPSOIL) != 0u) {
    let yr = f32(stepNo / YEAR);
    let years = f32(stepNo) / f32(YEAR);
    let q = upAt(p) * PLANET_R;
    let fertile = mix(0.25, 1.6, smoothstep(-0.3, 0.3, fbm3(q * 0.07 + vec3f(1.1, 0.6, 0.8) * years, 73u, 3u)));
    let deep = smoothstep(0.0, 0.5, fbm3(q * 0.08 + vec3f(yr * 37.1, yr * 19.7, yr * 23.3), 71u, 3u) + 0.25);
    f.y += (4e-4 * fertile * max(1.0 - f.y, 0.0) + leafFall(p, stepNo) * 1.0e-3 * deep * max(3.0 - f.y, 0.0)) * dt;
  }
  if ((wf & W_BARK) != 0u) { f.y += 1.2e-5 * max(3.0 - f.y, 0.0) * dt; }

  fieldB[i] = f;

  let sporeIn = f32(atomicExchange(&grid[gi(GF_SPORE_IN, i)], 0u)) / 256.0;
  let sporeOrg = atomicExchange(&grid[gi(GF_SPORE_ORG, i)], 0u);
  var air = 0.0;
  // (spore parents ride in the spore field's z: read from the last, written
  // to the next, so no thread reads what another is writing)
  var ids = u32(s.z + 0.5);
  // (inside soil, rock and wood there is no air to carry spores)
  if ((wf & W_SOLID) == 0u) {
    let src = p - (airFlow(p) - upAt(p) * SETTLE) * dt;
    air = airSample(src) * exp(-0.00075 * dt) + sporeIn;
    // parents travel with the air; settled ground keeps its own and may pick
    // up a second (in place: an upwind neighbour may already hold this pass's)
    let uc = f4Coord(src);
    let upwind = select(0u, u32(fieldA[F4N + f4Index(uc)].z + 0.5), inF4(uc));
    ids = select(upwind, mergeParents(ids, upwind), s.y > 0.02);
    ids = mergeParents(ids, sporeOrg);
    if ((wf & W_SHORE) != 0u) {
      // next to a surface: spores land
      let land = air * (1.0 - pow(0.52, dt / 4.0));
      air -= land;
      s.y += land;
    }
  }
  s.x = air;
  s.y *= exp(-0.000075 * dt);
  if (bite > 0.0) {
    slimeFeed(s.y * bite * 0.5);
    s.y -= s.y * bite;
  }
  if (s.x + s.y < 1e-4) { ids = 0u; }
  s.z = f32(ids);
  fieldB[F4N + i] = s;
}

// Food by material: soil is modest, wood is rich, rock and air have none.
@compute @workgroup_size(4, 4, 4)
fn resetFields(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= F4_RES)) { return; }
  let p = BOX_MIN + (vec3f(gid) + 0.5) * F4;
  let w = world(p);
  // (soil's food is in its top layer: it thins out a little way down)
  var nut = 0.0;
  if (w.x < 0.1) { nut = select(select(exp(min(w.x + 0.6, 0.0) / 0.5), 0.0, w.y > 0.5), 3.0, w.z > 0.5); }
  let i = f4Index(vec3i(gid));
  // (light, till the light pass: none outside the shell life keeps to)
  let lit = select(0.0, 1.0, inLife(p));
  let wet = climate(p, 0u);
  fieldA[i] = vec4f(wet, nut, lit, 0.0);
  fieldB[i] = vec4f(wet, nut, lit, 0.0);
  fieldA[F4N + i] = vec4f(0.0);
  fieldB[F4N + i] = vec4f(0.0);
  for (var k = 0u; k < GRID_FIELDS; k++) {
    if (k != GF_WORLD) { atomicStore(&grid[gi(k, i)], select(0u, NO_OWNER, k == GF_OWNER)); }
  }
  if (i < SNAP_WORDS) { atomicStore(&grid[GRID_SNAP + i], 0u); }
  if (i < PAT_SLOTS) { atomicStore(&grid[GRID_PATS + i], 0u); }
  if (i < MAX_ORGS * 4u) { atomicStore(&grid[GRID_HOSTS + i], 0u); }
  if (i < N_SUBS + N_BLOCKS) { atomicStore(&grid[GRID_ACT + i], 0u); }
  if (i < N_BLOCKS) { atomicStore(&grid[GRID_SIGNEW + i], 0u); }
  // (the signal slots hold f32 bits: 0u is 0.0)
  // the world is fixed until the next bake: the per-step passes read these
  // flags instead of sampling the volume
  var wf = 0u;
  if (w.x < 0.0) { wf |= W_SOLID; }
  if (w.x < F4) { wf |= W_SHORE; }
  if (w.x > -0.6 && w.x < 0.1 && w.y < 0.5 && w.z < 0.5) { wf |= W_TOPSOIL; }
  if (w.x < -1.0) { wf |= W_DEEP; }
  if (w.x > -0.6 && w.x < 0.1 && w.z > 0.5) { wf |= W_BARK; }
  // (and how open to the sky, in the air of the shell life keeps to)
  if (w.x > 0.0 && inLife(p)) { wf |= u32(openSky(p) * 255.0) << 8u; }
  atomicStore(&grid[gi(GF_WORLD, i)], wf);
}

// The player's brush (once a frame while they paint): water or dry the
// ground, or feed it or take its food away, in a ball around where they
// point. Food only goes where the soil or wood feeds.
@compute @workgroup_size(4, 4, 4)
fn brush(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= F4_RES)) { return; }
  let p = BOX_MIN + (vec3f(gid) + 0.5) * F4;
  let d = length(p - sim.brush.xyz);
  if (d > sim.brush.w) { return; }
  let k = smoothstep(sim.brush.w, sim.brush.w * 0.35, d) * sim.brushOp.y;
  let i = f4Index(vec3i(gid));
  let op = u32(sim.brushOp.x);
  // (watering and drying shift how damp it is from the weather's, held with
  // the spores')
  var f = fieldA[i];
  var s = fieldA[F4N + i];
  if (op == 1u) {
    s.w = min(s.w + k, 1.0);
    f.x = min(f.x + k, 1.0);
  } else if (op == 2u) {
    s.w = max(s.w - k, -1.0);
    f.x = max(f.x - k, 0.0);
  } else {
    if ((atomicLoad(&grid[gi(GF_WORLD, i)]) & (W_TOPSOIL | W_BARK)) == 0u) { return; }
    f.y = select(max(f.y - k * 2.0, 0.0), min(f.y + k * 2.0, 3.0), op == 3u);
  }
  fieldA[i] = f;
  fieldB[i] = f;
  fieldA[F4N + i] = s;
  fieldB[F4N + i] = s;
}

// ---------------------------------------------------------------- cells ---

// A filament knot: a cell of the fruiting phase, up from cell i
fn knot(i: u32, id: u32, org: u32, pos: vec3f, fphase: u32) {
  let k = allocCell();
  if (k == NO_PARENT) { return; }
  // (each a little askew its own way: knots set on one spot do not coincide)
  let tilt = vec3f(u01(hash3(id, stepNo, 5u)), u01(hash3(id, stepNo, 6u)), u01(hash3(id, stepNo, 7u))) - 0.5;
  let d0 = normalize(worldN(pos) + upAt(pos) + tilt * 0.4);
  let p0 = pos + d0 * CELL_LEN;
  cells[k] = Cell(vec4f(p0, PQ(org, fphase, Q_RADIUS)), vec4f(p0, childSeed(id, 4u)), vec4f(d0, 0.0),
                  vec4u(i, org, fphase | F_TIP, stepNo));
}

// A free slot: one handed back by a dead cell, else a new one.
fn allocCell() -> u32 {
  let n = atomicSub(&counters[3], 1u);
  if (n > 0u && n <= MAX_CELLS) { return atomicLoad(&grid[GRID_FREE + n - 1u]); }
  atomicAdd(&counters[3], 1u);
  let k = atomicAdd(&counters[0], 1u);
  return select(k, NO_PARENT, k >= MAX_CELLS);
}

fn rotateAway(d: vec3f, angle: f32) -> vec3f {
  let axis = unit(cross(d, rdir()));
  return normalize(d * cos(angle) + cross(axis, d) * sin(angle) + axis * dot(axis, d) * (1.0 - cos(angle)));
}

fn killCell(i: u32, flags: u32, org: u32, ph: u32, pos: vec3f) {
  cells[i].i.z = flags | F_DEAD | select(0u, F_DIED_ODD, (stepNo & 1u) != 0u);
  cells[i].d.w = f32(stepNo);
  if (ph == 0u) { atomicSub(&pools[org * 4u + 1u], 1); }
  dropLitter(pos, LITTER);
}

// An 8-bit count down over `len` steps (in whole units: slower than one a
// step, on every so many steps, staggered by `key`).
fn tick8(v: u32, len: f32, key: u32) -> u32 {
  if (v == 0u) { return 0u; }
  let per = 255.0 / max(len, 1.0);
  if (per >= 1.0) { return u32(max(f32(v) - per, 0.0)); }
  return select(v, v - 1u, (stepNo + key) % u32(1.0 / per) == 0u);
}

// Excitable tissue: what fired around a cell last step (GF_EXCITE, from the
// deposit) makes it fire in turn, or it fires of itself now and then, or as
// night falls. It flashes as it fires and must rest before it can fire
// again, so firing runs through the tissue in waves: rings from where it
// starts, spirals where waves break. Its state rides in its flags.
fn excite(flags: u32, org: u32, id: u32, pos: vec3f) -> u32 {
  var e = tick8(exciteOf(flags), PG(org, G_FLASH), id);
  var r = tick8(restOf(flags), PG(org, G_REST), id ^ 0x5bd1u);
  let ex = PG(org, G_EXCITE);
  if (r == 0u && ex > 0.0) {
    let k = f4Coord(pos);
    var stim = 0.0;
    if (inF4(k)) { stim = f32(atomicLoad(&grid[gi(GF_EXCITE, f4Index(k))])); }
    let p = PG(org, G_PACE) + ex * min(stim, 6.0) * 0.025 + PG(org, G_DUSK) * dusk(pos, stepNo);
    if (p > 0.0 && rr() < p) {
      e = 255u;
      r = 255u;
    }
  }
  return (flags & 0xffffu) | (e << 16u) | (r << 24u);
}

// What a tip senses, 0..1 (RULE_INPUTS in genes.ts; 0: always).
fn sense(k: u32, flags: u32, pos: vec3f, counter: f32, len: f32) -> f32 {
  if (k == 0u) { return 1.0; }
  let c = f4Coord(pos);
  if (!inF4(c)) { return 0.0; }
  let v = f4Index(c);
  switch (k) {
    case 1u: { return f32(exciteOf(flags)) / 255.0; }
    case 2u: { return nightness(pos, stepNo); }
    case 3u: { return clamp(sigAt(c), 0.0, 1.0); }
    case 4u: { return clamp(fieldA[v].y * 0.5, 0.0, 1.0); }
    case 5u: { return wetAt(v); }
    case 6u: { return clamp(fieldA[v].z, 0.0, 1.0); }
    case 7u: { return clamp(densAt(c) / 24.0, 0.0, 1.0); }
    case 8u: { return clamp(aboveGround(pos) / 6.0, 0.0, 1.0); }
    case 9u: { return clamp(counter / max(len, 1.0), 0.0, 1.0); }
    default: { return 0.0; }
  }
}

// A phase's rules: when its tips sense something they do something, as
// strongly as the rule says; what they do, summed (RULE_OUTPUTS: rise, seek
// light, follow signal, twist, branch, thicken, fire, move on, stop).
fn rules(org: u32, ph: u32, flags: u32, pos: vec3f, counter: f32) -> array<f32, 10> {
  var act = array<f32, 10>();
  let len = PQ(org, ph, Q_LENGTH);
  for (var r = 0u; r < 2u; r++) {
    let gain = PQ(org, ph, select(Q_R1_GAIN, Q_R2_GAIN, r == 1u));
    let o = u32(PQ(org, ph, select(Q_R1_OUT, Q_R2_OUT, r == 1u)));
    if (gain == 0.0 || o == 0u || o > 9u) { continue; }
    act[o] += gain * sense(u32(PQ(org, ph, select(Q_R1_IN, Q_R2_IN, r == 1u))), flags, pos, counter, len);
  }
  return act;
}

@compute @workgroup_size(256)
fn grow(@builtin(global_invocation_id) gid: vec3u) {
  stepNo = atomicLoad(&counters[5]);
  let i = gid.x;
  if (i >= atomicLoad(&counters[4])) { return; }
  var c = cells[i];
  var flags = c.i.z;
  if ((flags & F_DEAD) != 0u) { return; }
  // (a cell grown this step waits for the next: it may or may not have been
  // reached yet in this pass)
  if (c.i.w == stepNo) { return; }
  let org = c.i.y;
  let ph = phaseOf(flags);
  let id = cellId(c);
  rinit(id, stepNo);
  var pos = c.p.xyz;

  // hyphae age and die too: a colony holds its ground only by growing
  var life = PQ(org, ph, Q_LIFESPAN);
  if (ph == 0u) { life = min(life, HYPHA_LIFE * (0.7 + 0.6 * c.q.w)); }
  if (PG(org, G_ALIVE) < 0.5 || f32(stepNo - c.i.w) > life) {
    killCell(i, flags, org, ph, pos);
    return;
  }
  // a parent that died (before this step) lets go (its slot is about to be
  // reused). (When, from its flags alone: one load sees its death whole.)
  var pz = 0u;
  if (c.i.x != NO_PARENT) { pz = cells[c.i.x].i.z; }
  if ((pz & F_DEAD) != 0u && ((pz & F_DIED_ODD) != 0u) != ((stepNo & 1u) != 0u)) {
    cells[i].i.x = NO_PARENT;
    c.i.x = NO_PARENT;
  }
  // excitable tissue fires and rests (and any cell set firing fades)
  if (PG(org, G_EXCITE) > 0.0 || (flags >> 16u) != 0u) {
    let nf = excite(flags, org, id, pos);
    if (nf != flags) {
      flags = nf;
      cells[i].i.z = flags;
    }
  }

  let r = c.p.w;
  let age = f32(stepNo - c.i.w);
  let food = snapFood(org);
  let para = PG(org, G_PARASITE);

  if (ph == 0u) {
    let fi = f4Index(f4Coord(pos));
    // lost a fight here: the hyphae are overrun
    let kv = atomicLoad(&grid[gi(GF_KILL, fi)]);
    if ((kv & 127u) == org + 1u && stepNo - (kv >> 7u) < 400u && rr() < 0.03) {
      killCell(i, flags, org, ph, pos);
      return;
    }
    // Every FEED_EVERY steps (each hypha on a step of its own), and so
    // FEED_EVERY times as much or as likely: what lives off it, starving,
    // feeding.
    if ((id + stepNo) % FEED_EVERY == 0u) {
      let fe = f32(FEED_EVERY);
      // a mycoparasite among these hyphae lives off them, as far as the
      // colony's defences hold it off: it takes a share of what they gather
      // (below) and now and then digests one
      let pv = atomicLoad(&grid[gi(GF_PARA, fi)]);
      let pz = pv & 63u;
      let hit = select(0.0, f32(pv >> 6u) / 255.0 * (1.0 - 0.8 * PG(org, G_COMBAT)), pv != 0u && pz != org && PG(pz, G_ALIVE) > 0.5);
      if (hit > 0.0) {
        atomicAdd(&grid[GRID_HOSTS + org * 4u], FEED_EVERY);
        atomicMax(&grid[GRID_HOSTS + org * 4u + 1u], pv);
      }
      if (hit > 0.0 && rr() < 0.003 * fe * hit * mix(0.3, 1.0, wetAt(fi))) {
        // (the hypha, and its share of the colony's store)
        let share = i32(max(food, 0.0) / f32(max(snapHyphae(org), 1)) * POOL_FP);
        atomicSub(&pools[org * 4u], share);
        atomicAdd(&pools[pz * 4u], i32(0.2 * POOL_FP) + share);
        killCell(i, flags, org, ph, pos);
        return;
      }
      // food here, less what other hyphae have eaten since the fields updated
      // (many share a voxel: they eat from one tally)
      let nut = max(fieldA[fi].y - f32(atomicLoad(&grid[gi(GF_EATEN, fi)])) / 65536.0, 0.0);
      // a hungry colony dies back, first where it has eaten the ground bare
      let wet = wetAt(fi);
      if (food < 0.0 && rr() < select(0.00015, 0.001, nut < 0.05) * fe * mix(0.3, 1.0, wet) * smoothstep(150.0, 1500.0, age)) {
        killCell(i, flags, org, ph, pos);
        return;
      }
      // feeding: hyphae hold only so much (a colony with its store full
      // leaves the rest in the ground), and take it up only from damp ground
      let w = world(pos);
      // (a slime mould crawling over it eats mycelium near the surface)
      let sd = slimeAt(pos);
      if (sd > 0 && w.x > -0.5 && rr() < 0.00002 * fe * f32(min(sd, 20))) {
        slimeFeed(0.1);
        killCell(i, flags, org, ph, pos);
        return;
      }
      let pref = PG(org, G_SUBSTRATE);
      let eff = mix(1.0 - pref, pref, w.z) * 2.0;
      let full = food >= RESERVE * f32(max(snapHyphae(org), 60));
      // (a mycoparasite gets little from the ground)
      let take = select(min(nut, PG(org, G_ABSORB) * 0.0008 * fe * eff * mix(0.2, 1.0, wet) * (1.0 - 0.9 * para)), 0.0, full);
      if (take > 0.0) {
        atomicAdd(&grid[gi(GF_EATEN_NEW, fi)], u32(take * 65536.0));
        // (a parasite here, or one in the network, takes its share)
        let lost = take * 0.7 * hit;
        if (lost > 0.0) { atomicAdd(&pools[pz * 4u], i32(lost * POOL_FP + rr())); }
        let hp = hostParasite(org);
        let sapped = select(0.0, min(take * 0.3 * hostLoad(org), take - lost), hp > 0u);
        if (sapped > 0.0) { atomicAdd(&pools[(hp - 1u) * 4u], i32(sapped * POOL_FP + rr())); }
        // (in whole thousandths, rounded at random so a trickle still counts)
        atomicAdd(&pools[org * 4u], i32((take - lost - sapped) * POOL_FP + rr()));
      }
    }
  }

  // established mycelium near the surface starts fruiting: one initiation
  // per cooldown, mostly near the growing edge of the colony (so rings form).
  // It becomes a cluster of fruit bodies, or a filament knot whose signal
  // recruits neighbours.
  if (ph == 0u) {
    let fphase = u32(PG(org, G_FRUIT_PHASE));
    let filaments = fphase > 0u && fphase < PHASES;
    let body = PG(org, G_BODY);
    // the claim it won last step
    var knotted = false;
    if (atomicLoad(&grid[SNAP_WON + org]) == claimKey(id, stepNo - 1u)) {
      atomicStore(&pools[org * 4u + 2u], i32(stepNo));
      if (rr() < body) { spawnCluster(org, pos); }
      else if (filaments) {
        knot(i, id, org, pos, fphase);
        knotted = true;
      }
    }
    // (the chance first: on most steps no hypha starts one, and where is
    // costlier to find out)
    let prob = PG(org, G_FRUIT_PROB);
    let cooldown = clamp(2.0 / max(prob, 1e-4), 60.0, 20000.0);
    let last = snapFruited(org);
    let edge = 0.3 + exp(-age / 700.0);
    let tryBody = f32(i32(stepNo) - last) > cooldown && rr() < 0.002 * edge;
    let sig = select(0.0, sigAt(f4Coord(pos)), filaments);
    let tryKnot = !tryBody && filaments && sig > 0.4 && rr() < prob * sig * 0.5;
    // (only when the ground is wet enough for it, and on enough of a store:
    // a share of what it can hold, up to what a large colony can spare)
    let store = RESERVE * f32(clamp(snapHyphae(org), 60, 1200));
    if ((tryBody || tryKnot) && (filaments || body > 0.0) && food > max(PG(org, G_FRUIT_AT) * store, 20.0)
        && wetAt(f4Index(f4Coord(pos))) > PG(org, G_FRUIT_WET)) {
      // wood-lovers fruit only from wood
      let w0 = world(pos);
      let wd = w0.x;
      let wn = worldN(pos);
      let open = openAt(f4Index(f4Coord(pos + wn * F4)));
      // only where there is open air under the sky; shelf-formers take the
      // sides of things, not their tops
      let lat = PG(org, G_LATERAL);
      let wu = dot(wn, upAt(pos));
      let side = (wu > -0.4) && (wu < 0.6);
      let faceOk = select(wu > 0.3, side, lat > 0.5);
      if ((w0.z > 0.5 || PG(org, G_SUBSTRATE) < 0.6) && wd > -0.25 && wd < 0.08 && open > 0.15 && faceOk) {
        // a flush (one a cooldown): a claim, the strongest of this step's
        // fruiting next step; filament knots start at once
        // (one knot a step: two would be the same knot)
        if (tryBody) { atomicMax(&grid[SNAP_CLAIM + org], claimKey(id, stepNo)); }
        else if (filaments && !knotted) { knot(i, id, org, pos, fphase); }
      }
    }
  }

  // physics for free-standing tissue; it sets once grown
  if ((flags & F_FIXED) == 0u && stepNo - c.i.w > SET_AGE) {
    flags = flags | F_FIXED;
    cells[i].i.z = flags;
  }
  if ((flags & F_FIXED) == 0u) {
    let prev = c.q.xyz;
    var np = pos + (pos - prev) * 0.3 - upAt(pos) * 0.0012;
    let par = c.i.x;
    if (par != NO_PARENT) {
      let pp = cells[par].p.xyz;
      let cur = unit(np - pp);
      let stiff = PQ(org, ph, Q_STIFF);
      let want = unit(mix(cur, c.d.xyz, stiff) + upAt(np) * 1e-5);
      np = mix(np, pp + want * CELL_LEN, 0.85);
    }
    // crowding pushes tissue apart; bundling phases also draw it together
    let dg = densGrad(np);
    let crowd = PQ(org, ph, Q_CROWD);
    let gdir = unit(dg.xyz);
    np -= gdir * min(max(dg.w - 24.0, 0.0) * 0.0003, 0.006);
    np += gdir * max(crowd, 0.0) * 0.0008;
    // solids (tissue that hugs what it grows on runs over the moss, not
    // under it)
    let clear = r + select(0.0, mossAt(np), ph > 0u && PQ(org, ph, Q_SURFACE) < -0.5);
    let w = world(np).x;
    if (w < clear) { np += worldN(np) * (clear - w); }
    // (and stays within the shell life keeps to)
    let rn = length(np);
    np *= clamp(rn, LIFE_IN, LIFE_OUT) / max(rn, 1e-4);
    // (where it moves to, held aside till the next step's deposit: the
    // tissue it hangs from and on reads where it was)
    cells[i].q = vec4f(np, c.q.w);
    flags = flags | F_NEWPOS;
    cells[i].i.z = flags;
    pos = np;
  }

  // older mycelium of a well-fed colony sprouts again toward empty ground
  // next to it that still has food (where its own hyphae died, or nobody's
  // ever grew); a mycoparasite's, toward other colonies' ground
  if ((flags & F_TIP) == 0u && ph == 0u && age > f32(SET_AGE) && rr() < 0.003) {
    let sd = rdir();
    let su = upAt(pos);
    let dir = unit(level(sd, su) + su * dot(sd, su) * 0.3);
    let q = f4Coord(pos + dir * F4 * 1.5);
    let ow = gridAt(GF_OWNER, q);
    let hostGround = para > 0.3 && ow != NO_OWNER && (ow & 63u) != org && PG(ow & 63u, G_ALIVE) > 0.5;
    if (inF4(q) && ((densAt(q) < 0.5 && (gridAt(GF_WORLD, q) & (W_TOPSOIL | W_BARK)) != 0u
        && fieldA[f4Index(q)].y > 0.3 && vigour(org) > 1.0) || (hostGround && vigour(org) > 0.7))) {
      flags |= F_TIP;
      cells[i].i.z = flags;
      cells[i].d = vec4f(dir, 0.0);
      c.d = cells[i].d;
    }
  }
  if ((flags & F_TIP) == 0u) { return; }

  // growth slows almost to a stop in the dry of summer
  let here = wetAt(f4Index(f4Coord(pos)));
  let season = select(mix(0.35, 1.0, here), mix(0.12, 1.15, here), ph == 0u);
  if (rr() >= PQ(org, ph, Q_SPEED) * select(TEMPO_AERIAL, TEMPO_VEG, ph == 0u) * season) { return; }
  if (ph == 0u && snapHyphae(org) > ORG_CELLS) { return; }
  // (the garden holds only so much: growth falls away near the limit)
  if (ph == 0u && rr() * 0.03 * f32(GARDEN_CELLS) >= f32(GARDEN_CELLS) - f32(atomicLoad(&grid[SNAP_CELLS]))) { return; }
  // mycelium extends only on a surplus: while the colony's store stays at
  // least partly full (so it stops before it overdraws), as many tips as the
  // surplus pays for, by what they spent last step. A mycoparasite spends
  // down its store hunting for hosts.
  if (ph == 0u) {
    let surplus = food - 0.4 * RESERVE * f32(max(snapHyphae(org), 60)) * (1.0 - 0.6 * para);
    if (surplus < 0.0) { return; }
    let spent = f32(atomicLoad(&grid[SNAP_SPENT + org])) / POOL_FP;
    if (spent > surplus && rr() * spent >= surplus) { return; }
  }
  let rad = PQ(org, ph, Q_RADIUS);
  let cost = 0.18 + 0.75 * (rad / 0.04) * (rad / 0.04);
  let costFp = i32(cost * POOL_FP);
  if (food * POOL_FP < f32(costFp)) { return; }
  atomicSub(&pools[org * 4u], costFp);

  // tropisms
  var d = c.d.xyz;
  let dg = densGrad(pos);
  let up = upAt(pos);
  var acc = d * PQ(org, ph, Q_PERSIST)
          + up * PQ(org, ph, Q_UP)
          + rdir() * PQ(org, ph, Q_WANDER)
          + unit(dg.xyz) * PQ(org, ph, Q_CROWD)
          + unit(sigGrad(pos)) * PQ(org, ph, Q_SIGNAL)
          + unit(fieldGrad(pos, 1u)) * PQ(org, ph, Q_NUTRIENT)
          + unit(fieldGrad(pos, 2u)) * PQ(org, ph, Q_LIGHT)
          + worldN(pos) * PQ(org, ph, Q_SURFACE)
          + unit(cross(d, up)) * PQ(org, ph, Q_TWIST);
  // a mycoparasite's hyphae turn toward other colonies
  if (ph == 0u && para > 0.05) { acc += hostward(pos, org) * para * 2.5; }
  // and what its rules have it do
  var act = array<f32, 10>();
  if (PQ(org, ph, Q_R1_GAIN) != 0.0 || PQ(org, ph, Q_R2_GAIN) != 0.0) {
    act = rules(org, ph, flags, pos, c.d.w);
    acc += up * act[1] + unit(fieldGrad(pos, 2u)) * act[2] + unit(sigGrad(pos)) * act[3]
         + unit(cross(d, up)) * act[4];
  }
  var nd = unit(acc);
  if (dot(nd, nd) < 0.5) { nd = d; }

  if (ph == 0u) {
    // the mycelium stays inside its substrate
    let np = pos + nd * CELL_LEN;
    let w = world(np);
    if (w.x > 0.02) {
      let n = worldN(np);
      nd = unit(nd - n * (max(dot(nd, n), 0.0) * 1.6 + 0.2));
    }
    let p2 = pos + nd * CELL_LEN;
    let v2 = f4Index(f4Coord(p2));
    // ground another colony holds: they fight. The loser's hyphae there die
    // and the line between them darkens.
    var won = false;
    let ow = atomicLoad(&grid[gi(GF_OWNER, v2)]);
    // a mycoparasite grows along other colonies' hyphae, seldom out into
    // ground nobody holds
    if (ow == NO_OWNER && rr() < para * 1.2 - 0.2) {
      cells[i].d = vec4f(unit(-d + rdir() * 0.8), c.d.w);
      atomicAdd(&pools[org * 4u], costFp);
      return;
    }
    if (ow != NO_OWNER) {
      let rival = ow & 63u;
      if (rival != org && PG(rival, G_ALIVE) > 0.5) {
        if (rr() < para * (1.0 - 0.8 * PG(rival, G_COMBAT))) {
          // a mycoparasite slips in among the host's hyphae, or turns away to
          // try elsewhere; it does not fight
          won = true;
        } else if (para > 0.3) {
          cells[i].d = vec4f(unit(-d + rdir() * 0.8), c.d.w);
          atomicAdd(&pools[org * 4u], costFp);
          return;
        } else {
          atomicAdd(&grid[gi(GF_SCAR, v2)], 1u);
          // a well-fed colony pushes into a hungry one's ground
          let atk = PG(org, G_COMBAT) * (0.6 + 0.8 * rr()) * vigour(org);
          let def = (PG(rival, G_COMBAT) * (0.6 + 0.8 * rr()) + 0.12) * vigour(rival);
          if (atk > def) {
            atomicMax(&grid[gi(GF_KILL_NEW, v2)], (rival + 1u) | (stepNo << 7u));
            won = true;
          } else {
            cells[i].i.z = flags & ~F_TIP;
            atomicAdd(&pools[org * 4u], costFp);
            return;
          }
        }
      }
    }
    // colonised ground: the tip retires (hyphae fuse rather than pile up)
    if (!won && c.d.w > 6.0 && densAt(f4Coord(p2)) > VEG_CROWD) {
      cells[i].i.z = flags & ~F_TIP;
      atomicAdd(&pools[org * 4u], costFp);
      return;
    }
    let w2 = world(p2);
    if (w2.x > 0.05 || w2.y > 0.5 || !inLife(p2)) {
      // blocked (air or rock): turn around a little and try again later
      cells[i].d = vec4f(unit(-d + rdir() * 0.8), c.d.w);
      atomicAdd(&pools[org * 4u], costFp);
      return;
    }
  } else if (occAt(pos + nd * CELL_LEN) > 0u) {
    // aerial tissue does not grow into fruit bodies
    cells[i].d = vec4f(unit(-d + rdir() * 0.8), c.d.w);
    atomicAdd(&pools[org * 4u], costFp);
    return;
  }

  let counter = c.d.w + 1.0;
  var nphase = ph;
  var ncount = counter;
  var tip = F_TIP;
  let moveOn = act[8] > 0.0 && rr() < act[8] * 0.2;
  if (act[9] > 0.0 && rr() < act[9] * 0.2) { tip = 0u; }
  if (counter >= PQ(org, ph, Q_LENGTH) || moveOn) {
    nphase = u32(PQ(org, ph, Q_NEXT));
    ncount = 0.0;
    if (nphase >= PHASES) {
      nphase = ph;
      tip = 0u;
    }
  }
  let gen = (flags & F_GEN) >> F_GEN_SHIFT;
  let wantBranch = rr() < PQ(org, ph, Q_BRANCH) * pow(BRANCH_DECAY, f32(gen)) * max(1.0 + act[5], 0.0);
  let sphase = u32(PQ(org, ph, Q_SPAWN_PHASE));
  let wantSpawn = sphase < PHASES && rr() < PQ(org, ph, Q_SPAWN);
  let k = allocCell();
  if (k == NO_PARENT) { return; }
  if (nphase == 0u) { atomicAdd(&pools[org * 4u + 1u], 1); }
  if (ph == 0u) { atomicAdd(&grid[SNAP_SPEND + org], u32(costFp)); }

  let fixedBit = select(0u, F_FIXED, nphase == 0u);
  let np = pos + nd * CELL_LEN;
  let keepGen = select(0u, gen << F_GEN_SHIFT, nphase == ph);
  let childGen = min(gen + 1u, 15u) << F_GEN_SHIFT;
  // (thickened, or born firing, by its rules)
  let nr = PQ(org, nphase, Q_RADIUS) * clamp(1.0 + 0.5 * act[6], 0.3, 2.5);
  let fire = select(0u, 0xffff0000u, act[7] > 0.0 && rr() < act[7]);
  cells[k] = Cell(vec4f(np, nr), vec4f(np, childSeed(id, 1u)), vec4f(nd, ncount),
                  vec4u(i, org, nphase | tip | fixedBit | select(keepGen, childGen, wantBranch && nphase == ph) | fire, stepNo));
  cells[i].i.z = flags & ~F_TIP;
  // a program that ends can end in a head
  if (tip == 0u && ph != 0u && rr() < PG(org, G_TIP_HEADS)) {
    spawnFruit(org, np, nd, 0.0, FR_TIPHEAD, 0);
  }
  if (wantBranch) {
    let kb = allocCell();
    if (kb != NO_PARENT) {
      let bd = rotateAway(nd, PQ(org, ph, Q_ANGLE));
      let bp = pos + bd * CELL_LEN;
      cells[kb] = Cell(vec4f(bp, rad), vec4f(bp, childSeed(id, 2u)), vec4f(bd, counter),
                       vec4u(i, org, ph | F_TIP | select(0u, F_FIXED, ph == 0u) | childGen, stepNo));
      if (ph == 0u) { atomicAdd(&pools[org * 4u + 1u], 1); }
    }
  }
  if (wantSpawn) {
    let ks = allocCell();
    if (ks != NO_PARENT) {
      let rise = sign(PQ(org, sphase, Q_UP) + 1e-3);
      let sd = unit(nd * 0.25 + upAt(pos) * rise + rdir() * 0.15);
      let sp = pos + sd * CELL_LEN;
      cells[ks] = Cell(vec4f(sp, PQ(org, sphase, Q_RADIUS)), vec4f(sp, childSeed(id, 3u)), vec4f(sd, 0.0),
                       vec4u(i, org, sphase | F_TIP | select(0u, F_FIXED, sphase == 0u), stepNo));
      if (sphase == 0u) { atomicAdd(&pools[org * 4u + 1u], 1); }
    }
  }
}

// Dead cells hand their slots back once their children have let go.
@compute @workgroup_size(256)
fn reap(@builtin(global_invocation_id) gid: vec3u) {
  stepNo = atomicLoad(&counters[5]);
  let i = gid.x;
  if (i >= atomicLoad(&counters[4])) { return; }
  let fl = cells[i].i.z;
  if ((fl & F_DEAD) == 0u || (fl & F_FREED) != 0u) { return; }
  if (f32(stepNo) - cells[i].d.w < 8.0) { return; }
  let n = atomicAdd(&counters[3], 1u);
  if (n >= MAX_CELLS) {
    atomicSub(&counters[3], 1u);
    return;
  }
  cells[i].i.z = fl | F_FREED;
  atomicStore(&grid[GRID_FREE + n], i);
}

// Packing the cells, every PACK_EVERY steps: the cells at or above slot L
// (as many slots as there are cells) move down into the slots freed below
// it, so the per-cell passes cover only as many slots as there are cells.
// Cells below L stay where they are. While it runs the free stack (left
// empty after) holds the pairs: hole r at GRID_FREE + r, the cell that
// fills it at GRID_FREE + MAX_CELLS / 2 + r.
var<workgroup> packN: u32;
var<workgroup> packCnt: atomic<u32>;
var<workgroup> packSum: array<u32, 256>;

fn packUsed() -> u32 { return min(atomicLoad(&counters[0]), MAX_CELLS); }
fn packLive(i: u32, n: u32) -> u32 { return select(0u, 1u, i < n && (cells[i].i.z & F_FREED) == 0u); }

// inclusive prefix sum of packSum over the workgroup
fn packScanWg(li: u32) {
  for (var o = 1u; o < 256u; o <<= 1u) {
    var v = 0u;
    if (li >= o) { v = packSum[li - o]; }
    workgroupBarrier();
    packSum[li] += v;
    workgroupBarrier();
  }
}

// cells in each block of 256 slots
@compute @workgroup_size(256)
fn packCount(@builtin(global_invocation_id) gid: vec3u, @builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) {
    packN = packUsed();
    atomicStore(&packCnt, 0u);
  }
  let n = workgroupUniformLoad(&packN);
  if (wg.x * 256u >= n) { return; }
  if (packLive(gid.x, n) == 1u) { atomicAdd(&packCnt, 1u); }
  workgroupBarrier();
  if (li == 0u) { atomicStore(&grid[GRID_PACK + wg.x], atomicLoad(&packCnt)); }
}

// cells before each block; L, the cells below L and the holes below it
@compute @workgroup_size(256)
fn packScan(@builtin(local_invocation_index) li: u32) {
  if (li == 0u) { packN = packUsed(); }
  let n = workgroupUniformLoad(&packN);
  let blocks = (n + 255u) / 256u;
  let per = PACK_BLOCKS / 256u;
  var s = 0u;
  for (var k = 0u; k < per; k++) {
    let b = li * per + k;
    if (b < blocks) { s += atomicLoad(&grid[GRID_PACK + b]); }
  }
  packSum[li] = s;
  workgroupBarrier();
  packScanWg(li);
  var run = packSum[li] - s;
  for (var k = 0u; k < per; k++) {
    let b = li * per + k;
    if (b < blocks) {
      let c = atomicLoad(&grid[GRID_PACK + b]);
      atomicStore(&grid[GRID_PACK + b], run);
      run += c;
    }
  }
  storageBarrier();
  workgroupBarrier();
  if (li == 0u) {
    let L = packSum[255];
    var below = L;
    let wl = L / 256u;
    if (wl < blocks) {
      below = atomicLoad(&grid[GRID_PACK + wl]);
      for (var j = wl * 256u; j < L; j++) { below += packLive(j, n); }
    }
    atomicStore(&grid[PK_L], L);
    atomicStore(&grid[PK_BELOW], below);
    atomicStore(&grid[PK_HOLES], L - below);
  }
}

// each hole below L, and each cell at or above it, takes its place in line
@compute @workgroup_size(256)
fn packRank(@builtin(global_invocation_id) gid: vec3u, @builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) { packN = packUsed(); }
  let n = workgroupUniformLoad(&packN);
  if (wg.x * 256u >= n) { return; }
  let i = gid.x;
  let live = packLive(i, n);
  packSum[li] = live;
  workgroupBarrier();
  packScanWg(li);
  if (i >= n) { return; }
  // cells in the slots below this one
  let before = atomicLoad(&grid[GRID_PACK + wg.x]) + packSum[li] - live;
  let L = atomicLoad(&grid[PK_L]);
  if (i < L && live == 0u) { atomicStore(&grid[GRID_FREE + i - before], i); }
  if (i >= L && live == 1u) { atomicStore(&grid[GRID_FREE + MAX_CELLS / 2u + before - atomicLoad(&grid[PK_BELOW])], i); }
}

// the cells move down; their old slots say where to
@compute @workgroup_size(256)
fn packMove(@builtin(global_invocation_id) gid: vec3u) {
  let r = gid.x;
  if (r >= atomicLoad(&grid[PK_HOLES])) { return; }
  let dst = atomicLoad(&grid[GRID_FREE + r]);
  let src = atomicLoad(&grid[GRID_FREE + MAX_CELLS / 2u + r]);
  cells[dst] = cells[src];
  cells[src].i.x = dst;
  cells[src].i.z = F_DEAD | F_FREED | F_MOVED;
}

// children follow their parents; the slots above L are let go
@compute @workgroup_size(256)
fn packFix(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let L = atomicLoad(&grid[PK_L]);
  if (i == 0u) {
    atomicStore(&counters[0], L);
    atomicStore(&counters[3], 0u);
  }
  if (i >= L) { return; }
  let par = cells[i].i.x;
  if (par == NO_PARENT || par < L) { return; }
  let moved = (cells[par].i.z & F_MOVED) != 0u;
  cells[i].i.x = select(NO_PARENT, cells[par].i.x, moved);
}

// Seed organisms: a small star of mycelial tips at each requested point.
// Mode 0: in the soil below (x, z); 1: on the first surface below; 2: at
// (x, y, z) exactly.
@compute @workgroup_size(16)
fn inoculate(@builtin(local_invocation_index) li: u32) {
  stepNo = atomicLoad(&counters[5]);
  for (var j = 0u; j < sim.nInoc; j++) {
    let s = sim.inoc[j * 2u];
    let mode = u32(sim.inoc[j * 2u + 1u].x);
    let org = u32(s.w);
    var p = s.xyz;
    if (mode == 0u) {
      // a mycoparasite's spore germinates where a host nearby is growing
      // (its youngest ground)
      if (PG(org, G_PARASITE) > 0.3) {
        var best = 0u;
        var at = p;
        for (var k = 0u; k < 128u; k++) {
          let rad = 0.08 * f32(k);
          let a = f32(k) * 2.39996;
          let b = basisOf(upAt(p));
          var q = upAt(p + (b[0] * cos(a) + b[1] * sin(a)) * rad);
          q *= groundR(q) - 0.15;
          let ow = gridAt(GF_OWNER, f4Coord(q));
          if (ow != NO_OWNER && (ow & 63u) != org && PG(ow & 63u, G_ALIVE) > 0.5 && (ow >> 6u) + 1u > best) {
            best = (ow >> 6u) + 1u;
            at = q;
          }
        }
        p = at;
      }
      p = upAt(p) * (groundR(p) - 0.15);
    } else if (mode == 1u) {
      // drop from high above onto whatever is below, then sink just inside it
      let down = -upAt(p);
      p = -down * LIFE_OUT;
      for (var k = 0; k < 96; k++) {
        let d = world(p).x;
        if (d < 0.02) { break; }
        p += down * max(d * 0.9, 0.01);
      }
      p += down * 0.12;
    } else {
      p -= worldN(p) * (world(p).x + 0.12);
    }
    if (li == 0u) {
      atomicStore(&pools[org * 4u], i32(40.0 * POOL_FP));
      atomicStore(&pools[org * 4u + 1u], 0);
      atomicStore(&pools[org * 4u + 2u], i32(stepNo));
      atomicStore(&pools[org * 4u + 3u], 0);
      for (var k = 0u; k < 4u; k++) { atomicStore(&grid[GRID_HOSTS + org * 4u + k], 0u); }
      atomicStore(&grid[GRID_HOME + org], f4Index(f4Coord(p)));
    }
    storageBarrier();
    rinit(li, org * 977u + stepNo);
    let k = allocCell();
    if (k != NO_PARENT) {
      atomicAdd(&pools[org * 4u + 1u], 1);
      let a = (f32(li) + rr() * 0.5) / 16.0 * TAU;
      let b = basisOf(upAt(p));
      let d = normalize(b[0] * cos(a) + b[1] * sin(a) - upAt(p) * 0.15);
      let q = p + d * CELL_LEN * 0.5;
      cells[k] = Cell(vec4f(q, PQ(org, 0u, Q_RADIUS)), vec4f(q, u01(hash3(org * 977u + li, stepNo, j + 0x1d0u))), vec4f(d, 0.0),
                      vec4u(NO_PARENT, org, F_TIP | F_FIXED, stepNo));
    }
    storageBarrier();
  }
}

// -------------------------------------------------------------- fruits ---

// Settle a point onto the nearest surface.
fn toSurface(p: vec3f) -> vec3f {
  var q = p;
  for (var k = 0; k < 4; k++) { q -= worldN(q) * world(q).x; }
  return q;
}

// How well fed a colony is, 0.5 (starving) .. 1.5 (its store full).
fn vigour(org: u32) -> f32 {
  let store = RESERVE * f32(max(snapHyphae(org), 60));
  return clamp(0.5 + snapFood(org) / store, 0.5, 1.5);
}

// How much spore-bearing surface an organism's heads carry, relative to a
// plain one: gills (by count and depth), teeth, a spore mass, and the extra
// surface of a folded or pitted head. More spores, at a cost (spawnCluster).
fn hymenium(org: u32) -> f32 {
  return 0.4 + min(PG(org, G_GILLS) / 50.0, 2.0) * PG(org, G_GILL_DEPTH) / 0.2 + PG(org, G_TEETH) * 1.2
       + PG(org, G_GLEBA) * 1.5 + 0.8 * max(PG(org, G_EXCESS), 0.0) * PG(org, G_BEND);
}

// (`already`: how many the caller has set going this step; with how many
// the colony had as the step began, not the live count, whose last ones over
// would depend on the order threads run in)
fn spawnFruit(org: u32, p: vec3f, n: vec3f, delay: f32, flags: u32, already: i32) -> bool {
  if (snapFruits(org) + already >= select(ORG_FRUITS, 400, org == SLIME_ORG)) { return false; }
  // the next free slot round the ring
  var slot = MAX_FRUITS;
  for (var t = 0u; t < 8u; t++) {
    let k = atomicAdd(&counters[6], 1u) % MAX_FRUITS;
    if ((fruits[k].i.z & FR_ALIVE) == 0u) {
      slot = k;
      break;
    }
  }
  if (slot == MAX_FRUITS) { return false; }
  atomicAdd(&pools[org * 4u + 3u], 1);
  let seed = hash3(u32(rr() * 4294967040.0), stepNo, 0x5eedu);
  let up = upAt(p);
  var f: Fruit;
  f.base = vec4f(p, 0.75 + 0.5 * rr());
  f.norm = vec4f(n, f32(stepNo) + delay);
  f.dir = vec4f(normalize(mix(n, up, PG(org, G_UPRIGHT)) + rdir() * 0.15 + up * 1e-4), 0.0);
  // a head that will wrinkle claims a development slot (none left: it stays
  // smooth)
  var ds = NO_DEV;
  if (PG(org, G_EXCESS) > 0.0 || PG(org, G_EDGE) > 0.1) {
    let start = seed % PAT_SLOTS;
    for (var k = 0u; k < 48u; k++) {
      let c = (start + k) % PAT_SLOTS;
      if (atomicCompareExchangeWeak(&grid[GRID_PATS + c], 0u, slot + 1u).exchanged) {
        ds = c;
        break;
      }
    }
  }
  f.i = vec4u(org, stepNo, FR_ALIVE | flags | (ds << 16u), seed);
  fruits[slot] = f;
  return true;
}

// A flush of fruit bodies around a point, each paid for from the reserve,
// only where there is room overhead and no other fruit body.
fn spawnCluster(org: u32, pos: vec3f) {
  let n = u32(PG(org, G_CLUSTER));
  let spread = PG(org, G_SPREAD);
  let R = PG(org, G_SIZE);
  let L = PG(org, G_STEM_LEN) * R;
  // a head costs more the more spore-bearing surface it carries, a stem by
  // its length
  let cost = i32((1.0 + 20.0 * R * R * (0.6 + 0.2 * hymenium(org)) + 2.5 * R * PG(org, G_STEM_LEN)) * POOL_FP);
  let grow = PG(org, G_GROW);
  let c = toSurface(pos);
  let up = upAt(c);
  let nc = worldN(c);
  var store = i32(snapFood(org) * POOL_FP);
  var made = 0;
  for (var j = 0u; j < n; j++) {
    var q = c;
    if (j > 0u) { q = toSurface(c + unit(cross(nc, rdir())) * spread * sqrt(rr()) + nc * 0.2); }
    let w = world(q);
    if (abs(w.x) > 0.1 || w.y > 0.5 || !inLife(q)) { continue; }
    let qn = worldN(q);
    let rise = normalize(mix(qn, up, PG(org, G_UPRIGHT)) + up * 1e-4);
    if (world(q + rise * (L + R)).x < R * 0.7) { continue; }
    if (occAt(q + rise * (L * 0.6 + R * 0.5)) > 0u) { continue; }
    if (store < cost) { return; }
    if (spawnFruit(org, q, qn, f32(j) * grow * 0.08 * rr(), 0u, made)) {
      atomicSub(&pools[org * 4u], cost);
      store -= cost;
      made++;
    }
  }
}

fn killFruit(i: u32, org: u32) {
  let ds = devSlot(fruits[i]);
  if (ds < PAT_SLOTS) { atomicStore(&grid[GRID_PATS + ds], 0u); }
  atomicSub(&pools[org * 4u + 3u], select(1, ROT_COUNT, (fruits[i].i.z & FR_ROT) != 0u));
  fruits[i].i.z = 0u;
  fruits[i].bound.w = 0.0;
  // it rots where it stood, giving back part of what it cost
  // (less what a mycoparasite has eaten of it)
  let R = max(fruits[i].pa.y, fruits[i].head.w);
  dropLitter(fruits[i].base.xyz, (0.5 + 8.0 * R * R) * (1.0 - 0.8 * fruits[i].pb.y));
}

// The live fruit bodies, for neighbour tests (one workgroup).
@compute @workgroup_size(256)
fn listFruits(@builtin(local_invocation_index) li: u32) {
  if (li == 0u) { atomicStore(&counters[1], 0u); }
  storageBarrier();
  workgroupBarrier();
  for (var k = 0u; k < MAX_FRUITS / 256u; k++) {
    let i = li * (MAX_FRUITS / 256u) + k;
    if ((fruits[i].i.z & FR_ALIVE) != 0u && fruits[i].bound.w > 0.0) {
      let n = atomicAdd(&counters[1], 1u);
      atomicStore(&grid[GRID_FRUITS + n], i);
      let f = fruits[i];
      let b = GRID_FNB + n * 8u;
      atomicStore(&grid[b], bitcast<u32>(f.hb.x));
      atomicStore(&grid[b + 1u], bitcast<u32>(f.hb.y));
      atomicStore(&grid[b + 2u], bitcast<u32>(f.hb.z));
      atomicStore(&grid[b + 3u], bitcast<u32>(f.head.w));
      atomicStore(&grid[b + 4u], f.i.y);
      atomicStore(&grid[b + 5u], f.i.w);
      atomicStore(&grid[b + 6u], bitcast<u32>(f.pa.z));
    }
  }
}

// Age a fruit body and rebuild its shape: a closed button that rises on its
// stem, opens, matures, then withers, dissolves, bursts or keeps growing.
// Stems lean away from neighbours and from rock and wood overhead; heads stop
// growing when they run out of room. Ripe heads release spores.
// Runs once per frame, after the steps.
@compute @workgroup_size(64)
fn growFruits(@builtin(global_invocation_id) gid: vec3u) {
  stepNo = atomicLoad(&counters[5]);
  let i = gid.x;
  if (i >= MAX_FRUITS) { return; }
  var f = fruits[i];
  if ((f.i.z & FR_ALIVE) == 0u) { return; }
  let org = f.i.x;
  // a colony that has died or left keeps its program while its fruit bodies
  // rot (G_ALIVE 0.25); an empty slot has nothing to rot them with
  let alive = PG(org, G_ALIVE);
  let age = f32(stepNo) - f.norm.w;
  if (alive < 0.1 || (alive < 0.5 && age < 0.0)) {
    killFruit(i, org);
    return;
  }
  if (age < 0.0) {
    fruits[i].bound.w = 0.0;
    return;
  }
  let tip = (f.i.z & FR_TIPHEAD) != 0u;
  // up where it stands, and a frame along the ground there (its own: it does
  // not move)
  let up = upAt(f.base.xyz);
  let tb = basisOf(up);
  let steps = f32(sim.steps);
  let grow = PG(org, G_GROW) * select(1.0, 0.5, tip);
  let m = clamp(age / grow, 0.0, 1.0);
  let mode = u32(PG(org, G_DECAY) + 0.5);
  // dying takes about as long as growing did: dissolving heads go faster,
  // persistent ones crumble slowly
  let dlen = select(select(1.5, 0.6, mode == 1u), 2.0, mode == 3u) * grow + 500.0;
  // a fruit body whose colony has died rots, however long it would have lived
  // (the slime mould's sporangia have no mycelium); one a slime mould
  // crawls over is engulfed, and rots from it
  if (alive < 0.5 || (org != SLIME_ORG && atomicLoad(&pools[org * 4u + 1u]) <= 0)) { f.hc.z += f32(sim.steps); }
  let sd = slimeAt(f.base.xyz);
  if (org != SLIME_ORG && sd > 6 && slimeMode() == 1u) {
    let bite = f32(sim.steps) * 0.25 * min(f32(sd) / 16.0, 1.0);
    f.hc.z += bite;
    let R0 = max(f.pa.y, f.head.w);
    slimeFeed(8.0 * R0 * R0 * bite / dlen);
  }
  let dcy = clamp(max((age - grow - PG(org, G_LIFE)) / dlen, f.hc.z / dlen), 0.0, 1.0);
  // as it dies it slumps and rots down into the ground
  let rot = smoothstep(0.2, 1.0, dcy);
  // (a withering one wilts and falls first, and only then collapses)
  let collapse = select(rot, smoothstep(0.55, 1.0, dcy), mode == 0u);
  if (dcy > 0.0 && (f.i.z & FR_ROT) == 0u) {
    f.i.z |= FR_ROT;
    fruits[i].i.z = f.i.z;
    atomicAdd(&pools[org * 4u + 3u], ROT_COUNT - 1);
  }
  if (dcy >= 1.0) {
    killFruit(i, org);
    return;
  }

  // a mycoparasite living in its colony's mycelium (the more of it, the
  // likelier), or whose hyphae reach its base, or whose spores drift around
  // its head, can infect it (less often a well-defended colony's); it crusts
  // the body over in about as long as the body took to grow, feeding on it
  // as it goes
  var parOrg = u32(f.pb.x + 0.5);
  if (parOrg == 0u && org != SLIME_ORG) {
    let bc = f4Coord(f.base.xyz - f.norm.xyz * F4 * 0.5);
    var pv = 0u;
    for (var k = 0u; k < 27u; k++) {
      let q = bc + vec3i(i32(k % 3u), i32((k / 3u) % 3u), i32(k / 9u)) - 1;
      if (inF4(q)) { pv = max(pv, atomicLoad(&grid[gi(GF_PARA, f4Index(q))])); }
    }
    var p = pv & 63u;
    var rate = select(0.0, 0.003 * f32(pv >> 6u) / 255.0, pv != 0u);
    let hp = hostParasite(org);
    if (hp > 0u) {
      let rs = 0.0015 * hostLoad(org) * PG(hp - 1u, G_PARASITE);
      if (rs > rate) {
        rate = rs;
        p = hp - 1u;
      }
    }
    let hv = f4Coord(f.hb.xyz);
    if (inF4(hv)) {
      let v = f4Index(hv);
      let sp = fieldA[F4N + v];
      let ids = u32(sp.z + 0.5);
      for (var k = 0u; k < 2u; k++) {
        let q = (ids >> (7u * k)) & 127u;
        if (q == 0u || q - 1u == org || PG(q - 1u, G_ALIVE) < 0.5) { continue; }
        let r = 0.02 * min(sp.x + sp.y, 5.0) * PG(q - 1u, G_PARASITE);
        if (PG(q - 1u, G_PARASITE) > 0.3 && r > rate) {
          rate = r;
          p = q - 1u;
        }
      }
    }
    if (rate > 0.0 && p != org && PG(p, G_ALIVE) > 0.5) {
      let chance = 1.0 - exp(-f32(sim.steps) * rate * (1.0 - 0.7 * PG(org, G_COMBAT)));
      if (u01(hash3(i, stepNo, 0x9a7u)) < chance) {
        parOrg = p + 1u;
        // (its colour is kept, in case its organism is gone before this is)
        let q = vec3u(round(clamp(vec3f(PG(p, G_PIG_R), PG(p, G_PIG_G), PG(p, G_PIG_B)), vec3f(0.0), vec3f(1.0)) * 255.0));
        f.pb = vec4f(f32(parOrg), 0.0, f32(q.x | (q.y << 8u) | (q.z << 16u)), 0.0);
      }
    }
  }
  if (parOrg > 0u) {
    let par = parOrg - 1u;
    let dc = min(f32(sim.steps) / (0.5 * grow + 400.0), 1.0 - f.pb.y);
    f.pb.y += dc;
    if (PG(par, G_ALIVE) > 0.5 && dc > 0.0) {
      let R0 = max(f.pa.y, f.head.w);
      atomicAdd(&pools[par * 4u], i32(8.0 * R0 * R0 * dc * POOL_FP + u01(hash3(f.i.w, stepNo, 5u))));
    }
  }
  let crust = f.pb.y;
  let seedf = f32(f.i.w & 1023u);
  let prevR = f.head.w;
  let prevApex = f.apex.xyz;
  let lat = PG(org, G_LATERAL);

  // size; persistent heads keep adding margin
  var Rf = PG(org, G_SIZE) * f.base.w * select(1.0, 0.3, tip);
  if (mode == 3u) { Rf *= 1.0 + 0.3 * log(1.0 + max(age - grow, 0.0) / grow); }
  var R = Rf * (0.15 + 0.85 * smoothstep(0.0, 0.8, m));
  if (mode == 0u) { R *= 1.0 - 0.3 * smoothstep(0.2, 1.0, dcy); }
  if (mode == 2u) { R *= 1.0 - 0.2 * dcy; }

  // neighbours: lean away from heads that crowd this one, and fit around
  // older ones
  // (summed in fixed point: the same in whatever order the list has them)
  var pushFp = vec3i(0);
  var room = 1e9;
  // blinking: each pulls toward the blinks of those within a few cm
  // (fireflies fall into step so); summed in fixed point too
  let blink = PG(org, G_BLINK);
  var blinkAt = f.pa.z;
  if (blink > 0.0 && (blinkAt < 0.0 || prevR <= 0.0)) { blinkAt = u01(hash3(f.i.w, 0x611u, 0u)); }
  var pullFp = 0;
  if (prevR > 0.0) {
    let n = min(atomicLoad(&counters[1]), MAX_FRUITS);
    for (var j = 0u; j < n; j++) {
      let o = atomicLoad(&grid[GRID_FRUITS + j]);
      if (o == i) { continue; }
      // (as it was when the frame began)
      let nb = GRID_FNB + j * 8u;
      let ob = vec3f(bitcast<f32>(atomicLoad(&grid[nb])), bitcast<f32>(atomicLoad(&grid[nb + 1u])), bitcast<f32>(atomicLoad(&grid[nb + 2u])));
      let oR = bitcast<f32>(atomicLoad(&grid[nb + 3u]));
      let oBorn = atomicLoad(&grid[nb + 4u]);
      let oSeed = atomicLoad(&grid[nb + 5u]);
      let dv = f.hb.xyz - ob;
      let dist = length(dv);
      let oBlink = bitcast<f32>(atomicLoad(&grid[nb + 6u]));
      if (blink > 0.0 && oBlink >= 0.0 && dist < BLINK_REACH) {
        pullFp += i32(round(sin(TAU * (oBlink - blinkAt)) * 65536.0));
      }
      let touch = (prevR + oR) * 0.95;
      if (dist < touch + Rf * 0.6) {
        pushFp += vec3i(round(unit(dv + vec3f(1e-4, 0.0, 0.0)) * (touch + Rf * 0.6 - dist) / max(touch, 1e-3) * 65536.0));
        let older = oBorn < f.i.y || (oBorn == f.i.y && oSeed < f.i.w);
        if (older) { room = min(room, dist - oR * 0.9); }
      }
    }
  }

  // stem direction: upright, toward or away from the sun (where it stands at
  // noon here, while it is day), a slow wobble,
  // away from crowding neighbours and from what is overhead; withering stems
  // sag
  let n0 = f.norm.xyz;
  let d0 = normalize(mix(n0, up, PG(org, G_UPRIGHT)) + up * 1e-4);
  var want = d0 + noonSun(f.base.xyz, stepNo) * (1.0 - nightness(f.base.xyz, stepNo)) * PG(org, G_PHOTO) * 0.6;
  want += (tb[0] * sin(seedf + age * 0.004) + tb[1] * cos(seedf * 1.7 + age * 0.0031)) * PG(org, G_WOBBLE) * 0.7;
  want += vec3f(pushFp) / 65536.0 * 1.5;
  if (prevR > 0.0) {
    let above = prevApex + up * prevR;
    if (world(above).x < prevR * 1.5) { want += worldN(above) * 1.5; }
  }
  // dying, it wilts: a withering one topples over on its stem, the head
  // still on it, the way it already leans (or a way of its own, away from
  // what crowds it); the rest sag where they stand
  let wilt = select(rot * 0.85, smoothstep(0.05, 0.6, dcy), mode == 0u) * (1.0 - lat) * select(1.0, 0.0, mode == 3u);
  let sway = tb[0] * cos(seedf * 2.4) + tb[1] * sin(seedf * 2.4) + level(vec3f(pushFp) / 65536.0, up);
  let fallTo = unit(level(f.dir.xyz, up) * 2.0 + sway + tb[0] * 1e-4);
  want = mix(want, select(n0 * 0.3 - up, fallTo - up * 0.2, mode == 0u), wilt);
  let fallen = select(0.0, wilt, mode == 0u);
  let dir = normalize(mix(f.dir.xyz, unit(want + up * 1e-4), 1.0 - pow(0.98, steps)));

  // the stem stops rising when its head would meet rock or wood
  var L = PG(org, G_STEM_LEN) * Rf * smoothstep(0.05, 0.85, m) + 0.02 * Rf;
  let prevL = length(prevApex - f.base.xyz);
  if (prevR > 0.0 && L > prevL && world(prevApex + dir * prevR).x < prevR * 0.8) { L = prevL; }
  // (a falling stem keeps its length, leaning from its base, until it
  // collapses; at the last a withered body sinks into the litter)
  L *= 1.0 - select(0.85, 0.6, mode == 0u) * collapse;
  let base = f.base.xyz - n0 * select(0.0, smoothstep(0.75, 1.0, dcy) * (Rf + 0.3), mode == 0u);
  let ctrl = base + normalize(mix(d0, dir, fallen) + up * 1e-4) * L * 0.5;
  let apex = ctrl + dir * L * 0.5;
  let rs = PG(org, G_STEM_R) * Rf * (0.5 + 0.5 * smoothstep(0.0, 0.7, m)) * (1.0 - select(0.8, 0.6, mode == 0u) * collapse);
  let rTop = rs * PG(org, G_TAPER);
  let bulb = PG(org, G_BULB);

  // the spore-bearing face turns to the ground; a fallen head lies along
  // its stem
  let Y = normalize(mix(dir + up * (1.0 + 3.0 * lat), dir, fallen));
  var Xo = level(n0, up);
  if (dot(Xo, Xo) < 0.04) { Xo = tb[0] * cos(seedf) + tb[1] * sin(seedf); }
  let X = normalize(Xo - Y * dot(Xo, Y));
  // a burst puffball deflates
  let el = PG(org, G_ELONG) * select(1.0, 1.0 - 0.55 * smoothstep(0.2, 0.9, dcy), mode == 2u);
  let cone = PG(org, G_CONE);
  let rays = PG(org, G_RAYS);
  let curl = PG(org, G_CURL);
  var mfrac = 0.0;
  var turn = 0.0;
  if (rays > 0.5) {
    mfrac = 0.12 + 0.8 * PG(org, G_RAY_DEPTH);
    turn = PG(org, G_RAY_CURL) * smoothstep(0.5, 1.0, m);
  } else if (abs(curl) > 0.02) {
    mfrac = 0.2;
    turn = curl * 1.8;
  }

  // room: other heads, and rock or wood around this one (shelves are meant
  // to touch their wood); a head never shrinks for lack of it, it stops
  // (a head that sits on its substrate is always that close to it). Heads
  // that will stay closed sit on top of their stem.
  let closed = smoothstep(2.2, 2.8, 2.95 - PG(org, G_OPEN));
  let sessile = max(1.0 - smoothstep(0.15, 0.9, PG(org, G_STEM_LEN)), closed) * (1.0 - lat);
  if (prevR > 0.0 && lat < 0.5) { room = min(room, (world(f.hb.xyz).x + sessile * prevR) / 0.95); }
  let Rwant = R;
  if (R > room) { R = max(min(prevR, R), min(room, R)); }
  R = max(R, 0.15 * Rf) * (1.0 - 0.45 * collapse);
  let th = PG(org, G_THICK) * R * (1.0 + 0.6 * crust);

  // the head's shape develops: it starts as a closed button, and as it
  // expands its lower (spore-bearing) layer outgrows the upper one, so each
  // arc of its profile gives up turning, the crown's or the margin's first:
  // a dome, a bell, a flat cap, a funnel, a cup. Some go on after they are
  // grown, flattening and turning up. A head short of room opens less.
  // Withering heads curl in; dissolving ones turn up at the margin.
  var tn = array<f32, 8>(f.seg0.x, f.seg0.y, f.seg0.z, f.seg0.w, f.seg1.x, f.seg1.y, f.seg1.z, f.seg1.w);
  if (prevR <= 0.0) {
    for (var k = 0u; k < 8u; k++) { tn[k] = 2.95 / 8.0; }
  }
  let opened = (smoothstep(0.35, 1.0, m) - smoothstep(0.35, 1.0, f.dir.w)) * PG(org, G_OPEN);
  let crowd = mix(0.6, 1.0, clamp(R / max(Rwant, 1e-4), 0.0, 1.0));
  var late = 0.0;
  if (age > grow) { late = PG(org, G_UPTURN) * steps / grow; }
  let dd = dcy - f.life.x;
  let openFrom = PG(org, G_OPEN_AT);
  // heads that keep the night close a little by day and open at dusk
  var shut = f.pa.w;
  var reshut = 0.0;
  if (PG(org, G_NIGHT_OPEN) > 0.0 && m > 0.6 && dcy <= 0.0) {
    let want = PG(org, G_NIGHT_OPEN) * (1.0 - nightness(f.base.xyz, stepNo));
    let ns = shut + clamp(want - shut, -steps * 0.004, steps * 0.004);
    reshut = (ns - shut) * 0.75 * PG(org, G_OPEN);
    shut = ns;
  }
  var A0 = 0.0;
  for (var k = 0u; k < 8u; k++) { A0 += tn[k]; }
  var A = 0.0;
  for (var k = 0u; k < 8u; k++) {
    let x = (f32(k) + 0.5) / 8.0;
    let w = 2.0 * mix(1.0 - x, x, openFrom);
    var t = tn[k] - (opened * crowd + select(0.0, late, A0 > -2.4)) * w / 8.0 + reshut * w / 8.0;
    if (mode == 0u) { t += 0.5 * dd / 8.0; }
    if (mode == 1u) { t -= 1.4 * dd * 2.0 * x / 8.0; }
    tn[k] = clamp(t, -0.42, 0.42);
    A += tn[k];
  }
  f.seg0 = vec4f(tn[0], tn[1], tn[2], tn[3]);
  f.seg1 = vec4f(tn[4], tn[5], tn[6], tn[7]);

  f.head.w = R;
  f.marg.w = mfrac;
  f.prof.w = turn;
  f.axis.w = A;
  // its arc length is whatever spans radius R at this shape
  var pu = profileScaled(f, 1.0);
  let S = R / max(profBox(&pu).z, 0.05);
  var pr = profileScaled(f, S);
  if (blink > 0.0) {
    blinkAt = fract(blinkAt + steps * blink + clamp(steps * PG(org, G_SYNC) * f32(pullFp) / 65536.0, -0.2, 0.2));
  } else {
    blinkAt = -1.0;
  }
  f.pa = vec4f(S, max(f.pa.y, R), blinkAt, shut);
  let box = profBox(&pr);
  // (bounds: the lowest and widest together)
  let ymin = box.x * el - cone * min(box.z, R * 1.2);
  let ymax = box.y * el;

  // sessile heads sit on the ground, stalked ones on the stem top, shelves
  // stand out from the wood; heads that stay closed sit on top of the stem
  // rather than around it
  let ht = th * 0.5;
  let lift = mix(ht, max(ht, -profLow(&pr, el, cone, R) + ht * 0.5), sessile);
  let O = apex + Y * lift + X * (R * 0.92 * lat);

  let skirt = PG(org, G_SKIRT) * smoothstep(0.6, 1.0, m);
  let flare = PG(org, G_RING) * R * 0.3 * smoothstep(0.4, 0.9, m);
  let vr = PG(org, G_VOLVA) * (rs * 1.8 + 0.25 * Rf);
  let teeth = PG(org, G_TEETH) * R * (1.0 - crust);
  let gleba = PG(org, G_GLEBA);
  // the developed field's grid and wavelength follow the head as it grows
  let feats = PG(org, G_SCALE);
  let lam = 2.0 * S / feats;

  f.dir = vec4f(dir, m);
  f.apex = vec4f(apex, rTop);
  f.ctrl = vec4f(ctrl, rs);
  f.head = vec4f(O, R);
  f.axis = vec4f(Y, A);
  f.frame = vec4f(X, el);
  f.prof = vec4f(cone, th, PG(org, G_UMBO) * R, turn);
  f.marg = vec4f(lam, lam * PG(org, G_DEPTH), rays, mfrac);
  f.feat = vec4f(devGrid(org), PG(org, G_TEAR), PG(org, G_BEND), PG(org, G_TEXTURE));
  f.feat2 = vec4f(f.feat2.x, PG(org, G_GILLS), PG(org, G_GILL_DEPTH) * R * (1.0 - crust), teeth);
  f.ring = vec4f(mix(0.78, 0.95, PG(org, G_SKIRT)), flare, skirt, bulb);
  f.misc = vec4f(gleba, PG(org, G_EGGS), vr, lat);
  f.life = vec4f(dcy, max(abs(R - prevR), length(apex - prevApex)),
                 // (an inkcap's head dissolves from its margin; a withered
                 // one only crumbles a little at the last)
                 select(1.0 - 1.25 * select(smoothstep(0.45, 1.0, dcy), smoothstep(0.0, 0.75, dcy), mode == 1u),
                        1.0 - 0.4 * smoothstep(0.75, 1.0, dcy), mode == 0u),
                 select(0.0, 0.4 * dcy, mode == 2u));

  // bounds
  let hc = O + Y * (ymin + ymax) * 0.5;
  let hh = (ymax - ymin) * 0.5 + th + teeth + f.marg.y * 1.8 + f.prof.z + gleba * R;
  let hcr = box.z + th + f.marg.y * 1.8 + 0.01;
  let hch = hh + 0.01;
  f.hc = vec4f(hcr, hch, f.hc.z, 0.0);
  f.hb = vec4f(hc, sqrt(hcr * hcr + hch * hch));
  let mid = (base + apex) * 0.5;
  let sr = 0.5 * length(apex - base) + 0.5 * length(ctrl - mid) + max(rs * (1.0 + bulb), rTop) + flare + vr + 0.01;
  f.sb = vec4f(mid, sr);
  let bc = (f.hb.xyz + f.sb.xyz) * 0.5;
  f.bound = vec4f(bc, max(length(f.hb.xyz - bc) + f.hb.w, length(f.sb.xyz - bc) + f.sb.w));
  fruits[i] = f;

  // spores: ripe heads shed them from the spore-bearing face; puffballs keep
  // theirs until they burst, then puff them out of the crown
  // sporangia leave their spores in the ground for the slime mould to hatch
  // from (slimesim.wgsl)
  if (org == SLIME_ORG) {
    if (m > 0.8 && steps > 0.0 && dcy < 0.5) { atomicAdd(&grid[sl(SL_BANK + slimeCol(f.base.xyz))], u32(steps * 2.0)); }
    return;
  }
  // (a crusted body sheds the parasite's spores instead of its own)
  let out = PG(org, G_SPORE_OUT) * (1.0 - crust);
  if (parOrg > 0u && crust > 0.3 && steps > 0.0) {
    let par = parOrg - 1u;
    let amt = PG(par, G_SPORE_OUT) * R * R * steps * 0.024 * crust;
    let cc = f4Coord(O + Y * (ht + 0.05));
    if (amt > 0.0 && inF4(cc)) {
      let v = f4Index(cc);
      atomicAdd(&grid[gi(GF_SPORE_IN, v)], u32(min(amt, 1e6) * 256.0));
      atomicMax(&grid[gi(GF_SPORE_ORG, v)], parOrg);
    }
  }
  if (out > 0.0 && m > 0.8 && steps > 0.0) {
    var amt = out * R * R * steps * 0.012 * hymenium(org) * (1.0 - smoothstep(0.0, 0.3, dcy))
            * mix(1.0, 2.0 * nightness(f.base.xyz, stepNo), PG(org, G_NIGHT_SPORE));
    var at = O - Y * (th + f.feat2.z * 0.7 + teeth);
    if (A < 0.0) { at = O + Y * (ht + 0.05); }
    if (mode == 2u) {
      amt = select(0.0, out * R * R * R * steps * 240.0 / dlen, dcy > 0.02 && dcy < 0.6);
      at = O + Y * (ht + 0.05);
    }
    let cc = f4Coord(at);
    if (amt > 0.0 && inF4(cc)) {
      let v = f4Index(cc);
      atomicAdd(&grid[gi(GF_SPORE_IN, v)], u32(min(amt, 1e6) * 256.0));
      atomicMax(&grid[gi(GF_SPORE_ORG, v)], org + 1u);
    }
  }
}

@compute @workgroup_size(256)
fn clearOcc(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= F4N) { return; }
  atomicStore(&grid[gi(GF_OCC, gid.x)], 0u);
  if (gid.x < N_BLOCKS) { atomicStore(&grid[GRID_BLOCKS + gid.x], 0u); }
}

// Fruit bodies into the occupancy grid (after the last step of a frame): the
// head as a stack of discs, the stem as a line of points. The light field
// shades under caps from it, tips avoid it, and new flushes look for room.
fn depositPoint(p: vec3f, amt: u32) {
  let c = f4Coord(p);
  if (inF4(c)) { atomicAdd(&grid[gi(GF_OCC, f4Index(c))], amt); }
}

@compute @workgroup_size(64)
fn depositFruits(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= MAX_FRUITS) { return; }
  let f = fruits[i];
  if ((f.i.z & FR_ALIVE) == 0u || f.bound.w <= 0.0) { return; }
  let Y = f.axis.xyz;
  let X = f.frame.xyz;
  let Z = cross(Y, X);
  let hr = f.hb.w * 0.8;
  let step = F4 * 0.7;
  let n = i32(clamp(hr * 2.0 / step, 1.0, 14.0));
  let layers = i32(clamp(hr / F4, 1.0, 4.0));
  for (var l = 0; l < layers; l++) {
    let c = f.hb.xyz + Y * hr * 0.6 * ((f32(l) + 0.5) / f32(layers) * 2.0 - 1.0);
    for (var a = 0; a <= n; a++) {
      for (var b = 0; b <= n; b++) {
        let uv = (vec2f(f32(a), f32(b)) / f32(n) * 2.0 - 1.0) * hr;
        if (dot(uv, uv) > hr * hr) { continue; }
        depositPoint(c + X * uv.x + Z * uv.y, 45u);
      }
    }
  }
  let len = length(f.apex.xyz - f.base.xyz);
  let ns = i32(clamp(len / step, 1.0, 40.0));
  let amt = u32(clamp(f.ctrl.w / 0.04, 0.3, 1.0) * 45.0);
  for (var k = 0; k <= ns; k++) {
    depositPoint(bezPoint(f.base.xyz, f.ctrl.xyz, f.apex.xyz, f32(k) / f32(ns)), amt);
  }
}

// --------------------------------------------------------------- spores ---

// Settled spores germinate on bare, fed ground: an event for the CPU, which
// breeds the new organism from the parents and plants it here. A spore that
// landed alone mates with whatever mycelium is established next to it.
@compute @workgroup_size(4, 4, 4)
fn germinate(@builtin(global_invocation_id) gid: vec3u) {
  stepNo = atomicLoad(&counters[5]);
  if (any(gid >= F4_RES)) { return; }
  let i = f4Index(vec3i(gid));
  let bed = fieldA[F4N + i].y;
  if (bed < 0.15) { return; }
  rinit(i, stepNo ^ 0x9e3779b9u);
  let wet = wetAt(i);
  if (rr() > 2e-4 * f32(sim.steps) * wet * wet) { return; }
  let p = BOX_MIN + (vec3f(gid) + 0.5) * F4;
  let w = world(p);
  if (w.y > 0.5 || fieldA[i].y < 0.25) { return; }
  if (atomicLoad(&grid[gi(GF_DENS, i)]) > 0u || atomicLoad(&grid[gi(GF_OCC, i)]) > 0u) { return; }
  var ids = u32(fieldA[F4N + i].z + 0.5);
  if (ids == 0u) { return; }
  if ((ids >> 7u) == 0u) {
    for (var t = 0; t < 6; t++) {
      let gb = basisOf(upAt(BOX_MIN + (vec3f(gid) + 0.5) * F4));
      let q = vec3i(gid) + vec3i(round(gb[0] * (rr() * 6.0 - 3.0) + gb[1] * (rr() * 6.0 - 3.0) - gb[2] * (rr() * 2.0)));
      let ow = gridAt(GF_OWNER, q);
      if (ow != NO_OWNER && (ow & 63u) + 1u != ids) {
        ids = mergeParents(ids, (ow & 63u) + 1u);
        break;
      }
    }
  }
  let e = atomicAdd(&counters[2], 1u);
  if (e >= MAX_EVENTS) {
    atomicSub(&counters[2], 1u);
    return;
  }
  events[e] = vec4f(p, f32(ids));
  fieldA[F4N + i].y = bed * 0.2;
  fieldB[F4N + i].y = bed * 0.2;
}

// ----------------------------------------------------------- development ---

// A head's surface develops while it expands. Where it grows more than its
// shape has room for, the sheet wrinkles: a Swift-Hohenberg field over the
// head (a disk from the crown out to the margin), whose growth rate is the
// excess growth over the head and in a band at the margin (about a wave
// deep, as an edge buckles). The quadratic
// term tips its patterns from bumps through ridges to pits, damping along or
// across the radius gives radial pleats or concentric ridges, and the
// wavelength is set by how many features fit across the head. The field
// takes development time with the head's maturity and is fixed once the
// head is grown. fruit.wgsl turns it into folds, relief or tears.

const DEV_T: f32 = 30.0;          // development time over a head's growth

// Grid size for an organism's heads: about 7 cells per feature.
fn devGrid(org: u32) -> f32 { return clamp(ceil(PG(org, G_SCALE) * 7.0) + 2.0, 10.0, 62.0); }

// The sim never samples the atlas (fruitdev.wgsl does, for the renderer).
fn devAt(f: Fruit, s: f32, phi: f32) -> f32 { return 0.0; }

var<workgroup> devW: array<f32, 4000>;   // the field, rows of PAT_RES (62 used at most)
var<workgroup> devL: array<f32, 4000>;   // w + lap w
var<workgroup> devP: array<f32, 11>;     // N, h, dt, excess, edge, quadratic, radial and concentric damping, time, seed, margin band
var<workgroup> devRun: u32;              // iterations this frame
var<workgroup> devFi: u32;
var<workgroup> devFresh: u32;

@compute @workgroup_size(256)
fn develop(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let slot = wg.x;
  if (li == 0u) {
    devRun = 0u;
    let owner = atomicLoad(&grid[GRID_PATS + slot]);
    if (owner != 0u) {
      let fi = owner - 1u;
      let f = fruits[fi];
      let org = f.i.x;
      if ((f.i.z & FR_ALIVE) != 0u && devSlot(f) == slot && f.feat.x >= 8.0) {
        let N = f.feat.x;
        let lamPx = (N - 2.0) / PG(org, G_SCALE);
        let h = TAU / lamPx;
        let lmax = 8.0 / (h * h) - 1.0;
        let dt = min(0.05, 1.6 / (lmax * lmax));
        let tau = f.feat2.x;
        let want = DEV_T * clamp((f.dir.w - 0.1) / 0.85, 0.0, 1.0);
        if (want > tau) {
          devRun = u32(clamp(ceil((want - tau) / dt), 1.0, 64.0));
          devFi = fi;
          devFresh = select(0u, 1u, tau <= 0.0);
          let grain = PG(org, G_GRAIN);
          devP[0] = N;
          devP[1] = h;
          devP[2] = dt;
          devP[3] = PG(org, G_EXCESS);
          devP[4] = PG(org, G_EDGE);
          devP[5] = PG(org, G_TEXTURE);
          devP[6] = max(-grain, 0.0) * 1.5;
          devP[7] = max(grain, 0.0) * 1.5;
          devP[8] = tau;
          devP[9] = f32(f.i.w & 0xffffu);
          devP[10] = clamp(2.4 / PG(org, G_SCALE), 0.2, 1.0);
        }
      }
    }
  }
  let run = workgroupUniformLoad(&devRun);
  if (run == 0u) { return; }
  let Nf = devP[0];
  let N = u32(Nf);
  let ih2 = 1.0 / (devP[1] * devP[1]);
  let dt = devP[2];
  let base = slot * PAT_RES * PAT_RES;
  let fresh = devFresh == 1u;
  let seed = u32(devP[9]);
  let ctr = vec2f(Nf * 0.5);
  let rad = Nf * 0.5 - 1.0;

  // load the field (a new head starts from faint noise)
  for (var k = li; k < 4000u; k += 256u) {
    let x = k % PAT_RES;
    let y = k / PAT_RES;
    var v = 0.0;
    if (x < N && y < N) {
      v = select(devField[base + k], (u01(hash3(k, seed, 77u)) - 0.5) * 0.04, fresh);
    }
    devW[k] = v;
  }
  workgroupBarrier();

  var nxt: array<f32, 16>;
  for (var it = 0u; it < run; it++) {
    let tau = devP[8] + f32(it) * dt;
    // the excess builds up while the head expands
    let ramp = smoothstep(0.0, 0.35 * DEV_T, tau);
    // L = w + lap w (edges mirror)
    for (var j = 0u; j < 16u; j++) {
      let k = li + j * 256u;
      if (k >= 4000u) { break; }
      let x = k % PAT_RES;
      let y = k / PAT_RES;
      if (x >= N || y >= N) { continue; }
      let w = devW[k];
      let xl = select(k - 1u, k, x == 0u);
      let xr = select(k + 1u, k, x + 1u >= N);
      let yd = select(k - PAT_RES, k, y == 0u);
      let yu = select(k + PAT_RES, k, y + 1u >= N);
      devL[k] = w + (devW[xl] + devW[xr] + devW[yd] + devW[yu] - 4.0 * w) * ih2;
    }
    workgroupBarrier();
    for (var j = 0u; j < 16u; j++) {
      let k = li + j * 256u;
      if (k >= 4000u) { break; }
      let x = k % PAT_RES;
      let y = k / PAT_RES;
      if (x >= N || y >= N) { continue; }
      let w = devW[k];
      let xl = select(k - 1u, k, x == 0u);
      let xr = select(k + 1u, k, x + 1u >= N);
      let yd = select(k - PAT_RES, k, y == 0u);
      let yu = select(k + PAT_RES, k, y + 1u >= N);
      let L = devL[k];
      let L2 = L + (devL[xl] + devL[xr] + devL[yd] + devL[yu] - 4.0 * L) * ih2;
      let pq = vec2f(f32(x), f32(y)) + 0.5 - ctr;
      // the margin is a free edge: past it the growth goes on as at the edge
      // (nothing out there is drawn)
      let rho = min(length(pq) / rad, 1.0);
      let r = ramp * (devP[3] + devP[4] * clamp((rho - 1.0 + devP[10]) / devP[10], 0.0, 1.0));
      var du = r * w - L2 + devP[5] * w * w - w * w * w;
      if (devP[6] + devP[7] > 0.0) {
        // second derivatives along and across the radius
        let n = pq / max(length(pq), 1e-4);
        let wxx = (devW[xl] + devW[xr] - 2.0 * w) * ih2;
        let wyy = (devW[yd] + devW[yu] - 2.0 * w) * ih2;
        let xl2 = select(yu - 1u, yu, x == 0u);
        let xr2 = select(yu + 1u, yu, x + 1u >= N);
        let xl1 = select(yd - 1u, yd, x == 0u);
        let xr1 = select(yd + 1u, yd, x + 1u >= N);
        let wxy = (devW[xr2] + devW[xl1] - devW[xl2] - devW[xr1]) * 0.25 * ih2;
        let wrr = n.x * n.x * wxx + 2.0 * n.x * n.y * wxy + n.y * n.y * wyy;
        let wtt = n.y * n.y * wxx - 2.0 * n.x * n.y * wxy + n.x * n.x * wyy;
        du += devP[6] * wrr + devP[7] * wtt;
      }
      nxt[j] = w + dt * du;
    }
    workgroupBarrier();
    for (var j = 0u; j < 16u; j++) {
      let k = li + j * 256u;
      if (k >= 4000u) { break; }
      if (k % PAT_RES < N && k / PAT_RES < N) { devW[k] = nxt[j]; }
    }
    workgroupBarrier();
  }

  // keep it, and show it
  let ox = (slot % PAT_COLS) * PAT_RES;
  let oy = (slot / PAT_COLS) * PAT_RES;
  for (var k = li; k < 4000u; k += 256u) {
    let x = k % PAT_RES;
    let y = k / PAT_RES;
    if (x >= N || y >= N) { continue; }
    devField[base + k] = devW[k];
    textureStore(devOut, vec2u(ox + x, oy + y), vec4f(devW[k], 0.0, 0.0, 0.0));
  }
  if (li == 0u) { fruits[devFi].feat2.x = devP[8] + f32(run) * dt; }
}

// The developed heads shown again from their fields, as they are (after a
// checkpoint is restored: the atlas is not kept with it).
@compute @workgroup_size(256)
fn devPaint(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let slot = wg.x;
  let owner = atomicLoad(&grid[GRID_PATS + slot]);
  if (owner == 0u) { return; }
  let f = fruits[owner - 1u];
  if ((f.i.z & FR_ALIVE) == 0u || devSlot(f) != slot || f.feat.x < 8.0) { return; }
  let N = u32(f.feat.x);
  let base = slot * PAT_RES * PAT_RES;
  let ox = (slot % PAT_COLS) * PAT_RES;
  let oy = (slot / PAT_COLS) * PAT_RES;
  for (var k = li; k < 4000u; k += 256u) {
    let x = k % PAT_RES;
    let y = k / PAT_RES;
    if (x < N && y < N) { textureStore(devOut, vec2u(ox + x, oy + y), vec4f(devField[base + k], 0.0, 0.0, 0.0)); }
  }
}

// ----------------------------------------------------------------- light ---

// Light through fungal tissue and fruit bodies on the 4 mm grid:
// transmittance towards the sun (r) and straight up to the sky (g); outside
// the shell life keeps to, open. Rendering
// multiplies these with the baked world lighting. Also the fields the
// renderer reads: whose spores are here (b) and how much fighting (a); and
// airborne spores, settled spores, food and litter.
fn tissueTau(start: vec3f, d: vec3f) -> f32 {
  var q = start;
  var tau = 0.0;
  for (var i = 0; i < 90; i++) {
    let c = f4Coord(q);
    if (!inF4(c) || dot(q, q) > LIFE_OUT * LIFE_OUT) { break; }
    let v = f4Index(c);
    tau += f32(atomicLoad(&grid[gi(GF_DENS, v)]) + atomicLoad(&grid[gi(GF_OCC, v)])) * 0.03;
    q += d * F4 * 0.7;
  }
  return tau;
}

@compute @workgroup_size(4, 4, 4)
fn shadeTissue(@builtin(global_invocation_id) gid: vec3u) {
  stepNo = atomicLoad(&counters[5]);
  if (any(gid >= F4_RES)) { return; }
  let p = BOX_MIN + (vec3f(gid) + 0.5) * F4;
  let i = f4Index(vec3i(gid));
  let L = sunDir(stepNo);
  var ts = 0.0;
  var tk = 0.0;
  if ((atomicLoad(&grid[gi(GF_WORLD, i)]) & W_DEEP) == 0u && inLife(p)) {
    let up = upAt(p);
    ts = tissueTau(p + L * F4 * 0.5, L);
    tk = tissueTau(p + up * F4 * 0.5, up);
  }
  let ids = u32(fieldA[F4N + i].z + 0.5);
  let scar = 1.0 - exp(-f32(atomicLoad(&grid[gi(GF_SCAR, i)])) * 0.02);
  textureStore(tissueSun, gid, vec4f(exp(-ts), exp(-tk * 0.7), f32(ids & 127u), scar));
  let f = fieldA[i];
  let s = fieldA[F4N + i];
  textureStore(fieldTex, gid, vec4f(s.x, s.y, f.y, f.x));
  if (s.x > 0.003) { atomicMax(&grid[GRID_BLOCKS + blockIndex(vec3i(gid) / 8)], u32(min(s.x, 1000.0) * 1000.0)); }
}
