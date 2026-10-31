// ---- cells.wgsl: hyphal cells, organisms and the field grids ----
// A fungus is chains of cells. Each cell links to its parent; tips grow the
// chains. Cells are about a millimetre long.

const CELL_LEN: f32 = 0.1;          // cm
const POOL_FP: f32 = 1000.0;        // fixed point for organism resource pools

// cell flags (i.z): phase in the low bits
const F_PHASE: u32 = 0xfu;
const F_TIP: u32 = 0x10u;
const F_DEAD: u32 = 0x20u;
const F_FIXED: u32 = 0x40u;         // embedded in the substrate, or set tissue
const F_FREED: u32 = 0x80u;         // dead, and its slot handed back for reuse
const F_GEN_SHIFT: u32 = 8u;        // branch generation within the phase (4 bits)
const F_GEN: u32 = 0xf00u;
const F_MOVED: u32 = 0x1000u;       // a slot whose cell was packed down (sim.wgsl): i.x is where to
const F_NEWPOS: u32 = 0x2000u;      // moved this step: q.xyz holds where to (sim.wgsl deposit)
const F_DIED_ODD: u32 = 0x4000u;    // dead since an odd step (sim.wgsl killCell)
const NO_PARENT: u32 = 0xffffffffu;

struct Cell {
  p: vec4f,   // position, radius
  q: vec4f,   // previous position (or the next, F_NEWPOS); its own number (24 bits as a fraction), from its parent's, that its random rolls are drawn from
  d: vec4f,   // growth (rest) direction, cells grown in this phase
  i: vec4u,   // parent, organism, flags, birth step
}

// Fields over the glass box, 4 mm (F4, F4_RES, F4N: dims.ts).
fn f4Count() -> u32 { return F4_RES.x * F4_RES.y * F4_RES.z; }

// The grid buffer: per-voxel counters on the 4 mm grid, field after field,
// then the list of live fruit bodies and the stack of free cell slots.
// Float fields (fieldA/B) hold two vec4s per voxel: [v] how damp (the
// weather there and what was watered or dried), nutrient, light, litter;
// [F4N + v] airborne spores, settled spores, their parents, and how much
// wetter or drier than the weather it was made.
const GF_DENS: u32 = 0u;      // cells here this step
const GF_SIG: u32 = 1u;       // signal released this step (fixed point)
const GF_OWNER: u32 = 2u;     // oldest mycelium here: birth step << 6 | organism
const GF_OCC: u32 = 3u;       // fruit-body occupancy (rebuilt each frame)
const GF_KILL: u32 = 4u;      // organism + 1 that lost a fight here | step << 7
const GF_SCAR: u32 = 5u;      // fights here so far (zone lines)
const GF_SPORE_IN: u32 = 6u;  // spores released here (fixed point)
const GF_SPORE_ORG: u32 = 7u; // organism + 1 that released them
const GF_LITTER_IN: u32 = 8u; // dead tissue dropped here (fixed point)
const GF_IDS: u32 = 9u;       // (unused: spore parents ride in the spore field's z, sim.wgsl fields)
const GF_SIGNAL: u32 = 10u;   // the diffusing signal (f32 bits), two slots by step parity
const GF_EATEN: u32 = 12u;    // food eaten here since the fields last updated (1/65536)
const GF_PARA: u32 = 13u;     // the strongest mycoparasite here this step: parasitism (0..255) << 6 | organism
const GF_WORLD: u32 = 14u;    // what the (static) world is here, W_*, and (bits 8-15) how open to the sky (set on reset)
// eaten and fights lost here this step: merged into GF_EATEN and GF_KILL as
// the next begins (sim.wgsl clearFields), so this step's passes all read the
// same values whatever order the GPU runs them in
const GF_EATEN_NEW: u32 = 15u;
const GF_KILL_NEW: u32 = 16u;
const GF_EXCITE: u32 = 17u;   // firing tissue here and next to here this step (sim.wgsl, excite)
const W_SOLID: u32 = 1u;      // inside soil, rock or wood
const W_SHORE: u32 = 2u;      // within a voxel of a surface
const W_TOPSOIL: u32 = 4u;    // bare soil near the surface (fed from above)
const W_DEEP: u32 = 8u;       // well inside soil, rock or wood: never seen
const W_BARK: u32 = 16u;      // near the surface of wood
// (GRID_FIELDS of them, then their tables, as dims.ts lays them out:)
// GRID_FRUITS the live fruit bodies; GRID_FREE the free cell slots;
// GRID_BLOCKS most airborne spores in each 8-voxel block (x1000), so
// rendering can skip clean air; GRID_PATS the fruit body (index + 1) holding
// each head-development slot, or 0
fn gi(field: u32, v: u32) -> u32 { return field * F4N + v; }
fn blockIndex(b: vec3i) -> u32 { return u32((b.z * BLOCKS.y + b.y) * BLOCKS.x + b.x); }

fn f4Coord(p: vec3f) -> vec3i { return vec3i(floor((p - BOX_MIN) / F4)); }
fn f4Index(c: vec3i) -> u32 {
  let k = clamp(c, vec3i(0), vec3i(F4_RES) - 1);
  return (u32(k.z) * F4_RES.y + u32(k.y)) * F4_RES.x + u32(k.x);
}
fn inF4(c: vec3i) -> bool { return all(c >= vec3i(0)) && all(c < vec3i(F4_RES)); }

fn phaseOf(flags: u32) -> u32 { return flags & F_PHASE; }
// excitable tissue (sim.wgsl, excite): how lit a cell is as it fires,
// 0..255 (flags bits 16-23), and how long it must still rest (bits 24-31)
fn exciteOf(flags: u32) -> u32 { return (flags >> 16u) & 255u; }
fn restOf(flags: u32) -> u32 { return flags >> 24u; }

// The light glowing things cast, on an 8 mm grid over the box (glow.wgsl;
// GLOW_CELL, GLOW_RES, GLOW_N: dims.ts).
fn glowUVW(p: vec3f) -> vec3f { return (p - BOX_MIN) / (vec3f(GLOW_RES) * GLOW_CELL); }


