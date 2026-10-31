// The garden's sizes in one place: the domain, the grids over it, and where
// things lie in the grid buffer. renderer.ts takes them from here, and
// dimsWgsl() sets them down ahead of every shader (gpu/shaders.ts), so the
// two cannot drift apart.

type V3i = [number, number, number]
const prod = (v: number[]) => v.reduce((a, b) => a * b, 1)
const blocks = (v: V3i, n: number) => v.map((x) => Math.ceil(x / n)) as V3i

export const MAX_ORGS = 64
export const MAX_CELLS = 1 << 20
export const MAX_FRUITS = 4096
// germinations reported between readbacks
export const MAX_EVENTS = 4096
// head development (fruit.wgsl): slots of a 64 x 64 field, in an atlas
export const PAT_SLOTS = 768
export const PAT_RES = 64
export const PAT_COLS = 32
// steps between updates of the slow fields (sim.wgsl), and the slime mould's
export const FIELD_EVERY = 8
export const SLIME_EVERY = 4

// The planet (cm): its centre is the origin and its ground about PLANET_R from
// it. What lives on it stays between LIFE_IN and LIFE_OUT from the centre,
// and what is drawn of it within SKY_R. The domain is a cube around it, and
// the scenery's distance-field volume spans it; the ground map is the
// ground's height over the sphere, unfolded (world.wgsl octEncode).
export const PLANET_R = 13.5
export const LIFE_IN = 8.5
export const LIFE_OUT = 19.6
export const SKY_R = 20.6
export const BOX_MIN: V3i = [-20.8, -20.8, -20.8]
export const BOX_MAX: V3i = [20.8, 20.8, 20.8]
export const VOL: V3i = [288, 288, 288]
export const GROUND_RES: [number, number] = [1024, 1024]
// The fields, on a 4 mm grid; in 8-voxel blocks and 4-voxel sub-blocks.
export const F4 = 0.4
export const F4_RES: V3i = [104, 104, 104]
export const F4N = prod(F4_RES)
export const BLOCKS = blocks(F4_RES, 8)
export const N_BLOCKS = prod(BLOCKS)
export const SUBS = blocks(F4_RES, 4)
export const N_SUBS = prod(SUBS)
// The blocks life can reach (they cross the shell between LIFE_IN and
// LIFE_OUT, with a voxel or two to spare): the per-step block passes visit
// only these. Each packed x | y << 8 | z << 16, four to a vec4 of a uniform.
const shellBlocks = (() => {
  const out: number[] = []
  const side = 8 * F4
  const lo = LIFE_IN - 2 * F4
  const hi = LIFE_OUT + 2 * F4
  for (let z = 0; z < BLOCKS[2]; z++) {
    for (let y = 0; y < BLOCKS[1]; y++) {
      for (let x = 0; x < BLOCKS[0]; x++) {
        const a = [x, y, z].map((c, k) => BOX_MIN[k] + c * side)
        const near = Math.hypot(...a.map((v) => Math.max(v, Math.min(0, v + side))))
        const far = Math.hypot(...a.map((v) => Math.max(Math.abs(v), Math.abs(v + side))))
        if (far > lo && near < hi) out.push(x | (y << 8) | (z << 16))
      }
    }
  }
  return out
})()
export const N_SHELL = shellBlocks.length
export const SHELL = new Uint32Array(Math.ceil(N_SHELL / 4) * 4)
SHELL.set(shellBlocks)
// the baked light, at half the volume's resolution; and the light what glows
// casts, on an 8 mm grid
export const LIGHT: V3i = [144, 144, 144]
export const GLOW_CELL = 0.8
export const GLOW: V3i = [52, 52, 52]
export const GLOW_N = prod(GLOW)
// The slime mould: its agents, its trail grid over the planet (slime.wgsl:
// cells of about 1.2 mm, all the same area), its columns (four by four of
// its cells, about 4.6 mm), the grid buffer's words and the map's floats it
// has.
export const SLIME_MAX = 16384
export const SLIME_COL_RES = 104
export const SLIME_RES: [number, number] = [4 * SLIME_COL_RES, 4 * SLIME_COL_RES]
export const SLIME_N = prod(SLIME_RES)
export const SLIME_COLS = SLIME_COL_RES * SLIME_COL_RES
export const SLIME_WORDS = 32 + 3 * SLIME_COLS + SLIME_N
export const SLIME_FLOATS = 5 * SLIME_N + 2 * SLIME_COLS

// The grid buffer (u32 words): the per-voxel fields (cells.wgsl GF_*), then
// the tables after them in this order (sim.wgsl and slimesim.wgsl say what
// each holds).
export const GRID_FIELDS = 18
export const PACK_WORDS = MAX_CELLS / 256 + 4
export const SNAP_WORDS = 4 * MAX_ORGS + 4 + 4 * MAX_ORGS
export const FNB_WORDS = MAX_FRUITS * 8
const REGIONS = [
  ['VOXELS', GRID_FIELDS * F4N],
  ['FRUITS', MAX_FRUITS],
  ['FREE', MAX_CELLS],
  ['BLOCKS', N_BLOCKS],
  ['PATS', PAT_SLOTS],
  ['ACT', N_SUBS],
  ['SIGACT', N_BLOCKS],
  ['HOSTS', 4 * MAX_ORGS],
  ['PACK', PACK_WORDS],
  ['SNAP', SNAP_WORDS],
  ['FNB', FNB_WORDS],
  ['SIGNEW', N_BLOCKS],
  ['HOME', MAX_ORGS],
  ['SLIME', SLIME_WORDS],
] as const
export type GridRegion = (typeof REGIONS)[number][0]
// each region's first word, and its words
export const GRID = {} as Record<GridRegion, number>
export const GRID_SIZE = {} as Record<GridRegion, number>
let at = 0
for (const [name, words] of REGIONS) {
  GRID[name] = at
  GRID_SIZE[name] = words
  at += words
}
export const GRID_WORDS = at

// WGSL for all of the above (the grid's regions as GRID_<region>)
export function dimsWgsl(): string {
  const f = (x: number) => (Number.isInteger(x) ? x.toFixed(1) : `${x}`)
  const u = (x: number) => `${x}u`
  const v3f = (v: number[]) => `vec3f(${v.map(f).join(', ')})`
  const v3u = (v: number[]) => `vec3u(${v.map(u).join(', ')})`
  const v3i = (v: number[]) => `vec3i(${v.join(', ')})`
  const consts: [string, string, string][] = [
    ['MAX_ORGS', 'u32', u(MAX_ORGS)],
    ['MAX_CELLS', 'u32', u(MAX_CELLS)],
    ['MAX_FRUITS', 'u32', u(MAX_FRUITS)],
    ['MAX_EVENTS', 'u32', u(MAX_EVENTS)],
    ['PAT_SLOTS', 'u32', u(PAT_SLOTS)],
    ['PAT_RES', 'u32', u(PAT_RES)],
    ['PAT_COLS', 'u32', u(PAT_COLS)],
    ['FIELD_EVERY', 'u32', u(FIELD_EVERY)],
    ['SLIME_EVERY', 'u32', u(SLIME_EVERY)],
    ['PLANET_R', 'f32', f(PLANET_R)],
    ['LIFE_IN', 'f32', f(LIFE_IN)],
    ['LIFE_OUT', 'f32', f(LIFE_OUT)],
    ['SKY_R', 'f32', f(SKY_R)],
    ['BOX_MIN', 'vec3f', v3f(BOX_MIN)],
    ['BOX_MAX', 'vec3f', v3f(BOX_MAX)],
    ['VOL_RES', 'vec3u', v3u(VOL)],
    ['GROUND_RES', 'vec2u', `vec2u(${GROUND_RES.map(u).join(', ')})`],
    ['F4', 'f32', f(F4)],
    ['F4_RES', 'vec3u', v3u(F4_RES)],
    ['F4N', 'u32', u(F4N)],
    ['BLOCKS', 'vec3i', v3i(BLOCKS)],
    ['N_BLOCKS', 'u32', u(N_BLOCKS)],
    ['SUBS', 'vec3i', v3i(SUBS)],
    ['N_SUBS', 'u32', u(N_SUBS)],
    ['N_SHELL', 'u32', u(N_SHELL)],
    ['SHELL_VEC4', 'u32', u(SHELL.length / 4)],
    ['LIGHT_RES', 'vec3u', v3u(LIGHT)],
    ['GLOW_CELL', 'f32', f(GLOW_CELL)],
    ['GLOW_RES', 'vec3u', v3u(GLOW)],
    ['GLOW_N', 'u32', u(GLOW_N)],
    ['SLIME_MAX', 'u32', u(SLIME_MAX)],
    ['SLIME_RES', 'vec2u', `vec2u(${SLIME_RES.map(u).join(', ')})`],
    ['SLIME_N', 'u32', u(SLIME_N)],
    ['SLIME_COLS', 'u32', u(SLIME_COLS)],
    ['SLIME_COL_RES', 'u32', u(SLIME_COL_RES)],
    ['GRID_FIELDS', 'u32', u(GRID_FIELDS)],
    ['PACK_WORDS', 'u32', u(PACK_WORDS)],
    ['SNAP_WORDS', 'u32', u(SNAP_WORDS)],
    ['FNB_WORDS', 'u32', u(FNB_WORDS)],
    // (the voxels' fields from 0: cells.wgsl gi)
    ...REGIONS.slice(1).map(([name]) => [`GRID_${name}`, 'u32', u(GRID[name])] as [string, string, string]),
  ]
  return ['// ---- dims.ts: the garden\'s sizes ----', ...consts.map(([n, t, v]) => `const ${n}: ${t} = ${v};`)].join('\n')
}
