import { mulberry32, type Rng } from './genome/breed'
import { BOX_MAX, BOX_MIN, FLAT, PLANET_R } from './gpu/dims'
import { siteFrame, type Site } from './weather'

// The world's layout, generated from a seed: the lie of the ground, rocks,
// and branches, roots and twigs, placed by where over the ground they are
// and how high above it (the bake settles them onto it: bake_ground.wgsl).
// Units are cm. On the planet, its centre is the origin and its ground
// about PLANET_R from it; in the terrarium, the glass box's inside is
// BOX_MIN..BOX_MAX, the camera looking in from +x +z (the back is -x -z),
// and the box stands at a site on a planet (its sun and weather: weather.ts).

export type V3 = [number, number, number]
export const MAX_ROCKS = 16
export const MAX_ROOTS = 12
const MAX_BUMPS = 8
const ROCK_VEC4 = 6
const ROOT_VEC4 = 5
const TERRAIN_VEC4 = 2 + MAX_BUMPS + 2
// (and the terrarium's lie of the ground, at the end)
export const WORLD_BYTES = 16 + TERRAIN_VEC4 * 16 + MAX_ROCKS * ROCK_VEC4 * 16 + MAX_ROOTS * ROOT_VEC4 * 16 + 16

interface RockSpec {
  at: V3 // where (a direction)
  r: V3
  sink: number
  type: number
}

// a point of a piece of wood: in its direction, PLANET_R plus its height
// above the ground from the centre; and its radius
type Pt = [number, number, number, number]

interface RootSpec {
  pts: Pt[]
  type: number // < 0.5 bark, >= 0.5 driftwood
  rest?: [number, number] // rests on this rock, this far along
}

export interface World {
  seed: number
  buffer: ArrayBuffer
  rocks: { at: V3; r: number }[] // where (the planet: a direction; the terrarium: over the floor)
  wood: RootSpec[] // the first is the big branch or log
  site: Site | null // where on a planet the terrarium stands
}

const len = (v: number[]) => Math.hypot(...v)
const unit = (v: number[]) => {
  const l = len(v) || 1
  return v.map((x) => x / l) as V3
}
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
// how far apart two directions are over the ground (cm)
export const apart = (a: V3, b: V3) => PLANET_R * Math.acos(Math.max(-1, Math.min(1, dot(a, b))))
// a direction `s` cm on from `u` toward the direction (along the ground) `t`
const walk = (u: V3, t: V3, s: number) => {
  const a = s / PLANET_R
  return unit([0, 1, 2].map((k) => u[k] * Math.cos(a) + t[k] * Math.sin(a)))
}
// a direction along the ground from `u`, at angle `a`, in a frame of u's own
const toward = (u: V3, a: number): V3 => {
  const e = unit(cross(Math.abs(u[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0], u))
  const n = cross(u, e)
  return unit([0, 1, 2].map((k) => e[k] * Math.cos(a) + n[k] * Math.sin(a)))
}
const dirOf = (p: number[]) => unit(p.slice(0, 3))
// a wood point at direction `d`, `h` above the ground, radius `r`
const pt = (d: V3, h: number, r: number): Pt => [d[0] * (PLANET_R + h), d[1] * (PLANET_R + h), d[2] * (PLANET_R + h), r]
const heightOf = (p: Pt) => len(p.slice(0, 3)) - PLANET_R

function bez(p: Pt[], t: number): Pt {
  const it = 1 - t
  const w = [it * it * it, 3 * it * it * t, 3 * it * t * t, t * t * t]
  return [0, 1, 2, 3].map((k) => w.reduce((s, wi, i) => s + wi * p[i][k], 0)) as Pt
}

// Points along a piece of wood.
const samples = (rt: RootSpec, n = 12) => Array.from({ length: n + 1 }, (_, i) => bez(rt.pts, i / n))

function makePlanetWorld(seed: number): World {
  const rng = mulberry32(seed)
  const rand = (a: number, b: number) => a + rng() * (b - a)
  const anywhere = (): V3 => {
    const z = rng() * 2 - 1
    const a = rng() * Math.PI * 2
    const s = Math.sqrt(1 - z * z)
    return [s * Math.cos(a), z, s * Math.sin(a)]
  }

  // Highland and lowland over the whole planet, a broad rise somewhere, and
  // hollows and mounds.
  const bumps: [V3, number, number][] = []
  if (rng() < 0.6) bumps.push([anywhere(), rand(9, 13), rand(1.5, 3)])
  const hollows = 1 + Math.floor(rng() * 3)
  for (let i = 0; i < hollows; i++) bumps.push([anywhere(), rand(3, 5), -rand(0.8, 1.5)])
  const mounds = Math.floor(rng() * 4)
  for (let i = 0; i < mounds; i++) bumps.push([anywhere(), rand(2.5, 5.5), rand(0.7, 2)])

  const wood: RootSpec[] = []
  const rocks: RockSpec[] = []
  const nearWood = (d: V3, clear: number) => wood.some((rt) => samples(rt).some((p) => apart(d, dirOf(p)) < p[3] + clear))
  const rockClash = (d: V3, r: number, slack: number) => rocks.some((o) => apart(d, o.at) < (r + Math.max(o.r[0], o.r[2])) * slack)

  // Rocks: one geology per planet, with the odd stranger.
  const geology = rng()
  const rockType = () => (rng() < 0.15 ? rng() : Math.min(1, Math.max(0, geology + rand(-0.18, 0.18))))
  const addRock = (at: V3, r: V3, sink: number) => {
    if (rocks.length < MAX_ROCKS) rocks.push({ at, r, sink, type: rockType() })
  }

  // The landmark: a big rock, often on a knoll.
  {
    const rx = rand(2.6, 3.6)
    const at = anywhere()
    addRock(at, [rx, rx * rand(0.72, 0.85), rx * rand(0.75, 0.9)], rand(-0.35, -0.25))
    if (rng() < 0.6) bumps.push([at, rx * 1.6, rand(0.4, 1)])
  }
  const hero = rocks[0]

  // The main piece of wood: a weathered branch arching over the big rock, or
  // a fallen log lying beside it.
  {
    const arch = rng() < 0.65
    const hr = Math.max(hero.r[0], hero.r[2])
    const l = arch ? rand(11, 15) : rand(8, 12)
    const t = toward(hero.at, rng() * Math.PI * 2)
    const sideT = unit(cross(hero.at, t))
    let rt: RootSpec
    if (arch) {
      // the rock is a third of the way along
      const s = walk(hero.at, t, -l / 3)
      const at = (f: number, off: number) => walk(walk(s, t, l * f), sideT, off)
      const r0 = rand(0.95, 1.3)
      rt = {
        pts: [
          pt(s, rand(0.5, 0.9), r0),
          // lifted onto the rock when baked (bake_ground.wgsl)
          pt(at(1 / 3, 0), 3.4, r0 * rand(0.75, 0.88)),
          pt(at(2 / 3, rand(-1.1, 1.1)), rand(1.8, 3.3), r0 * rand(0.55, 0.7)),
          pt(at(1, 0), rand(0.12, 0.3), rand(0.35, 0.5)),
        ],
        type: rng() < 0.6 ? rand(0.55, 0.95) : rand(0.1, 0.4),
        rest: [0, 1 / 3],
      }
    } else {
      const off = (rng() < 0.5 ? -1 : 1) * (hr + rand(1.3, 2.1))
      const s = walk(walk(hero.at, sideT, off), t, -l * rand(0.2, 0.8))
      const bow = rand(-0.9, 0.9)
      const at = (f: number, b: number) => walk(walk(s, t, l * f), sideT, b)
      const r0 = rand(1.4, 2)
      const r3 = r0 * rand(0.65, 0.85)
      rt = {
        pts: [pt(at(0, 0), r0 * 0.4, r0), pt(at(1 / 3, bow), r0 * 0.45, r0 * 0.95), pt(at(2 / 3, bow), r3 * 0.45, r3 * 1.02), pt(at(1, 0), r3 * 0.35, r3)],
        type: rand(0.05, 0.35),
      }
    }
    if (rng() < 0.5) {
      rt.pts.reverse()
      if (rt.rest) rt.rest[1] = 1 - rt.rest[1]
    }
    wood.push(rt)
  }

  const place = (n: number, tries: number, r: () => number, ok: (d: V3, rx: number) => boolean, make: (d: V3, rx: number) => void) => {
    for (let i = 0, k = 0; i < n && k < tries && rocks.length < MAX_ROCKS; k++) {
      const rx = r()
      const d = anywhere()
      if (!ok(d, rx)) continue
      make(d, rx)
      i++
    }
  }
  // a second boulder, a flat slab, middling rocks
  if (rng() < 0.4) {
    place(1, 200, () => rand(1.9, 2.6), (d, rx) => !rockClash(d, rx, 1.1) && !nearWood(d, rx + 0.3),
      (d, rx) => addRock(d, [rx, rx * rand(0.7, 0.85), rx * rand(0.75, 0.9)], rand(-0.35, -0.25)))
  }
  if (rng() < 0.6) {
    place(1, 100, () => rand(1.6, 2.3), (d, rx) => !rockClash(d, rx, 1) && !nearWood(d, rx * 0.8),
      (d, rx) => addRock(d, [rx, rand(0.5, 0.75), rx * rand(0.7, 0.85)], -0.15))
  }
  place(4 + Math.floor(rng() * 3), 300, () => rand(0.9, 1.9), (d, rx) => !rockClash(d, rx, 1) && !nearWood(d, rx * 0.8),
    (d, rx) => addRock(d, [rx, rx * rand(0.7, 0.85), rx * rand(0.8, 0.95)], rand(-0.35, -0.25)))
  const pebbles = 6 + Math.floor(rng() * 4)
  for (let i = 0, k = 0; i < pebbles && k < 300 && rocks.length < MAX_ROCKS; k++) {
    const rx = rand(0.4, 0.8)
    let d = anywhere()
    if (rng() < 0.5) {
      // scree at the foot of a bigger rock
      const o = rocks[Math.floor(rng() * rocks.length)]
      d = walk(o.at, toward(o.at, rng() * Math.PI * 2), Math.max(o.r[0], o.r[2]) + rx * rand(0.6, 1))
    }
    if (rockClash(d, rx, 0.8)) continue
    addRock(d, [rx, rx * rand(0.65, 0.85), rx * rand(0.75, 0.95)], rand(-0.35, -0.2))
    i++
  }

  // Roots surfacing from the soil and diving back, often from where the
  // branch's thick end lies, and a few twigs.
  const clearOfRocks = (rt: RootSpec) => samples(rt).every((p) => rocks.every((o) => apart(dirOf(p), o.at) > Math.max(o.r[0], o.r[2]) * 0.9 + p[3]))
  const stick = (n: number, l: [number, number], make: (s: V3, t: V3, l: number) => RootSpec, from?: () => V3) => {
    for (let i = 0, k = 0; i < n && k < 200 && wood.length < MAX_ROOTS; k++) {
      const s = from && rng() < 0.5 ? from() : anywhere()
      const rt = make(s, toward(s, rng() * Math.PI * 2), rand(l[0], l[1]))
      if (!clearOfRocks(rt)) continue
      wood.push(rt)
      i++
    }
  }
  const ends = wood[0].pts
  const thick = dirOf(ends[0][3] > ends[3][3] ? ends[0] : ends[3])
  // four directions along a gently bent line from s
  const path = (s: V3, t: V3, l: number, bend: number) => {
    const side = unit(cross(s, t))
    return [0, 1 / 3, 2 / 3, 1].map((f) => walk(walk(s, t, l * f), side, bend * Math.sin(Math.PI * f)))
  }
  stick(
    2 + Math.floor(rng() * 2),
    [4, 8],
    (s, t, l) => {
      const p = path(s, t, l, rand(-1.1, 1.1))
      const r0 = rand(0.3, 0.45)
      return {
        pts: [pt(p[0], -0.8, r0), pt(p[1], rand(0.75, 1.05), r0 * 0.85), pt(p[2], rand(0.45, 0.7), r0 * 0.68), pt(p[3], -rand(0.45, 0.7), r0 * 0.5)],
        type: rand(0.15, 0.35),
      }
    },
    () => walk(thick, toward(thick, rng() * Math.PI * 2), rand(0, 1.1)),
  )
  if (rng() < 0.4) {
    // a fallen stick
    stick(1, [6, 10], (s, t, l) => {
      const r0 = rand(0.35, 0.5)
      return { pts: path(s, t, l, rand(-0.8, 0.8)).map((d, i) => pt(d, r0 * 0.35, r0 * (1 - i * 0.12))), type: rand(0.2, 0.9) }
    })
  }
  stick(1 + Math.floor(rng() * 3), [2.5, 4.5], (s, t, l) => {
    // crooked, lying on the moss
    const r0 = rand(0.13, 0.17)
    return {
      pts: path(s, t, l, rand(-0.6, 0.6)).map((d, i) => pt(walk(d, toward(d, rng() * Math.PI * 2), rand(0, 0.4)), rand(0.35, 0.45), r0 * (1 - i * 0.15))),
      type: rand(0.3, 0.9),
    }
  })

  // pack for world.wgsl (WorldDesc)
  const buf = new ArrayBuffer(WORLD_BYTES)
  const f = new Float32Array(buf)
  const u = new Uint32Array(buf)
  const nb = Math.min(bumps.length, MAX_BUMPS)
  u.set([rocks.length, wood.length, nb, 0], 0)
  f.set([rand(0.8, 1.6), rand(0.9, 1.5), rand(0.12, 0.22), 0], 4)
  u.set([Math.floor(rng() * 1e6), Math.floor(rng() * 1e6), Math.floor(rng() * 1e6), 0], 8)
  bumps.slice(0, nb).forEach(([d, r, h], i) => {
    f.set([d[0] * PLANET_R, d[1] * PLANET_R, d[2] * PLANET_R, r], 12 + i * 4)
    f[12 + MAX_BUMPS * 4 + i] = h
  })
  const rock0 = 4 + TERRAIN_VEC4 * 4
  rocks.forEach((rk, i) => {
    const o = rock0 + i * ROCK_VEC4 * 4
    f.set([rk.at[0] * PLANET_R, rk.at[1] * PLANET_R, rk.at[2] * PLANET_R, Math.floor(rng() * 1000)], o)
    f.set([...rk.r, rk.type], o + 4)
    f.set([rng() * Math.PI * 2, (rng() - 0.5) * 0.4, rk.sink, 0], o + 8)
    for (let k = 0; k < 3; k++) {
      // fracture planes: mostly sideways, one often near the top
      const up = k === 0 ? 0.6 + rng() * 0.4 : (rng() - 0.3) * 0.6
      const a = rng() * Math.PI * 2
      const n = unit([Math.cos(a), up, Math.sin(a)])
      f.set([...n, 0.45 + rng() * 0.35], o + 12 + k * 4)
    }
  })
  const root0 = rock0 + MAX_ROCKS * ROCK_VEC4 * 4
  wood.forEach((rt, i) => {
    const o = root0 + i * ROOT_VEC4 * 4
    rt.pts.forEach((p, k) => f.set(p, o + k * 4))
    f.set([Math.floor(rng() * 1000), rt.type, rt.rest ? rt.rest[0] + 1 : 0, rt.rest?.[1] ?? 0], o + 16)
  })
  return { seed, buffer: buf, rocks: rocks.map((r) => ({ at: r.at, r: Math.max(r.r[0], r.r[2]) })), wood, site: null }
}

// Planting spots over the whole planet (about `step` apart), clear of rocks
// and lying wood, ordered so that each next spot is as far as possible from
// those before it: a few colonies spread far apart. Each a point on the
// sphere of PLANET_R (the ground is found under it).
function planetSpots(world: World, seed: number, step = 3.8): V3[] {
  const rng = mulberry32(seed)
  const n = Math.round((4 * Math.PI * PLANET_R * PLANET_R) / (step * step))
  const all: V3[] = []
  for (let i = 0; i < n; i++) {
    // a Fibonacci sphere, jittered
    const y = 1 - (2 * (i + 0.5)) / n
    const a = i * 2.39996323
    const r = Math.sqrt(1 - y * y)
    const base = unit([r * Math.cos(a), y, r * Math.sin(a)])
    const d = walk(base, toward(base, rng() * Math.PI * 2), rng() * step * 0.35)
    if (world.rocks.some((o) => apart(d, o.at) < o.r + 1.4)) continue
    const onWood = world.wood.some((rt) => samples(rt).some((p) => heightOf(p) < p[3] + 1 && apart(d, dirOf(p)) < p[3] + 0.8))
    if (onWood) continue
    all.push(d)
  }
  const spots: V3[] = []
  if (!all.length) return spots
  spots.push(all.splice(Math.floor(rng() * all.length), 1)[0])
  const near = all.map((d) => apart(d, spots[0]))
  while (all.length) {
    let best = 0
    near.forEach((v, i) => {
      if (v > near[best]) best = i
    })
    const s = all.splice(best, 1)[0]
    near.splice(best, 1)
    spots.push(s)
    all.forEach((d, i) => (near[i] = Math.min(near[i], apart(d, s))))
  }
  return spots.map((d) => d.map((x) => x * PLANET_R) as V3)
}

// A direction as a point of the square [-1, 1]^2 (world.wgsl octEncode): where
// the ground map holds the ground there.
export function octEncode(n: V3): [number, number] {
  const l = Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2])
  const p: [number, number] = [n[0] / l, n[2] / l]
  if (n[1] >= 0) return p
  return [(1 - Math.abs(p[1])) * (p[0] >= 0 ? 1 : -1), (1 - Math.abs(p[0])) * (p[1] >= 0 ? 1 : -1)]
}

// A point `s` cm over the ground from over `p`, at angle `a` (in a frame of
// p's own): on the planet, on the sphere of PLANET_R; in the terrarium, over
// the floor, inside the glass.
export function beside(p: V3, s: number, a: number): V3 {
  if (FLAT) {
    const m = 1.5
    return [clampIn(p[0] + Math.cos(a) * s, 0, m), 0, clampIn(p[2] + Math.sin(a) * s, 2, m)]
  }
  const d = dirOf(p)
  return walk(d, toward(d, a), s).map((x) => x * PLANET_R) as V3
}

// A point along the big branch or log, for wood-dwelling strains (over it:
// the wood is found under it).
export function branchSpot(world: World, rng: Rng): V3 {
  const p = bez(world.wood[0].pts, 0.25 + rng() * 0.5)
  return p.slice(0, 3) as V3
}

export const makeWorld = (seed: number) => (FLAT ? makeBoxWorld(seed) : makePlanetWorld(seed))
export const plantingSpots = (world: World, seed: number) => (FLAT ? boxSpots(world, seed) : planetSpots(world, seed))

// ------------------------------------------------------------ terrarium ---

const clampIn = (v: number, k: number, m: number) => Math.min(BOX_MAX[k] - m, Math.max(BOX_MIN[k] + m, v))

// The terrarium's layout: the ground rising toward the back, a big rock on
// the high ground with a branch arching over it (or a log beside it), rocks,
// roots and twigs. Wood's points are [x, height above the ground, z,
// radius]; rocks are where over the floor.
function makeBoxWorld(seed: number): World {
  const rng = mulberry32(seed)
  const rand = (a: number, b: number) => a + rng() * (b - a)
  const B = { minX: BOX_MIN[0], maxX: BOX_MAX[0], minZ: BOX_MIN[2], maxZ: BOX_MAX[2] }
  const inBox = (x: number, z: number, m: number) => x > B.minX + m && x < B.maxX - m && z > B.minZ + m && z < B.maxZ - m
  const anywhere = (m: number): [number, number] => [rand(B.minX + m, B.maxX - m), rand(B.minZ + m, B.maxZ - m)]

  // The ground rises toward a side at the back, so the view in from the
  // front stays open, often into a hill in the back corner.
  const ra = Math.PI * 1.25 + rand(-1, 1)
  const back = [Math.cos(ra), Math.sin(ra)]
  const depth = (x: number, z: number) => x * back[0] + z * back[1] // + toward the back
  const bumps: [number, number, number, number][] = []
  if (rng() < 0.6) {
    const cx = Math.sign(back[0]) * (B.maxX - rand(4, 9))
    const cz = Math.sign(back[1]) * (B.maxZ - rand(4, 8))
    bumps.push([cx, cz, rand(8, 12), rand(1.5, 3)])
  }
  const hollows = 1 + Math.floor(rng() * 3)
  for (let i = 0; i < hollows; i++) {
    let [x, z] = anywhere(5)
    for (let k = 0; k < 20 && depth(x, z) > 4; k++) [x, z] = anywhere(5)
    bumps.push([x, z, rand(3.5, 5.5), -rand(0.8, 1.5)])
  }
  const mounds = Math.floor(rng() * 4)
  for (let i = 0; i < mounds; i++) {
    const [x, z] = anywhere(3)
    bumps.push([x, z, rand(3, 6.5), rand(0.7, 2)])
  }

  type BoxRock = { x: number; z: number; r: V3; sink: number; type: number }
  const wood: RootSpec[] = []
  const rocks: BoxRock[] = []
  const flat = (rt: RootSpec) => samples(rt).map((p) => ({ x: p[0], z: p[2], r: p[3], y: p[1] }))
  const nearWood = (x: number, z: number, clear: number) =>
    wood.some((rt) => flat(rt).some((p) => Math.hypot(x - p.x, z - p.z) < p.r + clear))
  const rockClash = (x: number, z: number, r: number, slack: number) =>
    !inBox(x, z, r + 0.5) || rocks.some((o) => Math.hypot(x - o.x, z - o.z) < (r + Math.max(o.r[0], o.r[2])) * slack)

  // Rocks: one geology per terrarium, with the odd stranger.
  const geology = rng()
  const rockType = () => (rng() < 0.15 ? rng() : Math.min(1, Math.max(0, geology + rand(-0.18, 0.18))))
  const addRock = (x: number, z: number, r: V3, sink: number) => {
    rocks.push({ x, z, r, sink, type: rockType() })
  }

  // The landmark: a big rock on the high ground, often on a knoll.
  {
    const rx = rand(3.5, 5)
    let [x, z] = anywhere(rx + 0.5)
    for (let k = 0; k < 100 && depth(x, z) < 5; k++) [x, z] = anywhere(rx + 0.5)
    addRock(x, z, [rx, rx * rand(0.72, 0.85), rx * rand(0.75, 0.9)], rand(-0.35, -0.25))
    if (rng() < 0.6) bumps.push([x, z, rx * 1.6, rand(0.4, 1)])
  }
  const hero = rocks[0]

  // The main piece of wood: a weathered branch arching over the big rock and
  // down toward the front, or a fallen log lying beside it.
  {
    const arch = rng() < 0.65
    const hr = Math.max(hero.r[0], hero.r[2])
    let len = arch ? rand(20, 28) : rand(13, 20)
    let s: [number, number] = [0, 0]
    let e: [number, number] = [0, 0]
    let side = 0
    for (let k = 0; k < 400; k++) {
      if (k % 100 === 99) len *= 0.85
      const a = rng() * Math.PI * 2
      const u = [Math.cos(a), Math.sin(a)]
      if (arch) {
        // the rock is a third of the way along
        s = [hero.x - (u[0] * len) / 3, hero.z - (u[1] * len) / 3]
      } else {
        side = (rng() < 0.5 ? -1 : 1) * (hr + rand(1.7, 2.8))
        const back0 = len * rand(0.2, 0.8)
        s = [hero.x - u[1] * side - u[0] * back0, hero.z + u[0] * side - u[1] * back0]
      }
      e = [s[0] + u[0] * len, s[1] + u[1] * len]
      if (inBox(s[0], s[1], 1.5) && inBox(e[0], e[1], 1.5) && depth(e[0], e[1]) < depth(hero.x, hero.z)) break
    }
    const along = (t: number, off: number): [number, number] => {
      const dx = e[0] - s[0]
      const dz = e[1] - s[1]
      const l = Math.hypot(dx, dz) || 1
      return [s[0] + dx * t - (dz / l) * off, s[1] + dz * t + (dx / l) * off]
    }
    let rt: RootSpec
    if (arch) {
      const r0 = rand(0.95, 1.3)
      const [x1, z1] = along(1 / 3, 0)
      const [x2, z2] = along(2 / 3, rand(-1.5, 1.5))
      rt = {
        pts: [
          [s[0], rand(0.6, 1.2), s[1], r0],
          // lifted onto the rock when baked (bake_ground.wgsl)
          [x1, 4.5, z1, r0 * rand(0.75, 0.88)],
          [x2, rand(2.5, 4.5), z2, r0 * rand(0.55, 0.7)],
          [e[0], rand(0.15, 0.4), e[1], rand(0.35, 0.5)],
        ],
        type: rng() < 0.6 ? rand(0.55, 0.95) : rand(0.1, 0.4),
        rest: [0, 1 / 3],
      }
    } else {
      const r0 = rand(1.4, 2)
      const r3 = r0 * rand(0.65, 0.85)
      const bow = rand(-1.2, 1.2)
      const [x1, z1] = along(1 / 3, bow)
      const [x2, z2] = along(2 / 3, bow)
      rt = {
        pts: [
          [s[0], r0 * 0.4, s[1], r0],
          [x1, r0 * 0.45, z1, r0 * 0.95],
          [x2, r3 * 0.45, z2, r3 * 1.02],
          [e[0], r3 * 0.35, e[1], r3],
        ],
        type: rand(0.05, 0.35),
      }
    }
    if (rng() < 0.5) {
      rt.pts.reverse()
      if (rt.rest) rt.rest[1] = 1 - rt.rest[1]
    }
    wood.push(rt)
  }

  if (rng() < 0.4) {
    // a second boulder
    for (let k = 0; k < 200; k++) {
      const rx = rand(2.6, 3.6)
      const [x, z] = anywhere(rx + 0.5)
      if (rockClash(x, z, rx, 1.1) || nearWood(x, z, rx + 0.3)) continue
      addRock(x, z, [rx, rx * rand(0.7, 0.85), rx * rand(0.75, 0.9)], rand(-0.35, -0.25))
      break
    }
  }
  if (rng() < 0.6) {
    // a flat slab
    for (let k = 0; k < 100; k++) {
      const rx = rand(2.2, 3.2)
      const [x, z] = anywhere(rx + 0.5)
      if (rockClash(x, z, rx, 1) || nearWood(x, z, rx * 0.8)) continue
      addRock(x, z, [rx, rand(0.7, 1), rx * rand(0.7, 0.85)], -0.15)
      break
    }
  }
  const medium = 3 + Math.floor(rng() * 2)
  for (let i = 0, k = 0; i < medium && k < 300; k++) {
    const rx = rand(1.2, 2.6)
    const [x, z] = anywhere(rx + 0.5)
    if (depth(x, z) < -12 || rockClash(x, z, rx, 1) || nearWood(x, z, rx * 0.8)) continue
    addRock(x, z, [rx, rx * rand(0.7, 0.85), rx * rand(0.8, 0.95)], rand(-0.35, -0.25))
    i++
  }
  const pebbles = 4 + Math.floor(rng() * 5)
  for (let i = 0, k = 0; i < pebbles && k < 300 && rocks.length < MAX_ROCKS; k++) {
    const rx = rand(0.55, 1.1)
    let [x, z] = anywhere(rx + 0.5)
    if (rng() < 0.5) {
      // scree at the foot of a bigger rock
      const o = rocks[Math.floor(rng() * rocks.length)]
      const a = rng() * Math.PI * 2
      const d = Math.max(o.r[0], o.r[2]) + rx * rand(0.6, 1)
      ;[x, z] = [o.x + Math.cos(a) * d, o.z + Math.sin(a) * d]
    }
    if (rockClash(x, z, rx, 0.8)) continue
    addRock(x, z, [rx, rx * rand(0.65, 0.85), rx * rand(0.75, 0.95)], rand(-0.35, -0.2))
    i++
  }

  // Roots surfacing from the soil and diving back, often from where the
  // branch's thick end lies, and a few twigs.
  const clearOfRocks = (rt: RootSpec) =>
    flat(rt).every((p) => inBox(p.x, p.z, p.r + 0.4) && rocks.every((o) => Math.hypot(p.x - o.x, p.z - o.z) > Math.max(o.r[0], o.r[2]) * 0.9 + p.r))
  const stick = (n: number, len: [number, number], make: (s: [number, number], a: number, l: number) => RootSpec, from?: () => [number, number]) => {
    for (let i = 0, k = 0; i < n && k < 200 && wood.length < MAX_ROOTS; k++) {
      const s = from && rng() < 0.5 ? from() : anywhere(2)
      const rt = make(s, rng() * Math.PI * 2, rand(len[0], len[1]))
      if (!clearOfRocks(rt)) continue
      wood.push(rt)
      i++
    }
  }
  const ends = wood[0].pts
  const thick = ends[0][3] > ends[3][3] ? ends[0] : ends[3]
  const path = (s: [number, number], a: number, l: number, bend: number) => {
    const c = Math.cos(a)
    const sn = Math.sin(a)
    return [0, 1 / 3, 2 / 3, 1].map((t) => {
      const side = bend * Math.sin(Math.PI * t)
      return [s[0] + c * l * t - sn * side, s[1] + sn * l * t + c * side] as [number, number]
    })
  }
  stick(
    2 + Math.floor(rng() * 2),
    [5, 10],
    (s, a, l) => {
      const p = path(s, a, l, rand(-1.5, 1.5))
      const r0 = rand(0.4, 0.6)
      return {
        pts: [
          [p[0][0], -1.1, p[0][1], r0],
          [p[1][0], rand(1, 1.4), p[1][1], r0 * 0.85],
          [p[2][0], rand(0.6, 0.9), p[2][1], r0 * 0.68],
          [p[3][0], -rand(0.6, 0.9), p[3][1], r0 * 0.5],
        ],
        type: rand(0.15, 0.35),
      }
    },
    () => [thick[0] + rand(-1.5, 1.5), thick[2] + rand(-1.5, 1.5)],
  )
  if (rng() < 0.4) {
    // a fallen stick
    stick(1, [8, 14], (s, a, l) => {
      const p = path(s, a, l, rand(-1, 1))
      const r0 = rand(0.45, 0.7)
      return { pts: p.map(([x, z], i) => [x, r0 * 0.35, z, r0 * (1 - i * 0.12)] as Pt), type: rand(0.2, 0.9) }
    })
  }
  stick(1 + Math.floor(rng() * 3), [3, 6], (s, a, l) => {
    // crooked, lying on the moss
    const p = path(s, a, l, rand(-0.8, 0.8)).map(([x, z]) => [x + rand(-0.5, 0.5), z + rand(-0.5, 0.5)])
    const r0 = rand(0.17, 0.22)
    return {
      pts: p.map(([x, z], i) => [x, rand(0.45, 0.6), z, r0 * (1 - i * 0.15)] as Pt),
      type: rand(0.3, 0.9),
    }
  })

  // pack for world.wgsl (WorldDesc; shape_box.wgsl reads the ground)
  const buf = new ArrayBuffer(WORLD_BYTES)
  const f = new Float32Array(buf)
  const u = new Uint32Array(buf)
  const nb = Math.min(bumps.length, MAX_BUMPS)
  u.set([rocks.length, wood.length, nb, 0], 0)
  const rise0 = rand(-5, 2)
  const base = [rand(0, 0.6), rand(2.5, 5)]
  const rise = [rise0, rise0 + rand(10, 17), rand(1.1, 1.8), rand(0.15, 0.28)]
  f.set([base[0], base[1], rise[2], rise[3]], 4)
  u.set([Math.floor(rng() * 1e6), Math.floor(rng() * 1e6), 0, 0], 8)
  bumps.slice(0, nb).forEach(([x, z, r, h], i) => {
    f.set([x, 0, z, r], 12 + i * 4)
    f[12 + MAX_BUMPS * 4 + i] = h
  })
  const rock0 = 4 + TERRAIN_VEC4 * 4
  rocks.forEach((rk, i) => {
    const o = rock0 + i * ROCK_VEC4 * 4
    f.set([rk.x, 0, rk.z, Math.floor(rng() * 1000)], o)
    f.set([...rk.r, rk.type], o + 4)
    f.set([rng() * Math.PI * 2, (rng() - 0.5) * 0.4, rk.sink, 0], o + 8)
    for (let k = 0; k < 3; k++) {
      // fracture planes: mostly sideways, one often near the top
      const up = k === 0 ? 0.6 + rng() * 0.4 : (rng() - 0.3) * 0.6
      const a = rng() * Math.PI * 2
      const n = unit([Math.cos(a), up, Math.sin(a)])
      f.set([...n, 0.45 + rng() * 0.35], o + 12 + k * 4)
    }
  })
  const root0 = rock0 + MAX_ROCKS * ROCK_VEC4 * 4
  wood.forEach((rt, i) => {
    const o = root0 + i * ROOT_VEC4 * 4
    rt.pts.forEach((p, k) => f.set(p, o + k * 4))
    f.set([Math.floor(rng() * 1000), rt.type, rt.rest ? rt.rest[0] + 1 : 0, rt.rest?.[1] ?? 0], o + 16)
  })
  f.set([back[0], back[1], rise[0], rise[1]], root0 + MAX_ROOTS * ROOT_VEC4 * 4)
  // where on a planet it stands: 30-55 degrees north or south
  const lat = (rand(30, 55) * Math.PI) / 180 * (rng() < 0.5 ? 1 : -1)
  return {
    seed, buffer: buf, wood, site: siteFrame(lat),
    rocks: rocks.map((r) => ({ at: [r.x, 0, r.z] as V3, r: Math.max(r.r[0], r.r[2]) })),
  }
}

// Planting spots on a jittered grid over the floor, clear of rocks and lying
// wood, ordered so that each next spot is as far as possible from those
// before it: a few colonies spread over the whole box.
function boxSpots(world: World, seed: number, step = 3.8): V3[] {
  const rng = mulberry32(seed)
  const all: V3[] = []
  for (let z = BOX_MIN[2] + 2.5; z < BOX_MAX[2] - 2; z += step) {
    for (let x = BOX_MIN[0] + 2.5; x < BOX_MAX[0] - 2; x += step) {
      const px = x + (rng() - 0.5) * 1.5
      const pz = z + (rng() - 0.5) * 1.3
      if (world.rocks.some((r) => Math.hypot(px - r.at[0], pz - r.at[2]) < r.r + 1.4)) continue
      const onWood = world.wood.some((rt) => samples(rt).some((p) => p[1] < p[3] + 1 && Math.hypot(px - p[0], pz - p[2]) < p[3] + 0.8))
      if (onWood) continue
      all.push([px, 0, pz])
    }
  }
  const spots: V3[] = []
  if (!all.length) return spots
  spots.push(all.splice(Math.floor(rng() * all.length), 1)[0])
  while (all.length) {
    let best = 0
    let bestD = -1
    all.forEach(([x, , z], i) => {
      const d = Math.min(...spots.map(([sx, , sz]) => Math.hypot(x - sx, z - sz)))
      if (d > bestD) [best, bestD] = [i, d]
    })
    spots.push(all.splice(best, 1)[0])
  }
  return spots
}
