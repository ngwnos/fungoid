import { GENES, NUM_GENES, STOP, phaseValue, type Genome } from './genes'

// (with its state to keep, and to go back to: history.ts)
export type Rng = (() => number) & { state?: () => number; restore?: (s: number) => void }

export function mulberry32(seed: number): Rng {
  let s = seed >>> 0
  const next: Rng = () => {
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  next.state = () => s
  next.restore = (v) => (s = v >>> 0)
  return next
}

const gauss = (rng: Rng) => {
  const u = Math.max(rng(), 1e-9)
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng())
}

// Linkage groups are contiguous runs of genes sharing a group.
const CHROMOSOMES: [number, number][] = (() => {
  const out: [number, number][] = []
  let start = 0
  for (let i = 1; i <= NUM_GENES; i++) {
    if (i === NUM_GENES || GENES[i].group !== GENES[start].group) {
      out.push([start, i])
      start = i
    }
  }
  return out
})()

// Meiosis in the gills: recombine the two nuclei into one haploid spore.
export function meiosis(parent: Genome, rng: Rng): Uint8Array {
  const out = new Uint8Array(NUM_GENES)
  for (const [lo, hi] of CHROMOSOMES) {
    let strand = rng() < 0.5 ? parent.a : parent.b
    const crossP = 1.3 / (hi - lo)
    for (let i = lo; i < hi; i++) {
      if (i > lo && rng() < crossP) strand = strand === parent.a ? parent.b : parent.a
      out[i] = strand[i]
    }
  }
  return out
}

export function mutateHaploid(h: Uint8Array, rng: Rng, rate = 0.04, scale = 0.08): Uint8Array {
  const out = h.slice()
  for (let i = 0; i < out.length; i++) {
    const r = rng()
    if (r < rate * 0.08) out[i] = Math.floor(rng() * 256) // rare macro-mutation
    else if (r < rate) out[i] = clampByte(out[i] + gauss(rng) * scale * 255)
  }
  return out
}

const clampByte = (v: number) => Math.max(0, Math.min(255, Math.round(v)))

// Two compatible monokaryons fuse into a dikaryon: one spore from each parent.
export function cross(p1: Genome, p2: Genome, rng: Rng): Genome {
  return { a: mutateHaploid(meiosis(p1, rng), rng), b: mutateHaploid(meiosis(p2, rng), rng) }
}

export function mutate(g: Genome, rng: Rng, strength = 1): Genome {
  return {
    a: mutateHaploid(g.a, rng, 0.25 * strength, 0.1 * strength),
    b: mutateHaploid(g.b, rng, 0.25 * strength, 0.1 * strength),
  }
}

// A wild fungus: everything about its form is random, but it keeps a working
// mycelium from `base` and fruits within a sensible range of reserves, so it
// lives long enough to show what it is.
const KEEP = new Set(['Mycelium'])
// (and most keep no night habits; about a third take each)
const HABITS = new Set(['excite', 'duskFire', 'nightOpen', 'nightSpore', 'blink'])
const RANGES: Record<string, [number, number]> = {
  absorb: [0.4, 0.9],
  fruitAt: [0.2, 0.5],
  fruitProb: [0.3, 0.7],
  body: [0.3, 1],
  cluster: [0, 0.8],
  size: [0.15, 0.75],
  grow: [0.15, 0.6],
  life: [0.2, 0.7],
}

export function randomGenome(rng: Rng, base?: Genome): Genome {
  const a = new Uint8Array(NUM_GENES)
  const b = new Uint8Array(NUM_GENES)
  for (let i = 0; i < NUM_GENES; i++) {
    const g = GENES[i]
    if (base && KEEP.has(g.group)) {
      a[i] = clampByte(base.a[i] + gauss(rng) * 8)
      b[i] = clampByte(base.b[i] + gauss(rng) * 8)
      continue
    }
    const [lo, hi] = RANGES[g.key] ?? [0, 1]
    let v = lo + (hi - lo) * rng()
    // most wild fungi fruit as bodies; filament programs eat the reserve
    if (g.key === 'fruitPhase') v = rng() < 0.65 ? phaseValue(STOP) : phaseValue(1 + Math.floor(rng() * 3))
    // and few live on other fungi
    if (g.key === 'parasite') v = rng() < 0.15 ? 0.5 + 0.5 * rng() : 0.1 * rng()
    if (HABITS.has(g.key)) v = rng() < 0.35 ? 0.3 + 0.7 * rng() : 0
    a[i] = clampByte(v * 255)
    // both nuclei close, so the wild type breeds roughly true
    b[i] = clampByte((v + gauss(rng) * 0.08) * 255)
  }
  return { a, b }
}
