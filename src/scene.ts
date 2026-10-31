import { geneValues, homozygous, type Genome } from './genome/genes'
import { colors, express, PHENO_STRIDE } from './genome/phenotype'
import { PRESETS, SLIME_SPORANGIA } from './genome/presets'
import { MAX_ORGS } from './gpu/renderer'

export interface Organism {
  id: number
  name: string
  genome: Genome
  slot: number // index into the GPU program table
  // where: a point over the ground (it starts on the ground under it), or
  // `exact`ly where (it grew from a spore)
  x: number
  y: number
  z: number
  onSurface: boolean // on the first thing under it (wood) rather than the soil
  exact?: boolean
}

let nextId = 1
export const newId = () => nextId++
// (the next id, to keep and go back to: history.ts)
export const ids = { get: () => nextId, set: (v: number) => (nextId = v) }

export function fromPreset(i: number): Omit<Organism, 'slot' | 'x' | 'y' | 'z' | 'onSurface'> {
  const p = PRESETS[i]
  return { id: newId(), name: p.name, genome: homozygous(p.genes) }
}

export const prefersWood = (g: Genome) => geneValues(g).substrate > 0.6

// The last program slot belongs to the slime mould's sporangia (slime.wgsl).
export const SLIME_ORG = MAX_ORGS - 1
const SLIME_PROGRAM = homozygous(SLIME_SPORANGIA)

export function fillPrograms(list: Organism[], rotting: Organism[] = []): Float32Array {
  const out = new Float32Array(MAX_ORGS * PHENO_STRIDE)
  for (const o of list) express(o.genome, out, o.slot * PHENO_STRIDE)
  for (const o of rotting) express(o.genome, out, o.slot * PHENO_STRIDE, 0.25)
  express(SLIME_PROGRAM, out, SLIME_ORG * PHENO_STRIDE)
  return out
}

export function swatch(g: Genome): string {
  const c = colors(geneValues(g)).pig
  const srgb = c.map((x) => Math.round(255 * (x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055)))
  return `rgb(${srgb.join(',')})`
}
