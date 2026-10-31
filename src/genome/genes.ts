// Gene table. A fungus is a growth program: a few global genes plus up to
// PHASES phases, each a set of rules for its growing tips. Phase 0 is the
// vegetative mycelium in the substrate; fruiting switches tips into
// `fruitPhase`, and each phase names the phase that follows it and the phase
// of tips it spawns. Every fungus carries two alleles per gene (a dikaryon).

export type Dominance = 'add' | 'max' | 'min' | 'pick'

export interface GeneDef {
  key: string
  label: string
  group: string
  dom: Dominance
  def: number
}

export const PHASES = 6
export const STOP = PHASES // "next phase" value meaning: stop growing

// Per-phase rule genes (0..1 in the genome; ranges applied in expression).
export const PHASE_GENES: [key: string, label: string, def: number][] = [
  ['speed', 'Growth rate', 0.5],
  ['up', 'Down ↔ up', 0.5],
  ['persist', 'Persistence', 0.5],
  ['wander', 'Wander', 0.3],
  ['crowd', 'Avoid ↔ bundle', 0.5],
  ['signal', 'Flee ↔ follow signal', 0.5],
  ['emit', 'Emit signal', 0.0],
  ['nutrient', 'Follow food', 0.5],
  ['light', 'Shun ↔ seek light', 0.5],
  ['surface', 'Hug ↔ leave surfaces', 0.5],
  ['twist', 'Twist', 0.5],
  ['branch', 'Branching', 0.1],
  ['angle', 'Branch angle', 0.4],
  ['length', 'Length', 0.3],
  ['next', 'Next phase', 1.0],
  ['spawn', 'Spawning', 0.0],
  ['spawnPhase', 'Spawn phase', 1.0],
  ['stiff', 'Stiffness', 0.5],
  ['radius', 'Thickness', 0.4],
  ['pigment', 'Pigment', 0.5],
  ['lifespan', 'Lifespan', 1.0],
  ['glow', 'Glow', 0.0],
  // two rules a phase: when its tips sense something, they do something
  // (RULE_INPUTS, RULE_OUTPUTS; sim.wgsl, rules)
  ['r1In', 'Rule 1 senses', 0.0],
  ['r1Out', 'Rule 1 does', 0.0],
  ['r1Gain', 'Rule 1 strength', 0.5],
  ['r2In', 'Rule 2 senses', 0.0],
  ['r2Out', 'Rule 2 does', 0.0],
  ['r2Gain', 'Rule 2 strength', 0.5],
]

export const RULE_INPUTS = ['always', 'firing', 'night', 'signal', 'food', 'damp', 'light', 'crowding', 'height', 'progress'] as const
export const RULE_OUTPUTS = ['nothing', 'rise', 'seek light', 'follow signal', 'twist', 'branch', 'thicken', 'fire', 'move on', 'stop'] as const
export const ruleIndex = (v: number, n: number) => Math.min(n - 1, Math.floor(v * n))
export const ruleValue = (k: number, n: number) => (k + 0.5) / n

// Fruit bodies: an organ with a stem and a head whose shape is set by these
// genes and by its age. The head is a curved sheet that can carry gills,
// teeth and rays. It starts as a closed button and opens as its lower layer
// outgrows its upper one, into a dome, a bell, a flat cap, a funnel or a cup
// ('open', 'openAt', 'upturn'; sim.wgsl, growFruits). Its surface develops
// as it expands: where it grows more than its shape has room for it wrinkles,
// and the Cap genes from 'excess' on set how (sim.wgsl, develop).
export const DECAY_MODES = ['wither', 'deliquesce', 'burst', 'persist'] as const
export const decayValue = (k: number) => (k + 0.5) / DECAY_MODES.length
export const decayIndex = (v: number) => Math.min(DECAY_MODES.length - 1, Math.floor(v * DECAY_MODES.length))

const g = (key: string, label: string, group: string, def: number, dom: Dominance = 'add'): GeneDef => ({ key, label, group, dom, def })

const BODY_GENES: GeneDef[] = [
  g('body', 'Filaments ↔ fruit body', 'Form', 1),
  g('cluster', 'Cluster', 'Form', 0.2),
  g('spread', 'Cluster spread', 'Form', 0.3),
  g('size', 'Size', 'Form', 0.4),
  g('grow', 'Growth time', 'Form', 0.4),
  g('life', 'Lifespan', 'Form', 0.4),
  g('decay', 'Decay', 'Form', decayValue(0), 'pick'),
  g('lateral', 'Central ↔ lateral', 'Form', 0),
  g('upright', 'Upright', 'Form', 0.8),
  g('photo', 'Shun ↔ seek light', 'Form', 0.6),
  g('tipHeads', 'Heads on tips', 'Form', 0),
  g('stemLen', 'Length', 'Stem', 0.4),
  g('stemThick', 'Thickness', 'Stem', 0.3),
  g('taper', 'Taper ↔ club', 'Stem', 0.4),
  g('bulb', 'Bulb', 'Stem', 0.1),
  g('volva', 'Volva', 'Stem', 0),
  g('ring', 'Ring', 'Stem', 0),
  g('skirt', 'Skirt', 'Stem', 0),
  g('wobble', 'Wobble', 'Stem', 0.2),
  g('open', 'Opening', 'Cap', 0.23),
  g('openAt', 'Opens from crown ↔ margin', 'Cap', 0.5),
  g('upturn', 'Upturns with age', 'Cap', 0),
  g('elong', 'Flat ↔ tall', 'Cap', 0.35),
  g('cone', 'Conical', 'Cap', 0.1),
  g('thick', 'Flesh', 'Cap', 0.35),
  g('umbo', 'Umbo', 'Cap', 0.1),
  g('curl', 'Margin out ↔ in', 'Cap', 0.5),
  g('rays', 'Rays', 'Cap', 0),
  g('rayDepth', 'Ray depth', 'Cap', 0.5),
  g('rayCurl', 'Ray curl', 'Cap', 0.5),
  g('excess', 'Excess growth', 'Cap', 0),
  g('edge', 'Margin growth', 'Cap', 0),
  g('texture', 'Bumps ↔ ridges ↔ pits', 'Cap', 0.5),
  g('grain', 'Radial ↔ concentric', 'Cap', 0.5),
  g('scale', 'Fine ↔ coarse', 'Cap', 0.5),
  g('bend', 'Relief ↔ folding', 'Cap', 0.5),
  g('depth', 'Relief depth', 'Cap', 0.4),
  g('tear', 'Tearing', 'Cap', 0),
  g('gills', 'Gills', 'Underside', 0.6),
  g('gillDepth', 'Gill depth', 'Underside', 0.4),
  g('teeth', 'Teeth', 'Underside', 0),
  g('gleba', 'Spore mass', 'Underside', 0),
  g('eggs', 'Eggs', 'Underside', 0),
  g('sporeOut', 'Spore output', 'Underside', 0.5),
]

const GLOBAL_GENES: GeneDef[] = [
  { key: 'substrate', label: 'Soil ↔ wood', group: 'Life', dom: 'add', def: 0.3 },
  { key: 'absorb', label: 'Absorption', group: 'Life', dom: 'add', def: 0.5 },
  { key: 'fruitAt', label: 'Store before fruiting', group: 'Life', dom: 'add', def: 0.4 },
  { key: 'fruitProb', label: 'Fruiting urge', group: 'Life', dom: 'add', def: 0.3 },
  { key: 'fruitWet', label: 'Fruits in the dry', group: 'Life', dom: 'add', def: 0.44 },
  { key: 'fruitPhase', label: 'Fruit program', group: 'Life', dom: 'pick', def: 1 / 7 },
  { key: 'combat', label: 'Combat', group: 'Life', dom: 'max', def: 0.5 },
  // feeding on other fungi: their hyphae, and the fruit bodies it infects
  { key: 'parasite', label: 'Parasitism', group: 'Life', dom: 'add', def: 0 },
  ...BODY_GENES,
  { key: 'flesh', label: 'Flesh tone', group: 'Color', dom: 'add', def: 0.3 },
  { key: 'mel', label: 'Melanin', group: 'Color', dom: 'add', def: 0.4 },
  { key: 'car', label: 'Carotenoid', group: 'Color', dom: 'max', def: 0.3 },
  { key: 'red', label: 'Red pigment', group: 'Color', dom: 'max', def: 0.2 },
  { key: 'blue', label: 'Blue pigment', group: 'Color', dom: 'min', def: 0.1 },
  { key: 'spore', label: 'Spore melanin', group: 'Color', dom: 'add', def: 0.5 },
  g('stemPig', 'Stem pigment', 'Color', 0.2),
  g('zones', 'Zones', 'Color', 0),
  g('streak', 'Fibrils', 'Color', 0.3),
  g('bodyGlow', 'Glow', 'Color', 0),
  g('glowHue', 'Glow colour', 'Color', 0.5),
  g('jelly', 'Jelly', 'Color', 0),
  // tissue that fires and rests like nerve or heart, the firing spreading
  // through it as waves of light (sim.wgsl, excite)
  g('excite', 'Excitable', 'Behaviour', 0),
  g('pace', 'Fires unprompted', 'Behaviour', 0),
  g('rest', 'Rest after firing', 'Behaviour', 0.4),
  g('flashLen', 'Flash length', 'Behaviour', 0.4),
  g('pulseGlow', 'Flash brightness', 'Behaviour', 0.5),
  g('duskFire', 'Fires at dusk', 'Behaviour', 0),
  // fruit bodies that keep the night (sim.wgsl, growFruits)
  g('nightOpen', 'Opens at night', 'Behaviour', 0),
  g('nightSpore', 'Sheds spores at night', 'Behaviour', 0),
  g('blink', 'Blinks', 'Behaviour', 0),
  g('sync', 'Blinks in step', 'Behaviour', 0),
]

const phaseGroup = (p: number) => (p === 0 ? 'Mycelium' : `Phase ${p}`)

export const GENES: GeneDef[] = [
  ...GLOBAL_GENES,
  ...Array.from({ length: PHASES }, (_, p) =>
    PHASE_GENES.map(([key, label, def]) => ({
      key: `p${p}.${key}`,
      label,
      group: phaseGroup(p),
      dom: (key === 'next' || key === 'spawnPhase' || key.endsWith('In') || key.endsWith('Out') ? 'pick' : 'add') as Dominance,
      def,
    })),
  ).flat(),
]

export const GROUP_ORDER = ['Life', 'Form', 'Stem', 'Cap', 'Underside', 'Color', 'Behaviour', ...Array.from({ length: PHASES }, (_, p) => phaseGroup(p))]
export const GENE_INDEX: Record<string, number> = Object.fromEntries(GENES.map((d, i) => [d.key, i]))
export const NUM_GENES = GENES.length

export interface Genome {
  a: Uint8Array
  b: Uint8Array
}

export type GeneValues = Record<string, number>

export function expressGene(genome: Genome, i: number): number {
  const a = genome.a[i] / 255
  const b = genome.b[i] / 255
  switch (GENES[i].dom) {
    case 'max':
      return Math.max(a, b)
    case 'min':
      return Math.min(a, b)
    case 'pick':
      // discrete choices: the first nucleus is dominant
      return a
    default:
      return (a + b) * 0.5
  }
}

export function geneValues(genome: Genome): GeneValues {
  const out: GeneValues = {}
  for (let i = 0; i < NUM_GENES; i++) out[GENES[i].key] = expressGene(genome, i)
  return out
}

export function homozygous(values: Partial<GeneValues>): Genome {
  const a = new Uint8Array(NUM_GENES)
  for (let i = 0; i < NUM_GENES; i++) {
    const v = values[GENES[i].key] ?? GENES[i].def
    a[i] = Math.round(Math.min(1, Math.max(0, v)) * 255)
  }
  return { a, b: a.slice() }
}

export function setGene(genome: Genome, i: number, v: number): Genome {
  const a = genome.a.slice()
  const b = genome.b.slice()
  a[i] = b[i] = Math.round(Math.min(1, Math.max(0, v)) * 255)
  return { a, b }
}

// Phase indices are stored as k / 7 so that both allele values map cleanly.
export const phaseValue = (k: number) => (k + 0.5) / (PHASES + 1)
export const phaseIndex = (v: number) => Math.min(PHASES, Math.floor(v * (PHASES + 1)))
