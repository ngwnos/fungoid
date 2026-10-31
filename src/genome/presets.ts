import { DECAY_MODES, RULE_INPUTS, RULE_OUTPUTS, decayValue, phaseValue, ruleValue, STOP, type GeneValues } from './genes'

// Starting strains, written in natural units and converted to gene values.
// Filament programs grow shapes out of rules; fruit bodies are organs whose
// form comes from the body genes and their age.

interface PhaseSpec {
  speed?: number // extension probability per step
  up?: number // -2..2
  persist?: number // 0..4
  wander?: number // 0..1.5
  crowd?: number // -3 (spread) .. 3 (bundle)
  signal?: number // -3..3
  emit?: number // 0..1
  nutrient?: number // -2..2
  light?: number // -2..2
  surface?: number // -2 (hug) .. 2 (leave)
  twist?: number // -1.5..1.5
  branch?: number // 0..0.5 per extension
  angle?: number // radians
  length?: number // cells
  next?: number // phase index or STOP
  spawn?: number // 0..0.7 per extension
  spawnPhase?: number
  stiff?: number // 0..1
  radius?: number // cm
  pigment?: number // 0 flesh .. 0.66 pigment .. 1 spore colour
  lifespan?: number // steps (Infinity = forever)
  glow?: number // 0..3
  // up to two: when its tips sense this, they do that, this strongly (-2..2)
  rules?: [(typeof RULE_INPUTS)[number], (typeof RULE_OUTPUTS)[number], number][]
}

// What it does by day and night, as gene values (0..1; genes.ts, Behaviour).
interface BehaviourSpec {
  excite?: number
  pace?: number
  rest?: number
  flashLen?: number
  pulseGlow?: number
  duskFire?: number
  nightOpen?: number
  nightSpore?: number
  blink?: number
  sync?: number
}

const sq = (x: number) => Math.sqrt(Math.max(0, x))
const signed = (x: number, r: number) => x / (2 * r) + 0.5

// Fruit body genes in natural units (R is the head radius).
interface BodySpec {
  body?: number // chance a fruiting event makes fruit bodies, not filaments
  cluster?: number // fruit bodies per flush
  spread?: number // cm
  size?: number // head radius, cm
  grow?: number // steps to mature
  life?: number // steps mature (Infinity: perennial)
  decay?: (typeof DECAY_MODES)[number]
  lateral?: number // 0 central stem .. 1 shelf
  upright?: number
  photo?: number // -1 shun .. 1 seek the sun
  tipHeads?: number // chance a filament that stops grows a head
  stemLen?: number // × R
  stemThick?: number // radius × R
  taper?: number // top / base
  bulb?: number
  volva?: number
  ring?: number
  skirt?: number
  wobble?: number
  // the head starts as a closed button (its profile turning by 2.95) and its
  // lower layer outgrows the upper one as it expands: `open` is the turning
  // given up (2.95: flat, more: a funnel or cup), from the crown (0) or the
  // margin (1) first; `upturn` goes on after it is grown (per growth time)
  open?: number
  openAt?: number
  upturn?: number
  elong?: number // vertical stretch
  cone?: number
  thick?: number // × R
  umbo?: number // × R
  curl?: number // -1 flared .. 1 inrolled margin
  rays?: number // 0, or 3..12 arms
  rayDepth?: number
  rayCurl?: number // radians the arms turn
  // how the head's surface develops (all 0..1): excess growth over the head
  // and at its margin, bumps (0) ↔ ridges ↔ pits (1), radial (0) ↔
  // concentric (1) grain, fine ↔ coarse, relief (0) ↔ folding (1), depth,
  // tearing
  excess?: number
  edge?: number
  texture?: number
  grain?: number
  scale?: number
  bend?: number
  depth?: number
  tear?: number
  gills?: number // count
  gillDepth?: number // × R
  teeth?: number // × R
  gleba?: number
  eggs?: number
  stemPig?: number
  zones?: number
  streak?: number
  glow?: number // 0..3
  jelly?: number
  sporeOut?: number // 0..2
}

const id = (x: number) => x
const BODY_GENE: Record<string, [string, (x: number) => number]> = {
  body: ['body', id],
  cluster: ['cluster', (x) => sq((x - 1) / 9)],
  spread: ['spread', (x) => (x - 0.1) / 2.4],
  size: ['size', (x) => Math.pow(Math.max(0, (x - 0.12) / 2.6), 1 / 1.6)],
  grow: ['grow', (x) => sq((x - 150) / 3000)],
  life: ['life', (x) => (Number.isFinite(x) ? Math.min(0.98, sq((x - 200) / 6000)) : 1)],
  lateral: ['lateral', id],
  upright: ['upright', id],
  photo: ['photo', (x) => x / 2 + 0.5],
  tipHeads: ['tipHeads', sq],
  stemLen: ['stemLen', (x) => sq(x / 8)],
  stemThick: ['stemThick', (x) => sq((x - 0.04) / 0.56)],
  taper: ['taper', (x) => (x - 0.4) / 1.2],
  bulb: ['bulb', (x) => x / 1.2],
  volva: ['volva', id],
  ring: ['ring', id],
  skirt: ['skirt', id],
  wobble: ['wobble', id],
  open: ['open', (x) => x / 5.35],
  openAt: ['openAt', id],
  upturn: ['upturn', (x) => x / 1.5],
  elong: ['elong', (x) => (x - 0.5) / 2],
  cone: ['cone', (x) => x / 1.5],
  thick: ['thick', (x) => (x - 0.04) / 0.36],
  umbo: ['umbo', (x) => x / 0.45],
  curl: ['curl', (x) => x / 2 + 0.5],
  rays: ['rays', (x) => (x < 3 ? 0 : 0.16 + ((x - 3) / 9) * 0.84)],
  rayDepth: ['rayDepth', id],
  rayCurl: ['rayCurl', (x) => x / 6 + 0.5],
  excess: ['excess', id],
  edge: ['edge', id],
  texture: ['texture', id],
  grain: ['grain', id],
  scale: ['scale', id],
  bend: ['bend', id],
  depth: ['depth', id],
  tear: ['tear', id],
  gills: ['gills', (x) => sq(x / 90)],
  gillDepth: ['gillDepth', (x) => (x - 0.05) / 0.5],
  teeth: ['teeth', (x) => x / 1.2],
  gleba: ['gleba', id],
  eggs: ['eggs', (x) => (x <= 1 ? 0 : 0.1 + ((x - 2) / 6) * 0.9)],
  stemPig: ['stemPig', id],
  zones: ['zones', id],
  streak: ['streak', id],
  glow: ['bodyGlow', (x) => sq(x / 3)],
  jelly: ['jelly', id],
  sporeOut: ['sporeOut', (x) => sq(x / 2)],
}

function bodyGenes(b: BodySpec): GeneValues {
  const g: GeneValues = {}
  for (const [k, v] of Object.entries(b)) {
    if (k === 'decay') {
      g.decay = decayValue(DECAY_MODES.indexOf(v as (typeof DECAY_MODES)[number]))
      continue
    }
    const [key, f] = BODY_GENE[k]
    g[key] = Math.min(1, Math.max(0, f(v as number)))
  }
  return g
}

interface StrainSpec {
  substrate?: number
  absorb?: number
  fruitAt?: number
  fruitProb?: number
  fruitWet?: number // how wet the ground must be for it to fruit: 0.15 (summer too) .. 0.6 (only the wettest)
  fruitPhase?: number
  flesh?: number
  mel?: number
  car?: number
  red?: number
  blue?: number
  spore?: number
  combat?: number
  parasite?: number // 0 feeds on the ground .. 1 only on other fungi
  glowHue?: number // 0 violet, 0.25 blue, 0.5 green, 0.75 yellow, 1 amber
  behaviour?: BehaviourSpec
  phases: PhaseSpec[]
  body?: BodySpec
}


function phaseGenes(p: number, s: PhaseSpec): GeneValues {
  const g: GeneValues = {}
  const set = (k: string, v: number | undefined, f: (x: number) => number) => {
    if (v !== undefined) g[`p${p}.${k}`] = Math.min(1, Math.max(0, f(v)))
  }
  set('speed', s.speed, (x) => sq((x - 0.02) / 0.98))
  set('up', s.up, (x) => signed(x, 2))
  set('persist', s.persist, (x) => x / 4)
  set('wander', s.wander, (x) => x / 1.5)
  set('crowd', s.crowd, (x) => signed(x, 3))
  set('signal', s.signal, (x) => signed(x, 3))
  set('emit', s.emit, (x) => x)
  set('nutrient', s.nutrient, (x) => signed(x, 2))
  set('light', s.light, (x) => signed(x, 2))
  set('surface', s.surface, (x) => signed(x, 2))
  set('twist', s.twist, (x) => signed(x, 1.5))
  set('branch', s.branch, (x) => sq(x / 0.5))
  set('angle', s.angle, (x) => (x - 0.15) / 1.3)
  set('length', s.length, (x) => sq((x - 1) / 150))
  set('next', s.next ?? STOP, phaseValue)
  set('spawn', s.spawn, (x) => x / 0.7)
  set('spawnPhase', s.spawnPhase ?? STOP, phaseValue)
  set('stiff', s.stiff, (x) => x)
  set('radius', s.radius, (x) => (x - 0.012) / 0.06)
  set('pigment', s.pigment, (x) => x)
  set('lifespan', s.lifespan ?? Infinity, (x) => (Number.isFinite(x) ? sq((x - 20) / 3000) : 1))
  set('glow', s.glow, (x) => sq(x / 3))
  s.rules?.forEach(([sense, act, gain], r) => {
    g[`p${p}.r${r + 1}In`] = ruleValue(RULE_INPUTS.indexOf(sense), RULE_INPUTS.length)
    g[`p${p}.r${r + 1}Out`] = ruleValue(RULE_OUTPUTS.indexOf(act), RULE_OUTPUTS.length)
    g[`p${p}.r${r + 1}Gain`] = Math.min(1, Math.max(0, gain / 4 + 0.5))
  })
  return g
}

function strain(s: StrainSpec): GeneValues {
  const g: GeneValues = {
    substrate: s.substrate ?? 0.2,
    absorb: s.absorb ?? 0.5,
    fruitAt: sq(((s.fruitAt ?? 120) - 20) / 600),
    fruitProb: sq((s.fruitProb ?? 0.01) / 0.04),
    fruitWet: (0.6 - (s.fruitWet ?? 0.4)) / 0.45,
    fruitPhase: phaseValue(s.fruitPhase ?? 1),
    flesh: s.flesh ?? 0.3,
    mel: s.mel ?? 0.3,
    car: s.car ?? 0.2,
    red: s.red ?? 0.2,
    blue: s.blue ?? 0.1,
    spore: s.spore ?? 0.5,
    combat: s.combat ?? 0.5,
    parasite: s.parasite ?? 0,
    ...(s.glowHue !== undefined ? { glowHue: s.glowHue } : {}),
    ...s.behaviour,
  }
  s.phases.forEach((ph, p) => Object.assign(g, phaseGenes(p, ph)))
  Object.assign(g, bodyGenes(s.body ?? { body: 0 }))
  return g
}

// A strain that fruits only as fruit bodies, from soil mycelium.
const fruiting = (colour: Omit<StrainSpec, 'phases' | 'body'>, body: BodySpec, mycelium = soilMycelium): StrainSpec => ({
  fruitPhase: STOP,
  ...colour,
  phases: [mycelium],
  body: { body: 1, ...body },
})

// Mycelium that spreads through soil just under the surface.
const soilMycelium: PhaseSpec = {
  speed: 0.5, up: 0, persist: 2, wander: 0.6, crowd: -1.2, nutrient: 1.4, surface: 0.6,
  branch: 0.05, angle: 0.8, length: 150, next: 0, stiff: 0.2, radius: 0.016, pigment: 0,
}

const woodMycelium: PhaseSpec = { ...soilMycelium, surface: 0.3, nutrient: 1.8 }

// Mycelium that hunts: quicker and wandering, hardly drawn by food in the
// ground (its hyphae turn toward other colonies; sim.wgsl, grow).
const parasiteMycelium: PhaseSpec = { ...soilMycelium, speed: 0.65, wander: 0.9, nutrient: 0.3 }

export const PRESETS: { name: string; genes: GeneValues }[] = [
  {
    name: 'Fly agaric',
    genes: strain(fruiting(
      { fruitWet: 0.35, red: 0.95, car: 0.7, mel: 0.05, flesh: 0.05, spore: 0.05 },
      { cluster: 2, spread: 2, size: 1.1, grow: 900, life: 2000, stemLen: 2.4, stemThick: 0.17, taper: 0.85, bulb: 0.9,
        volva: 0.5, ring: 0.6, skirt: 0.12, open: 1.75, openAt: 0.35, upturn: 0.15, elong: 0.75, thick: 0.18,
        excess: 0.7, texture: 0, scale: 0.12, bend: 0, depth: 0.55,
        gills: 80, gillDepth: 0.2, streak: 0.1, stemPig: 0 },
    )),
  },
  {
    name: 'Parasol',
    genes: strain(fruiting(
      { fruitWet: 0.3, mel: 0.55, car: 0.3, red: 0.1, flesh: 0.2, spore: 0.1 },
      { size: 1.0, grow: 1200, life: 2500, stemLen: 3.5, stemThick: 0.1, taper: 0.75, bulb: 0.8, ring: 0.5,
        open: 2.35, upturn: 0.1, elong: 0.7, thick: 0.1, umbo: 0.18, excess: 0.8, texture: 0.05, grain: 0.75, scale: 0.15,
        bend: 0, depth: 0.45, gills: 70, gillDepth: 0.15, streak: 0.6,
        stemPig: 0.5, photo: 0.3, wobble: 0.1 },
    )),
  },
  {
    name: 'Shaggy inkcap',
    genes: strain(fruiting(
      { fruitWet: 0.55, flesh: 0.02, mel: 0.12, car: 0.05, red: 0, spore: 1 },
      { cluster: 3, spread: 0.9, size: 0.6, grow: 700, life: 600, decay: 'deliquesce', stemLen: 3.5, stemThick: 0.18,
        taper: 0.85, bulb: 0.3, ring: 0.3, open: 0.09, openAt: 0.8, elong: 2.4, cone: 0.1, thick: 0.1,
        excess: 0.6, texture: 0.15, grain: 0.8, scale: 0.35, bend: 0, depth: 0.6,
        gills: 60, gillDepth: 0.35, streak: 0.4 },
    )),
  },
  {
    name: 'Bonnets',
    genes: strain(fruiting(
      { mel: 0.5, car: 0.15, red: 0, flesh: 0.3, spore: 0.2 },
      { cluster: 8, spread: 0.9, size: 0.28, grow: 500, life: 1500, stemLen: 8, stemThick: 0.09, upright: 0.9,
        photo: 0.6, wobble: 0.6, open: 0.67, openAt: 0.75, elong: 1, cone: 0.5, thick: 0.08, gills: 20,
        gillDepth: 0.25, streak: 0.8, glow: 1.5 },
    )),
  },
  {
    name: 'Puffball',
    genes: strain(fruiting(
      { fruitWet: 0.22, flesh: 0.45, mel: 0.15, car: 0.15, red: 0, spore: 0.6 },
      { cluster: 5, spread: 1.4, size: 0.75, grow: 800, life: 1500, decay: 'burst', stemLen: 0.25, stemThick: 0.5,
        taper: 0.7, open: 0, elong: 0.95, thick: 0.08, excess: 0.65, texture: 0, scale: 0.05, bend: 0, depth: 0.15,
        gills: 0, gleba: 0.8, sporeOut: 2 },
    )),
  },
  {
    name: 'Basket stinkhorn',
    genes: strain(fruiting(
      { fruitWet: 0.25, red: 1, car: 0.55, mel: 0.05, flesh: 0.1, spore: 0.75 },
      { size: 1.3, grow: 600, life: 900, stemLen: 0, stemThick: 0.3, open: 0, elong: 1.05, thick: 0.14,
        excess: 0.8, texture: 1, scale: 0.62, bend: 0.3, depth: 0.3, tear: 0.8, gleba: 0.65, volva: 0.6, gills: 0 },
    )),
  },
  {
    name: 'Earthstar',
    genes: strain(fruiting(
      { fruitWet: 0.3, mel: 0.55, car: 0.35, red: 0.1, flesh: 0.4, spore: 0.7 },
      { cluster: 2, spread: 1.5, size: 1.2, grow: 900, life: 3000, stemLen: 0, open: 1.82, elong: 0.7, thick: 0.1,
        rays: 7, rayDepth: 0.8, rayCurl: 2.4, gleba: 0.6, gills: 0 },
    )),
  },
  {
    name: 'Morel',
    genes: strain(fruiting(
      { fruitWet: 0.5, mel: 0.65, car: 0.35, red: 0.1, flesh: 0.35, spore: 0.4 },
      { cluster: 2, spread: 1.2, size: 0.75, grow: 1000, life: 2000, stemLen: 1.4, stemThick: 0.55, taper: 0.95,
        bulb: 0.3, open: 0, elong: 1.8, cone: 0.35, thick: 0.12, excess: 0.85, texture: 0.85, scale: 0,
        bend: 0.7, depth: 0.85, gills: 0, stemPig: 0.05 },
    )),
  },
  {
    name: 'Bird’s nest',
    genes: strain(fruiting(
      { fruitWet: 0.45, mel: 0.55, car: 0.4, red: 0.05, flesh: 0.45, spore: 0.25 },
      { cluster: 9, spread: 1.4, size: 0.35, grow: 900, life: 4000, stemLen: 0, stemThick: 0.3, open: 4.85,
        elong: 1.2, thick: 0.08, gleba: 0.6, eggs: 5, gills: 0, streak: 0.5 },
    )),
  },
  {
    name: 'Anemone',
    genes: strain(fruiting(
      { fruitWet: 0.3, red: 0.95, car: 0.4, mel: 0.05, flesh: 0.15, spore: 0.8 },
      { size: 1.1, grow: 600, life: 900, stemLen: 1.8, stemThick: 0.35, taper: 1.1, volva: 0.6, open: 2.85,
        elong: 0.6, thick: 0.1, rays: 8, rayDepth: 0.7, rayCurl: 0.6, gleba: 0.35, gills: 0, stemPig: 0.25 },
    )),
  },
  {
    name: 'Veiled lady',
    genes: strain(fruiting(
      { fruitWet: 0.2, flesh: 0.02, spore: 0.85, mel: 0.25, car: 0.15, red: 0 },
      { size: 0.7, grow: 400, life: 700, stemLen: 5.5, stemThick: 0.42, taper: 0.85, volva: 0.65, ring: 1, skirt: 0.9,
        open: 0.16, elong: 1.25, cone: 0.45, thick: 0.12, excess: 0.75, texture: 0.8, scale: 0.05, bend: 0.6, depth: 0.7,
        gleba: 0.45, gills: 0, stemPig: 0 },
    )),
  },
  {
    name: 'Chanterelle',
    genes: strain(fruiting(
      { fruitWet: 0.2, car: 0.9, red: 0.15, mel: 0.08, flesh: 0.6, spore: 0.15 },
      { cluster: 3, spread: 1.6, size: 1, grow: 1200, life: 3000, stemLen: 1.6, stemThick: 0.3, taper: 1.6,
        open: 3.65, openAt: 0.25, elong: 0.8, thick: 0.14, edge: 0.75, scale: 0.92, bend: 1, depth: 0.6, curl: -0.3, gills: 30,
        gillDepth: 0.12, stemPig: 0.9, streak: 0.1 },
    )),
  },
  {
    name: 'Fingers',
    genes: strain(fruiting(
      { mel: 1, spore: 1, flesh: 0.2, car: 0.1, red: 0 },
      { cluster: 6, spread: 1.2, size: 0.16, grow: 1200, life: 4000, stemLen: 8, stemThick: 0.6, taper: 1.6, wobble: 0.7,
        photo: -0.3, open: 0, elong: 1.6, thick: 0.3, gills: 0, stemPig: 1 },
    )),
  },
  {
    name: 'Turkey tail',
    genes: strain(fruiting(
      { fruitWet: 0.3, substrate: 0.95, fruitAt: 160, mel: 0.3, blue: 0.85, red: 0.3, car: 0.25, flesh: 0.5, spore: 0.1, combat: 0.85 },
      { cluster: 6, spread: 1.3, size: 0.8, grow: 1500, life: Infinity, decay: 'persist', lateral: 1, stemLen: 0,
        stemThick: 0.1, open: 1.85, elong: 0.45, thick: 0.07, edge: 0.65, scale: 0.95, bend: 1, depth: 0.5, zones: 0.95, gills: 0,
        streak: 0.2 },
      woodMycelium,
    )),
  },
  {
    name: 'Oyster',
    genes: strain(fruiting(
      { fruitWet: 0.5, substrate: 0.95, mel: 0.4, car: 0.1, red: 0, blue: 0.75, flesh: 0.25, spore: 0.15 },
      { cluster: 6, spread: 1, size: 1.2, grow: 1000, life: 2500, lateral: 0.85, stemLen: 0.5, stemThick: 0.25,
        open: 1.5, elong: 0.55, thick: 0.13, curl: 0.45, edge: 0.6, scale: 1, bend: 1, depth: 0.5, gills: 70, gillDepth: 0.18,
        stemPig: 0.1 },
      woodMycelium,
    )),
  },
  {
    name: 'Lion’s mane',
    genes: strain(fruiting(
      { fruitWet: 0.45, substrate: 0.95, flesh: 0, mel: 0, car: 0.05, red: 0, spore: 0.1 },
      { size: 1.1, grow: 1500, life: 3000, lateral: 0.5, stemLen: 0, open: 1.75, elong: 0.7, thick: 0.35, teeth: 1.1,
        gills: 0 },
      woodMycelium,
    )),
  },
  {
    name: 'Brain jelly',
    genes: strain(fruiting(
      { fruitWet: 0.55, substrate: 0.95, car: 0.95, red: 0.1, mel: 0, flesh: 0.6, spore: 0.2 },
      { cluster: 2, spread: 1, size: 0.9, grow: 1200, life: 3000, lateral: 0.45, stemLen: 0, open: 0.45, elong: 0.8,
        thick: 0.25, excess: 1, texture: 0.5, scale: 0.22, bend: 1, depth: 0.4, jelly: 0.85, gills: 0 },
      woodMycelium,
    )),
  },
  // (coral and spire tissue lives about as long as a mushroom, withering
  // through the last quarter of it)
  {
    name: 'Coral',
    genes: strain({
      fruitWet: 0.35, mel: 0.05, car: 0.9, red: 0.25, flesh: 0.6,
      phases: [
        soilMycelium,
        { speed: 0.8, up: 1.4, crowd: 1.5, signal: 1.5, emit: 0.6, branch: 0.2, angle: 0.3, length: 3, next: 2, stiff: 0.8, radius: 0.04, pigment: 0.3, lifespan: 2800 },
        { speed: 0.7, up: 1.6, persist: 1.5, wander: 0.3, crowd: -0.8, branch: 0.32, angle: 0.45, length: 14, next: 3, stiff: 0.85, radius: 0.045, pigment: 0.55, lifespan: 2800 },
        { speed: 0.7, up: 1.2, persist: 1.2, wander: 0.3, crowd: -1, branch: 0.22, angle: 0.5, length: 8, stiff: 0.8, radius: 0.03, pigment: 0.7, lifespan: 2800 },
      ],
    }),
  },
  {
    name: 'Spire',
    genes: strain({
      blue: 0.92, red: 0.6, mel: 0.15, car: 0.1, spore: 0.2,
      phases: [
        soilMycelium,
        { speed: 0.8, up: 1.4, crowd: 1.6, signal: 1.5, emit: 0.6, branch: 0.22, angle: 0.3, length: 4, next: 2, stiff: 0.85, radius: 0.04, pigment: 0.3, lifespan: 2800 },
        { speed: 0.8, up: 1.6, persist: 2, wander: 0.1, crowd: 2, twist: 1.4, branch: 0.04, angle: 0.3, length: 44, next: 3, stiff: 0.92, radius: 0.04, pigment: 0.6, glow: 0.4, lifespan: 2800 },
        { speed: 0.8, up: 0, persist: 1.5, crowd: -2, branch: 0.3, angle: 0.6, length: 6, stiff: 0.7, radius: 0.03, pigment: 1, glow: 1.2, lifespan: 2800 },
      ],
      // tiny glowing caps on the ends of the twisted stalks
      body: { body: 0, tipHeads: 0.6, size: 0.7, stemLen: 0.3, stemThick: 0.2, open: 1.0, thick: 0.12,
        gills: 14, gillDepth: 0.2, glow: 2 },
    }),
  },
  {
    name: 'Crawler',
    genes: strain({
      red: 0.9, car: 0.7, mel: 0.05, flesh: 0.8, fruitAt: 60, fruitProb: 0.02,
      phases: [
        soilMycelium,
        { speed: 0.7, up: 0, persist: 3, wander: 0.45, crowd: -1, surface: -2, branch: 0.08, angle: 0.7, length: 150, next: 1, stiff: 0.3, radius: 0.05, pigment: 0.7, lifespan: 500, glow: 0.2 },
      ],
    }),
  },
  {
    name: 'Cordweb',
    genes: strain({
      car: 0.45, mel: 0.1, red: 0.05, flesh: 0.7, fruitPhase: 0, absorb: 0.8,
      phases: [{ speed: 0.6, up: -0.1, persist: 2.5, wander: 0.25, crowd: 1.3, nutrient: 2, surface: 1.6, branch: 0.05, angle: 0.7, length: 150, next: 0, stiff: 0.3, radius: 0.035, pigment: 0.35 }],
    }),
  },
  // Kinds of the night: they fire, blink and open in the dark.
  {
    // a net over the moss that fires as night falls, and now and then of
    // itself: rings of blue light running out through it
    name: 'Lanternweb',
    genes: strain({
      blue: 0.85, car: 0.05, mel: 0.45, flesh: 0.5, red: 0.1, fruitWet: 0.35, fruitProb: 0.006, fruitAt: 260, absorb: 0.75, combat: 0.6, glowHue: 0.28,
      behaviour: { excite: 1, pace: 0.35, rest: 0.5, flashLen: 0.3, pulseGlow: 0.95, duskFire: 0.7 },
      phases: [
        soilMycelium,
        { speed: 0.75, up: 0, persist: 2, wander: 0.55, crowd: -0.5, surface: -2, branch: 0.12, angle: 0.9, length: 70, stiff: 0.3, radius: 0.03, pigment: 0.35, lifespan: 2600, glow: 0.04 },
      ],
    }),
  },
  {
    // a dense mat that keeps its own rhythm: waves start here and there,
    // break on each other and curl into spirals of violet light
    name: 'Whorl',
    genes: strain({
      blue: 0.6, red: 0.55, car: 0.05, mel: 0.55, flesh: 0.4, fruitWet: 0.35, fruitProb: 0.004, fruitAt: 260, absorb: 0.75, combat: 0.6, glowHue: 0.02,
      behaviour: { excite: 0.85, pace: 0.6, rest: 0.35, flashLen: 0.45, pulseGlow: 0.8 },
      phases: [
        soilMycelium,
        { speed: 0.8, up: 0, persist: 1.6, wander: 0.5, crowd: -0.6, surface: -2, branch: 0.14, angle: 1.0, length: 90, stiff: 0.3, radius: 0.02, pigment: 0.5, lifespan: 3000 },
      ],
    }),
  },
  {
    // tendrils that rise toward the open air and coil in the dark, and
    // branch where a pulse runs through them
    name: 'Nightreach',
    genes: strain({
      blue: 0.75, car: 0.25, mel: 0.4, flesh: 0.5, fruitWet: 0.35, fruitAt: 200, absorb: 0.7, combat: 0.6, glowHue: 0.4,
      behaviour: { excite: 0.9, pace: 0.25, rest: 0.4, flashLen: 0.35, pulseGlow: 0.9, duskFire: 0.4 },
      phases: [
        soilMycelium,
        { speed: 0.7, up: 1.3, persist: 2, wander: 0.25, crowd: -0.8, light: 0.6, branch: 0.03, angle: 0.5, length: 80, stiff: 0.85, radius: 0.035, pigment: 0.6, lifespan: 2800, glow: 0.05,
          rules: [['night', 'twist', 1.6], ['firing', 'branch', 2]] },
      ],
    }),
  },
  {
    // little bonnets in crowds that blink, each at its own pace at first,
    // falling into step with those around it
    name: 'Fireflies',
    genes: strain(fruiting(
      { mel: 0.25, car: 0.35, red: 0, flesh: 0.6, spore: 0.2, absorb: 0.65, glowHue: 0.62, behaviour: { blink: 0.55, sync: 0.8 } },
      { cluster: 10, spread: 1.6, size: 0.34, grow: 500, life: 2500, stemLen: 5, stemThick: 0.1, upright: 0.9,
        photo: 0.3, wobble: 0.5, open: 0.9, openAt: 0.7, elong: 0.9, cone: 0.3, thick: 0.1, gills: 16,
        gillDepth: 0.25, glow: 2.4 },
    )),
  },
  {
    // a pale funnel that opens wide at dusk, glows amber and sheds its
    // spores in the dark, and closes again by morning
    name: 'Moonlily',
    genes: strain(fruiting(
      { mel: 0.05, car: 0.2, red: 0.05, flesh: 0.1, spore: 0.35, glowHue: 0.92, behaviour: { nightOpen: 0.9, nightSpore: 0.9 } },
      { cluster: 3, spread: 2, size: 1.0, grow: 900, life: 3500, stemLen: 3.2, stemThick: 0.09, taper: 0.8,
        open: 3.6, openAt: 0.85, elong: 0.8, thick: 0.08, curl: -0.4, gills: 40, gillDepth: 0.2, glow: 1.4, upright: 1, wobble: 0.15 },
    )),
  },
  // Mycoparasites: they make no fruit bodies of their own. Their hyphae
  // feed on other colonies' and they crust over the fruit bodies they reach,
  // shedding their spores from them.
  {
    // Hypomyces lactifluorum, which turns its hosts into lobster mushrooms
    name: 'Lobster mould',
    genes: strain({
      car: 0.95, red: 0.62, mel: 0.05, flesh: 0.6, spore: 0.15, combat: 0.1, parasite: 0.95, fruitPhase: STOP,
      phases: [parasiteMycelium],
    }),
  },
  {
    // Trichoderma: half a decomposer, and it poisons the hyphae it feeds on
    name: 'Green mould',
    genes: strain({
      car: 0.75, blue: 0.87, mel: 0.12, flesh: 0.4, spore: 0.3, combat: 0.85, absorb: 0.7, parasite: 0.6, fruitPhase: STOP,
      phases: [{ ...parasiteMycelium, branch: 0.08 }],
    }),
  },
]

// The slime mould's sporangia (Physarum): tiny closed heads on thin stalks,
// greyish yellow, in crowds where the plasmodium gathered to fruit. They are
// fruit bodies grown from the slime mould, not from a colony (slimesim.wgsl).
export const SLIME_SPORANGIA: GeneValues = strain({
  flesh: 0.45, mel: 0.3, car: 0.55, red: 0.08, blue: 0.05, spore: 0.7, fruitPhase: STOP,
  phases: [soilMycelium],
  body: { body: 1, size: 0.14, grow: 400, life: 3000, stemLen: 3, stemThick: 0.18, taper: 0.7, upright: 1, wobble: 0.3,
    open: 0, elong: 1.1, thick: 0.3, gills: 0, stemPig: 0.15 },
})

const presetIndex = (n: string) => PRESETS.findIndex((p) => p.name === n)

// The garden a new terrarium starts with.
const STARTER_NAMES = [
  'Fly agaric', 'Parasol', 'Shaggy inkcap', 'Bonnets', 'Puffball', 'Basket stinkhorn', 'Earthstar', 'Morel',
  'Bird’s nest', 'Anemone', 'Veiled lady', 'Chanterelle', 'Fingers', 'Turkey tail', 'Oyster', 'Lion’s mane',
  'Brain jelly', 'Coral', 'Spire',
  'Lanternweb', 'Whorl', 'Nightreach', 'Fireflies', 'Moonlily',
]
export const STARTERS = STARTER_NAMES.map(presetIndex)
// What blows in from outside later: the same kinds, and mycoparasites.
export const ARRIVALS = [...STARTER_NAMES, 'Lobster mould', 'Green mould'].map(presetIndex)
