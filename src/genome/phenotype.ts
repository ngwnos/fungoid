import { PHASES, PHASE_GENES, RULE_INPUTS, RULE_OUTPUTS, decayIndex, geneValues, phaseIndex, ruleIndex, type Genome } from './genes'

// Expressed program, laid out for the GPU: a global block, then one block per
// phase. WGSL constants are generated from these lists.
export const GLOBAL_FIELDS = [
  'SUBSTRATE', 'ABSORB', 'FRUIT_AT', 'FRUIT_PROB', 'FRUIT_PHASE', 'ALIVE', 'PARASITE', 'FRUIT_WET',
  'FLESH_R', 'FLESH_G', 'FLESH_B', 'PIG_R', 'PIG_G', 'PIG_B', 'SPORE_R', 'SPORE_G', 'SPORE_B', 'GLOW_R', 'GLOW_G', 'GLOW_B',
  // fruit bodies
  'BODY', 'CLUSTER', 'SPREAD', 'SIZE', 'GROW', 'LIFE', 'DECAY', 'LATERAL', 'UPRIGHT', 'PHOTO', 'TIP_HEADS',
  'STEM_LEN', 'STEM_R', 'TAPER', 'BULB', 'VOLVA', 'RING', 'SKIRT', 'WOBBLE',
  'OPEN', 'OPEN_AT', 'UPTURN', 'ELONG', 'CONE', 'THICK', 'UMBO', 'CURL',
  'RAYS', 'RAY_DEPTH', 'RAY_CURL', 'EXCESS', 'EDGE', 'TEXTURE', 'GRAIN', 'SCALE', 'BEND', 'DEPTH', 'TEAR',
  'GILLS', 'GILL_DEPTH', 'TEETH', 'GLEBA', 'EGGS',
  'STEM_PIG', 'ZONES', 'STREAK', 'BODY_GLOW', 'JELLY',
  'COMBAT', 'SPORE_OUT',
  // behaviour
  'EXCITE', 'PACE', 'REST', 'FLASH', 'PULSE_GLOW', 'DUSK', 'NIGHT_OPEN', 'NIGHT_SPORE', 'BLINK', 'SYNC',
] as const
export const PHASE_FIELDS = PHASE_GENES.map(([k]) => k.replace(/[A-Z]/g, (c) => '_' + c).toUpperCase())

export const PHASE_BASE = Math.ceil(GLOBAL_FIELDS.length / 4) * 4
export const PHASE_STRIDE = Math.ceil(PHASE_FIELDS.length / 4) * 4
export const PHENO_STRIDE = PHASE_BASE + PHASES * PHASE_STRIDE

// whether a program glows at all (its fruit bodies, or any phase's tissue)
export function glows(data: Float32Array, offset: number) {
  if (data[offset + GLOBAL_FIELDS.indexOf('BODY_GLOW')] > 0) return true
  const q = PHASE_FIELDS.indexOf('GLOW')
  for (let p = 0; p < PHASES; p++) if (data[offset + PHASE_BASE + p * PHASE_STRIDE + q] > 0) return true
  return false
}

const G: Record<string, number> = Object.fromEntries(GLOBAL_FIELDS.map((k, i) => [k, i]))
const Q: Record<string, number> = Object.fromEntries(PHASE_FIELDS.map((k, i) => [k, i]))

export function phenoWgsl(): string {
  const lines = [
    ...GLOBAL_FIELDS.map((k, i) => `const G_${k}: u32 = ${i}u;`),
    ...PHASE_FIELDS.map((k, i) => `const Q_${k}: u32 = ${i}u;`),
    `const PHASES: u32 = ${PHASES}u;`,
    `const PHASE_BASE: u32 = ${PHASE_BASE}u;`,
    `const PHASE_STRIDE: u32 = ${PHASE_STRIDE}u;`,
    `const PH_STRIDE: u32 = ${PHENO_STRIDE}u;`,
  ]
  return lines.join('\n')
}

const sat = (x: number) => Math.min(1, Math.max(0, x))
const smooth = (e0: number, e1: number, x: number) => {
  const t = sat((x - e0) / (e1 - e0))
  return t * t * (3 - 2 * t)
}

// Subtractive pigment model (matches the WGSL `pigment`).
export function pigment(flesh: number[], mel: number, car: number, red: number, blue: number) {
  const a = [
    mel * 0.3 + red * 0.05 + blue * 1.15,
    mel * 0.55 + car * 0.2 + red * 1.25 + blue * 0.5,
    mel * 0.95 + car * 1.25 + red * 0.8 + blue * 0.05,
  ]
  return flesh.map((x, i) => x * Math.exp(-a[i]))
}

export function colors(g: Record<string, number>) {
  const f = g.flesh * g.flesh
  const flesh = [0.93 + 0.02 * f, 0.91 - 0.07 * f, 0.87 - 0.27 * f]
  const pig = pigment(
    flesh,
    Math.pow(g.mel, 1.4) * 3.4,
    smooth(0.25, 1, g.car) * 3.2,
    smooth(0.5, 1, g.red) * 2.6,
    smooth(0.78, 1, g.blue) * 2.6,
  )
  const sm = Math.pow(g.spore, 1.3) * 3.2
  const spore = pigment([0.93, 0.91, 0.87], sm, sm * 0.3, 0, 0)
  return { flesh, pig, spore }
}

// The colour of its light: violet, blue, the green of real foxfire (the
// middle), yellow, amber.
const GLOW_STOPS = [[0.62, 0.36, 1], [0.3, 0.62, 1], [0.45, 1, 0.55], [0.85, 1, 0.32], [1, 0.62, 0.24]]
export function glowColour(hue: number) {
  const x = sat(hue) * (GLOW_STOPS.length - 1)
  const i = Math.min(Math.floor(x), GLOW_STOPS.length - 2)
  const t = x - i
  return GLOW_STOPS[i].map((a, k) => a + (GLOW_STOPS[i + 1][k] - a) * t)
}

// Genome -> program. Runs when an individual is created or edited.
// alive: 1 a living colony, 0.25 one that has gone but whose fruit bodies
// are still rotting
export function express(genome: Genome, out: Float32Array, offset: number, alive = 1) {
  const g = geneValues(genome)
  const set = (i: number, v: number) => (out[offset + i] = v)

  set(G.SUBSTRATE, g.substrate)
  set(G.ABSORB, 0.2 + g.absorb * 1.6)
  // how full its store must be before a colony fruits
  set(G.FRUIT_AT, 0.15 + 0.8 * g.fruitAt)
  set(G.FRUIT_PROB, g.fruitProb * g.fruitProb * 0.04)
  set(G.FRUIT_PHASE, phaseIndex(g.fruitPhase))
  // how wet the ground must be for it to fruit
  set(G.FRUIT_WET, 0.6 - 0.45 * g.fruitWet)
  set(G.ALIVE, alive)
  const c = colors(g)
  c.flesh.forEach((v, i) => set(G.FLESH_R + i, v))
  c.pig.forEach((v, i) => set(G.PIG_R + i, v))
  c.spore.forEach((v, i) => set(G.SPORE_R + i, v))
  glowColour(g.glowHue).forEach((v, i) => set(G.GLOW_R + i, v))

  set(G.BODY, g.body)
  set(G.CLUSTER, 1 + Math.round(g.cluster * g.cluster * 9))
  set(G.SPREAD, 0.1 + g.spread * 2.4)
  set(G.SIZE, 0.12 + 2.6 * Math.pow(g.size, 1.6))
  set(G.GROW, 150 + g.grow * g.grow * 3000)
  set(G.LIFE, g.life >= 0.99 ? 1e9 : 200 + g.life * g.life * 6000)
  set(G.DECAY, decayIndex(g.decay))
  set(G.LATERAL, g.lateral)
  set(G.UPRIGHT, g.upright)
  set(G.PHOTO, (g.photo - 0.5) * 2)
  set(G.TIP_HEADS, g.tipHeads * g.tipHeads)
  set(G.STEM_LEN, 8 * g.stemLen * g.stemLen)
  set(G.STEM_R, 0.04 + 0.56 * g.stemThick * g.stemThick)
  set(G.TAPER, 0.4 + 1.2 * g.taper)
  set(G.BULB, g.bulb * 1.2)
  set(G.VOLVA, g.volva)
  set(G.RING, g.ring)
  set(G.SKIRT, g.skirt)
  set(G.WOBBLE, g.wobble)
  // head development: turning the button gives up over its expansion (2.95:
  // flat; more: a cup), where it opens first, and turning per growth time
  // given up after it is grown
  set(G.OPEN, g.open * 5.35)
  set(G.OPEN_AT, g.openAt)
  set(G.UPTURN, g.upturn * 1.5)
  set(G.ELONG, 0.5 + 2 * g.elong)
  set(G.CONE, g.cone * 1.5)
  set(G.THICK, 0.04 + 0.36 * g.thick)
  set(G.UMBO, g.umbo * 0.45)
  set(G.CURL, (g.curl - 0.5) * 2)
  set(G.RAYS, g.rays < 0.15 ? 0 : Math.round(3 + ((g.rays - 0.15) / 0.85) * 9))
  set(G.RAY_DEPTH, g.rayDepth)
  set(G.RAY_CURL, (g.rayCurl - 0.5) * 6)
  // head development (sim.wgsl, develop): growth rate of the wrinkling over
  // the whole head and in a band at its margin, the bump/ridge/pit balance,
  // the grain (< 0 radial, > 0 concentric), features across the head
  set(G.EXCESS, -0.15 + 0.65 * g.excess)
  set(G.EDGE, g.edge * 0.9)
  set(G.TEXTURE, (0.5 - g.texture) * 2.4)
  set(G.GRAIN, (g.grain - 0.5) * 2)
  set(G.SCALE, 1.5 + (1 - g.scale) * 8.5)
  set(G.BEND, g.bend)
  // displacement per unit of field, as a fraction of the wavelength
  set(G.DEPTH, 0.08 + 0.42 * g.depth)
  set(G.TEAR, g.tear)
  set(G.GILLS, Math.round(g.gills * g.gills * 90))
  set(G.GILL_DEPTH, 0.05 + 0.5 * g.gillDepth)
  set(G.TEETH, g.teeth * 1.2)
  set(G.GLEBA, g.gleba)
  set(G.EGGS, g.eggs < 0.1 ? 1 : 2 + Math.round(((g.eggs - 0.1) / 0.9) * 6))
  set(G.STEM_PIG, g.stemPig)
  set(G.ZONES, g.zones)
  set(G.STREAK, g.streak)
  set(G.BODY_GLOW, g.bodyGlow * g.bodyGlow * 3)
  set(G.JELLY, g.jelly)
  set(G.COMBAT, g.combat)
  set(G.PARASITE, g.parasite)
  set(G.SPORE_OUT, g.sporeOut * g.sporeOut * 2)
  // (off below a little: a gene a mutation nudges from zero stays off)
  const on = (x: number, at = 0.08) => (x < at ? 0 : (x - at) / (1 - at))
  // how readily it fires from what fires around it; how often unprompted
  // (per step); steps it rests after firing and its flash lasts; how bright
  // the flash; how readily it fires as night falls (per step)
  set(G.EXCITE, on(g.excite))
  set(G.PACE, on(g.pace) ** 2 * 4e-5)
  set(G.REST, 30 + g.rest * g.rest * 2000)
  set(G.FLASH, 4 + g.flashLen * g.flashLen * 150)
  set(G.PULSE_GLOW, g.pulseGlow * g.pulseGlow * 3)
  set(G.DUSK, on(g.duskFire) * 0.0005)
  // fruit bodies: closing by day, shedding at night; blinking (cycles per
  // step) and pulling into step with blinking neighbours (per step)
  set(G.NIGHT_OPEN, on(g.nightOpen))
  set(G.NIGHT_SPORE, on(g.nightSpore))
  set(G.BLINK, g.blink < 0.08 ? 0 : 1 / (30 + (1 - g.blink) ** 2 * 600))
  set(G.SYNC, on(g.sync) ** 2 * 0.02)

  for (let p = 0; p < PHASES; p++) {
    const v = (k: string) => g[`p${p}.${k}`]
    const o = PHASE_BASE + p * PHASE_STRIDE
    const s = (k: string, x: number) => set(o + Q[k], x)
    const signed = (k: string, range: number) => (v(k) - 0.5) * 2 * range
    s('SPEED', 0.02 + 0.98 * v('speed') * v('speed'))
    s('UP', signed('up', 2))
    s('PERSIST', v('persist') * 4)
    s('WANDER', v('wander') * 1.5)
    s('CROWD', signed('crowd', 3))
    s('SIGNAL', signed('signal', 3))
    s('EMIT', v('emit'))
    s('NUTRIENT', signed('nutrient', 2))
    s('LIGHT', signed('light', 2))
    s('SURFACE', signed('surface', 2))
    s('TWIST', signed('twist', 1.5))
    s('BRANCH', v('branch') * v('branch') * 0.5)
    s('ANGLE', 0.15 + v('angle') * 1.3)
    s('LENGTH', Math.round(1 + v('length') * v('length') * 150))
    s('NEXT', phaseIndex(v('next')))
    s('SPAWN', v('spawn') * 0.7)
    s('SPAWN_PHASE', phaseIndex(v('spawnPhase')))
    s('STIFF', v('stiff'))
    s('RADIUS', 0.012 + v('radius') * 0.06)
    s('PIGMENT', v('pigment'))
    s('LIFESPAN', v('lifespan') >= 0.99 ? 1e9 : 20 + v('lifespan') * v('lifespan') * 3000)
    s('GLOW', v('glow') * v('glow') * 3)
    // rules: what a rule senses and does (0: nothing) and how strongly
    // (-2..2; none near the middle)
    for (const r of ['R1', 'R2']) {
      const k = r.toLowerCase()
      const gain = (v(`${k}Gain`) - 0.5) * 4
      s(`${r}_IN`, ruleIndex(v(`${k}In`), RULE_INPUTS.length))
      s(`${r}_OUT`, ruleIndex(v(`${k}Out`), RULE_OUTPUTS.length))
      s(`${r}_GAIN`, Math.abs(gain) < 0.1 ? 0 : gain)
    }
  }
}
