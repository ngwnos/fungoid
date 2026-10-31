import { compile } from './shaders'
import { add, halton, normalize, scale, sub, type Basis, type V3 } from './camera'
import { PHENO_STRIDE, glows } from '../genome/phenotype'
import { MAX_ROCKS, MAX_ROOTS, WORLD_BYTES, octEncode } from '../world'
import { sunDir } from '../weather'
import {
  F4_RES, F4N, FIELD_EVERY, GLOW, GLOW_N, GRID, GRID_WORDS, GROUND_RES, LIGHT, N_SHELL, PLANET_R, SHELL, MAX_CELLS, MAX_EVENTS, MAX_FRUITS, MAX_ORGS,
  PAT_COLS, PAT_RES, PAT_SLOTS, SLIME_COLS, SLIME_EVERY, SLIME_FLOATS, SLIME_MAX, SLIME_RES, SLIME_WORDS, VOL,
} from './dims'

export { MAX_ORGS }
const CELL_BYTES = 64
const FRUIT_BYTES = 24 * 16
// (which fruit body holds each development field; the slime mould's counters
// and columns: dims.ts)
const GRID_DEV = GRID.PATS
const GRID_SLIME = GRID.SLIME
const VIEW_BYTES = 96
const SCENE_BYTES = 160
const SIMU_BYTES = 32 + 128 * 16 + 48
// (all set down at once, whatever their number: a garden run again sets
// them down at the same step)
const MAX_INOC = 64
// The garden is kept (births, deaths, arrivals: App.tsx, tend) every
// KEEP_EVERY steps, at those steps whatever the speed: the steps stop there
// till its keeping is done, so a garden run again from a checkpoint grows
// as it did.
export const KEEP_EVERY = 1000
// and a sample of it taken whenever it has moved on (its counts, colony by
// colony: the timeline), read back as it goes

// The garden between two steps (Renderer.captureGarden): what the steps after it
// need, and the renderer's own count of it.
export interface GardenState {
  step: number
  keepAt: number
  fieldPar: number
  slimeReqs: number
  pendingInoc: Inoculum[]
  slimeReq: [number, number, number] | null
  cells: ArrayBuffer // the slots in use
  grid: ArrayBuffer
  field: ArrayBuffer // the current one of the two (the other is all rewritten before it is read)
  fruits: ArrayBuffer
  devSlots: number[] // the development fields a head holds, and theirs
  dev: ArrayBuffer
  slimeAgents: ArrayBuffer
  slimeMap: ArrayBuffer
  pools: ArrayBuffer
  counters: ArrayBuffer
  events: ArrayBuffer
}
const LIGHT_EVERY = 32 // steps between light sweeps (a multiple of FIELD_EVERY)
const PACK_EVERY = 256 // steps between packing the cells into the lowest slots
const WABS_BYTES = (MAX_ROCKS + MAX_ROOTS * 6) * 16

// workgroups of `n` per side over a grid
const groups = (v: readonly number[], n: number) => v.map((x) => Math.ceil(x / n)) as [number, number, number]

const PASSES = ['sim', 'tissue', 'shade', 'taa', 'post'] as const
type PassName = (typeof PASSES)[number]

export interface FrameInput {
  basis: Basis
  target: V3
  height: number
  sunRadius: number
  // how much day it is where the view is (0 night .. 1 day): the eye opens
  // up in the dark
  viewDay: number
  exposure: number
  debug: number
  bounces: number
  selected: number // organism slot or -1
  second: number
  alphaFloor: number
  aperture: number
  simSteps: number
  // a perspective camera through a thin lens (else orthographic): vertical
  // field of view (rad), the lens's radius and the distance it focuses at
  // (cm); each frame looks through another point of the lens, so frames
  // averaged together blur what is out of focus
  lens?: Lens | null
  // nothing changes from frame to frame: history is the plain average
  accumulate?: boolean
  fineMoss: boolean
  // the player's tools: the ground tinted by moisture (1) or food (2), the
  // brush ring, and the brush itself this frame
  overlay: number
  cursor: [number, number, number, number] | null
  brush: Brush | null
}

export interface Lens {
  fov: number
  aperture: number
  focus: number
  // frames averaged per image: what is out of focus is blurred a little
  // more, across the gaps between that many points on the lens
  samples: number
}
// points on the lens (a golden-angle spiral), taken from the frame history
// last started over in bit-reversed order, so any first few cover the lens
const LENS_POINTS = 32
const lensPoint = (i: number) => {
  let j = 0
  for (let b = 0; b < 5; b++) j |= ((i >> b) & 1) << (4 - b)
  return j
}

// What the brush does where the player points (sim.wgsl, brush).
export const BRUSH_OPS = { water: 1, dry: 2, feed: 3, deplete: 4 } as const
export interface Brush {
  at: V3
  radius: number
  op: number
  strength: number
}

// What is under a pixel: the organism there (or -1), and the point.
export interface Pick {
  slot: number
  point: V3 | null
}

export const SUN_COLOR: [number, number, number] = [5.6, 0.84 * 5.6, 0.64 * 5.6]

// The sky by day (shade.wgsl warms it as the sun sets or rises, and darkens
// it at night, point by point).
const ZEN: V3 = [0.34, 0.5, 0.95]
const HOR: V3 = [0.75, 0.82, 0.95]
// The sun's light is baked for one place of the sun every SUN_EVERY steps
// (bake_sun.wgsl bakeSunVis) and blended between the two either side of now.
const SUN_EVERY = 24

export interface Inoculum {
  org: number
  // where: on the ground under this point, or on the first surface (wood)
  // under it, or exactly here (a germinating spore)
  x: number
  y: number
  z: number
  onSurface: boolean
  exact?: boolean
}

// Where simulation kernels are recorded: all into one compute pass, or (to
// time them) each into a pass of its own.
interface Passes {
  begin(name: string): GPUComputePassEncoder
  end(p: GPUComputePassEncoder): void
}
const onePass = (p: GPUComputePassEncoder): Passes => ({ begin: () => p, end: () => {} })

// What the garden reports back to the CPU every few frames.
export interface Ecology {
  // per organism: food, mycelium cells, live fruit bodies
  food: Float32Array
  cells: Int32Array
  fruits: Int32Array
  // spores that germinated: where, and their parents (organism slots, -1 none)
  births: { x: number; y: number; z: number; a: number; b: number }[]
  // the slime mould: 0 none, 1 feeding, 2 fruiting; its agents
  slime: { mode: number; agents: number }
}

// A finished frame read back from the GPU (rows of `row` bytes).
export interface Grab {
  data: Uint8Array
  width: number
  height: number
  row: number
  format: GPUTextureFormat
}

export interface RenderStats {
  gpuMs: Partial<Record<PassName, number>>
  gpuTotal: number
  cells: number
  step: number
}

const BUF = GPUBufferUsage
const TEX = GPUTextureUsage
type Res = GPUBuffer | GPUTextureView | GPUSampler

export class Renderer {
  device!: GPUDevice
  ctx!: GPUCanvasContext
  format!: GPUTextureFormat
  width = 0
  height = 0
  frame = 0
  stats: RenderStats = { gpuMs: {}, gpuTotal: 0, cells: 0, step: 0 }

  private pipes: Record<string, GPUComputePipeline> = {}
  private presentPipe!: GPURenderPipeline
  private simLayout!: GPUBindGroupLayout
  private slimeLayout!: GPUBindGroupLayout
  private glowLayout!: GPUBindGroupLayout
  private tissuePipe!: GPURenderPipeline
  private fruitPipe!: GPURenderPipeline
  private b: Record<string, GPUBuffer> = {}
  private ground!: GPUTexture
  private vol!: GPUTexture
  private aux!: GPUTexture
  private light!: GPUTexture
  private bounce!: GPUTexture
  private tissueLight!: GPUTexture
  private sampler!: GPUSampler
  private sized: {
    rad: GPUTexture
    alb: GPUTexture
    geo: GPUTexture[]
    col: GPUTexture[]
    mom: GPUTexture[]
    final: GPUTexture
    tGeo: GPUTexture
    tId: GPUTexture
    tDepth: GPUTexture
    ids: GPUBuffer
  } | null = null
  private groups: Record<string, GPUBindGroup> = {}
  private cur = 0
  private fieldPar = 0
  private simStep = 0
  private prevView: ArrayBuffer | null = null
  private pendingInoc: Inoculum[] = []
  private pendingReset = true
  private flushHistory = false
  private grabRequest: { resolve: (g: Grab) => void; scale: number } | null = null
  private grabTex: GPUTexture | null = null
  private grabSmall: GPUTexture | null = null
  private lensFrame = 0

  private querySet: GPUQuerySet | null = null
  private queryResolve: GPUBuffer | null = null
  private queryRead: GPUBuffer | null = null
  // the frame's own stats (cells, step, GPU time), now and then
  private statsPending = false
  private statFrame = 0
  private statsGen = 0
  private statRead!: GPUBuffer
  // the next step the garden is kept at, and its keeping under way
  private keepAt = KEEP_EVERY
  private keeping: Promise<void> | null = null
  // while above 0, nothing runs that would change the garden (a checkpoint
  // is being taken)
  holds = 0
  private countRead!: GPUBuffer
  private poolRead!: GPUBuffer
  private eventRead!: GPUBuffer
  private fieldTex!: GPUTexture
  private glowTex!: GPUTexture
  // whether anything glows, and whether the light field still holds glow
  private glowing = false
  private glowLit = false
  private glowDirty = true
  // how much day the frame shows, and dusk
  // the sun's light baked for two places of the sun: the step each is for
  private sunVis!: GPUTexture[]
  private visAt = [NaN, NaN]
  private devTex!: GPUTexture
  private slimeTex!: GPUTexture
  private slimeRead!: GPUBuffer
  private slimeReq: [number, number, number] | null = null
  private slimeReqs = 0
  private slimeFrame = false
  private slimeShow = false
  ecology: Ecology = {
    food: new Float32Array(MAX_ORGS), cells: new Int32Array(MAX_ORGS), fruits: new Int32Array(MAX_ORGS), births: [],
    slime: { mode: 0, agents: 0 },
  }
  private pickRead!: GPUBuffer
  private pickRequest: { x: number; y: number; resolve: (v: Pick) => void } | null = null
  private pickPending = false

  static async create(canvas: HTMLCanvasElement): Promise<Renderer> {
    const r = new Renderer()
    await r.init(canvas)
    return r
  }

  private buf(name: string, size: number, usage: number) {
    this.b[name] = this.device.createBuffer({ label: name, size, usage })
    return this.b[name]
  }

  private async init(canvas: HTMLCanvasElement) {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.')
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) throw new Error('No WebGPU adapter.')
    const features: GPUFeatureName[] = []
    if (adapter.features.has('timestamp-query')) features.push('timestamp-query')
    const lim = adapter.limits
    this.device = await adapter.requestDevice({
      requiredFeatures: features,
      requiredLimits: {
        maxStorageBuffersPerShaderStage: lim.maxStorageBuffersPerShaderStage,
        maxStorageTexturesPerShaderStage: lim.maxStorageTexturesPerShaderStage,
        maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
        maxBufferSize: lim.maxBufferSize,
        maxSampledTexturesPerShaderStage: lim.maxSampledTexturesPerShaderStage,
        maxUniformBufferBindingSize: lim.maxUniformBufferBindingSize,
        maxBindingsPerBindGroup: lim.maxBindingsPerBindGroup,
        // head development holds a 62 x 62 field and its Laplacian (sim.wgsl)
        maxComputeWorkgroupStorageSize: lim.maxComputeWorkgroupStorageSize,
      },
    })
    this.device.lost.then((info) => console.error('WebGPU device lost:', info.message))
    // (the grid buffer is bound whole: dims.ts)
    if (GRID_WORDS * 4 > this.device.limits.maxStorageBufferBindingSize) throw new Error(`the grid buffer (${GRID_WORDS * 4} bytes) is larger than this GPU binds`)
    let errors = 0
    this.device.addEventListener('uncapturederror', (e) => {
      if (errors++ < 8) console.error('WebGPU error:', (e as GPUUncapturedErrorEvent).error.message)
    })
    this.ctx = canvas.getContext('webgpu') as GPUCanvasContext
    this.format = navigator.gpu.getPreferredCanvasFormat()
    this.ctx.configure({ device: this.device, format: this.format, alphaMode: 'opaque' })

    const d = this.device
    const S = BUF.STORAGE
    this.buf('world', WORLD_BYTES, BUF.UNIFORM | BUF.COPY_DST)
    this.buf('wabs', WABS_BYTES, S)
    this.buf('view', VIEW_BYTES, BUF.UNIFORM | BUF.COPY_DST)
    this.buf('prevView', VIEW_BYTES, BUF.UNIFORM | BUF.COPY_DST)
    this.buf('scene', SCENE_BYTES, BUF.UNIFORM | BUF.COPY_DST)
    this.buf('simU', SIMU_BYTES, BUF.UNIFORM | BUF.COPY_DST)
    // (the blocks life can reach: dims.ts)
    d.queue.writeBuffer(this.buf('shell', SHELL.byteLength, BUF.UNIFORM | BUF.COPY_DST), 0, SHELL)
    this.buf('sunU', 32, BUF.UNIFORM | BUF.COPY_DST)
    this.buf('pheno', MAX_ORGS * PHENO_STRIDE * 4, S | BUF.COPY_DST | BUF.COPY_SRC)
    this.buf('cells', MAX_CELLS * CELL_BYTES, S | BUF.COPY_SRC | BUF.COPY_DST)
    this.buf('counters', 32, S | BUF.COPY_DST | BUF.COPY_SRC)
    this.buf('args', 32, S | BUF.INDIRECT | BUF.COPY_DST)
    this.buf('pools', MAX_ORGS * 16, S | BUF.COPY_DST | BUF.COPY_SRC)
    this.buf('fruits', MAX_FRUITS * FRUIT_BYTES, S | BUF.COPY_DST | BUF.COPY_SRC)
    this.buf('grid', GRID_WORDS * 4, S | BUF.COPY_SRC | BUF.COPY_DST)
    this.buf('fieldA', F4N * 32, S | BUF.COPY_SRC | BUF.COPY_DST)
    this.buf('fieldB', F4N * 32, S | BUF.COPY_SRC | BUF.COPY_DST)
    this.buf('events', MAX_EVENTS * 16, S | BUF.COPY_SRC | BUF.COPY_DST)
    this.countRead = d.createBuffer({ size: 32, usage: BUF.COPY_DST | BUF.MAP_READ })
    this.statRead = d.createBuffer({ size: 32, usage: BUF.COPY_DST | BUF.MAP_READ })
    this.poolRead = d.createBuffer({ size: MAX_ORGS * 16, usage: BUF.COPY_DST | BUF.MAP_READ })
    this.eventRead = d.createBuffer({ size: MAX_EVENTS * 16, usage: BUF.COPY_DST | BUF.MAP_READ })
    this.slimeRead = d.createBuffer({ size: 64, usage: BUF.COPY_DST | BUF.MAP_READ })
    this.buf('slimeAgents', SLIME_MAX * 16, S | BUF.COPY_DST | BUF.COPY_SRC)
    this.buf('slimeMap', SLIME_FLOATS * 4, S | BUF.COPY_DST | BUF.COPY_SRC)
    // the id there, then (at 256) its depth
    this.pickRead = d.createBuffer({ size: 272, usage: BUF.COPY_DST | BUF.MAP_READ })

    this.ground = d.createTexture({ label: 'ground', size: GROUND_RES, format: 'rgba16float', usage: TEX.STORAGE_BINDING | TEX.TEXTURE_BINDING | TEX.COPY_SRC })
    this.vol = d.createTexture({ label: 'volume', size: VOL, dimension: '3d', format: 'rgba16float', usage: TEX.STORAGE_BINDING | TEX.TEXTURE_BINDING })
    this.aux = d.createTexture({ label: 'aux', size: VOL, dimension: '3d', format: 'rgba8snorm', usage: TEX.STORAGE_BINDING | TEX.TEXTURE_BINDING })
    const vol3 = (label: string, size: [number, number, number]) =>
      d.createTexture({ label, size, dimension: '3d', format: 'rgba16float', usage: TEX.STORAGE_BINDING | TEX.TEXTURE_BINDING })
    this.light = vol3('light', LIGHT)
    this.bounce = vol3('bounce', LIGHT)
    this.sunVis = [0, 1].map((k) =>
      d.createTexture({ label: `sunVis${k}`, size: LIGHT, dimension: '3d', format: 'rgba8unorm', usage: TEX.STORAGE_BINDING | TEX.TEXTURE_BINDING }))
    for (const k of [0, 1]) this.buf(`sunVisU${k}`, 32, BUF.UNIFORM | BUF.COPY_DST)
    this.tissueLight = vol3('tissueLight', F4_RES)
    this.fieldTex = vol3('fieldTex', F4_RES)
    this.glowTex = vol3('glow', GLOW)
    this.buf('glowAcc', GLOW_N * 12, S)
    this.buf('glowBlur', GLOW_N * 32, S | BUF.COPY_SRC)
    this.slimeTex = d.createTexture({ label: 'slime', size: SLIME_RES, format: 'rgba16float', usage: TEX.STORAGE_BINDING | TEX.TEXTURE_BINDING })
    this.buf('devField', PAT_SLOTS * PAT_RES * PAT_RES * 4, S | BUF.COPY_SRC | BUF.COPY_DST)
    this.devTex = d.createTexture({
      label: 'development',
      size: [PAT_COLS * PAT_RES, (PAT_SLOTS / PAT_COLS) * PAT_RES],
      format: 'rgba16float',
      usage: TEX.STORAGE_BINDING | TEX.TEXTURE_BINDING,
    })
    this.sampler = d.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    })

    if (features.includes('timestamp-query')) {
      const n = PASSES.length * 2
      this.querySet = d.createQuerySet({ type: 'timestamp', count: n })
      this.queryResolve = d.createBuffer({ size: n * 8, usage: BUF.QUERY_RESOLVE | BUF.COPY_SRC })
      this.queryRead = d.createBuffer({ size: n * 8, usage: BUF.COPY_DST | BUF.MAP_READ })
    }

    await this.createPipelines()
  }

  private async createPipelines() {
    const d = this.device
    const C = GPUShaderStage.COMPUTE
    const st = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({ binding, visibility: C, buffer: { type } })
    this.simLayout = d.createBindGroupLayout({
      label: 'sim',
      entries: [
        st(0, 'uniform'),
        st(1, 'read-only-storage'),
        ...[2, 3, 4, 5, 6, 7, 8, 9].map((i) => st(i, 'storage')),
        { binding: 10, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 11, visibility: C, texture: { sampleType: 'float' } },
        { binding: 12, visibility: C, sampler: { type: 'filtering' } },
        ...[13, 14].map((binding) => ({
          binding,
          visibility: C,
          storageTexture: { access: 'write-only' as const, format: 'rgba16float' as const, viewDimension: '3d' as const },
        })),
        st(15, 'storage'),
        { binding: 16, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' } },
        // (moss, for tissue that runs over it)
        { binding: 20, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        st(21, 'uniform'),
      ],
    })
    // the slime mould's passes: what they use of the simulation's, and their own
    this.slimeLayout = d.createBindGroupLayout({
      label: 'slime',
      entries: [
        st(0, 'uniform'),
        st(1, 'read-only-storage'),
        ...[3, 4, 5, 6, 8, 17, 18].map((i) => st(i, 'storage')),
        { binding: 10, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 11, visibility: C, texture: { sampleType: 'float' } },
        { binding: 12, visibility: C, sampler: { type: 'filtering' } },
        { binding: 19, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' } },
      ],
    })
    // what glows and the light it casts (glow.wgsl)
    this.glowLayout = d.createBindGroupLayout({
      label: 'glow',
      entries: [
        ...[0, 1, 2, 3].map((i) => st(i, 'read-only-storage')),
        st(4, 'storage'),
        st(5, 'storage'),
        { binding: 6, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
      ],
    })
    const names = ['bakeGround', 'bakeVolume', 'bakeAux', 'bakeSun', 'sim', 'args', 'tissue', 'fruitRaster', 'shade', 'temporal', 'post', 'present', 'downsample', 'glow'] as const
    const mods = Object.fromEntries(await Promise.all(names.map(async (n) => [n, await compile(d, n)] as const)))
    const auto = (label: string, module: GPUShaderModule, entryPoint = 'main') =>
      d.createComputePipelineAsync({ label, layout: 'auto', compute: { module, entryPoint } })
    const explicit = (layout: GPUBindGroupLayout, module: GPUShaderModule, entryPoint: string) =>
      d.createComputePipelineAsync({
        label: entryPoint,
        layout: d.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module, entryPoint },
      })
    const jobs: [string, Promise<GPUComputePipeline>][] = [
      ['bakeGround', auto('bakeGround', mods.bakeGround)],
      ['bakeVolume', auto('bakeVolume', mods.bakeVolume)],
      ['bakeAux', auto('bakeAux', mods.bakeAux)],
      ['bakeLight', auto('bakeLight', mods.bakeSun, 'bakeLight')],
      ['bakeBounce', auto('bakeBounce', mods.bakeSun, 'bakeBounce')],
      ['bakeSunVis', auto('bakeSunVis', mods.bakeSun, 'bakeSunVis')],
      ['shade', auto('shade', mods.shade)],
      ['temporal', auto('temporal', mods.temporal)],
      ['post', auto('post', mods.post)],
      ['downsample', auto('downsample', mods.downsample)],
      ['finalize', auto('finalize', mods.args)],
      ['recount', auto('recount', mods.args, 'recount')],
    ]
    const simEntries = ['clearFields', 'deposit', 'signal', 'fields', 'grow', 'reap', 'inoculate', 'resetFields', 'shadeTissue',
      'listFruits', 'growFruits', 'develop', 'clearOcc', 'depositFruits', 'germinate', 'brush', 'light',
      'packCount', 'packScan', 'packRank', 'packMove', 'packFix', 'devPaint']
    for (const e of simEntries) {
      jobs.push([e, explicit(this.simLayout, mods.sim, e)])
    }
    for (const e of ['slimeHatch', 'slimeMove', 'slimeTrail', 'slimeColumns', 'slimePop', 'slimePaint']) {
      jobs.push([e, explicit(this.slimeLayout, mods.sim, e)])
    }
    for (const e of ['glowCells', 'glowFruits', 'glowBlurX', 'glowBlurY', 'glowBlurZ']) {
      jobs.push([e, explicit(this.glowLayout, mods.glow, e)])
    }
    const done = await Promise.all(jobs.map((j) => j[1]))
    jobs.forEach(([n], i) => (this.pipes[n] = done[i]))
    this.tissuePipe = await d.createRenderPipelineAsync({
      label: 'tissue',
      layout: 'auto',
      vertex: { module: mods.tissue, entryPoint: 'vs' },
      fragment: { module: mods.tissue, entryPoint: 'fs', targets: [{ format: 'rgba32float' }, { format: 'r32uint' }] },
      primitive: { topology: 'triangle-strip' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    })
    this.fruitPipe = await d.createRenderPipelineAsync({
      label: 'fruits',
      layout: 'auto',
      vertex: { module: mods.fruitRaster, entryPoint: 'vs' },
      fragment: { module: mods.fruitRaster, entryPoint: 'fs', targets: [{ format: 'rgba32float' }, { format: 'r32uint' }] },
      primitive: { topology: 'triangle-strip' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    })
    this.presentPipe = await d.createRenderPipelineAsync({
      label: 'present',
      layout: 'auto',
      vertex: { module: mods.present, entryPoint: 'vs' },
      fragment: { module: mods.present, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    })

    const v = (t: GPUTexture) => t.createView()
    const b = this.b
    const L = (n: string) => this.pipes[n].getBindGroupLayout(0)
    this.groups.bakeGround = this.group(L('bakeGround'), [b.world, b.wabs, v(this.ground)])
    this.groups.bakeVolume = this.group(L('bakeVolume'), [b.world, b.wabs, v(this.ground), this.sampler, v(this.vol)])
    this.groups.bakeAux = this.group(L('bakeAux'), [b.world, b.wabs, v(this.vol), v(this.aux)])
    const at = (layout: GPUBindGroupLayout, entries: [number, Res][]) =>
      d.createBindGroup({
        layout,
        entries: entries.map(([binding, r]) => ({ binding, resource: r instanceof GPUBuffer ? { buffer: r } : r })),
      })
    this.groups.bakeLight = at(L('bakeLight'), [[1, v(this.vol)], [2, v(this.aux)], [3, this.sampler], [4, v(this.light)]])
    this.groups.bakeBounce = at(L('bakeBounce'), [
      [0, b.sunU], [1, v(this.vol)], [2, v(this.aux)], [3, this.sampler], [5, v(this.light)], [6, v(this.bounce)],
    ])
    for (const k of [0, 1]) {
      this.groups[`bakeSunVis${k}`] = at(L('bakeSunVis'), [[0, b[`sunVisU${k}`]], [1, v(this.vol)], [2, v(this.aux)], [3, this.sampler], [7, v(this.sunVis[k])]])
    }
    this.groups.finalize = this.group(L('finalize'), [b.counters, b.args])
    this.groups.recount = this.group(L('recount'), [b.counters, b.args])
    this.groups.tissue = this.group(this.tissuePipe.getBindGroupLayout(0), [b.view, b.cells, b.pheno, b.counters])
    this.groups.glow = this.group(this.glowLayout, [b.pheno, b.cells, b.counters, b.fruits, b.glowAcc, b.glowBlur, v(this.glowTex)])
    this.groups.fruits = this.group(this.fruitPipe.getBindGroupLayout(0), [b.view, b.fruits, v(this.devTex), this.sampler])
    for (let p = 0; p < 2; p++) {
      const [fa, fb] = p === 0 ? [b.fieldA, b.fieldB] : [b.fieldB, b.fieldA]
      this.groups[`sim${p}`] = at(this.simLayout, [
        ...[b.simU, b.pheno, b.cells, b.counters, b.pools, b.grid, fa, fb, b.fruits, b.events, v(this.vol), v(this.ground), this.sampler,
          v(this.tissueLight), v(this.fieldTex), b.devField, v(this.devTex)].map((r, i) => [i, r] as [number, Res]),
        [20, v(this.aux)],
        [21, b.shell],
      ])
      this.groups[`slime${p}`] = at(this.slimeLayout, [
        [0, b.simU], [1, b.pheno], [3, b.counters], [4, b.pools], [5, b.grid], [6, fa], [8, b.fruits],
        [10, v(this.vol)], [11, v(this.ground)], [12, this.sampler], [17, b.slimeAgents], [18, b.slimeMap], [19, v(this.slimeTex)],
      ])
    }
  }

  private group(layout: GPUBindGroupLayout, resources: Res[]): GPUBindGroup {
    return this.device.createBindGroup({
      layout,
      entries: resources.map((r, i) => ({ binding: i, resource: r instanceof GPUBuffer ? { buffer: r } : r })),
    })
  }

  // The sun's light for step `t`: the places of the sun to bake it for now
  // (slot, step), and how far from the first slot's to the second's `t` is.
  // Two keyframes either side of t, SUN_EVERY steps apart; when a frame
  // takes more steps than that, the sun's light at t itself.
  private sunVisFor(t: number, steps: number) {
    const bake: [number, number][] = []
    if (steps >= SUN_EVERY) {
      bake.push([0, t])
      this.visAt = [t, NaN]
      return { bake, blend: 0 }
    }
    const k0 = Math.floor(t / SUN_EVERY) * SUN_EVERY
    const k1 = k0 + SUN_EVERY
    for (const k of [k0, k1]) {
      if (this.visAt.includes(k)) continue
      // (into the slot holding neither)
      const slot = this.visAt[0] === k0 || this.visAt[0] === k1 ? 1 : 0
      this.visAt[slot] = k
      bake.push([slot, k])
    }
    const f = (t - k0) / SUN_EVERY
    return { bake, blend: this.visAt[1] === k1 ? f : 1 - f }
  }

  private bakeSunVis(enc: GPUCommandEncoder, bake: [number, number][]) {
    if (!bake.length) return
    const p = enc.beginComputePass({ label: 'sun light' })
    p.setPipeline(this.pipes.bakeSunVis)
    for (const [slot, step] of bake) {
      this.device.queue.writeBuffer(this.b[`sunVisU${slot}`], 0, new Float32Array([...sunDir(step), 0, ...SUN_COLOR, 0]))
      p.setBindGroup(0, this.groups[`bakeSunVis${slot}`])
      p.dispatchWorkgroups(...groups(LIGHT, 4))
    }
    p.end()
  }

  async bakeWorld(world: ArrayBuffer) {
    const d = this.device
    d.queue.writeBuffer(this.b.world, 0, world)
    d.queue.writeBuffer(this.b.sunU, 0, new Float32Array([0, 1, 0, 0, ...SUN_COLOR, 0]))
    // (the sun's light is baked again for the new world)
    this.visAt = [NaN, NaN]
    const enc = d.createCommandEncoder()
    const p = enc.beginComputePass({ label: 'bake' })
    p.setPipeline(this.pipes.bakeGround)
    p.setBindGroup(0, this.groups.bakeGround)
    p.dispatchWorkgroups(Math.ceil(GROUND_RES[0] / 8), Math.ceil(GROUND_RES[1] / 8))
    p.setPipeline(this.pipes.bakeVolume)
    p.setBindGroup(0, this.groups.bakeVolume)
    p.dispatchWorkgroups(...groups(VOL, 4))
    p.setPipeline(this.pipes.bakeAux)
    p.setBindGroup(0, this.groups.bakeAux)
    p.dispatchWorkgroups(...groups(VOL, 4))
    p.end()
    // lighting: a separate pass so the bounce reads the finished visibility
    const q = enc.beginComputePass({ label: 'bake light' })
    q.setPipeline(this.pipes.bakeLight)
    q.setBindGroup(0, this.groups.bakeLight)
    q.dispatchWorkgroups(...groups(LIGHT, 4))
    q.end()
    const r = enc.beginComputePass({ label: 'bake bounce' })
    r.setPipeline(this.pipes.bakeBounce)
    r.setBindGroup(0, this.groups.bakeBounce)
    r.dispatchWorkgroups(...groups(LIGHT, 4))
    r.end()
    d.queue.submit([enc.finish()])
    // the old scenery must not linger in the temporal history
    this.flushHistory = true
    await d.queue.onSubmittedWorkDone()
  }

  resize(w: number, h: number) {
    w = Math.max(8, Math.floor(w))
    h = Math.max(8, Math.floor(h))
    if (w === this.width && h === this.height && this.sized) return
    this.width = w
    this.height = h
    const d = this.device
    if (this.sized) {
      const s = this.sized
      ;[s.rad, s.alb, s.final, s.tGeo, s.tId, s.tDepth, ...s.geo, ...s.col, ...s.mom].forEach((t) => t.destroy())
      s.ids.destroy()
    }
    const tex = (label: string, format: GPUTextureFormat) =>
      d.createTexture({ label, size: [w, h], format, usage: TEX.STORAGE_BINDING | TEX.TEXTURE_BINDING })
    this.sized = {
      rad: tex('radiance', 'rgba16float'),
      alb: d.createTexture({ label: 'albedo', size: [w, h], format: 'rgba16float', usage: TEX.STORAGE_BINDING | TEX.TEXTURE_BINDING | TEX.COPY_SRC }),
      geo: [0, 1].map((i) =>
        d.createTexture({ label: `geo${i}`, size: [w, h], format: 'rgba32float', usage: TEX.STORAGE_BINDING | TEX.TEXTURE_BINDING | TEX.COPY_SRC }),
      ),
      // full float: long histories lose small updates at half precision
      col: [0, 1].map((i) =>
        d.createTexture({ label: `col${i}`, size: [w, h], format: 'rgba32float', usage: TEX.STORAGE_BINDING | TEX.TEXTURE_BINDING | TEX.COPY_SRC }),
      ),
      mom: [tex('mom0', 'rgba32float'), tex('mom1', 'rgba32float')],
      final: tex('final', 'rgba16float'),
      tGeo: d.createTexture({ label: 'tissueGeo', size: [w, h], format: 'rgba32float', usage: TEX.RENDER_ATTACHMENT | TEX.TEXTURE_BINDING }),
      tId: d.createTexture({ label: 'tissueId', size: [w, h], format: 'r32uint', usage: TEX.RENDER_ATTACHMENT | TEX.TEXTURE_BINDING }),
      tDepth: d.createTexture({ label: 'tissueDepth', size: [w, h], format: 'depth32float', usage: TEX.RENDER_ATTACHMENT }),
      ids: d.createBuffer({ label: 'ids', size: w * h * 4, usage: BUF.STORAGE | BUF.COPY_SRC }),
    }
    const s = this.sized
    const b = this.b
    const v = (t: GPUTexture) => t.createView()
    const L = (n: string) => this.pipes[n].getBindGroupLayout(0)
    for (let c = 0; c < 2; c++) {
      const pv = 1 - c
      this.groups[`shade${c}`] = this.group(L('shade'), [
        b.view, b.scene, b.pheno, b.cells, v(this.vol), v(this.aux), v(this.ground), v(s.tGeo), this.sampler,
        v(s.rad), v(s.geo[c]), v(s.alb), s.ids, v(s.tId), b.grid, v(this.light), v(this.bounce), v(this.tissueLight),
        b.counters, b.fruits, v(this.fieldTex), v(this.devTex), this.sampler, v(this.slimeTex), v(this.glowTex),
        v(this.sunVis[0]), v(this.sunVis[1]),
      ])
      this.groups[`temporal${c}`] = this.group(L('temporal'), [
        b.view, b.prevView, b.scene, v(s.rad), v(s.geo[c]), v(s.geo[pv]), v(s.col[pv]), v(s.mom[pv]), v(s.col[c]), v(s.mom[c]),
        v(s.alb),
      ])
      this.groups[`post${c}`] = this.group(L('post'), [b.view, b.scene, v(s.col[c]), v(s.geo[c]), v(s.final)])
      this.groups[`present${c}`] = this.group(this.presentPipe.getBindGroupLayout(0), [
        v(s.final), v(s.alb), v(s.rad), v(s.geo[c]), s.ids, b.scene, b.view, v(this.fieldTex), this.sampler,
      ])
    }
    this.prevView = null
  }

  setPrograms(data: Float32Array, count: number) {
    const n = Math.min(count, MAX_ORGS)
    if (n > 0) this.device.queue.writeBuffer(this.b.pheno, 0, data.buffer, data.byteOffset, n * PHENO_STRIDE * 4)
    this.glowing = false
    for (let o = 0; o < n && !this.glowing; o++) this.glowing = glows(data, o * PHENO_STRIDE)
    this.glowDirty = true
  }

  inoculate(list: Inoculum[]) {
    this.pendingInoc.push(...list)
  }

  // Clear every cell and restore the soil's food.
  resetSim() {
    this.pendingReset = true
    this.pendingInoc = []
    // (the garden's readbacks and keeping start over with it; any still
    // under way are of the garden before)
    this.statFrame = 0
    this.statsGen++
    this.simStep = 0
    this.keepAt = KEEP_EVERY
    this.keeping = null
    this.sampleAt = 0
    // (and its frames' jitter and noise, so that its renders repeat too)
    this.frame = 0
    this.ecology.births.length = 0
    this.onReset?.()
  }
  // Called when the garden starts over.
  onReset: (() => void) | null = null

  // Called to keep the garden, with its ecology read back at a keeping
  // step; and a promise of the keeping under way, if any.
  onKeep: (() => void) | null = null
  // Called with each brush stroke laid down as it is drawn, and its step;
  // and with each run of steps the garden takes in one go (its frames: what
  // runs once a frame does so once a run), from the step it starts at.
  onStroke: ((step: number, brush: Brush) => void) | null = null
  onChunk: ((from: number, steps: number) => void) | null = null
  // Called between two submissions of steps: the garden as it is then can
  // be captured.
  onBetween: (() => void) | null = null
  get keepingNow() {
    return this.keeping !== null
  }
  // nothing set down by hand is still waiting to land
  get settled() {
    return this.pendingInoc.length === 0 && this.slimeReq === null && !this.slimeFrame
  }
  // Called with each sample of the garden: its step, counters, colonies'
  // pools and the slime mould's counters.
  onSample: ((step: number, counters: Uint32Array, pools: Int32Array, slime: Int32Array) => void) | null = null
  // (the step the next sample is due at: the first, the garden as planted)
  private sampleAt = 0
  private sampleBufs: GPUBuffer[] = []
  private samplesOut = 0
  // after a restore: what a checkpoint does not keep is drawn again
  private repaint = false
  private captureBufs: Record<string, GPUBuffer> | null = null
  private capturing: Promise<unknown> | null = null
  statsSettled(): Promise<void> {
    return this.keeping ?? Promise.resolve()
  }
  // Steps a frame may run of `want`: none while the garden is kept or held,
  // and none past its next keeping.
  private stepsNow(want: number) {
    return this.keeping || this.holds > 0 ? 0 : Math.min(want, this.keepAt - (this.pendingReset ? 0 : this.simStep))
  }
  // After a frame's steps: at a keeping step, read the garden back to keep it.
  private keepIfDue(enc: GPUCommandEncoder) {
    if (this.simStep !== this.keepAt) return false
    enc.copyBufferToBuffer(this.b.counters, 0, this.countRead, 0, 32)
    enc.copyBufferToBuffer(this.b.pools, 0, this.poolRead, 0, MAX_ORGS * 16)
    enc.copyBufferToBuffer(this.b.events, 0, this.eventRead, 0, MAX_EVENTS * 16)
    // (the events copied out are taken: the next are counted afresh)
    enc.clearBuffer(this.b.counters, 8, 4)
    enc.copyBufferToBuffer(this.b.grid, GRID_SLIME * 4, this.slimeRead, 0, 64)
    this.keepAt += KEEP_EVERY
    return true
  }
  // After a frame's steps: a sample, when one is due (none while one is
  // already read back for a while: it is taken later).
  private sampleIfDue(enc: GPUCommandEncoder) {
    if (!this.onSample || this.step < this.sampleAt || this.samplesOut >= 4) return null
    const buf = this.sampleBufs.pop() ?? this.device.createBuffer({ size: 32 + MAX_ORGS * 16 + 64, usage: BUF.COPY_DST | BUF.MAP_READ })
    enc.copyBufferToBuffer(this.b.counters, 0, buf, 0, 32)
    enc.copyBufferToBuffer(this.b.pools, 0, buf, 32, MAX_ORGS * 16)
    enc.copyBufferToBuffer(this.b.grid, GRID_SLIME * 4, buf, 32 + MAX_ORGS * 16, 64)
    this.sampleAt = this.step + 1
    return { buf, step: this.step }
  }
  private readSample(s: { buf: GPUBuffer; step: number }) {
    const gen = this.statsGen
    this.samplesOut++
    void s.buf.mapAsync(GPUMapMode.READ).then(() => {
      const a = s.buf.getMappedRange().slice(0)
      s.buf.unmap()
      this.samplesOut--
      this.sampleBufs.push(s.buf)
      if (gen !== this.statsGen) return
      this.onSample?.(s.step, new Uint32Array(a, 0, 8), new Int32Array(a, 32, MAX_ORGS * 4), new Int32Array(a, 32 + MAX_ORGS * 16, 16))
    })
  }

  pick(x: number, y: number): Promise<Pick> {
    return new Promise((resolve) => {
      this.pickRequest?.resolve({ slot: -1, point: null })
      this.pickRequest = { x, y, resolve }
    })
  }

  private encodeSim(src: Passes, steps: number, inoculating: boolean, brushing: boolean) {
    const fx = Math.ceil(F4N / 256)
    const f3 = F4_RES.map((n) => Math.ceil(n / 4)) as [number, number, number]
    const k = this.kernels(src)
    const args = (p: GPUComputePassEncoder) => p.dispatchWorkgroupsIndirect(this.b.args, 0)
    if (this.pendingReset) k('resetFields', f3)
    if (inoculating) {
      k('inoculate', [1])
      k('recount', [1], this.groups.recount)
    }
    if (brushing) k('brush', f3)
    // a slime mould set down while paused still appears
    if (this.slimeFrame && steps === 0) this.encodeSlime(src)
    this.slimeFrame = false
    if (this.pendingReset) {
      this.simStep = 0
      // (and which field buffer is current: the same physical state each time)
      this.fieldPar = 0
    }
    if (steps > 0) this.onChunk?.(this.simStep, steps)
    for (let s = 0; s < steps; s++) {
      const st = this.simStep++
      k('clearFields', [N_SHELL])
      k('deposit', args)
      k('signal', [N_SHELL])
      if (st % SLIME_EVERY === 0) this.encodeSlime(src)
      if (st % FIELD_EVERY === 0) {
        if (st % LIGHT_EVERY === 0) k('light', f3)
        k('fields', f3)
        // the fresh fields are now in the other buffer
        this.fieldPar = 1 - this.fieldPar
      }
      k('grow', args)
      // (dead cells' slots are handed back every other step)
      if (st % 2 === 0) k('reap', args)
      if (st % PACK_EVERY === PACK_EVERY - 1) {
        const all = MAX_CELLS / 256
        k('packCount', [all])
        k('packScan', [1])
        k('packRank', [all])
        k('packMove', [all / 2])
        k('packFix', [all])
      }
      k('finalize', [1], this.groups.finalize)
    }
    // once per frame: fruit bodies, germination, and the fields for rendering
    if (steps > 0) {
      k('listFruits', [1])
      k('growFruits', [MAX_FRUITS / 64])
      k('develop', [PAT_SLOTS])
      k('clearOcc', [fx])
      k('depositFruits', [MAX_FRUITS / 64])
      k('germinate', f3)
    }
    // light through tissue: nothing changes while paused
    // (and, after a checkpoint is restored, what it does not keep)
    if (this.repaint) {
      k('devPaint', [PAT_SLOTS])
      k('slimePaint', [Math.ceil(SLIME_RES[0] / 16), Math.ceil(SLIME_RES[1] / 16)], this.groups[`slime${this.fieldPar}`])
      this.glowDirty = true
    }
    const changed = steps > 0 || inoculating || brushing || this.pendingReset || this.frame < 3 || this.repaint
    this.repaint = false
    if (changed) k('shadeTissue', f3)
    // and the light of what glows (or once more, to clear what glowed)
    if ((changed || this.glowDirty) && (this.glowing || this.glowLit)) {
      const g = this.groups.glow
      const g3 = GLOW.map((n) => Math.ceil(n / 4)) as [number, number, number]
      k('glowCells', args, g)
      k('glowFruits', [MAX_FRUITS / 64], g)
      k('glowBlurX', g3, g)
      k('glowBlurY', g3, g)
      k('glowBlurZ', g3, g)
      this.glowLit = this.glowing
      this.glowDirty = false
    }
  }

  // Dispatches a simulation kernel (with the current fields' bind group
  // unless given another) through `src`.
  private kernels(src: Passes) {
    return (name: string, size: number[] | ((p: GPUComputePassEncoder) => void), group?: GPUBindGroup) => {
      const p = src.begin(name)
      p.setBindGroup(0, group ?? this.groups[`sim${this.fieldPar}`])
      p.setPipeline(this.pipes[name])
      if (typeof size === 'function') size(p)
      else p.dispatchWorkgroups(...(size as [number, number?, number?]))
      src.end(p)
    }
  }

  // Dev: GPU milliseconds per simulation kernel over `steps` steps (each in
  // a pass of its own, timed), then per step for the per-step kernels.
  async profileSim(steps = 16): Promise<Record<string, number> | null> {
    const d = this.device
    if (!d.features.has('timestamp-query')) return null
    const names: string[] = []
    const qs = d.createQuerySet({ type: 'timestamp', count: 4096 })
    const enc = d.createCommandEncoder()
    const inoc = this.writeSim(steps)
    if (this.pendingReset) this.clearForReset(enc)
    this.encodeSim({
      begin: (name) => {
        const i = names.push(name) - 1
        return enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } })
      },
      end: (p) => p.end(),
    }, steps, inoc.length > 0, false)
    this.pendingReset = false
    const n = names.length * 2
    const resolve = d.createBuffer({ size: n * 8, usage: BUF.QUERY_RESOLVE | BUF.COPY_SRC })
    const read = d.createBuffer({ size: n * 8, usage: BUF.COPY_DST | BUF.MAP_READ })
    enc.resolveQuerySet(qs, 0, n, resolve, 0)
    enc.copyBufferToBuffer(resolve, 0, read, 0, n * 8)
    d.queue.submit([enc.finish()])
    await read.mapAsync(GPUMapMode.READ)
    const t = new BigInt64Array(read.getMappedRange().slice(0))
    read.destroy()
    resolve.destroy()
    qs.destroy()
    const out: Record<string, number> = {}
    names.forEach((name, i) => (out[name] = (out[name] ?? 0) + Number(t[2 * i + 1] - t[2 * i]) / 1e6))
    return out
  }

  // A fresh garden: no fruit bodies, no slime mould.
  private clearForReset(enc: GPUCommandEncoder) {
    for (const n of ['fruits', 'slimeAgents', 'slimeMap']) enc.clearBuffer(this.b[n])
    enc.clearBuffer(this.b.grid, GRID_SLIME * 4, SLIME_WORDS * 4)
  }

  // The slime mould's update (slimesim.wgsl).
  private encodeSlime(src: Passes) {
    const k = this.kernels(src)
    const g = this.groups[`slime${this.fieldPar}`]
    k('slimeHatch', [1], g)
    k('slimeMove', [SLIME_MAX / 64], g)
    k('slimeTrail', [Math.ceil(SLIME_RES[0] / 16), Math.ceil(SLIME_RES[1] / 16)], g)
    k('slimeColumns', [Math.ceil(SLIME_COLS / 64)], g)
    k('slimePop', [SLIME_MAX / 64], g)
  }

  // A slime mould hatches over here as `agents` agents (shown at once while
  // paused, if set down by hand).
  addSlime(at: V3, agents = 3000, shown = true) {
    this.slimeReq = [...octEncode(at), agents]
    this.slimeShow = shown
  }

  // The simulation's uniforms for a frame of `steps`, with the pending
  // inoculations (returned) and a pending reset.
  private writeSim(steps: number, brush: Brush | null = null) {
    const d = this.device
    // (held: nothing is set down)
    const held = this.holds > 0
    const inoc = held ? [] : this.pendingInoc.splice(0, MAX_INOC)
    const simU = new ArrayBuffer(SIMU_BYTES)
    const uu = new Uint32Array(simU)
    const uf = new Float32Array(simU)
    uu.set([inoc.length, 0, steps, 0], 0)
    inoc.forEach((q, i) => {
      uf.set([q.x, q.y, q.z, q.org], 8 + i * 8)
      uf.set([q.exact ? 2 : q.onSurface ? 1 : 0, 0, 0, 0], 12 + i * 8)
    })
    if (brush) {
      uf.set([...brush.at, brush.radius], 520)
      uf.set([brush.op, brush.strength, 0, 0], 524)
    }
    // (a slime mould is set down by its next update: held till a frame that
    // runs one, or shows one while paused)
    const s0 = this.pendingReset ? 0 : this.simStep
    const slimeDue = steps > 0 ? Math.ceil(s0 / SLIME_EVERY) * SLIME_EVERY < s0 + steps : this.slimeShow
    if (this.slimeReq && !held && slimeDue) {
      uf.set([...this.slimeReq, ++this.slimeReqs], 528)
      this.slimeReq = null
      this.slimeFrame = this.slimeShow
    }
    d.queue.writeBuffer(this.b.simU, 0, simU)
    if (this.pendingReset) {
      d.queue.writeBuffer(this.b.counters, 0, new Uint32Array(8))
      d.queue.writeBuffer(this.b.args, 0, new Uint32Array([0, 1, 1, 0]))
      d.queue.writeBuffer(this.b.pools, 0, new Int32Array(MAX_ORGS * 4))
    }
    return inoc
  }

  // The next frame, read back instead of shown.
  // The next frame drawn, read back; `scale` > 1: a supersampled frame, each
  // scale x scale block of it averaged into one pixel.
  grabNext(scale = 1): Promise<Grab> {
    return new Promise((resolve) => (this.grabRequest = { resolve, scale }))
  }

  // The next frame starts its temporal history afresh (after a jump).
  resetHistory() {
    this.flushHistory = true
    this.lensFrame = 0
  }

  // Frames count from 0 again (their noise and jitter follow the count), so
  // a video drawn twice comes out the same.
  restartFrames() {
    this.frame = 0
    this.resetHistory()
  }

  // The steps run since the garden began.
  get step() {
    return this.pendingReset ? 0 : this.simStep
  }

  // Advance the garden `steps` without drawing it, kept on the way as it is
  // when drawn; laying down `strokes` (in order) at their steps, before
  // those steps run; in runs of `chunks` steps if given (as it ran before:
  // a garden run again grows as it did), else of up to 250.
  async simulate(steps: number, strokes: { step: number; brush: Brush }[] = [], chunks?: number[]) {
    const d = this.device
    let si = 0
    let ci = 0
    while (strokes[si] && strokes[si].step < this.step) si++
    while (steps > 0 || strokes[si]?.step === this.step) {
      if (this.keeping) await this.keeping
      const at = this.step
      const stroke = strokes[si]?.step === at ? strokes[si++].brush : null
      let n = stroke ? 0 : Math.min(steps, chunks ? (chunks[ci++] ?? steps) : 250, this.keepAt - at)
      if (strokes[si] && !chunks) n = Math.min(n, strokes[si].step - at)
      const inoc = this.writeSim(n, stroke)
      const enc = d.createCommandEncoder()
      if (this.pendingReset) this.clearForReset(enc)
      const planted = at === 0 ? this.sampleIfDue(enc) : null
      const p = enc.beginComputePass({ label: 'sim' })
      this.encodeSim(onePass(p), n, inoc.length > 0, stroke !== null)
      p.end()
      this.pendingReset = false
      const keep = this.keepIfDue(enc)
      const sample = this.sampleIfDue(enc)
      d.queue.submit([enc.finish()])
      if (keep) this.keeping = this.readKeep()
      if (planted) this.readSample(planted)
      if (sample) this.readSample(sample)
      this.onBetween?.()
      steps -= n
    }
  }

  // The garden as it is now, between two steps: what the steps after it need
  // (nothing that can be drawn again from it), off the GPU, with the
  // renderer's own count of it. Copied aside in one go, then read back
  // while the garden runs on.
  async captureGarden(): Promise<GardenState> {
    while (this.keeping) await this.keeping
    while (this.capturing) await this.capturing
    const d = this.device
    const b = this.b
    const names = ['cells', 'grid', 'fruits', 'devField', 'slimeAgents', 'slimeMap', 'pools', 'counters', 'events']
    const field = this.fieldPar === 0 ? b.fieldA : b.fieldB
    const cb = (this.captureBufs ??= Object.fromEntries(
      [...names, 'field'].map((n) => [n, d.createBuffer({ size: (n === 'field' ? field : b[n]).size, usage: BUF.COPY_DST | BUF.MAP_READ })]),
    ))
    const enc = d.createCommandEncoder()
    for (const n of names) enc.copyBufferToBuffer(b[n], 0, cb[n], 0, b[n].size)
    enc.copyBufferToBuffer(field, 0, cb.field, 0, field.size)
    d.queue.submit([enc.finish()])
    const own = {
      step: this.step, keepAt: this.keepAt, fieldPar: this.fieldPar, slimeReqs: this.slimeReqs,
      pendingInoc: this.pendingInoc.map((q) => ({ ...q })), slimeReq: this.slimeReq,
    }
    const read = (async () => {
      await Promise.all([...names, 'field'].filter((n) => n !== 'cells').map((n) => cb[n].mapAsync(GPUMapMode.READ)))
      const take = (n: string) => {
        const a = cb[n].getMappedRange().slice(0)
        cb[n].unmap()
        return a
      }
      const counters = take('counters')
      const cells = new Uint32Array(counters)[0]
      const size = Math.max(cells, 1) * CELL_BYTES
      await cb.cells.mapAsync(GPUMapMode.READ, 0, size)
      const cellData = cb.cells.getMappedRange(0, size).slice(0)
      cb.cells.unmap()
      const grid = take('grid')
      // (only the development fields a head holds: the rest start afresh)
      const pats = new Uint32Array(grid, GRID_DEV * 4, PAT_SLOTS)
      const devSlots = [...pats.keys()].filter((k) => pats[k] !== 0)
      const devAll = new Float32Array(take('devField'))
      const dev = new Float32Array(devSlots.length * PAT_RES * PAT_RES)
      devSlots.forEach((slot, k) => dev.set(devAll.subarray(slot * PAT_RES * PAT_RES, (slot + 1) * PAT_RES * PAT_RES), k * PAT_RES * PAT_RES))
      return {
        ...own, cells: cellData, grid, field: take('field'), fruits: take('fruits'), devSlots, dev: dev.buffer,
        slimeAgents: take('slimeAgents'), slimeMap: take('slimeMap'), pools: take('pools'), counters, events: take('events'),
      } as GardenState
    })()
    this.capturing = read
    try {
      return await read
    } finally {
      this.capturing = null
    }
  }

  // Put the garden back as a capture had it.
  restoreGarden(st: GardenState) {
    const q = this.device.queue
    const b = this.b
    q.writeBuffer(b.cells, 0, st.cells)
    q.writeBuffer(b.grid, 0, st.grid)
    this.fieldPar = st.fieldPar
    q.writeBuffer(st.fieldPar === 0 ? b.fieldA : b.fieldB, 0, st.field)
    q.writeBuffer(b.fruits, 0, st.fruits)
    const slotBytes = PAT_RES * PAT_RES * 4
    st.devSlots.forEach((slot, k) => q.writeBuffer(b.devField, slot * slotBytes, st.dev, k * slotBytes, slotBytes))
    q.writeBuffer(b.slimeAgents, 0, st.slimeAgents)
    q.writeBuffer(b.slimeMap, 0, st.slimeMap)
    q.writeBuffer(b.pools, 0, st.pools)
    q.writeBuffer(b.counters, 0, st.counters)
    q.writeBuffer(b.events, 0, st.events)
    // (the dispatch of the cells in use, as the step's end left it)
    const n = new Uint32Array(st.counters)[0]
    q.writeBuffer(b.args, 0, new Uint32Array([Math.ceil(n / 256), 1, 1, 0, 4, n, 0, 0]))
    this.simStep = st.step
    this.keepAt = st.keepAt
    this.sampleAt = st.step + 1
    this.slimeReqs = st.slimeReqs
    this.pendingInoc = st.pendingInoc.map((q) => ({ ...q }))
    this.slimeReq = st.slimeReq
    this.pendingReset = false
    this.keeping = null
    // (what was read back of the garden before is not of this one)
    this.statsGen++
    this.statsPending = false
    const c = new Uint32Array(st.counters)
    this.stats = { ...this.stats, cells: c[0] - c[3], step: c[5] }
    this.repaint = true
    this.resetHistory()
  }

  render(fi: FrameInput) {
    if (!this.sized) return
    const d = this.device
    const W = this.width
    const H = this.height
    this.frame++

    const lens = fi.lens ?? null
    const k = (this.frame % 64) + 1
    let jx = halton(k, 2) - 0.5
    let jy = halton(k, 3) - 0.5
    let ppu = H / fi.height
    let eye = sub(fi.target, scale(fi.basis.fwd, 600))
    // (perspective: `height` is the height of the frame at the target)
    const focal = lens ? H / 2 / Math.tan(lens.fov / 2) : 0
    if (lens) {
      eye = sub(fi.target, scale(fi.basis.fwd, (fi.height / 2) / Math.tan(lens.fov / 2)))
      ppu = focal / lens.focus
      // this frame's point on the lens; the image of the focal plane stays put
      const j = lensPoint(this.lensFrame++ % LENS_POINTS)
      const r = Math.sqrt((j + 0.5) / LENS_POINTS) * lens.aperture
      const a = j * 2.39996323
      const lx = r * Math.cos(a)
      const ly = r * Math.sin(a)
      eye = add(eye, add(scale(fi.basis.right, lx), scale(fi.basis.up, ly)))
      jx += lx * ppu
      jy -= ly * ppu
    }
    const view = new ArrayBuffer(VIEW_BYTES)
    const f = new Float32Array(view)
    const u = new Uint32Array(view)
    f.set([eye[0], eye[1], eye[2], ppu], 0)
    f.set([...fi.basis.right, W], 4)
    f.set([...fi.basis.up, H], 8)
    f.set([...fi.basis.fwd, 1 / ppu], 12)
    // (post.wgsl: the extra blur, as a share of how far out of focus)
    const gaps = lens && lens.aperture > 0 ? 0.9 / Math.sqrt(Math.min(lens.samples, LENS_POINTS)) : 0
    f.set([jx, jy, focal, gaps], 16)
    u.set([this.frame, 0, 0, 0], 20)
    d.queue.writeBuffer(this.b.view, 0, view)
    d.queue.writeBuffer(this.b.prevView, 0, this.prevView ?? view)
    this.prevView = view

    const sc = new ArrayBuffer(SCENE_BYTES)
    const sf = new Float32Array(sc)
    const su = new Uint32Array(sc)
    const steps = this.stepsNow(fi.simSteps)
    // the sun where it stands as this frame's steps end, and its light
    // (the frame shows the garden then)
    const shown = this.step + steps
    const vis = this.sunVisFor(shown, steps)
    // (the eye opens up as the light goes: by starlight, about six times)
    const vd = Math.min(1, Math.max(0, fi.viewDay))
    sf.set([...sunDir(shown), fi.sunRadius], 0)
    sf.set([...SUN_COLOR, 0], 4)
    sf.set([...ZEN, 0], 8)
    sf.set([...HOR, 0], 12)
    sf.set([fi.exposure * Math.min(6, (vd + 0.012 * (1 - vd)) ** -0.4), performance.now() / 1000, fi.debug, fi.bounces], 16)
    sf.set([vis.blend, 0, 0, 1 - vd], 36)
    su.set([fi.selected + 1, fi.second + 1, this.frame, 0], 20)
    // (in focus: the ground nearest the eye when looking at the planet whole,
    // or what is looked at)
    const focus = 600 - Math.max(0, PLANET_R - Math.hypot(...fi.target))
    sf.set([this.flushHistory ? 1 : fi.alphaFloor, lens ? lens.aperture : fi.aperture, lens ? lens.focus : focus, fi.fineMoss ? 1 : 0], 24)
    this.flushHistory = false
    // (no stroke while the garden is kept or held: it lands between steps)
    const brush = this.keeping || this.holds > 0 ? null : fi.brush
    sf.set([steps > 0 ? 0.05 : 0, steps, fi.overlay, fi.accumulate ? 1 : 0], 28)
    sf.set(fi.cursor ?? [0, 0, 0, 0], 32)
    d.queue.writeBuffer(this.b.scene, 0, sc)

    const inoc = this.writeSim(steps, brush)
    if (brush) this.onStroke?.(this.step, brush)
    const s = this.sized
    const c = this.cur
    const enc = d.createCommandEncoder()
    if (this.pendingReset) this.clearForReset(enc)
    const tsw = (name: PassName): GPUComputePassTimestampWrites | undefined => {
      if (!this.querySet) return undefined
      const i = PASSES.indexOf(name)
      return { querySet: this.querySet, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 }
    }
    const pass = (name: PassName, fn: (p: GPUComputePassEncoder) => void) => {
      const p = enc.beginComputePass({ label: name, timestampWrites: tsw(name) })
      fn(p)
      p.end()
    }
    // (the garden as planted, before its first step)
    const planted = this.step === 0 ? this.sampleIfDue(enc) : null
    pass('sim', (p) => this.encodeSim(onePass(p), steps, inoc.length > 0, brush !== null))
    this.pendingReset = false
    const keep = this.keepIfDue(enc)
    const sample = this.sampleIfDue(enc)
    const tp = enc.beginRenderPass({
      label: 'tissue',
      colorAttachments: [
        { view: s.tGeo.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 1, 0, 1e9] },
        { view: s.tId.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
      ],
      depthStencilAttachment: { view: s.tDepth.createView(), depthLoadOp: 'clear', depthStoreOp: 'discard', depthClearValue: 1 },
      timestampWrites: this.querySet
        ? { querySet: this.querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 }
        : undefined,
    })
    tp.setPipeline(this.tissuePipe)
    tp.setBindGroup(0, this.groups.tissue)
    tp.drawIndirect(this.b.args, 16)
    tp.setPipeline(this.fruitPipe)
    tp.setBindGroup(0, this.groups.fruits)
    tp.draw(4, MAX_FRUITS * 2)
    tp.end()
    this.bakeSunVis(enc, vis.bake)
    pass('shade', (p) => {
      p.setPipeline(this.pipes.shade)
      p.setBindGroup(0, this.groups[`shade${c}`])
      p.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8))
    })
    pass('taa', (p) => {
      p.setPipeline(this.pipes.temporal)
      p.setBindGroup(0, this.groups[`temporal${c}`])
      p.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8))
    })
    pass('post', (p) => {
      p.setPipeline(this.pipes.post)
      p.setBindGroup(0, this.groups[`post${c}`])
      p.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8))
    })
    // a grabbed frame is drawn off screen and read back instead (it does not
    // depend on the page being shown)
    const grab = this.grabRequest
    this.grabRequest = null
    if (grab && (this.grabTex?.width !== W || this.grabTex?.height !== H)) {
      this.grabTex?.destroy()
      this.grabTex = d.createTexture({ size: [W, H], format: this.format, usage: TEX.RENDER_ATTACHMENT | TEX.COPY_SRC | TEX.TEXTURE_BINDING })
    }
    const target = grab ? this.grabTex! : this.ctx.getCurrentTexture()
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    })
    rp.setPipeline(this.presentPipe)
    rp.setBindGroup(0, this.groups[`present${c}`])
    rp.draw(3)
    rp.end()
    this.cur = 1 - c
    const gs = grab?.scale ?? 1
    const [GW, GH] = [Math.floor(W / gs), Math.floor(H / gs)]
    let grabFrom = this.grabTex
    if (grab && gs > 1) {
      if (this.grabSmall?.width !== GW || this.grabSmall?.height !== GH) {
        this.grabSmall?.destroy()
        this.grabSmall = d.createTexture({ size: [GW, GH], format: 'rgba8unorm', usage: TEX.STORAGE_BINDING | TEX.COPY_SRC })
      }
      const p = enc.beginComputePass({ label: 'downsample' })
      p.setPipeline(this.pipes.downsample)
      p.setBindGroup(0, this.group(this.pipes.downsample.getBindGroupLayout(0), [this.grabTex!.createView(), this.grabSmall.createView()]))
      p.dispatchWorkgroups(Math.ceil(GW / 8), Math.ceil(GH / 8))
      p.end()
      grabFrom = this.grabSmall
    }
    const grabRow = Math.ceil((GW * 4) / 256) * 256
    const grabBuf = grab ? d.createBuffer({ size: grabRow * GH, usage: BUF.COPY_DST | BUF.MAP_READ }) : null
    if (grabBuf) enc.copyTextureToBuffer({ texture: grabFrom! }, { buffer: grabBuf, bytesPerRow: grabRow }, [GW, GH])

    // (steps run without drawing are read on the next frame drawn, however
    // many are drawn of each, so a scripted run keeps its garden the same)
    const doStats = !this.statsPending && this.statFrame++ % 10 === 0
    if (doStats) {
      enc.copyBufferToBuffer(this.b.counters, 0, this.statRead, 0, 32)
      if (this.querySet && this.queryResolve && this.queryRead) {
        enc.resolveQuerySet(this.querySet, 0, PASSES.length * 2, this.queryResolve, 0)
        enc.copyBufferToBuffer(this.queryResolve, 0, this.queryRead, 0, PASSES.length * 16)
      }
    }
    const pr = !this.pickPending ? this.pickRequest : null
    const px = pr ? Math.min(W - 1, Math.max(0, Math.floor(pr.x))) : 0
    const py = pr ? Math.min(H - 1, Math.max(0, Math.floor(pr.y))) : 0
    if (pr) {
      enc.copyBufferToBuffer(s.ids, (py * W + px) * 4, this.pickRead, 0, 4)
      enc.copyTextureToBuffer({ texture: s.geo[c], origin: { x: px, y: py } }, { buffer: this.pickRead, offset: 256, bytesPerRow: 256 }, [1, 1])
    }
    d.queue.submit([enc.finish()])

    if (grab && grabBuf) {
      grabBuf.mapAsync(GPUMapMode.READ).then(() => {
        const data = new Uint8Array(grabBuf.getMappedRange().slice(0))
        grabBuf.destroy()
        grab.resolve({ data, width: GW, height: GH, row: grabRow, format: gs > 1 ? 'rgba8unorm' : this.format })
      })
    }
    if (keep) this.keeping = this.readKeep()
    if (planted) this.readSample(planted)
    if (sample) this.readSample(sample)
    if (doStats) void this.readStats()
    this.onBetween?.()
    if (pr) {
      this.pickRequest = null
      this.pickPending = true
      const ox = (px + 0.5 - 0.5 * W) / ppu
      const oy = (py + 0.5 - 0.5 * H) / ppu
      const ro = lens ? eye : [0, 1, 2].map((k) => eye[k] + fi.basis.right[k] * ox - fi.basis.up[k] * oy)
      const rd = lens
        ? normalize(add(scale(fi.basis.fwd, focal), add(scale(fi.basis.right, px + 0.5 - 0.5 * W), scale(fi.basis.up, -(py + 0.5 - 0.5 * H)))))
        : fi.basis.fwd
      this.pickRead.mapAsync(GPUMapMode.READ).then(() => {
        const range = this.pickRead.getMappedRange()
        const v = new Uint32Array(range, 0, 1)[0]
        const t = new Float32Array(range, 256, 4)[3]
        this.pickRead.unmap()
        this.pickPending = false
        const point = t < 1e5 ? (ro.map((o, k) => o + rd[k] * t) as V3) : null
        pr.resolve({ slot: v - 1, point })
      })
    }
  }

  // Dev: the spore fields (airborne, settled) of the current field buffer.
  async readSpores() {
    const d = this.device
    const src = this.fieldPar === 0 ? this.b.fieldA : this.b.fieldB
    const rb = d.createBuffer({ size: F4N * 16, usage: BUF.COPY_DST | BUF.MAP_READ })
    const enc = d.createCommandEncoder()
    enc.copyBufferToBuffer(src, F4N * 16, rb, 0, F4N * 16)
    d.queue.submit([enc.finish()])
    await rb.mapAsync(GPUMapMode.READ)
    const f = new Float32Array(rb.getMappedRange().slice(0))
    rb.destroy()
    let air = 0, bed = 0, maxAir = 0, maxBed = 0, n = 0
    for (let i = 0; i < F4N; i++) {
      air += f[i * 4]
      bed += f[i * 4 + 1]
      maxAir = Math.max(maxAir, f[i * 4])
      maxBed = Math.max(maxBed, f[i * 4 + 1])
      if (f[i * 4] > 0.01) n++
    }
    return { air, bed, maxAir, maxBed, voxelsWithAir: n }
  }

  // Dev: the ground's height above PLANET_R (the ground map's first channel),
  // GROUND_RES[0] x GROUND_RES[1], row by row (world.wgsl groundUV).
  async readGround() {
    const d = this.device
    const row = GROUND_RES[0] * 8
    const rb = d.createBuffer({ size: row * GROUND_RES[1], usage: BUF.COPY_DST | BUF.MAP_READ })
    const enc = d.createCommandEncoder()
    enc.copyTextureToBuffer({ texture: this.ground }, { buffer: rb, bytesPerRow: row }, GROUND_RES)
    d.queue.submit([enc.finish()])
    await rb.mapAsync(GPUMapMode.READ)
    const h = new Uint16Array(rb.getMappedRange().slice(0))
    rb.destroy()
    const half = (b: number) => {
      const e = (b >> 10) & 31
      const m = b & 1023
      const v = e === 0 ? m / 1024 / 16384 : (1 + m / 1024) * 2 ** (e - 15)
      return b & 0x8000 ? -v : v
    }
    const out = new Float32Array(GROUND_RES[0] * GROUND_RES[1])
    for (let i = 0; i < out.length; i++) out[i] = half(h[i * 4])
    return { heights: out, size: GROUND_RES }
  }

  // Dev: fruit bodies as raw floats (FRUIT_BYTES / 4 per fruit).
  async readFruits() {
    const d = this.device
    const rb = d.createBuffer({ size: MAX_FRUITS * FRUIT_BYTES, usage: BUF.COPY_DST | BUF.MAP_READ })
    const enc = d.createCommandEncoder()
    enc.copyBufferToBuffer(this.b.fruits, 0, rb, 0, MAX_FRUITS * FRUIT_BYTES)
    d.queue.submit([enc.finish()])
    await rb.mapAsync(GPUMapMode.READ)
    const ab = rb.getMappedRange().slice(0)
    rb.destroy()
    return { f: new Float32Array(ab), u: new Uint32Array(ab), stride: FRUIT_BYTES / 4 }
  }

  // Dev: the last antialiased frame (row-padded) and the organism ids.
  async capture() {
    const d = this.device
    const s = this.sized!
    const W = this.width
    const H = this.height
    const row = Math.ceil((W * 16) / 256) * 256
    const cb = d.createBuffer({ size: row * H, usage: BUF.COPY_DST | BUF.MAP_READ })
    const ib = d.createBuffer({ size: W * H * 4, usage: BUF.COPY_DST | BUF.MAP_READ })
    const arow = Math.ceil((W * 8) / 256) * 256
    const ab = d.createBuffer({ size: arow * H, usage: BUF.COPY_DST | BUF.MAP_READ })
    const enc = d.createCommandEncoder()
    enc.copyTextureToBuffer({ texture: s.alb }, { buffer: ab, bytesPerRow: arow }, [W, H])
    enc.copyTextureToBuffer({ texture: s.col[1 - this.cur] }, { buffer: cb, bytesPerRow: row }, [W, H])
    enc.copyBufferToBuffer(s.ids, 0, ib, 0, W * H * 4)
    d.queue.submit([enc.finish()])
    await Promise.all([cb.mapAsync(GPUMapMode.READ), ib.mapAsync(GPUMapMode.READ), ab.mapAsync(GPUMapMode.READ)])
    const col = new Float32Array(cb.getMappedRange().slice(0))
    const ids = new Uint32Array(ib.getMappedRange().slice(0))
    const alb = new Uint16Array(ab.getMappedRange().slice(0))
    cb.destroy()
    ib.destroy()
    ab.destroy()
    return { col, ids, alb, W, H, row: row / 4, arow: arow / 2 }
  }

  // The frame's stats (cells, step, GPU time per pass), now and then.
  private async readStats() {
    this.statsPending = true
    const gen = this.statsGen
    const qr = this.queryRead
    await Promise.all([this.statRead.mapAsync(GPUMapMode.READ), ...(qr ? [qr.mapAsync(GPUMapMode.READ)] : [])])
    const cnt = new Uint32Array(this.statRead.getMappedRange().slice(0))
    this.statRead.unmap()
    const gpuMs: RenderStats['gpuMs'] = {}
    let total = 0
    if (qr) {
      const t = new BigInt64Array(qr.getMappedRange().slice(0))
      qr.unmap()
      PASSES.forEach((n, i) => {
        const ms = Number(t[i * 2 + 1] - t[i * 2]) / 1e6
        if (ms >= 0 && ms < 1000) {
          gpuMs[n] = ms
          total += ms
        }
      })
    }
    if (gen === this.statsGen) this.stats = { gpuMs, gpuTotal: total, cells: cnt[0] - cnt[3], step: cnt[5] }
    this.statsPending = false
  }

  // A keeping step's readback: the garden's ecology, then its keeping.
  private async readKeep() {
    const gen = this.statsGen
    await Promise.all([
      this.countRead.mapAsync(GPUMapMode.READ),
      this.poolRead.mapAsync(GPUMapMode.READ),
      this.eventRead.mapAsync(GPUMapMode.READ),
      this.slimeRead.mapAsync(GPUMapMode.READ),
    ])
    const cnt = new Uint32Array(this.countRead.getMappedRange().slice(0))
    this.countRead.unmap()
    const pools = new Int32Array(this.poolRead.getMappedRange().slice(0))
    this.poolRead.unmap()
    const ev = new Float32Array(this.eventRead.getMappedRange().slice(0))
    this.eventRead.unmap()
    const sl = new Int32Array(this.slimeRead.getMappedRange().slice(0))
    this.slimeRead.unmap()
    // (of the garden before a reset: not kept)
    if (gen !== this.statsGen) return
    const eco = this.ecology
    // (live agents: the count of whichever update finished last)
    eco.slime = { mode: sl[4], agents: Math.max(sl[1], sl[2]) }
    for (let o = 0; o < MAX_ORGS; o++) {
      eco.food[o] = pools[o * 4] / 1000
      eco.cells[o] = pools[o * 4 + 1]
      // (living fruit bodies in the low 16 bits, dying ones above)
      eco.fruits[o] = (pools[o * 4 + 3] & 0xffff) + (pools[o * 4 + 3] >> 16)
    }
    // (in a fixed order, whatever order the GPU wrote them in)
    const nEv = Math.min(cnt[2], MAX_EVENTS)
    const births = []
    for (let e = 0; e < nEv; e++) {
      const ids = Math.round(ev[e * 4 + 3])
      births.push({ x: ev[e * 4], y: ev[e * 4 + 1], z: ev[e * 4 + 2], a: (ids & 127) - 1, b: ((ids >> 7) & 127) - 1 })
    }
    births.sort((p, q) => p.x - q.x || p.z - q.z || p.y - q.y || p.a - q.a || p.b - q.b)
    eco.births.push(...births)
    this.stats = { ...this.stats, cells: cnt[0] - cnt[3], step: cnt[5] }
    this.onKeep?.()
    this.keeping = null
  }
}
