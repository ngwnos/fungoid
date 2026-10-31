import { compile } from './shaders'
import { ROW, type Samples } from '../history'

// The timeline's graph (graph.wgsl), drawn on the renderer's device into a
// canvas of its own: the samples go up as they come (only the new rows), and
// each frame a column a pixel is gathered from them, then drawn.

// what the graph shows, by its index in the shader (mycelium, mushrooms,
// damp, food stored, colonies, slime mould)
export const GRAPH_RGB: [number, number, number][] = [
  [217, 89, 38],
  [25, 158, 112],
  [57, 135, 229],
  [201, 133, 0],
  [144, 133, 233],
  [213, 81, 129],
]
// the columns start this far into the canvas (pixels), for the playhead's
// knob at the ends (the canvas reaches as far out beyond them: styles.css)
export const GRAPH_PAD = 7
const COL = 8 * 16
const BUF = GPUBufferUsage

export interface GraphView {
  head: number
  first: number
  at: number
  // the step under the pointer (-1: none)
  hover: number
  hidden: number
  focus: number
  // the chart above the bar, in CSS pixels
  chart: number
  // where on the planet the nights, damp and seasons are of (a direction)
  look: [number, number, number]
}

export class Graph {
  private uni: GPUBuffer
  private peaks: GPUBuffer
  private scales: GPUBuffer
  private easeGroup: GPUBindGroup
  private drawn = 0
  private rows: GPUBuffer | null = null
  private cps: GPUBuffer | null = null
  private bins: GPUBuffer | null = null
  private binGroup: GPUBindGroup | null = null
  private drawGroup: GPUBindGroup | null = null
  private uploaded = 0
  private cpCount = 0
  private data = new ArrayBuffer(176)
  private f = new Float32Array(this.data)
  private u = new Uint32Array(this.data)
  private i = new Int32Array(this.data)

  private constructor(
    private device: GPUDevice,
    private ctx: GPUCanvasContext,
    private binPipe: GPUComputePipeline,
    private easePipe: GPUComputePipeline,
    private drawPipe: GPURenderPipeline,
  ) {
    const d = device
    this.uni = d.createBuffer({ label: 'graph', size: 176, usage: BUF.UNIFORM | BUF.COPY_DST })
    this.peaks = d.createBuffer({ label: 'graph peaks', size: 32, usage: BUF.STORAGE | BUF.COPY_DST | BUF.COPY_SRC })
    this.scales = d.createBuffer({ label: 'graph scales', size: 64, usage: BUF.STORAGE | BUF.COPY_SRC })
    this.easeGroup = d.createBindGroup({
      layout: easePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 4, resource: { buffer: this.peaks } },
        { binding: 7, resource: { buffer: this.scales } },
      ],
    })
    GRAPH_RGB.forEach((c, k) => this.f.set([...c.map((x) => x / 255), 1], 20 + k * 4))
  }

  static async create(device: GPUDevice, canvas: HTMLCanvasElement, format: GPUTextureFormat) {
    const ctx = canvas.getContext('webgpu')!
    ctx.configure({ device, format, alphaMode: 'premultiplied' })
    const module = await compile(device, 'graph')
    const binPipe = await device.createComputePipelineAsync({ label: 'graph bin', layout: 'auto', compute: { module, entryPoint: 'bin' } })
    const easePipe = await device.createComputePipelineAsync({ label: 'graph ease', layout: 'auto', compute: { module, entryPoint: 'ease' } })
    const drawPipe = await device.createRenderPipelineAsync({
      label: 'graph draw',
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    })
    return new Graph(device, ctx, binPipe, easePipe, drawPipe)
  }

  // room for the samples, checkpoints and columns (anew, all of it, when
  // they outgrow it)
  private room(samples: Samples, cps: number, cols: number) {
    const d = this.device
    let changed = false
    const need = Math.max(1024, samples.count) * ROW * 4
    if (!this.rows || this.rows.size < need) {
      this.rows?.destroy()
      this.rows = d.createBuffer({ label: 'graph samples', size: 2 ** Math.ceil(Math.log2(need)), usage: BUF.STORAGE | BUF.COPY_DST })
      this.uploaded = 0
      changed = true
    }
    if (!this.cps || this.cps.size < Math.max(256, cps * 4)) {
      this.cps?.destroy()
      this.cps = d.createBuffer({ label: 'graph checkpoints', size: 2 ** Math.ceil(Math.log2(Math.max(256, cps * 4))), usage: BUF.STORAGE | BUF.COPY_DST })
      changed = true
    }
    if (!this.bins || this.bins.size < cols * COL) {
      this.bins?.destroy()
      this.bins = d.createBuffer({ label: 'graph columns', size: 2 ** Math.ceil(Math.log2(Math.max(1024, cols) * COL)), usage: BUF.STORAGE | BUF.COPY_SRC })
      changed = true
    }
    if (changed || !this.binGroup) {
      const entry = (binding: number, buffer: GPUBuffer) => ({ binding, resource: { buffer } })
      this.binGroup = d.createBindGroup({
        layout: this.binPipe.getBindGroupLayout(0),
        entries: [entry(0, this.uni), entry(1, this.rows), entry(2, this.cps), entry(3, this.bins), entry(4, this.peaks)],
      })
      this.drawGroup = d.createBindGroup({
        layout: this.drawPipe.getBindGroupLayout(0),
        entries: [entry(0, this.uni), entry(5, this.bins), entry(8, this.scales)],
      })
    }
    return changed
  }

  // `cpsChanged`: the checkpoints are not as they were last drawn
  draw(samples: Samples, checkpoints: { step: number; action: boolean }[], cpsChanged: boolean, v: GraphView) {
    const now = performance.now()
    // (a garden begun again: its scales start where it does)
    const dt = samples.count < this.uploaded && samples.count <= 1 ? -1 : this.drawn ? (now - this.drawn) / 1000 : 0
    this.drawn = now
    const canvas = this.ctx.canvas as HTMLCanvasElement
    const dpr = window.devicePixelRatio || 1
    const W = Math.max(1, Math.round(canvas.clientWidth * dpr))
    const H = Math.max(1, Math.round(canvas.clientHeight * dpr))
    if (canvas.width !== W || canvas.height !== H) {
      canvas.width = W
      canvas.height = H
    }
    const pad = Math.round(GRAPH_PAD * dpr)
    const cols = Math.max(1, W - 2 * pad)
    const q = this.device.queue
    const grown = this.room(samples, checkpoints.length, cols)
    // the new samples (all of them into a new buffer)
    const from = Math.min(this.uploaded, samples.dirty)
    if (samples.count > from) q.writeBuffer(this.rows!, from * ROW * 4, samples.buffer, from * ROW * 4, (samples.count - from) * ROW * 4)
    this.uploaded = samples.count
    samples.dirty = samples.count
    if (cpsChanged || grown) {
      const a = new Uint32Array(Math.max(1, checkpoints.length))
      checkpoints.forEach((c, k) => (a[k] = (c.step | (c.action ? 0x80000000 : 0)) >>> 0))
      q.writeBuffer(this.cps!, 0, a)
      this.cpCount = checkpoints.length
    }
    const head = Math.max(1, v.head)
    const perPx = head / cols
    const { f, u, i } = this
    f[0] = W
    f[1] = H
    f[2] = Math.round(v.chart * dpr)
    f[3] = dpr
    f[4] = perPx
    u[5] = cols
    f[6] = pad
    u[7] = samples.count
    u[8] = v.head
    u[9] = v.first
    f[10] = pad + v.at / perPx
    f[11] = v.hover < 0 ? -1 : pad + v.hover / perPx
    u[12] = this.cpCount
    u[13] = v.hidden
    i[14] = v.focus
    f[15] = dt
    f.set([...v.look, 0], 16)
    q.writeBuffer(this.uni, 0, this.data)
    const enc = this.device.createCommandEncoder({ label: 'graph' })
    enc.clearBuffer(this.peaks)
    const cp = enc.beginComputePass({ label: 'graph bin' })
    cp.setPipeline(this.binPipe)
    cp.setBindGroup(0, this.binGroup!)
    cp.dispatchWorkgroups(Math.ceil(cols / 64))
    cp.setPipeline(this.easePipe)
    cp.setBindGroup(0, this.easeGroup)
    cp.dispatchWorkgroups(1)
    cp.end()
    const rp = enc.beginRenderPass({
      label: 'graph draw',
      colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
    })
    rp.setPipeline(this.drawPipe)
    rp.setBindGroup(0, this.drawGroup!)
    rp.draw(3)
    rp.end()
    q.submit([enc.finish()])
  }

  // (dev) the columns as last gathered: per column, each measure's least,
  // most, mean and whether it has one, then the column's own; the peaks;
  // and the scales as drawn
  async readColumns(n: number) {
    const d = this.device
    const bytes = Math.min(n * COL, this.bins!.size)
    const rb = d.createBuffer({ size: bytes + 96, usage: BUF.COPY_DST | BUF.MAP_READ })
    const enc = d.createCommandEncoder()
    enc.copyBufferToBuffer(this.bins!, 0, rb, 0, bytes)
    enc.copyBufferToBuffer(this.peaks, 0, rb, bytes, 32)
    enc.copyBufferToBuffer(this.scales, 0, rb, bytes + 32, 64)
    d.queue.submit([enc.finish()])
    await rb.mapAsync(GPUMapMode.READ)
    const a = new Float32Array(rb.getMappedRange().slice(0))
    rb.destroy()
    const o = bytes / 4
    return { cols: a.subarray(0, o), peaks: a.subarray(o, o + 8), scales: Array.from({ length: 8 }, (_, k) => Math.exp(a[o + 8 + k * 2])) }
  }

  destroy() {
    this.ctx.unconfigure()
    for (const b of [this.uni, this.peaks, this.scales, this.rows, this.cps, this.bins]) b?.destroy()
  }
}
