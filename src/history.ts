// The garden's history: a timeline that grows as the garden does, and can be
// gone back over. The garden runs the same each time from the same state, so
// the history keeps only checkpoints (the whole garden now and then, in the
// browser's own file store) and what is needed to run again from one: the
// runs of steps the garden took (what runs once a frame does so once a run)
// and the brush strokes laid down. Anything else done to the garden (a
// colony planted, a gene changed) is a checkpoint of its own just after.
// Going back loads the checkpoint before a step and runs on to it; playing
// on from the past runs the history again as it was; doing something there
// cuts off what came after.

import type { Brush, GardenState, Renderer } from './gpu/renderer'

// what the garden's keeper holds (App.tsx): its colonies and so on
export type GardenKeeping = Record<string, unknown>

export interface GardenIO {
  get: () => GardenKeeping
  set: (k: GardenKeeping) => void
  // how many colonies there are (for the timeline)
  colonies: () => number
}

// The garden sampled as it goes (each frame it moves on), for the timeline:
// a row of ROW words a sample, its step and then its measures (FIELDS), kept
// as they go to the GPU (gpu/graph.ts).
export const FIELDS = ['cells', 'mycelium', 'fruits', 'food', 'colonies', 'slime'] as const
export const ROW = 8

export class Samples {
  count = 0
  // the rows from here on changed since the timeline last took them
  dirty = 0
  buffer = new ArrayBuffer(ROW * 4 * 1024)
  private steps = new Uint32Array(this.buffer)
  private values = new Float32Array(this.buffer)

  push(step: number, values: number[]) {
    if ((this.count + 1) * ROW * 4 > this.buffer.byteLength) {
      const old = new Uint8Array(this.buffer)
      this.buffer = new ArrayBuffer(this.buffer.byteLength * 2)
      new Uint8Array(this.buffer).set(old)
      this.steps = new Uint32Array(this.buffer)
      this.values = new Float32Array(this.buffer)
    }
    const i = this.count++ * ROW
    this.steps[i] = step
    for (let k = 0; k < FIELDS.length; k++) this.values[i + 1 + k] = values[k]
  }
  step(i: number) {
    return this.steps[i * ROW]
  }
  // the row's FIELDS[k]
  value(i: number, k: number) {
    return this.values[i * ROW + 1 + k]
  }
  // the last row at or before `step` (-1: none)
  at(step: number) {
    let lo = 0
    let hi = this.count
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (this.steps[mid * ROW] <= step) lo = mid + 1
      else hi = mid
    }
    return lo - 1
  }
  // the rows after `step` go
  cut(step: number) {
    this.count = this.at(step) + 1
    this.dirty = Math.min(this.dirty, this.count)
  }
  clear() {
    this.count = 0
    this.dirty = 0
  }
}

export interface Checkpoint {
  step: number
  // made just after something was done to the garden by hand
  action: boolean
  name: string
  bytes: number
}

// how far apart the garden's own checkpoints are, in steps or in runs
const EVERY_STEPS = 3000
const EVERY_RUNS = 600
// the file store kept for checkpoints; the oldest go first
const BUDGET = 8 * 1024 ** 3
const DIR = 'fungoid-history'

// runs of steps, run-length: `count` runs of `size` steps from `from`
interface Runs {
  from: number
  size: number
  count: number
}

export class History {
  // the furthest step recorded, and the earliest one still to be had
  head = 0
  checkpoints: Checkpoint[] = []
  samples = new Samples()
  // counts changes to the checkpoints (the timeline shows them)
  version = 0
  private runs: Runs[] = []
  private strokes: { step: number; brush: Brush }[] = []
  private dir: Promise<FileSystemDirectoryHandle>
  private writing: Promise<void> = Promise.resolve()
  private bytes = 0
  // the store kept (no more than the browser allows)
  budget = BUDGET
  // (a checkpoint as soon as the garden begins)
  private sinceSteps = EVERY_STEPS
  private sinceRuns = 0
  private actionDue = false
  private busy = false
  // a seek or a replay under way
  private moving: Promise<void> | null = null
  private seekTo: { step: number; quick: boolean } | null = null
  // timelines restart with the garden; work for one gone is dropped
  private gen = 0
  onChange: (() => void) | null = null

  constructor(
    private r: Renderer,
    private garden: GardenIO,
  ) {
    this.dir = navigator.storage.getDirectory().then(async (root) => {
      // (a new session: nothing kept from the last)
      await root.removeEntry(DIR, { recursive: true }).catch(() => {})
      return root.getDirectoryHandle(DIR, { create: true })
    })
    r.onChunk = (from, n) => this.chunk(from, n)
    r.onStroke = (step, brush) => {
      if (this.live) this.strokes.push({ step, brush: { ...brush, at: [...brush.at] as Brush['at'] } })
    }
    r.onBetween = () => this.between()
    r.onReset = () => this.reset()
    r.onSample = (step, cnt, pools, slime) => this.sample(step, cnt, pools, slime)
    void navigator.storage.persist?.()
    void navigator.storage.estimate?.().then((e) => {
      if (e.quota) this.budget = Math.min(BUDGET, e.quota * 0.8)
    })
  }

  // the earliest step that can be gone back to
  get first() {
    return this.checkpoints[0]?.step ?? 0
  }
  // the step the garden is at
  get at() {
    return this.r.step
  }
  // at the head: what the garden does now is recorded
  get live() {
    return this.r.step >= this.head
  }
  // a seek or a replay under way
  get busyMoving() {
    return this.moving !== null
  }

  // A new timeline: the garden started over.
  reset() {
    this.gen++
    this.head = 0
    this.runs = []
    this.strokes = []
    this.samples.clear()
    this.sinceSteps = EVERY_STEPS
    this.sinceRuns = 0
    this.actionDue = false
    const old = this.checkpoints
    this.checkpoints = []
    this.bytes = 0
    for (const c of old) void this.remove(c.name)
    this.changedView()
  }

  // Something is about to be done to the garden by hand: in the past, what
  // came after is cut off.
  willChange() {
    if (!this.live) this.cut(this.r.step)
  }
  // Something was done to the garden by hand: a checkpoint just after.
  changed() {
    this.actionDue = true
  }

  // a sample of the garden came back
  private sample(step: number, cnt: Uint32Array, pools: Int32Array, slime: Int32Array) {
    const S = this.samples
    if (!this.live && step <= this.head) return
    if (S.count && step <= S.step(S.count - 1)) return
    let mycelium = 0
    let fruits = 0
    let food = 0
    for (let o = 0; o < pools.length; o += 4) {
      food += Math.max(pools[o], 0) / 1000
      mycelium += Math.max(pools[o + 1], 0)
      // (living fruit bodies in the low 16 bits, dying ones above)
      fruits += (pools[o + 3] & 0xffff) + (pools[o + 3] >> 16)
    }
    S.push(step, [cnt[0] - cnt[3], mycelium, fruits, food, this.garden.colonies(), Math.max(slime[1], slime[2])])
  }

  private changedView() {
    this.version++
    this.onChange?.()
  }

  // the garden took a run of steps
  private chunk(from: number, n: number) {
    if (from < this.head) return
    const last = this.runs[this.runs.length - 1]
    if (last && last.size === n && last.from + last.size * last.count === from) last.count++
    else this.runs.push({ from, size: n, count: 1 })
    this.head = from + n
    this.sinceSteps += n
    this.sinceRuns++
  }

  // between two runs (or frames): a checkpoint now, if one is due
  private between() {
    if (!this.live || this.busy || this.moving) return
    const due = this.actionDue || this.sinceSteps >= EVERY_STEPS || this.sinceRuns >= EVERY_RUNS
    // (not while the garden is being kept, nor before what was done by hand
    // has landed)
    if (!due || this.r.keepingNow || (this.actionDue && !this.r.settled)) return
    if (this.checkpoints.some((c) => c.step === this.r.step && !this.actionDue)) return
    const action = this.actionDue
    this.actionDue = false
    this.sinceSteps = 0
    this.sinceRuns = 0
    this.busy = true
    const gen = this.gen
    const keeping = this.garden.get()
    const step = this.r.step
    const st = this.r.captureGarden()
    void st
      .then((garden) => {
        if (gen !== this.gen) return
        return this.save(step, action, keeping, garden)
      })
      .finally(() => (this.busy = false))
  }

  private async save(step: number, action: boolean, keeping: GardenKeeping, st: GardenState) {
    const blob = await pack(keeping, st)
    const name = `cp-${step}-${action ? 'a' : 'p'}.bin`
    const write = async () => {
      const dir = await this.dir
      const f = await dir.getFileHandle(name, { create: true })
      const w = await f.createWritable()
      await w.write(blob)
      await w.close()
    }
    let ok = true
    this.writing = this.writing.then(async () => {
      try {
        await write()
      } catch {
        // (the store is full: the oldest goes, and once more)
        const old = this.checkpoints.length > 1 ? this.checkpoints.shift() : undefined
        if (old) {
          this.bytes -= old.bytes
          await this.remove(old.name)
        }
        await write().catch(() => (ok = false))
      }
    })
    await this.writing
    if (!ok) return
    // (a checkpoint at this step already: replaced)
    const same = this.checkpoints.findIndex((c) => c.step === step)
    if (same >= 0) {
      this.bytes -= this.checkpoints[same].bytes
      if (this.checkpoints[same].name !== name) void this.remove(this.checkpoints[same].name)
      this.checkpoints.splice(same, 1)
    }
    const at = this.checkpoints.findIndex((c) => c.step > step)
    const cp = { step, action, name, bytes: blob.size }
    if (at < 0) this.checkpoints.push(cp)
    else this.checkpoints.splice(at, 0, cp)
    this.bytes += blob.size
    // over the budget: the oldest go
    while (this.bytes > this.budget && this.checkpoints.length > 1) {
      const old = this.checkpoints.shift()!
      this.bytes -= old.bytes
      void this.remove(old.name)
    }
    this.changedView()
  }

  private async remove(name: string) {
    const dir = await this.dir
    await dir.removeEntry(name).catch(() => {})
  }

  private async load(cp: Checkpoint) {
    await this.writing
    const dir = await this.dir
    const file = await (await dir.getFileHandle(cp.name)).getFile()
    const { keeping, st } = await unpack(file)
    this.r.restoreGarden(st)
    this.garden.set(keeping)
  }

  // Cut the history off after `step`: what was recorded beyond it goes.
  private cut(step: number) {
    this.head = step
    this.runs = this.runs.flatMap((r) => {
      if (r.from >= step) return []
      const n = Math.min(r.count, Math.floor((step - r.from) / r.size))
      return n > 0 ? [{ ...r, count: n }] : []
    })
    this.strokes = this.strokes.filter((s) => s.step < step)
    this.samples.cut(step)
    const gone = this.checkpoints.filter((c) => c.step > step)
    this.checkpoints = this.checkpoints.filter((c) => c.step <= step)
    for (const c of gone) {
      this.bytes -= c.bytes
      void this.remove(c.name)
    }
    this.sinceSteps = 0
    this.sinceRuns = 0
    this.changedView()
  }

  // the recorded runs from `from` up to `to` (sizes, in order)
  private plan(from: number, to: number) {
    const out: number[] = []
    for (const r of this.runs) {
      const end = r.from + r.size * r.count
      if (end <= from || r.from >= to) continue
      for (let k = 0; k < r.count; k++) {
        const s = r.from + k * r.size
        if (s >= from && s + r.size <= to) out.push(r.size)
      }
    }
    return out
  }

  // Run the garden on as it ran before, from where it is to `to` (loading a
  // checkpoint of something done by hand on the way when it is reached).
  private async runTo(to: number) {
    while (this.r.step < to) {
      const at = this.r.step
      const next = this.checkpoints.find((c) => c.step > at && c.step <= to && c.action)
      const end = next ? next.step : to
      const plan = this.plan(at, end)
      const steps = plan.reduce((a, b) => a + b, 0)
      if (steps > 0) await this.r.simulate(steps, this.strokes, plan)
      if (next && this.r.step === next.step) await this.load(next)
      if (steps === 0 && !next) break
    }
  }

  // Go to `step`: from the checkpoint before it (or from where the garden
  // is, if that is on the way), run on to it. Seeks made while one is under
  // way go to the last asked for. `quick`: only as far as the checkpoint
  // (while the playhead is dragged).
  seek(step: number, quick = false) {
    this.seekTo = { step: Math.max(this.first, Math.min(this.head, Math.round(step))), quick }
    if (this.moving) return this.moving
    const gen = this.gen
    this.moving = (async () => {
      while (this.seekTo !== null && gen === this.gen) {
        const { step: to, quick: q } = this.seekTo
        this.seekTo = null
        const at = this.r.step
        const cp = [...this.checkpoints].reverse().find((c) => c.step <= to)
        if (!cp) break
        const between = this.checkpoints.some((c) => c.action && c.step > at && c.step <= to)
        // (from where it is, when that is past the checkpoint and nothing
        // was done by hand on the way)
        if (!(at >= cp.step && at <= to && !between)) await this.load(cp)
        if (!q) await this.runTo(to)
        this.changedView()
      }
    })().finally(() => (this.moving = null))
    return this.moving
  }

  // Played on in the past: about `want` steps of the history again.
  replay(want: number) {
    if (this.moving || this.live) return
    const gen = this.gen
    const at = this.r.step
    // (whole runs, at least one)
    let to = at
    for (const n of this.plan(at, this.head)) {
      if (to > at && to - at + n > want) break
      to += n
    }
    this.moving = this.runTo(to)
      .then(() => {
        if (gen === this.gen) this.changedView()
      })
      .finally(() => (this.moving = null))
  }
}

// ------------------------------------------------------------- packing ---

// A checkpoint as one compressed file: a JSON header (the keeper's state,
// the renderer's own, the sections' sizes), then the sections.
const SECTIONS = ['cells', 'grid', 'field', 'fruits', 'dev', 'slimeAgents', 'slimeMap', 'pools', 'counters', 'events'] as const

async function pack(keeping: GardenKeeping, st: GardenState): Promise<Blob> {
  const own = Object.fromEntries(Object.entries(st).filter(([k]) => !(SECTIONS as readonly string[]).includes(k)))
  const head = new TextEncoder().encode(JSON.stringify({ keeping, own, sizes: SECTIONS.map((k) => st[k].byteLength) }))
  const len = new Uint32Array([head.byteLength])
  const raw = new Blob([len, head, ...SECTIONS.map((k) => st[k])])
  return new Response(raw.stream().pipeThrough(new CompressionStream('gzip'))).blob()
}

async function unpack(file: Blob): Promise<{ keeping: GardenKeeping; st: GardenState }> {
  const raw = await new Response(file.stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()
  const len = new Uint32Array(raw, 0, 1)[0]
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(raw, 4, len)))
  let at = 4 + len
  const st = { ...head.own } as GardenState
  SECTIONS.forEach((k, i) => {
    const n = head.sizes[i]
    ;(st as unknown as Record<string, ArrayBuffer>)[k] = raw.slice(at, at + n)
    at += n
  })
  return { keeping: head.keeping, st }
}
