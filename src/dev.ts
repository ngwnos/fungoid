// Dev-only helpers on window.fungoid for inspecting the garden from the console.
import { mulberry32, mutate } from './genome/breed'
import { GENES } from './genome/genes'
import { fromPreset, newId } from './scene'
import { octEncode, type V3 as WV3 } from './world'
import { BOX_MAX, BOX_MIN, FLAT, PLANET_R } from './gpu/dims'
import { dot, normalize, sub, type V3 } from './gpu/camera'
import { F4N, GRID, GRID_FIELDS, GRID_SIZE, GRID_WORDS, MAX_FRUITS } from './gpu/dims'
import { YEAR } from './weather'

// the whole world in view: the planet, or the terrarium on its table
const WHOLE = FLAT ? { target: [0, 3, 0], height: 84 } : { target: [0, 0, 0], height: 40 }

type Fungoid = Record<string, any>

// what a fruit body looks like (fungoid.mutants), apart from when it fruits
// and for how long
const FORM = new Set(['Form', 'Stem', 'Cap', 'Underside', 'Color'])
const HELD = new Set(['body', 'grow', 'life', 'decay'])

export function installDevTools(fungoid: Fungoid) {
  const wait = (ms: number) => new Promise((r) => setTimeout(r, ms))
  const renderer = () => fungoid.renderer.current

  // A copy of part of a GPU buffer (by its name in the renderer), for poking at
  fungoid.read = async (name: string, offset: number, bytes: number) => {
    const r = renderer()
    const d: GPUDevice = r.device
    const rb = d.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    const enc = d.createCommandEncoder()
    enc.copyBufferToBuffer(r.b[name], offset, rb, 0, bytes)
    d.queue.submit([enc.finish()])
    await rb.mapAsync(GPUMapMode.READ)
    const out = rb.getMappedRange().slice(0)
    rb.destroy()
    return out
  }

  // A fingerprint of the garden as it is, to check that a change leaves it
  // as it was, or that it repeats: the fields, the colonies' pools, the slime
  // mould, head development and counters as they lie; per field, the
  // voxels' grid fields; the grid's tables that carry over from step to
  // step; and the cells and fruit bodies as sets (which slot each has, and so
  // a cell's parent's slot, varies with the order the GPU runs threads in).
  // Each part's hash, and one of them all.
  fungoid.hash = async () => {
    const fnv = (u8: Uint8Array, h = 2166136261) => {
      for (let k = 0; k < u8.length; k++) h = Math.imul(h ^ u8[k], 16777619)
      return h >>> 0
    }
    const r = renderer()
    const out: Record<string, number> = {}
    for (const n of ['fieldA', 'fieldB', 'pools', 'slimeAgents', 'slimeMap', 'devField', 'counters', 'pheno']) out[n] = fnv(new Uint8Array(await fungoid.read(n, 0, r.b[n].size)))
    const grid = new Uint8Array(await fungoid.read('grid', 0, GRID_WORDS * 4))
    for (let g = 0; g < GRID_FIELDS; g++) out[`gf${g}`] = fnv(grid.subarray(g * F4N * 4, (g + 1) * F4N * 4))
    for (const k of ['ACT', 'SIGACT', 'HOSTS', 'SNAP', 'SLIME'] as const) out[k] = fnv(grid.subarray(GRID[k] * 4, (GRID[k] + GRID_SIZE[k]) * 4))
    const cnt = new Uint32Array(await fungoid.read('counters', 0, 32))
    const cells = new Uint32Array(await fungoid.read('cells', 0, Math.max(1, cnt[0]) * 64))
    const ch: number[] = []
    for (let i = 0; i < cnt[0]; i++) {
      const o = i * 16
      if (cells[o + 14] & 0x20) continue
      const rec = cells.slice(o, o + 16)
      rec[12] = 0
      ch.push(fnv(new Uint8Array(rec.buffer)))
    }
    out.cells = fnv(new Uint8Array(new Uint32Array(ch.sort((a, b) => a - b)).buffer))
    out.nCells = ch.length
    const fr = new Uint32Array(await fungoid.read('fruits', 0, r.b.fruits.size))
    const stride = fr.length / MAX_FRUITS
    const fh: number[] = []
    for (let j = 0; j < MAX_FRUITS; j++) {
      const o = j * stride
      if (!(fr[o + 86] & 1)) continue
      const rec = fr.slice(o, o + stride)
      rec[64] = 0
      fh.push(fnv(new Uint8Array(rec.buffer)))
    }
    out.fruits = fnv(new Uint8Array(new Uint32Array(fh.sort((a, b) => a - b)).buffer))
    out.nFruits = fh.length
    out.all = fnv(new Uint8Array(new Uint32Array(Object.values(out)).buffer))
    return out
  }

  // the fingerprint of a garden grown from `seed` for `steps` (the same
  // every time, while nothing that shapes the garden has changed)
  fungoid.hashRun = async (seed = 860443, steps = 6000) => {
    await fungoid.garden(seed)
    await fungoid.advance(steps)
    return fungoid.hash()
  }

  // live fruit bodies: bound centre and radius, head radius, maturity, shape
  fungoid.fruits = async () => {
    const fr = await renderer().readFruits()
    const out = []
    for (let j = 0; j < 4096; j++) {
      const o = j * fr.stride
      if (!(fr.u[o + 86] & 1) || fr.f[o + 83] <= 0) continue
      out.push({
        i: j, org: fr.u[o + 84], c: [fr.f[o + 80], fr.f[o + 81], fr.f[o + 82]], r: fr.f[o + 83],
        hc: [fr.f[o + 72], fr.f[o + 73], fr.f[o + 74]], hr: fr.f[o + 75], R: fr.f[o + 23], m: fr.f[o + 11], A: fr.f[o + 27],
        decay: fr.f[o + 56], par: fr.f[o + 64] - 1, crust: fr.f[o + 65],
      })
    }
    return out
  }

  // grow one preset alone, then frame its fruit bodies
  // i < 0: a random genome from `seed`
  fungoid.look = async (i: number, frames = 220, zoom = 3.2, seed = 0) => {
    fungoid.solo(i, undefined, seed)
    return growAndFrame(frames, zoom)
  }
  // grow for a while, then pause: the live fruit bodies
  const grow = async (frames: number) => {
    await wait(150)
    fungoid.setSpeed(5)
    await wait(100)
    await fungoid.frames(frames)
    fungoid.setSpeed(0)
    await wait(100)
    fungoid.setSelected(null)
    return fungoid.fruits()
  }
  const growAndFrame = async (frames: number, zoom: number) => {
    const pts = await grow(frames)
    if (!pts.length) return 'no fruits'
    const c = [0, 1, 2].map((j) => pts.reduce((a: number, p: any) => a + p.c[j], 0) / pts.length)
    let ext = 0
    for (const p of pts) ext = Math.max(ext, Math.hypot(p.c[0] - c[0], p.c[1] - c[1], p.c[2] - c[2]) + p.r)
    Object.assign(fungoid.cam, { target: c, height: Math.max(2, ext * zoom), yaw: Math.PI / 4, pitch: 0.42 })
    await fungoid.frames(50)
    return `n=${pts.length} ext=${ext.toFixed(2)}`
  }

  // Mutant lineages of one preset, each grown alone at the same spot, its
  // biggest ripe fruit body shown on its own and saved as
  // renders/<name>-<k>.png. `rounds` is one entry per still: how many times
  // Mutate is pressed down that lineage (0: the preset itself). Only form and
  // colour genes mutate (the life cycle is held, so every lineage fruits).
  fungoid.mutants = async (o: { preset?: number; rounds?: number[]; seed?: number; size?: number; frames?: number; zoom?: number; pitch?: number; at?: WV3; name?: string } = {}) => {
    const { preset = 0, rounds = [0, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8], seed = 1, size = 512, frames = 400, zoom = 2.3, pitch = 0.3, name = 'mutants' } = o
    const rng = mulberry32(seed)
    const base = fromPreset(preset)
    const free = GENES.map((g) => FORM.has(g.group) && !HELD.has(g.key))
    const spot = o.at ?? fungoid.spots()[0]
    const r = renderer()
    const out: string[] = []
    let redone = 0
    for (let k = 0; k < rounds.length; k++) {
      for (let tries = 0; ; tries++) {
        let g = base.genome
        for (let j = 0; j < rounds[k]; j++) g = mutate(g, rng)
        const genome = { a: g.a.map((v, i) => (free[i] ? v : base.genome.a[i])), b: g.b.map((v, i) => (free[i] ? v : base.genome.b[i])) }
        fungoid.solo({ id: newId(), name: base.name, genome }, spot)
        fungoid.fixedSize = [size, size]
        const pts = await grow(frames)
        // ripe and not rotting
        const ripe = pts.filter((p: any) => p.m >= 1 && p.decay < 0.15)
        const pool = ripe.length ? ripe : pts
        if (!pool.length && tries < 4) {
          redone++
          continue
        }
        if (pool.length) {
          const best = pool.reduce((a: any, p: any) => (p.R > a.R ? p : a))
          // the others are not drawn (until the garden grows again)
          for (const p of pts) if (p !== best) r.device.queue.writeBuffer(r.b.fruits, (p.i * 24 * 4 + 83) * 4, new Float32Array([0]))
          Object.assign(fungoid.cam, { target: best.c, height: Math.max(0.5, best.r * zoom), pitch, yaw: Math.PI / 4 })
        }
        break
      }
      out.push(await fungoid.snap(`${name}-${String(k).padStart(2, '0')}`, size, size, 60))
    }
    return { out, redone }
  }

  // frame the k-th largest fruit body (whole, or just its head)
  fungoid.closeup = async (k = 0, zoom = 3.5, pitch = 0.35, yaw = Math.PI / 4, head = false) => {
    fungoid.setSelected(null)
    const pts = (await fungoid.fruits()).sort((a: any, b: any) => b.R - a.R)
    if (!pts.length) return 'no fruits'
    const p = pts[Math.min(k, pts.length - 1)]
    const [c, r] = head ? [p.hc, p.hr] : [p.c, p.r]
    Object.assign(fungoid.cam, { target: c, height: Math.max(0.6, r * zoom), pitch, yaw })
    await fungoid.frames(60)
    return JSON.stringify({ n: pts.length, R: p.R.toFixed(2), m: p.m.toFixed(2), A: p.A.toFixed(2), r: p.r.toFixed(2) })
  }

  // frame-to-frame brightness change (stops) on tissue vs everything else
  fungoid.flicker = async (n = 12, speed = 0) => {
    fungoid.setSpeed(speed)
    await wait(50)
    await fungoid.frames(speed === 0 ? 60 : 4)
    let sum: Float64Array | null = null
    let sum2: Float64Array | null = null
    let diff: Float64Array | null = null
    let any: Uint8Array | null = null
    let prev: Float64Array | null = null
    let W = 0
    let H = 0
    for (let k = 0; k < n; k++) {
      await fungoid.frames(1)
      const c = await renderer().capture()
      W = c.W
      H = c.H
      sum ??= new Float64Array(W * H)
      sum2 ??= new Float64Array(W * H)
      diff ??= new Float64Array(W * H)
      any ??= new Uint8Array(W * H)
      const L = new Float64Array(W * H)
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++) {
          const o = y * c.row + x * 4
          L[y * W + x] = Math.log2(Math.max(0.2126 * c.col[o] + 0.7152 * c.col[o + 1] + 0.0722 * c.col[o + 2], 1e-3))
        }
      for (let i = 0; i < W * H; i++) {
        sum[i] += L[i]
        sum2[i] += L[i] * L[i]
        if (prev) diff[i] += Math.abs(L[i] - prev[i])
        if (c.ids[i] > 0) any[i] = 1
      }
      prev = L
    }
    const g = { tissue: [0, 0, 0], other: [0, 0, 0] }
    for (let i = 0; i < W * H; i++) {
      const m = sum![i] / n
      const b = any![i] ? g.tissue : g.other
      b[0]++
      b[1] += Math.sqrt(Math.max(sum2![i] / n - m * m, 0))
      b[2] += diff![i] / (n - 1)
    }
    const f = (b: number[]) => `sd ${(b[1] / b[0]).toFixed(3)} frameDiff ${(b[2] / b[0]).toFixed(3)}`
    return `tissue ${f(g.tissue)} | other ${f(g.other)}`
  }

  // H.264 encoded on the GPU (WebCodecs) and streamed to the dev server,
  // which wraps it in renders/<name>.mp4 (vite.config.ts). Progress of a
  // recording: fungoid.recording.
  const videoOut = async (name: string, width: number, height: number, fps: number, bitrate: number, ss = 1) => {
    const config: VideoEncoderConfig = {
      codec: 'avc1.640033', width, height, bitrate, framerate: fps,
      hardwareAcceleration: 'prefer-hardware', latencyMode: 'quality', avc: { format: 'annexb' },
    }
    if (!(await VideoEncoder.isConfigSupported(config)).supported) throw new Error('H.264 encoding is not supported here')
    let pending: Uint8Array[] = []
    let pendingBytes = 0
    let failed: unknown = null
    const send = async () => {
      if (!pending.length) return
      const body = new Uint8Array(pendingBytes)
      let at = 0
      for (const p of pending) {
        body.set(p, at)
        at += p.byteLength
      }
      pending = []
      pendingBytes = 0
      await fetch('/__record/data', { method: 'POST', body })
    }
    const encoder = new VideoEncoder({
      output: (chunk) => {
        const b = new Uint8Array(chunk.byteLength)
        chunk.copyTo(b)
        pending.push(b)
        pendingBytes += b.byteLength
      },
      error: (e) => (failed = e),
    })
    encoder.configure(config)
    const out = await (await fetch(`/__record/start?fps=${fps}&name=${name}`)).text()
    return {
      out,
      // the next frame drawn, as frame i of the video
      add: async (i: number) => {
        if (failed) throw failed
        const r = renderer()
        const grabbed = r.grabNext(ss)
        await fungoid.step()
        const g = await grabbed
        const frame = new VideoFrame(g.data, {
          format: g.format === 'bgra8unorm' ? 'BGRX' : 'RGBX',
          codedWidth: g.width,
          codedHeight: g.height,
          layout: [{ offset: 0, stride: g.row }],
          timestamp: Math.round((i * 1e6) / fps),
          duration: Math.round(1e6 / fps),
        })
        encoder.encode(frame, { keyFrame: i % (fps * 2) === 0 })
        frame.close()
        while (encoder.encodeQueueSize > 6) await new Promise((ok) => encoder.addEventListener('dequeue', ok, { once: true }))
        if (pendingBytes > 4e6) await send()
      },
      // ffmpeg's exit code
      end: async () => {
        try {
          if (!failed) {
            await encoder.flush()
            await send()
          }
        } finally {
          encoder.close()
        }
        return (await fetch('/__record/end')).text()
      },
    }
  }

  // Record the garden growing while the camera turns: a square video at `fps`
  // of `speed` steps per frame. Frames are read back from the GPU as soon as
  // they are drawn (the page need not be on screen). Each frame is the
  // average of `samples` renders with the garden held still, drawn at `ss`
  // times the size and shrunk (samples 1: one live frame each, ss 1: at size).
  fungoid.record = async (o: { seconds?: number; fps?: number; size?: number; turns?: number; speed?: number; samples?: number; ss?: number; bitrate?: number; name?: string; cam?: Record<string, unknown> } = {}) => {
    const { seconds = 60, fps = 60, size = 1080, turns = 1, speed = 5, samples = 32, ss = 2, bitrate = 30e6, name = 'terrarium' } = o
    const n = Math.round(seconds * fps)
    const r = renderer()
    fungoid.fixedSize = [size * ss, size * ss]
    fungoid.opts.motionFloor = 0.05
    fungoid.opts.accumulate = samples > 1
    fungoid.setSelected(null)
    fungoid.setSpeed(samples > 1 ? 0 : speed)
    Object.assign(fungoid.cam, { ...WHOLE, pitch: 0.5, yaw: Math.PI / 4, ...(o.cam ?? {}) })
    const yaw0 = fungoid.cam.yaw
    await wait(200)
    fungoid.hold = true
    r.restartFrames()
    // settle the new size before the first frame
    for (let k = 0; k < 2; k++) await fungoid.step()
    const video = await videoOut(name, size, size, fps, bitrate, ss)
    fungoid.recording = { frame: 0, of: n, out: video.out, done: false, started: performance.now() }
    try {
      for (let i = 0; i < n; i++) {
        fungoid.cam.yaw = yaw0 + (turns * 2 * Math.PI * i) / n
        if (samples > 1) {
          if (i > 0) {
            await r.simulate(speed)
            r.resetHistory()
          }
          for (let k = 1; k < samples; k++) await fungoid.step()
        }
        await video.add(i)
        fungoid.recording.frame = i + 1
      }
    } finally {
      fungoid.recording.exit = await video.end()
      fungoid.recording.done = true
      fungoid.recording.seconds = (performance.now() - fungoid.recording.started) / 1000
      fungoid.hold = false
      fungoid.fixedSize = undefined
      fungoid.opts.motionFloor = 0.15
      fungoid.opts.accumulate = false
      fungoid.setSpeed(0)
    }
    return fungoid.recording
  }

  // A time-lapse: `every` steps of the garden between frames, simulated
  // without drawing, then each frame the average of `samples` renders (thin
  // hyphae only resolve over many) drawn at `ss` times the size and shrunk;
  // meanwhile the garden also breeds the colonies its spores started.
  // `regrow` starts from fresh soil; the camera turns `turns` times over the
  // whole.
  // `width` (default: square) widens the frame; `onFrame(i)` runs before
  // frame i is drawn (to set something down partway through)
  fungoid.timelapse = async (o: { frames?: number; every?: number; fps?: number; size?: number; width?: number; turns?: number; samples?: number; ss?: number; regrow?: boolean; bitrate?: number; name?: string; cam?: Record<string, unknown>; onFrame?: (i: number) => unknown } = {}) => {
    const { frames = 240, every = 1000, fps = 30, size = 1080, turns = 0.25, samples = 32, ss = 2, regrow = true, bitrate = 30e6, name = 'timelapse' } = o
    const width = o.width ?? size
    const r = renderer()
    fungoid.fixedSize = [width * ss, size * ss]
    fungoid.opts.accumulate = true
    fungoid.setSelected(null)
    fungoid.setSpeed(0)
    Object.assign(fungoid.cam, { ...WHOLE, pitch: 0.5, yaw: Math.PI / 4, ...(o.cam ?? {}) })
    const yaw0 = fungoid.cam.yaw
    if (regrow) fungoid.regrow()
    await wait(200)
    fungoid.hold = true
    r.restartFrames()
    for (let k = 0; k < 2; k++) await fungoid.step()
    const video = await videoOut(name, width, size, fps, bitrate, ss)
    fungoid.recording = { frame: 0, of: frames, out: video.out, done: false, started: performance.now() }
    try {
      for (let i = 0; i < frames; i++) {
        fungoid.cam.yaw = yaw0 + (turns * 2 * Math.PI * i) / frames
        if (i > 0) {
          await r.simulate(every)
          r.resetHistory()
        }
        await o.onFrame?.(i)
        for (let k = 1; k < samples; k++) await fungoid.step()
        await video.add(i)
        fungoid.recording.frame = i + 1
      }
    } finally {
      fungoid.recording.exit = await video.end()
      fungoid.recording.done = true
      fungoid.recording.seconds = (performance.now() - fungoid.recording.started) / 1000
      fungoid.recording.steps = fungoid.renderer.current.stats.step
      fungoid.hold = false
      fungoid.fixedSize = undefined
      fungoid.opts.accumulate = false
    }
    return fungoid.recording
  }

  // A garden from `seed` (its layout and its life, from its first kinds),
  // started in step with the scripted frames that follow (the page's own
  // frames held), so that from one seed it grows the same every time.
  fungoid.garden = async (seed: number) => {
    fungoid.hold = true
    fungoid.setSpeed(0)
    await wait(100)
    fungoid.terrarium(seed, true)
    await wait(100)
    for (let k = 0; k < 3; k++) await fungoid.step()
  }

  // The ground's level under `p` (cm: on the planet, how far from its centre
  // the ground is in p's direction; in the terrarium, its height), from the
  // ground map as last read (fungoid.loadGround(), again after the world
  // changes).
  let ground: { heights: Float32Array; size: [number, number] } | null = null
  fungoid.loadGround = async () => {
    ground = await renderer().readGround()
  }
  fungoid.groundAt = (p: WV3) => {
    if (!ground) throw new Error('fungoid.loadGround() first')
    const [W, H] = ground.size
    const l = Math.hypot(...p) || 1
    const [ex, ey] = FLAT
      ? [0, 2].map((k) => ((p[k] - BOX_MIN[k]) / (BOX_MAX[k] - BOX_MIN[k])) * 2 - 1)
      : octEncode([p[0] / l, p[1] / l, p[2] / l])
    const u = Math.min(W - 1.001, Math.max(0, ((ex + 1) / 2) * W - 0.5))
    const v = Math.min(H - 1.001, Math.max(0, ((ey + 1) / 2) * H - 0.5))
    const i = Math.floor(u)
    const j = Math.floor(v)
    const h = (a: number, b: number) => ground!.heights[b * W + a]
    const [fu, fv] = [u - i, v - j]
    return (FLAT ? 0 : PLANET_R) + (h(i, j) * (1 - fu) + h(i + 1, j) * fu) * (1 - fv) + (h(i, j + 1) * (1 - fu) + h(i + 1, j + 1) * fu) * fv
  }

  // Points the camera from `eye` at `look` with a vertical field of view of
  // `fov` (rad); returns the direction it looks.
  fungoid.aim = (eye: V3, look: V3, fov: number): V3 => {
    const d = sub(look, eye)
    const dist = Math.hypot(...d)
    const f = normalize(d)
    Object.assign(fungoid.cam, { target: look, pitch: Math.asin(-f[1]), yaw: Math.atan2(-f[0], -f[2]), height: 2 * dist * Math.tan(fov / 2) })
    return f
  }

  // A film: shots one after another, each `frames` frames long. Before each
  // frame the garden runs on speed(t) steps (t from 0 to 1 through the shot;
  // fractions carry over), then the frame is the average of `samples`
  // renders through the lens cam(t) describes: where the camera is and what
  // it looks at, its vertical field of view (rad), the lens's radius and what
  // it focuses on (a point, or a distance; default: what it looks at), in
  // cm, drawn at `ss` times the size. A shot's setup() runs as it starts (to
  // choose what to look at). Progress: fungoid.recording.
  fungoid.film = async (o: {
    shots: {
      frames: number
      speed: number | ((t: number) => number)
      setup?: () => unknown
      cam: (t: number) => { eye: V3; look: V3; fov: number; aperture?: number; focus?: V3 | number }
    }[]
    fps?: number; size?: number; ss?: number; samples?: number; bitrate?: number; name?: string
  }) => {
    const { shots, fps = 60, size = 1080, ss = 2, samples = 32, bitrate = 40e6, name = 'film' } = o
    const r = renderer()
    const of = shots.reduce((a, s) => a + s.frames, 0)
    fungoid.fixedSize = [size * ss, size * ss]
    fungoid.setSelected(null)
    fungoid.setSpeed(0)
    fungoid.opts.accumulate = true
    await wait(200)
    fungoid.hold = true
    r.restartFrames()
    for (let k = 0; k < 2; k++) await fungoid.step()
    const video = await videoOut(name, size, size, fps, bitrate, ss)
    fungoid.recording = { frame: 0, of, shot: 0, out: video.out, done: false, started: performance.now() }
    let frame = 0
    let carry = 0
    try {
      for (const [si, shot] of shots.entries()) {
        fungoid.recording.shot = si
        await shot.setup?.()
        for (let i = 0; i < shot.frames; i++) {
          const t = shot.frames > 1 ? i / (shot.frames - 1) : 0
          carry += typeof shot.speed === 'number' ? shot.speed : shot.speed(t)
          let n = Math.floor(carry)
          carry -= n
          await r.simulate(n)
          const c = shot.cam(t)
          const fwd = fungoid.aim(c.eye, c.look, c.fov)
          const focus = typeof c.focus === 'number' ? c.focus : dot(sub(c.focus ?? c.look, c.eye), fwd)
          fungoid.opts.lens = { fov: c.fov, aperture: c.aperture ?? 0, focus: Math.max(focus, 0.5), samples }
          r.resetHistory()
          for (let k = 1; k < samples; k++) await fungoid.step()
          await video.add(frame++)
          fungoid.recording.frame = frame
        }
      }
    } finally {
      fungoid.recording.exit = await video.end()
      fungoid.recording.done = true
      fungoid.recording.seconds = (performance.now() - fungoid.recording.started) / 1000
      fungoid.recording.steps = renderer().stats.step
      fungoid.hold = false
      fungoid.fixedSize = undefined
      fungoid.opts.lens = null
      fungoid.opts.accumulate = false
    }
    return fungoid.recording
  }

  // Run the garden `steps` forward without drawing most of it: bursts of
  // `every` steps, each followed by enough drawn frames for the garden's
  // bookkeeping (stats, births, deaths, fungoid.ecoLog) to catch up.
  fungoid.advance = async (steps: number, every = 2000) => {
    const r = renderer()
    const hold = fungoid.hold
    fungoid.hold = true
    fungoid.setSpeed(0)
    try {
      for (let done = 0; done < steps; done += every) {
        await r.simulate(Math.min(every, steps - done))
        for (let k = 0; k < 16; k++) await fungoid.step()
      }
    } finally {
      fungoid.hold = hold
    }
    return r.stats.step
  }

  // The garden year by year from fungoid.ecoLog: colonies alive (mean), living
  // cells (least..most), fruit bodies (mean), births, deaths, lineages seen
  fungoid.ecoSummary = (year = YEAR) => {
    const years: Record<number, { n: number; alive: number; lo: number; hi: number; fruits: number; born: number; died: number; lineages: Set<string> }> = {}
    let prev = new Set<number>()
    for (const e of (fungoid.ecoLog ?? []) as { step: number; orgs: [number, string, number, number][] }[]) {
      const y = Math.floor(e.step / year) + 1
      const s = (years[y] ??= { n: 0, alive: 0, lo: Infinity, hi: 0, fruits: 0, born: 0, died: 0, lineages: new Set() })
      const ids = new Set(e.orgs.map((o) => o[0]))
      const cells = e.orgs.reduce((a, o) => a + o[2], 0)
      s.n++
      s.alive += e.orgs.length
      s.lo = Math.min(s.lo, cells)
      s.hi = Math.max(s.hi, cells)
      s.fruits += e.orgs.reduce((a, o) => a + o[3], 0)
      if (prev.size) for (const id of ids) if (!prev.has(id)) s.born++
      for (const id of prev) if (!ids.has(id)) s.died++
      for (const o of e.orgs) s.lineages.add(o[1])
      prev = ids
    }
    return Object.entries(years).map(([y, s]) => ({
      year: +y, alive: Math.round(s.alive / s.n), cells: `${Math.round(s.lo / 1000)}-${Math.round(s.hi / 1000)}k`,
      fruits: Math.round(s.fruits / s.n), born: s.born, died: s.died, lineages: s.lineages.size,
    }))
  }

  // Steps per second with nothing drawn, `chunk` steps per submission
  // (fruit bodies, germination and the light through tissue update once per
  // submission, as once per frame).
  fungoid.simRate = async (chunk = 64, seconds = 4) => {
    const r = renderer()
    const hold = fungoid.hold
    fungoid.hold = true
    await r.device.queue.onSubmittedWorkDone()
    const t0 = performance.now()
    let steps = 0
    while (performance.now() - t0 < seconds * 1000) {
      for (let k = 0; k < 4; k++) await r.simulate(chunk)
      steps += 4 * chunk
      await r.device.queue.onSubmittedWorkDone()
    }
    const s = (performance.now() - t0) / 1000
    fungoid.hold = hold
    return Math.round(steps / s)
  }

  // The timeline's graph as a PNG in renders/, over the panel's colour (drawn
  // again and read in the same task: a WebGPU canvas keeps nothing after)
  fungoid.timelineShot = async (name = 'timeline') => {
    const cv = document.querySelector('.history canvas') as HTMLCanvasElement
    const out = document.createElement('canvas')
    out.width = cv.width
    out.height = cv.height
    const ctx = out.getContext('2d')!
    ctx.fillStyle = '#1d1a17'
    ctx.fillRect(0, 0, out.width, out.height)
    fungoid.timeline.current?.(true)
    ctx.drawImage(cv, 0, 0)
    const blob = await new Promise<Blob | null>((ok) => out.toBlob(ok, 'image/png'))
    if (!blob) throw new Error('no image')
    return (await fetch(`/__record/png?name=${encodeURIComponent(name)}`, { method: 'POST', body: blob })).text()
  }

  // A still at a fixed size, saved to renders/<name>.png.
  fungoid.snap = async (name = 'still', w = 1600, h = 1200, frames = 24) => {
    fungoid.fixedSize = [w, h]
    await fungoid.frames(frames)
    const canvas = document.querySelector('canvas')!
    // read the canvas in the task that drew it
    const shot = new Promise<Blob | null>((ok) => {
      fungoid.step()
      canvas.toBlob(ok, 'image/png')
    })
    const blob = await shot
    fungoid.fixedSize = undefined
    if (!blob) throw new Error('no image')
    return (await fetch(`/__record/png?name=${encodeURIComponent(name)}`, { method: 'POST', body: blob })).text()
  }

  // GPU ms per pass, averaged over several stat readbacks (they lag a frame or two)
  // GPU ms per simulation kernel over `steps` steps (each timed in a pass of
  // its own, so the total runs a little over the real thing)
  fungoid.profile = async (steps = 64) => {
    const t: Record<string, number> = (await renderer().profileSim(steps)) ?? {}
    const total = Object.values(t).reduce((a, b) => a + b, 0)
    const rows = Object.entries(t).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${v.toFixed(2)}`)
    return `${steps} steps: ${total.toFixed(1)} ms (${(total / steps).toFixed(2)} per step) | ${rows.join(', ')}`
  }

  // The garden's speed, to compare before and after a change: grown from
  // `seed` to `steps`, its cells and fruit bodies then, the sim's time per
  // kernel (fungoid.profile), the GPU time per pass of a drawn frame
  // (fungoid.timing), then its steps per second in runs of 16 and of 256 (each
  // read `reads` times: the readings vary by a tenth or more).
  fungoid.bench = async (seed = 860443, steps = 6000, reads = 3) => {
    await fungoid.garden(seed)
    await fungoid.advance(steps)
    const r = renderer()
    const out: Record<string, unknown> = { step: r.stats.step, cells: r.stats.cells }
    out.kernels = await fungoid.profile(64)
    // (a frame's passes looking at the whole planet, and close in over its
    // ground)
    const cam = { ...fungoid.cam, target: [...fungoid.cam.target] }
    Object.assign(fungoid.cam, { ...WHOLE, yaw: Math.PI / 4, pitch: 0.52 })
    out.frame = await fungoid.timing()
    Object.assign(fungoid.cam, { target: FLAT ? [0, 2, 0] : [0, PLANET_R + 1, 0], height: 12, yaw: Math.PI / 4, pitch: 1.1 })
    out.close = await fungoid.timing()
    Object.assign(fungoid.cam, cam)
    for (const chunk of [16, 256]) {
      const v: number[] = []
      for (let k = 0; k < reads; k++) v.push(await fungoid.simRate(chunk, 3))
      out[`steps/s ${chunk}`] = v
    }
    return out
  }

  fungoid.timing = async (reads = 6) => {
    await fungoid.frames(12)
    const acc: Record<string, number> = {}
    for (let k = 0; k < reads; k++) {
      const st = await fungoid.frames(10)
      for (const [p, v] of Object.entries(st.gpuMs as Record<string, number>)) acc[p] = (acc[p] ?? 0) + v / reads
    }
    const total = Object.values(acc).reduce((a, b) => a + b, 0)
    return Object.entries(acc).map(([p, v]) => `${p} ${v.toFixed(2)}`).join(', ') + ` | total ${total.toFixed(2)}`
  }
}
