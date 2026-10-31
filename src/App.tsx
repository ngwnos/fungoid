import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Renderer, MAX_ORGS, BRUSH_OPS, type Lens, type RenderStats } from './gpu/renderer'
import { YEAR, dayPhase, nightness, seasonAt, sunElev } from './weather'
import { IsoCamera, type V3 } from './gpu/camera'
import { DECAY_MODES, GENES, GENE_INDEX, GROUP_ORDER, PHASES, RULE_INPUTS, RULE_OUTPUTS, decayIndex, expressGene, phaseIndex, ruleIndex, setGene } from './genome/genes'
import { cross, mulberry32, mutate, randomGenome } from './genome/breed'
import { ARRIVALS, PRESETS, STARTERS } from './genome/presets'
import { installDevTools } from './dev'
import { SLIME_ORG, fillPrograms, fromPreset, ids, newId, prefersWood, swatch, type Organism } from './scene'
import { History, type GardenKeeping } from './history'
import { Timeline } from './Timeline'
import { beside, branchSpot, makeWorld, plantingSpots } from './world'
import { PLANET_R } from './gpu/dims'

// new layouts on each visit; the garden's life (where colonies land,
// arrivals, breeding) from a generator seeded by its layout's seed, so a
// seed gives the same garden
const visit = mulberry32((Date.now() ^ 0x5eed) >>> 0)
let rng = visit
const seedLife = (seed: number) => (rng = mulberry32((Math.imul(seed, 0x9e3779b1) ^ 0x5eed) >>> 0))
const DEBUG_VIEWS = ['Lit', 'Albedo', 'Normals', 'No AA']
const SCALES = [1, 0.75, 0.5, 2]
const SPEEDS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
const SEASONS = ['autumn', 'winter', 'spring', 'summer']
// the time of day over direction d (its own: the sun's noon there)
const timeOfDay = (d: V3, step: number) => {
  if (sunElev(d, step) < -0.05) return 'night'
  const t = (dayPhase(step) - Math.atan2(d[2], d[0]) / (2 * Math.PI) + 2) % 1
  return t < 0.42 ? 'morning' : t < 0.62 ? 'midday' : 'evening'
}
// Where on the planet the view is: under its middle, or over what it looks at.
const lookAt = (cam: IsoCamera): V3 => {
  const t = cam.target
  const l = Math.hypot(...t)
  if (l > 1) return [t[0] / l, t[1] / l, t[2] / l]
  const f = cam.basis().fwd
  return [-f[0], -f[1], -f[2]]
}
// Tools: look around (and pick fungi), or paint the ground
const TOOLS = ['look', 'water', 'dry', 'feed', 'deplete', 'slime'] as const
type Tool = (typeof TOOLS)[number]
// The terrarium: a new layout every visit (or fungoid.terrarium(seed) in dev).
let WORLD = makeWorld(Math.floor(visit() * 1e6))
let SPOTS = plantingSpots(WORLD, WORLD.seed)
seedLife(WORLD.seed)
function layWorld(seed: number) {
  WORLD = makeWorld(seed)
  SPOTS = plantingSpots(WORLD, seed)
}

// Live render knobs (exposed on window.fungoid in dev).
// (dev: lens and accumulate are for recordings, gpu/renderer.ts FrameInput)
const OPTS = { bounces: 2, exposure: 0.95, sunRadius: 0.025, aperture: 0.006, fineMoss: true, motionFloor: 0.15, lens: null as Lens | null, accumulate: false }
const DEV: Record<string, unknown> = { opts: OPTS }
if (import.meta.env.DEV) {
  Object.assign(window, { fungoid: DEV })
  installDevTools(DEV)
}

type Seed = Omit<Organism, 'slot' | 'x' | 'y' | 'z' | 'onSurface'>

// Give a new organism a program slot and a free place to grow (the slots of
// colonies whose fruit bodies are still rotting stay taken).
function place(list: Organism[], seed: Seed, rotting: Organism[]): Organism | null {
  const used = new Set([...list, ...rotting].map((o) => o.slot))
  let slot = 0
  while (used.has(slot)) slot++
  if (slot >= SLIME_ORG) return null
  // a mycoparasite starts in a colony it can live on
  const parasitic = (g: Organism['genome']) => expressGene(g, GENE_INDEX.parasite) > 0.5
  if (parasitic(seed.genome)) {
    const hosts = list.filter((o) => !parasitic(o.genome))
    if (hosts.length) {
      const h = hosts[Math.floor(rng() * hosts.length)]
      const [x, y, z] = beside([h.x, h.y, h.z], 0.6, rng() * 2 * Math.PI)
      return { ...seed, slot, x, y, z, onSurface: h.onSurface }
    }
  }
  if (prefersWood(seed.genome)) {
    const [x, y, z] = branchSpot(WORLD, rng)
    return { ...seed, slot, x, y, z, onSurface: true }
  }
  const taken = (p: number[]) => list.some((o) => Math.hypot(o.x - p[0], o.y - p[1], o.z - p[2]) < 2.5)
  const free = SPOTS.filter((p) => !taken(p))
  const [x, y, z] = free.length ? free[0] : beside(SPOTS[0] ?? [0, 1, 0], rng() * 40, rng() * 2 * Math.PI)
  return { ...seed, slot, x, y, z, onSurface: false }
}

const initialSeeds = () => STARTERS.map((i) => fromPreset(i))
const shortName = (o: Organism) => o.name.split(/[ ×]/)[0]

// The garden keeps at most this many colonies; spores that land when it is
// full do not take.
const MAX_LIVE = 56
// steps between spores arriving from outside the garden
const ARRIVAL_EVERY = 6000
// steps per colony born from a germinated spore (the rest don't take), however
// many steps each frame runs
const BIRTH_EVERY = 1000

// A spore-grown child: a cross between two lineages, or the next generation
// of one.
const lineage = (o: Organism) => o.name.replace(/ F\d+$/, '')
const lineageShort = (o: Organism) => {
  const w = lineage(o).split(' × ')[0].split(' ')
  return w[0].length > 3 ? w[0] : w.slice(0, 2).join(' ')
}
function offspringName(a: Organism, b: Organism | undefined): string {
  if (b && lineage(a) !== lineage(b)) return `${lineageShort(a)} × ${lineageShort(b)}`
  const gen = Number(a.name.match(/ F(\d+)$/)?.[1] ?? 1)
  return `${lineage(a)} F${gen + 1}`
}

export function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rendererRef = useRef<Renderer | null>(null)
  const camRef = useRef(new IsoCamera())
  const [error, setError] = useState<string | null>(null)
  const [orgs, setOrgs] = useState<Organism[]>([])
  const [selected, setSelected] = useState<number | null>(null)
  const [second, setSecond] = useState<number | null>(null)
  const [speed, setSpeed] = useState(1)
  const [debug, setDebug] = useState(0)
  const [renderScale, setRenderScale] = useState(1)
  const [preset, setPreset] = useState(0)
  const [tool, setTool] = useState<Tool>('look')
  const [brushSize, setBrushSize] = useState(2.5)
  const [stats, setStats] = useState<{ fps: number; r: RenderStats; w: number; h: number } | null>(null)

  const live = useRef({ orgs, selected, second, debug, renderScale, speed, tool, brushSize })
  live.current = { orgs, selected, second, debug, renderScale, speed, tool, brushSize }
  // the pointer over the canvas (canvas pixels), whether it paints, and the
  // ground under it
  const pointer = useRef({ x: 0, y: 0, inside: false, painting: false })
  const ground = useRef<V3 | null>(null)
  // the garden's history (history.ts), and the timeline it shows
  const historyRef = useRef<History | null>(null)
  const [garden, setGarden] = useState<{ renderer: Renderer; history: History } | null>(null)
  const timelineFrame = useRef<((force?: boolean) => void) | null>(null)
  // where on the planet the view is (lookAt), for the clock and the timeline
  const look = useRef<V3>([0, 1, 0])
  Object.assign(DEV, { cam: camRef.current, renderer: rendererRef, live, setDebug, setSelected, setSpeed, history: historyRef, timeline: timelineFrame })
  const programsDirty = useRef(true)
  // colonies that have gone while their fruit bodies stand: they keep their
  // slots and programs until those have rotted away
  const rotting = useRef<Organism[]>([])
  // the garden was started over (its keeping in the frame loop resets)
  const restarted = useRef(false)
  useEffect(() => {
    programsDirty.current = true
  }, [orgs])

  // The colonies, now: the next keeping and checkpoint see them whether or
  // not React has rendered, and their programs go up at once.
  const commit = useCallback((list: Organism[]) => {
    setOrgs(list)
    live.current.orgs = list
    rendererRef.current?.setPrograms(fillPrograms(list, rotting.current), MAX_ORGS)
  }, [])
  // Something done to the garden by hand: in the past, what came after it
  // goes; a checkpoint just after it.
  const byHand = (fn: () => void) => {
    historyRef.current?.willChange()
    fn()
    historyRef.current?.changed()
  }

  // plant: give organisms slots and inoculate them
  const plant = useCallback((seeds: Seed[], base: Organism[]) => {
    const list = [...base]
    const added: Organism[] = []
    for (const s of seeds) {
      const o = place(list, s, rotting.current)
      if (!o) break
      list.push(o)
      added.push(o)
    }
    // (their programs go up with their spores: a colony without one dies)
    rendererRef.current?.setPrograms(fillPrograms(list, rotting.current), MAX_ORGS)
    rendererRef.current?.inoculate(added.map((o) => ({ org: o.slot, x: o.x, y: o.y, z: o.z, onSurface: o.onSurface, exact: o.exact })))
    return { list, added }
  }, [])

  // render loop
  useEffect(() => {
    const canvas = canvasRef.current!
    let raf = 0
    let alive = true
    let still = 0
    let lastCam = ''
    let last = performance.now()
    let fps = 60
    let statTick = 0
    Renderer.create(canvas)
      .then(async (r) => {
        await r.bakeWorld(WORLD.buffer)
        if (!alive) return
        rendererRef.current = r
        // first garden
        const { list } = plant(initialSeeds(), [])
        r.setPrograms(fillPrograms(list, rotting.current), MAX_ORGS)
        programsDirty.current = false
        setOrgs(list)
        const tick = (now: number) => {
          const dt = Math.min(0.1, (now - last) / 1000)
          last = now
          fps = fps * 0.95 + (1 / Math.max(dt, 1e-3)) * 0.05
          const L = live.current
          // dev: a fixed render size (recording)
          const fixed = DEV.fixedSize as [number, number] | undefined
          const w = fixed?.[0] ?? Math.max(1, Math.round(canvas.clientWidth * L.renderScale))
          const h = fixed?.[1] ?? Math.max(1, Math.round(canvas.clientHeight * L.renderScale))
          if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w
            canvas.height = h
          }
          r.resize(w, h)
          if (restarted.current) {
            // a garden started over: its keeping starts over too (before
            // its first frame, which its history begins with)
            restarted.current = false
            statTick = 0
            firstSeen.clear()
            lastArrival = 0
            lastSlime = -1
            lastTend = 0
            birthBudget = 0
            r.ecology.births.length = 0
          }
          if (programsDirty.current) {
            r.setPrograms(fillPrograms(L.orgs, rotting.current), MAX_ORGS)
            programsDirty.current = false
          }
          let steps = SPEEDS[L.speed]
          // in the past, the history plays again as it was (and while the
          // playhead moves, the garden waits)
          const hist = historyRef.current
          if (hist && (!hist.live || hist.busyMoving)) {
            if (steps > 0 && !hist.busyMoving) hist.replay(steps)
            steps = 0
          }
          const cam = camRef.current
          const key = `${cam.yaw},${cam.pitch},${cam.height},${cam.target.join()},${w},${h}`
          const camMoved = key !== lastCam
          if (camMoved || steps > 0) still = 0
          else still++
          lastCam = key
          const slot = (id: number | null) => L.orgs.find((o) => o.id === id)?.slot ?? -1
          // a tool follows the ground under the pointer, and paints it
          const P = pointer.current
          const painting = L.tool !== 'look'
          if (painting && P.inside) r.pick(P.x, P.y).then((h) => (ground.current = h.point))
          const at = painting && P.inside ? ground.current : null
          // (painting in the past cuts off what came after)
          if (at && P.painting && hist && !hist.live) hist.willChange()
          look.current = lookAt(cam)
          // (close in, the eye adjusts to the light there; the whole planet
          // in view, to its day side)
          const close = Math.min(1, Math.max(0, (32 - cam.height) / 20))
          r.render({
            basis: cam.basis(),
            target: cam.target,
            height: cam.height,
            sunRadius: OPTS.sunRadius,
            viewDay: 1 - close * nightness(look.current, r.step),
            exposure: OPTS.exposure,
            debug: L.debug,
            bounces: OPTS.bounces,
            selected: slot(L.selected),
            second: slot(L.second),
            // reprojected history blurs while the camera moves
            alphaFloor: camMoved ? OPTS.motionFloor : 0,
            aperture: OPTS.aperture,
            lens: OPTS.lens,
            accumulate: OPTS.accumulate,
            simSteps: steps,
            fineMoss: OPTS.fineMoss,
            overlay: L.tool === 'water' || L.tool === 'dry' ? 1 : L.tool === 'feed' || L.tool === 'deplete' ? 2 : 0,
            cursor: at ? [...at, L.tool === 'slime' ? 1 : L.brushSize] : null,
            brush: at && P.painting ? { at, radius: L.brushSize, op: BRUSH_OPS[L.tool as keyof typeof BRUSH_OPS], strength: 0.05 } : null,
          })
          if (++statTick % 15 === 0) {
            setStats({ fps, r: r.stats, w, h })
          }
          timelineFrame.current?.()
        }
        // the garden's own turnover: colonies that died out leave, spores
        // that germinated become new colonies bred from their parents
        const firstSeen = new Map<number, number>()
        let lastArrival = 0
        let lastSlime = -1
        let lastTend = 0
        let birthBudget = 0
        const tend = (r: Renderer) => {
          const eco = r.ecology
          const births = eco.births.splice(0)
          const step = r.stats.step
          const list = live.current.orgs
          for (const o of list) if (!firstSeen.has(o.id)) firstSeen.set(o.id, step)
          const rotten = rotting.current.filter((o) => eco.fruits[o.slot] <= 0)
          if (rotten.length) {
            rotting.current = rotting.current.filter((o) => !rotten.includes(o))
            programsDirty.current = true
          }
          const dead = list.filter(
            (o) => step - (firstSeen.get(o.id) ?? step) > 600 && eco.cells[o.slot] <= 0 && eco.fruits[o.slot] <= 0,
          )
          let next = dead.length ? list.filter((o) => !dead.includes(o)) : list
          const busy = new Set([...list, ...rotting.current].map((o) => o.slot))
          birthBudget = Math.min(birthBudget + Math.max(step - lastTend, 0) / BIRTH_EVERY, 4)
          lastTend = step
          for (const b of births) {
            if (birthBudget < 1) break
            const pa = next.find((o) => o.slot === b.a)
            if (!pa) continue
            const pb = next.find((o) => o.slot === b.b && o !== pa)
            if (next.length >= MAX_LIVE) {
              // a full garden: the newcomer takes the place of the weakest
              // struggling colony, if there is one
              const weak = next
                .filter((o) => o !== pa && o !== pb && step - (firstSeen.get(o.id) ?? step) > 600)
                .sort((x, y) => eco.cells[x.slot] - eco.cells[y.slot])[0]
              if (!weak || eco.cells[weak.slot] > 300) continue
              next = next.filter((o) => o !== weak)
              dead.push(weak)
              if (eco.fruits[weak.slot] > 0) rotting.current.push(weak)
            }
            let slot = 0
            while (busy.has(slot)) slot++
            if (slot >= SLIME_ORG) break
            busy.add(slot)
            const child: Organism = {
              id: newId(), name: offspringName(pa, pb), genome: cross(pa.genome, (pb ?? pa).genome, rng),
              slot, x: b.x, y: b.y, z: b.z, onSurface: false, exact: true,
            }
            next = [...next, child]
            r.inoculate([{ org: slot, x: b.x, y: b.y, z: b.z, onSurface: false, exact: true }])
            birthBudget -= 1
          }
          // in the wet of some half-years a slime mould turns up, somewhere it
          // is autumn or winter
          const half = Math.floor(step / (YEAR / 2))
          if (eco.slime.mode === 0 && (step % (YEAR / 2)) / (YEAR / 2) > 0.1 && half !== lastSlime) {
            lastSlime = half
            if (rng() < 0.35) {
              const wet = SPOTS.filter((p) => seasonAt(p, step) > 0.05 && seasonAt(p, step) < 0.35)
              if (wet.length) r.addSlime(wet[Math.floor(rng() * wet.length)], 3000, false)
            }
          }
          // spores blow in from outside now and then (more often into a sparse
          // garden): a variant of one of the garden's founding kinds or of a
          // mycoparasite, or something wild
          const arrivalEvery = ARRIVAL_EVERY * Math.min(Math.max(next.length / 24, 1 / 3), 1)
          if (step - lastArrival > arrivalEvery && next.length < MAX_LIVE) {
            lastArrival = step
            const kin = fromPreset(ARRIVALS[Math.floor(rng() * ARRIVALS.length)])
            const seed: Seed = rng() < 0.6
              ? { id: newId(), name: kin.name, genome: mutate(kin.genome, rng) }
              : { id: newId(), name: 'Wild', genome: randomGenome(rng, kin.genome) }
            const o = place(next, seed, rotting.current)
            if (o) {
              next = [...next, o]
              r.inoculate([{ org: o.slot, x: o.x, y: o.y, z: o.z, onSurface: o.onSurface }])
            }
          }
          if (next !== list) {
            const gone = new Set(dead.map((o) => o.id))
            if (gone.has(live.current.selected ?? -1)) setSelected(null)
            r.setPrograms(fillPrograms(next, rotting.current), MAX_ORGS)
            setOrgs(next)
            // (the next keeping sees it whether or not React has rendered)
            live.current.orgs = next
          }
          // dev: the garden over time, colony by colony (id, lineage, cells, fruit bodies)
          if (import.meta.env.DEV) {
            const log = (DEV.ecoLog ??= []) as unknown[]
            log.push({ step, orgs: next.map((o) => [o.id, lineage(o), eco.cells[o.slot], eco.fruits[o.slot]]) })
            if (log.length > 20000) log.shift()
          }
        }
        // the garden is kept every KEEP_EVERY steps, with what it read back
        r.onKeep = () => tend(r)
        // and its history kept, with what the keeping holds
        const orgOut = (o: Organism) => ({ ...o, genome: { a: Array.from(o.genome.a), b: Array.from(o.genome.b) } })
        const orgIn = (o: ReturnType<typeof orgOut>): Organism => ({ ...o, genome: { a: Uint8Array.from(o.genome.a), b: Uint8Array.from(o.genome.b) } })
        historyRef.current = new History(r, {
          get: (): GardenKeeping => ({
            orgs: live.current.orgs.map(orgOut), rotting: rotting.current.map(orgOut), firstSeen: [...firstSeen],
            lastArrival, lastSlime, lastTend, birthBudget, rng: rng.state?.() ?? 0, nextId: ids.get(),
          }),
          set: (k: GardenKeeping) => {
            const g = k as { orgs: ReturnType<typeof orgOut>[]; rotting: ReturnType<typeof orgOut>[]; firstSeen: [number, number][];
              lastArrival: number; lastSlime: number; lastTend: number; birthBudget: number; rng: number; nextId: number }
            rotting.current = g.rotting.map(orgIn)
            firstSeen.clear()
            for (const [id, step] of g.firstSeen) firstSeen.set(id, step)
            lastArrival = g.lastArrival
            lastSlime = g.lastSlime
            lastTend = g.lastTend
            birthBudget = g.birthBudget
            rng.restore?.(g.rng)
            ids.set(g.nextId)
            commit(g.orgs.map(orgIn))
            programsDirty.current = false
          },
          colonies: () => live.current.orgs.length,
        })
        setGarden({ renderer: r, history: historyRef.current })
        const loop = (now: number) => {
          if (!alive) return
          // dev: a recording drives the frames itself
          if (!DEV.hold) tick(now)
          raf = requestAnimationFrame(loop)
        }
        raf = requestAnimationFrame(loop)
        // step frames by hand when the page is in the background
        DEV.readCells = async (n: number) => {
          const d = r.device
          const bytes = n * 64
          const rb = d.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
          const enc = d.createCommandEncoder()
          enc.copyBufferToBuffer((r as unknown as { b: Record<string, GPUBuffer> }).b.cells, 0, rb, 0, bytes)
          d.queue.submit([enc.finish()])
          await rb.mapAsync(GPUMapMode.READ)
          const ab = rb.getMappedRange().slice(0)
          rb.unmap()
          rb.destroy()
          return { f: new Float32Array(ab), u: new Uint32Array(ab) }
        }
        const win = window as unknown as { fungoid?: Record<string, unknown> }
        if (win.fungoid) win.fungoid.readCells = DEV.readCells
        if (win.fungoid) win.fungoid.capture = () => r.capture()
        if (win.fungoid)
          win.fungoid.frames = DEV.frames = async (n: number) => {
            for (let i = 0; i < n; i++) {
              tick(performance.now())
              await r.device.queue.onSubmittedWorkDone()
              await r.statsSettled()
            }
            await new Promise((res) => setTimeout(res, 50))
            return r.stats
          }
        // dev: one frame, as soon as the GPU has it (and the garden has been
        // kept on what it read back, so scripted runs repeat)
        if (win.fungoid)
          win.fungoid.step = async () => {
            tick(performance.now())
            await r.device.queue.onSubmittedWorkDone()
            await r.statsSettled()
          }
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
    return () => {
      alive = false
      cancelAnimationFrame(raf)
    }
  }, [plant])

  // Esc puts the brush down
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setTool('look')
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // camera input
  const drag = useRef<{ x: number; y: number; button: number; moved: number; shift: boolean } | null>(null)
  const toCanvas = (e: React.PointerEvent) => {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    pointer.current.x = ((e.clientX - rect.left) / rect.width) * canvas.width
    pointer.current.y = ((e.clientY - rect.top) / rect.height) * canvas.height
    pointer.current.inside = true
  }
  const onPointerDown = (e: React.PointerEvent) => {
    ;(e.target as Element).setPointerCapture(e.pointerId)
    toCanvas(e)
    // with a tool, the left button paints, or sets a slime mould down (the
    // right one turns the view)
    if (tool !== 'look' && e.button === 0 && !e.shiftKey && !e.altKey) {
      const g = ground.current
      if (tool === 'slime') {
        if (g) byHand(() => rendererRef.current?.addSlime(g))
      } else {
        pointer.current.painting = true
      }
      return
    }
    drag.current = { x: e.clientX, y: e.clientY, button: e.button, moved: 0, shift: e.shiftKey }
  }
  const onPointerMove = (e: React.PointerEvent) => {
    toCanvas(e)
    const d = drag.current
    if (!d) return
    const dx = e.clientX - d.x
    const dy = e.clientY - d.y
    d.x = e.clientX
    d.y = e.clientY
    d.moved += Math.abs(dx) + Math.abs(dy)
    if (d.moved < 4) return
    const cam = camRef.current
    const orbit = d.button === (tool === 'look' ? 0 : 2) && !e.altKey && !d.shift
    if (orbit) cam.orbit(dx, dy)
    else cam.pan(dx, dy, canvasRef.current!.clientHeight)
  }
  const onPointerUp = (e: React.PointerEvent) => {
    pointer.current.painting = false
    const d = drag.current
    drag.current = null
    if (!d || d.moved >= 4 || d.button !== 0 || tool !== 'look') return
    const r = rendererRef.current
    const canvas = canvasRef.current
    if (!r || !canvas) return
    const rect = canvas.getBoundingClientRect()
    const sx = ((e.clientX - rect.left) / rect.width) * canvas.width
    const sy = ((e.clientY - rect.top) / rect.height) * canvas.height
    const shift = e.shiftKey
    r.pick(sx, sy).then(({ slot }) => {
      const o = live.current.orgs.find((q) => q.slot === slot)
      if (shift) setSecond(o ? o.id : null)
      else setSelected(o ? o.id : null)
    })
  }
  const onWheel = (e: React.WheelEvent) => {
    camRef.current.zoom(Math.exp(e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)))
  }
  const focus = useCallback((o: Organism | undefined) => {
    if (!o) return
    const cam = camRef.current
    // (just over it, looking down on it at a slant)
    const r = Math.hypot(o.x, o.y, o.z) || 1
    const up: V3 = [o.x / r, o.y / r, o.z / r]
    cam.target = [up[0] * (PLANET_R + 1), up[1] * (PLANET_R + 1), up[2] * (PLANET_R + 1)]
    cam.height = 12
    const side = Math.hypot(up[0], up[2]) > 1e-3 ? [up[2], 0, -up[0]].map((x) => x / Math.hypot(up[0], up[2])) : [1, 0, 0]
    const back = up.map((x, k) => x * 0.75 + side[k] * 0.66)
    const bl = Math.hypot(...back)
    cam.yaw = Math.atan2(back[0], back[2])
    cam.pitch = Math.asin(back[1] / bl)
  }, [])

  const sel = orgs.find((o) => o.id === selected)
  const sec = orgs.find((o) => o.id === second)
  const add = (seeds: Seed[]) =>
    byHand(() => {
      const { list, added } = plant(seeds, live.current.orgs)
      commit(list)
      if (added.length) setSelected(added[0].id)
    })
  const actions = {
    addPreset: () => add([fromPreset(preset)]),
    random: () => add([{ id: newId(), name: 'Wild', genome: randomGenome(rng, fromPreset(0).genome) }]),
    mutate: () => sel && add([{ id: newId(), name: `${shortName(sel)} mutant`, genome: mutate(sel.genome, rng) }]),
    remove: () => {
      if (!sel) return
      byHand(() => {
        if ((rendererRef.current?.ecology.fruits[sel.slot] ?? 0) > 0) rotting.current.push(sel)
        commit(live.current.orgs.filter((o) => o.id !== sel.id))
      })
      setSelected(null)
    },
    cross: () => {
      if (!sel || !sec) return
      const name = `${shortName(sel)} × ${shortName(sec)}`
      add(Array.from({ length: 3 }, (_, k) => ({ id: newId(), name: `${name} ${k + 1}`, genome: cross(sel.genome, sec.genome, rng) })))
      setSecond(null)
    },
    // same organisms (or these), fresh soil; the garden's life starts over
    // from its seed
    regrow: (seeds?: Seed[]) => {
      rendererRef.current?.resetSim()
      rotting.current = []
      restarted.current = true
      seedLife(WORLD.seed)
      const { list } = plant(seeds ?? orgs.map(({ id, name, genome }) => ({ id, name, genome })), [])
      commit(list)
    },
    // a new layout, replanted with the same organisms (fresh: with the
    // garden's first kinds, as on a first visit)
    terrarium: (seed = Math.floor(visit() * 1e6), fresh = false) => {
      const r = rendererRef.current
      if (!r) return
      layWorld(seed)
      // queued ahead of the regrown garden's first step
      void r.bakeWorld(WORLD.buffer)
      actions.regrow(fresh ? initialSeeds() : undefined)
    },
  }
  DEV.terrarium = actions.terrarium
  DEV.regrow = actions.regrow
  DEV.world = () => WORLD
  DEV.spots = () => SPOTS

  // dev: set a gene (0..1) on every organism
  DEV.gene = (key: string, v: number) =>
    setOrgs((list) => list.map((o) => ({ ...o, genome: setGene(o.genome, GENE_INDEX[key], v) })))

  // dev: regrow a single preset at a chosen spot
  // (or a given organism: { id, name, genome })
  // dev: plant presets as if from the Add button
  DEV.add = (...i: number[]) => add(i.map(fromPreset))
  DEV.solo = (i: number | Seed, at: number[] = SPOTS[0], randomSeed = 0) => {
    rendererRef.current?.resetSim()
    rotting.current = []
    const seed =
      typeof i === 'object' ? i
      : i < 0 ? { id: newId(), name: 'Wild', genome: randomGenome(mulberry32(randomSeed), fromPreset(0).genome) }
      : fromPreset(i)
    const wood = prefersWood(seed.genome)
    const [x, y, z] = wood ? branchSpot(WORLD, rng) : at
    const o: Organism = { ...seed, slot: 0, x, y, z, onSurface: wood }
    commit([o])
    rendererRef.current?.inoculate([{ org: 0, x, y, z, onSurface: o.onSurface }])
    setSelected(o.id)
  }

  const setGeneValue = (i: number, v: number) => {
    if (!sel) return
    byHand(() => commit(live.current.orgs.map((o) => (o.id === sel.id ? { ...o, genome: setGene(o.genome, i, v) } : o))))
  }

  const groups = useMemo(() => {
    const out: Record<string, number[]> = {}
    GENES.forEach((g, i) => (out[g.group] ??= []).push(i))
    return out
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === 'INPUT') return
      if (e.key === 'f') focus(sel)
      if (e.key === ' ') {
        e.preventDefault()
        setSpeed((s) => (s === 0 ? 1 : 0))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [sel, focus])

  const phaseLabel = (v: number) => {
    const k = phaseIndex(v)
    return k >= PHASES ? 'stop' : k === 0 ? 'mycelium' : `phase ${k}`
  }
  const pickLabel = (key: string, v: number) =>
    key === 'decay' ? DECAY_MODES[decayIndex(v)]
    : key.endsWith('In') ? RULE_INPUTS[ruleIndex(v, RULE_INPUTS.length)]
    : key.endsWith('Out') ? RULE_OUTPUTS[ruleIndex(v, RULE_OUTPUTS.length)]
    : phaseLabel(v)

  return (
    <div className="app">
      <canvas
        ref={canvasRef}
        className="viewport"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => {
          pointer.current.inside = false
          pointer.current.painting = false
        }}
        onWheel={onWheel}
        onContextMenu={(e) => e.preventDefault()}
        onDoubleClick={() => focus(sel)}
      />
      {error && <div className="error">{error}</div>}

      <aside className="panel left">
        <header>
          <h1>Fungoid</h1>
          {stats && (
            <div className="stats" title={Object.entries(stats.r.gpuMs).map(([k, v]) => `${k} ${v.toFixed(2)} ms`).join('\n')}>
              {stats.fps.toFixed(0)} fps · GPU {stats.r.gpuTotal.toFixed(1)} ms
              <br />
              {(stats.r.cells / 1000).toFixed(0)}k cells · year {Math.floor(stats.r.step / YEAR) + 1}, {SEASONS[Math.floor(seasonAt(look.current, stats.r.step) * 4) % 4]}, {timeOfDay(look.current, stats.r.step)}
            </div>
          )}
        </header>
        <ul className="shelf">
          {orgs.map((o) => (
            <li
              key={o.id}
              className={o.id === selected ? 'sel' : o.id === second ? 'sec' : ''}
              onClick={(e) => (e.shiftKey ? setSecond(o.id) : setSelected(o.id))}
              onDoubleClick={() => focus(o)}
            >
              <span className="swatch" style={{ background: swatch(o.genome) }} />
              {o.name}
            </li>
          ))}
        </ul>
        <div className="actions">
          <div className="row">
            <select value={preset} onChange={(e) => setPreset(+e.target.value)}>
              {PRESETS.map((p, i) => (
                <option key={p.name} value={i}>
                  {p.name}
                </option>
              ))}
            </select>
            <button onClick={actions.addPreset}>Add</button>
            <button onClick={actions.random}>Random</button>
          </div>
          <div className="row">
            <button disabled={!sel || !sec} onClick={actions.cross} title="Select one parent, shift-select the other">
              Cross
            </button>
            <button disabled={!sel} onClick={actions.mutate}>Mutate</button>
            <button disabled={!sel} onClick={actions.remove}>Remove</button>
          </div>
          <div className="row">
            <select value={debug} onChange={(e) => setDebug(+e.target.value)}>
              {DEBUG_VIEWS.map((v, i) => (
                <option key={v} value={i}>
                  {v}
                </option>
              ))}
            </select>
            <select value={renderScale} onChange={(e) => setRenderScale(+e.target.value)}>
              {SCALES.map((v) => (
                <option key={v} value={v}>
                  {v * 100}%
                </option>
              ))}
            </select>
          </div>
        </div>
      </aside>

      <aside className="panel right">
        {sel ? (
          <>
            <h2>{sel.name}</h2>
            {GROUP_ORDER.map((gk) => (
              <details key={gk} open={gk === 'Life' || gk === 'Color'}>
                <summary>{gk}</summary>
                {groups[gk]?.map((i) => {
                  const v = expressGene(sel.genome, i)
                  const pick = GENES[i].dom === 'pick'
                  return (
                    <label key={GENES[i].key} className="gene">
                      <span>{pick ? `${GENES[i].label}: ${pickLabel(GENES[i].key, v)}` : GENES[i].label}</span>
                      <div className="track">
                        <input type="range" min={0} max={1} step={1 / 255} value={v} onChange={(e) => setGeneValue(i, +e.target.value)} />
                        <i style={{ left: `${(sel.genome.a[i] / 255) * 100}%` }} />
                        <i style={{ left: `${(sel.genome.b[i] / 255) * 100}%` }} />
                      </div>
                    </label>
                  )
                })}
              </details>
            ))}
          </>
        ) : (
          <p className="empty">Click a fungus to edit its growth program. Shift-click a second to cross them.</p>
        )}
      </aside>

      <div className="tools">
        {TOOLS.map((t) => (
          <button key={t} aria-pressed={tool === t} onClick={() => setTool(t)}>
            {t[0].toUpperCase() + t.slice(1)}
          </button>
        ))}
        {tool !== 'look' && tool !== 'slime' && (
          <label>
            <input type="range" min={0.5} max={8} step={0.5} value={brushSize} onChange={(e) => setBrushSize(+e.target.value)} />
            <span>{brushSize} cm</span>
          </label>
        )}
      </div>

      <footer className="timeline">
        <button onClick={() => setSpeed((s) => (s === 0 ? 1 : 0))}>{speed === 0 ? 'Grow' : 'Pause'}</button>
        <select value={speed} onChange={(e) => setSpeed(+e.target.value)}>
          {SPEEDS.map((s, i) => (
            <option key={s} value={i}>
              {s === 0 ? 'Paused' : `${s} steps / frame`}
            </option>
          ))}
        </select>
        <button onClick={() => actions.regrow()}>Regrow</button>
        <button onClick={() => actions.terrarium()}>New terrarium</button>
      </footer>

      <Timeline history={garden?.history ?? null} renderer={garden?.renderer ?? null} frame={timelineFrame} look={look} onSeek={(s, quick) => void historyRef.current?.seek(s, quick)} />
    </div>
  )
}
