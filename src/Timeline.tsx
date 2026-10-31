import { useEffect, useRef, useState, type MutableRefObject } from 'react'
import type { Renderer } from './gpu/renderer'
import { GRAPH_PAD, GRAPH_RGB, Graph } from './gpu/graph'
import type { History } from './history'
import { YEAR, climate, nightness, seasonAt } from './weather'

// The garden's history as a bar as long as it is, with its measures graphed
// over it (gpu/graph.ts): mycelium, mushrooms and how damp it is filled; the
// food stored, the colonies and the slime mould as lines; nights and seasons
// (where on the planet the view is), checkpoints and years. The playhead is where the garden is, dragged to go
// back and forth (history.ts); what has gone from the store at the start is
// hatched and cannot be gone back to. The legend reads the graphs at the
// playhead (or the pointer); hovering an entry brings its graph forward,
// clicking it hides it. All of it is drawn once a frame with the garden
// (App.tsx calls `frame`), not by React.

const SERIES = [
  { label: 'Mycelium', field: 1, mark: 'fill' },
  { label: 'Mushrooms', field: 2, mark: 'fill' },
  { label: 'Damp', field: -1, mark: 'fill' },
  { label: 'Food stored', field: 3, mark: 'line' },
  { label: 'Colonies', field: 4, mark: 'dash' },
  { label: 'Slime mould', field: 5, mark: 'line' },
]
const SLIME = 5
const SEASONS = ['autumn', 'winter', 'spring', 'summer']
// the chart above the bar, and the bar (CSS pixels)
const CHART = 96
const BAR = 18

const fmt = (n: number) => (n >= 10000 ? `${Math.round(n / 1000)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${Math.round(n)}`)

export function Timeline({ history, renderer, frame, look, onSeek }: {
  history: History | null
  renderer: Renderer | null
  // set to what draws it (`force`: even if nothing changed)
  frame: MutableRefObject<((force?: boolean) => void) | null>
  // where on the planet the view is (a direction)
  look: { current: [number, number, number] }
  onSeek: (step: number, quick: boolean) => void
}) {
  const root = useRef<HTMLDivElement>(null)
  const area = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const years = useRef<HTMLDivElement>(null)
  const now = useRef<HTMLDivElement>(null)
  const values = useRef<(HTMLElement | null)[]>([])
  const graph = useRef<Graph | null>(null)
  const [hidden, setHidden] = useState(0)
  const [focus, setFocus] = useState(-1)
  const [slime, setSlime] = useState(false)
  // what the pointer is doing (read each frame)
  const view = useRef({ drag: null as number | null, hover: -1, hidden: 0, focus: -1 })
  view.current.hidden = hidden | (slime ? 0 : 1 << SLIME)
  view.current.focus = focus

  // the controls above sit clear of it
  useEffect(() => {
    const el = root.current!
    const ro = new ResizeObserver(() => document.documentElement.style.setProperty('--history-h', `${el.offsetHeight}px`))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    if (!renderer || !canvas.current) return
    let gone = false
    void Graph.create(renderer.device, canvas.current, renderer.format).then((g) => {
      if (gone) g.destroy()
      else graph.current = g
      // (dev: fungoid.graph.readColumns)
      if (import.meta.env.DEV) Object.assign((window as { fungoid?: object }).fungoid ?? {}, { graph: g })
    })
    return () => {
      gone = true
      graph.current?.destroy()
      graph.current = null
    }
  }, [renderer])

  useEffect(() => {
    if (!history) return
    let last = ''
    // (drawn on for a while after anything changes: the scales follow
    // gradually, graph.wgsl `ease`)
    let settle = 0
    let version = -1
    let read = -1
    let readCount = -1
    let readLook = ''
    let yearsShown = ''
    let slimeTo = 0
    let slimeSeen = false
    frame.current = (force = false) => {
      const g = graph.current
      const cv = canvas.current
      if (!g || !cv) return
      const V = view.current
      const S = history.samples
      const head = history.head
      const at = V.drag ?? history.at
      // the slime mould's graph, once there has been any
      if (S.count < slimeTo) {
        slimeTo = 0
        slimeSeen = false
        setSlime(false)
      }
      for (; slimeTo < S.count && !slimeSeen; slimeTo++) {
        if (S.value(slimeTo, 5) > 0) {
          slimeSeen = true
          setSlime(true)
        }
      }
      if (slimeSeen) slimeTo = S.count
      const lk = look.current
      const key = `${at},${head},${history.first},${S.count},${history.version},${V.hover},${V.hidden},${V.focus},${cv.clientWidth},${cv.clientHeight},${devicePixelRatio},${lk.map((x) => x.toFixed(2))}`
      const t = performance.now()
      if (key !== last || force) {
        last = key
        settle = t + 4000
      }
      if (t < settle) {
        g.draw(S, history.checkpoints, history.version !== version, { head, first: history.first, at, hover: V.hover, hidden: V.hidden, focus: V.focus, chart: CHART, look: lk })
        version = history.version
      }
      // the legend, at the playhead or the pointer
      const step = V.drag ?? (V.hover >= 0 ? V.hover : at)
      const lkKey = lk.map((x) => x.toFixed(2)).join()
      if (step !== read || S.count !== readCount || lkKey !== readLook) {
        read = step
        readCount = S.count
        readLook = lkKey
        const i = S.at(step)
        SERIES.forEach((s, k) => {
          const el = values.current[k]
          if (!el) return
          const t = s.field < 0 ? `${Math.round(climate(look.current, step) * 100)}%` : i < 0 ? '' : fmt(S.value(i, s.field))
          if (el.textContent !== t) el.textContent = t
        })
        const t = `year ${Math.floor(step / YEAR) + 1} · ${SEASONS[Math.floor(seasonAt(lk, step) * 4) % 4]} · ${nightness(lk, step) > 0.5 ? 'night' : 'day'}`
        if (now.current && now.current.textContent !== t) now.current.textContent = t
      }
      // the years
      const box = years.current
      const n = Math.max(0, Math.ceil(head / YEAR) - 1)
      const yk = `${n},${head},${cv.clientWidth}`
      if (box && yk !== yearsShown) {
        yearsShown = yk
        while (box.children.length < n) box.appendChild(document.createElement('i')).appendChild(document.createElement('span'))
        while (box.children.length > n) box.lastChild!.remove()
        const every = Math.max(1, Math.ceil(n / 12))
        const w = cv.clientWidth - 2 * GRAPH_PAD
        for (let y = 1; y <= n; y++) {
          const el = box.children[y - 1] as HTMLElement
          el.style.transform = `translateX(${((y * YEAR) / Math.max(1, head)) * w}px)`
          const label = y % every === 0 ? `year ${y + 1}` : ''
          if (el.firstChild!.textContent !== label) el.firstChild!.textContent = label
        }
      }
    }
    return () => {
      frame.current = null
    }
  }, [history, frame, look])

  const stepAt = (x: number) => {
    const r = area.current!.getBoundingClientRect()
    const t = (x - r.left) / Math.max(1, r.width)
    return Math.round(Math.min(1, Math.max(0, t)) * (history?.head ?? 0))
  }

  return (
    <div className="history" ref={root}>
      <div className="legend">
        {SERIES.map((s, k) =>
          k === SLIME && !slime ? null : (
            <button
              key={s.label}
              className={(hidden >> k) & 1 ? 'off' : undefined}
              onClick={() => setHidden(hidden ^ (1 << k))}
              onPointerEnter={() => setFocus(k)}
              onPointerLeave={() => setFocus(-1)}
            >
              <i className={s.mark} style={{ color: `rgb(${GRAPH_RGB[k].join(', ')})` }} />
              <span>{s.label}</span>
              <em ref={(el) => void (values.current[k] = el)} />
            </button>
          ),
        )}
        <div className="now" ref={now} />
      </div>
      <div
        className="area"
        ref={area}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          const st = stepAt(e.clientX)
          view.current.drag = st
          onSeek(st, true)
        }}
        onPointerMove={(e) => {
          const st = stepAt(e.clientX)
          view.current.hover = st
          if (view.current.drag === null) return
          view.current.drag = st
          onSeek(st, true)
        }}
        onPointerUp={(e) => {
          if (view.current.drag === null) return
          onSeek(stepAt(e.clientX), false)
          view.current.drag = null
        }}
        onPointerLeave={() => (view.current.hover = -1)}
      >
        <canvas ref={canvas} style={{ height: CHART + BAR }} />
        <div className="years" ref={years} />
      </div>
    </div>
  )
}
