export type V3 = [number, number, number]

export const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
export const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
export const scale = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s]
export const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
export const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
export const normalize = (a: V3): V3 => scale(a, 1 / Math.hypot(a[0], a[1], a[2]))

export interface Basis {
  fwd: V3
  right: V3
  up: V3
}

// (up: which way is up on screen, as near as it can be; world y where it
// cannot)
export function basisFromDir(fwd: V3, up: V3 = [0, 1, 0]): Basis {
  let side = cross(fwd, up)
  if (Math.hypot(...side) < 1e-3) side = cross(fwd, [0, 1, 0])
  const right = normalize(side)
  return { fwd, right, up: cross(right, fwd) }
}

// Orthographic orbit camera, about the planet's axis (world y). `height` is
// the visible world height in cm. Close in over the ground, up on screen is
// up there rather than along the axis.
export class IsoCamera {
  target: V3 = [0, 0, 0]
  yaw = Math.PI * 0.25
  pitch = 0.52
  height = 40

  basis(): Basis {
    const cp = Math.cos(this.pitch)
    const back: V3 = [cp * Math.sin(this.yaw), Math.sin(this.pitch), cp * Math.cos(this.yaw)]
    const r = Math.hypot(...this.target)
    const k = r > 5 ? Math.min(1, Math.max(0, (26 - this.height) / 16)) : 0
    const up = normalize(add(scale([0, 1, 0], 1 - k), scale(this.target, k / Math.max(r, 1e-3))))
    return basisFromDir(scale(back, -1), up)
  }

  orbit(dx: number, dy: number) {
    this.yaw -= dx * 0.006
    this.pitch = Math.min(1.5, Math.max(-1.5, this.pitch + dy * 0.005))
  }

  pan(dxPx: number, dyPx: number, viewportH: number) {
    const b = this.basis()
    const k = this.height / viewportH
    // (across the view)
    this.target = add(this.target, add(scale(b.right, -dxPx * k), scale(b.up, dyPx * k)))
  }

  zoom(factor: number) {
    this.height = Math.min(400, Math.max(1.5, this.height * factor))
  }
}

export function halton(i: number, b: number) {
  let f = 1
  let r = 0
  while (i > 0) {
    f /= b
    r += f * (i % b)
    i = Math.floor(i / b)
  }
  return r
}
