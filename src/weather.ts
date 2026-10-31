// The planet's calendar and weather, as the simulation has them (world.wgsl):
// the sun's place in the sky over any point at a step, how dark it is there,
// the season, and how damp the planet is.

type V3 = [number, number, number]

const pcg = (v: number) => {
  const s = (Math.imul(v, 747796405) + 2891336453) >>> 0
  const w = Math.imul((s >>> ((s >>> 28) + 4)) ^ s, 277803737) >>> 0
  return ((w >>> 22) ^ w) >>> 0
}
const hash2 = (a: number, b: number) => pcg((a ^ pcg((b + 0x9e3779b9) >>> 0)) >>> 0)
const u01 = (h: number) => (h >>> 8) / 16777216
const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)))
  return t * t * (3 - 2 * t)
}

// A day: SOL steps from one midnight to the next at longitude 0 (0.5 its
// noon; a garden starts in the morning there). A year: YEAR steps round the
// sun from the northern autumn's equinox. The planet spins about y, tilted
// TILT to its orbit.
export const SOL = 4800
export const YEAR = 96000
const TILT = 0.41

export const yearPhase = (step: number) => (step % YEAR) / YEAR
export const dayPhase = (step: number) => ((step + SOL * 0.3) % SOL) / SOL
// towards the sun
export function sunDir(step: number): V3 {
  const sd = -Math.sin(TILT) * Math.sin(2 * Math.PI * yearPhase(step))
  const cd = Math.sqrt(1 - sd * sd)
  const a = 2 * Math.PI * (dayPhase(step) - 0.5)
  return [cd * Math.cos(a), sd, cd * Math.sin(a)]
}
// A terrarium's site: latitude `lat` (radians; longitude 0), as its frame in
// the planet's: up there, and the box's x and z axes, turned so the sun at
// its noon stands behind the box and to the left as seen from the front (+x
// +z: as a lamp once stood), north or south of the equator. shape_box.wgsl
// turns the planet's sun into the box with these.
const NOON = [Math.cos(2.83), Math.sin(2.83)]
export interface Site {
  lat: number
  up: V3
  x: V3
  z: V3
}
export function siteFrame(lat: number): Site {
  const U: V3 = [Math.cos(lat), Math.sin(lat), 0]
  const N: V3 = [-Math.sin(lat), Math.cos(lat), 0]
  // (toward the equator, and along the sun's way across the sky)
  const Q = N.map((v) => (lat >= 0 ? -v : v)) as V3
  const K: V3 = [U[1] * Q[2] - U[2] * Q[1], U[2] * Q[0] - U[0] * Q[2], U[0] * Q[1] - U[1] * Q[0]]
  // (Q to the noon direction; K, the way the sun goes on, a quarter turn
  // from it)
  const [wx, wz] = NOON
  return { lat, up: U, x: [0, 1, 2].map((k) => wx * Q[k] + wz * K[k]) as V3, z: [0, 1, 2].map((k) => wz * Q[k] - wx * K[k]) as V3 }
}
const dot3 = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
// a direction in the planet's frame, in the terrarium's
export const toWorld = (f: Site, v: V3): V3 => [dot3(f.x, v), dot3(f.up, v), dot3(f.z, v)]
// towards the sun, in the world's frame (the terrarium's, if at a site)
export const sunWorld = (step: number, site: Site | null): V3 => (site ? toWorld(site, sunDir(step)) : sunDir(step))

// how high the sun stands over direction `d` (the sine of its elevation)
export function sunElev(d: V3, step: number) {
  const s = sunDir(step)
  const l = Math.hypot(...d) || 1
  return (d[0] * s[0] + d[1] * s[1] + d[2] * s[2]) / l
}
// 1 in the dark of night .. 0 by day
export const nightness = (d: V3, step: number) => 1 - smooth(-0.12, 0.22, sunElev(d, step))
// the season over `d` (0 the start of autumn, 0.25 winter...): the other way
// about in the south
export const seasonAt = (d: V3, step: number) => (yearPhase(step) + (d[1] < 0 ? 0.5 : 0)) % 1

// The weather over direction d. The seasons there, the more marked the
// nearer the poles: autumn and winter soaking, summer drier, damp all year
// near the equator.
const seasonal = (d: V3) => smooth(0.05, 0.6, Math.abs(d[1]) / (Math.hypot(...d) || 1))
// 0.57 in high summer .. 1 in winter (0.82 near the equator)
const seasonWet = (d: V3, step: number) => 0.82 + 0.25 * seasonal(d) * Math.cos(2 * Math.PI * (seasonAt(d, step) - 0.25))
// Storms carried round the planet, as world.wgsl rainWet has them.
const STORMS = 6
const STORM_LIFE = 2400
const PLANET_R = 13.5
function rainWet(d: V3, step: number) {
  const l = Math.hypot(...d) || 1
  const lat = Math.asin(Math.max(-1, Math.min(1, d[1] / l)))
  const lon = Math.atan2(d[2], d[0])
  const TAU = 2 * Math.PI
  let r = 0
  for (let n = 0; n < STORMS; n++) {
    const t = step + n * Math.floor(STORM_LIFE / STORMS)
    const g = Math.floor(t / STORM_LIFE)
    for (let k = 0; k < 2 && k <= g; k++) {
      const gen = g - k
      const age = t - gen * STORM_LIFE
      const h = hash2((gen * STORMS + n) >>> 0, 0x5707)
      const sLat = Math.asin(u01(h) * 1.8 - 0.9)
      const lon0 = u01(pcg(h)) * TAU
      const w = (3 + 3 * u01(pcg((h + 1) >>> 0))) / PLANET_R
      const spin = ((0.5 + 0.7 * u01(pcg((h + 2) >>> 0))) * (u01(pcg((h + 3) >>> 0)) < 0.5 ? 1 : -1)) / STORM_LIFE
      const strength = (0.35 + 0.45 * u01(pcg((h + 4) >>> 0))) * (0.45 + 0.55 * (seasonWet([Math.cos(sLat), Math.sin(sLat), 0], step) - 0.57) / 0.5)
      const f = ((lon - lon0) / TAU) * Math.sign(spin) + 1
      const ahead = (f - Math.floor(f)) * TAU
      const when = ahead / Math.abs(spin)
      const gone = Math.min(age, STORM_LIFE)
      let dLon = 0
      let at = when
      if (when > gone) {
        const back = TAU - ahead
        dLon = Math.min(ahead - gone * Math.abs(spin), back)
        at = back < ahead - gone * Math.abs(spin) ? 0 : gone
      }
      const dd = (lat - sLat) ** 2 + (dLon * Math.cos(lat)) ** 2
      const life = Math.sin(Math.PI * Math.max(0, Math.min(1, at / STORM_LIFE)))
      r += strength * life * Math.exp(-dd / (w * w)) * Math.exp(-Math.max(age - at, 0) / 900)
    }
  }
  return r
}
// 1 wet .. 0 dry
export const climate = (d: V3, step: number) => Math.min(1, Math.max(0, seasonWet(d, step) + rainWet(d, step)))
