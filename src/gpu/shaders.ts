import common from '../shaders/common.wgsl?raw'
import world from '../shaders/world.wgsl?raw'
import worldbake from '../shaders/worldbake.wgsl?raw'
import cells from '../shaders/cells.wgsl?raw'
import cellprog from '../shaders/cellprog.wgsl?raw'
import fruit from '../shaders/fruit.wgsl?raw'
import fruitRaster from '../shaders/fruitraster.wgsl?raw'
import fruitDev from '../shaders/fruitdev.wgsl?raw'
import bakeGround from '../shaders/bake_ground.wgsl?raw'
import bakeVolume from '../shaders/bake_volume.wgsl?raw'
import bakeAux from '../shaders/bake_aux.wgsl?raw'
import sim from '../shaders/sim.wgsl?raw'
import slime from '../shaders/slime.wgsl?raw'
import slimeSim from '../shaders/slimesim.wgsl?raw'
import args from '../shaders/args.wgsl?raw'
import tissue from '../shaders/tissue.wgsl?raw'
import shade from '../shaders/shade.wgsl?raw'
import bakeSun from '../shaders/bake_sun.wgsl?raw'
import temporal from '../shaders/temporal.wgsl?raw'
import post from '../shaders/post.wgsl?raw'
import present from '../shaders/present.wgsl?raw'
import downsample from '../shaders/downsample.wgsl?raw'
import glow from '../shaders/glow.wgsl?raw'
import graph from '../shaders/graph.wgsl?raw'
import shapePlanet from '../shaders/shape_planet.wgsl?raw'
import shapeBox from '../shaders/shape_box.wgsl?raw'
import { phenoWgsl } from '../genome/phenotype'
import { FLAT, dimsWgsl } from './dims'

// The world's shape comes in sections (shape_*.wgsl), each only in the
// shaders that bind what it uses: pure (all), ground (the ground map), bake
// (the ground bake), sim, shade.
function sections(src: string) {
  const out: Record<string, string> = { pure: '', ground: '', bake: '', sim: '', shade: '' }
  let at = 'pure'
  for (const line of src.split('\n')) {
    const m = /^\/\/== (\w+)/.exec(line)
    if (m) at = m[1]
    else out[at] += line + '\n'
  }
  return out
}
const shape = sections(FLAT ? shapeBox : shapePlanet)

// WGSL has no includes; modules are concatenated in dependency order.
const pre = [dimsWgsl(), common, phenoWgsl(), world, shape.pure].join('\n')
const join = (...parts: string[]) => [pre, ...parts].join('\n')

export const SHADERS = {
  bakeGround: join(shape.bake, bakeGround),
  bakeVolume: join(worldbake, shape.ground, bakeVolume),
  bakeAux: join(worldbake, bakeAux),
  sim: join(cells, cellprog, fruit, slime, shape.ground, shape.sim, sim, slimeSim),
  args: join(cells, args),
  tissue: join(cells, cellprog, tissue),
  glow: join(cells, cellprog, fruit, glow),
  fruitRaster: join(fruit, fruitDev, fruitRaster),
  shade: join(cells, cellprog, fruit, fruitDev, slime, shape.ground, shape.shade, shade),
  bakeSun: join(bakeSun),
  temporal: join(temporal),
  post: join(post),
  present: join(present),
  downsample,
  graph: join(graph),
}

export type ShaderName = keyof typeof SHADERS

export async function compile(device: GPUDevice, name: ShaderName): Promise<GPUShaderModule> {
  const code = SHADERS[name]
  const module = device.createShaderModule({ label: name, code })
  const info = await module.getCompilationInfo()
  const lines = code.split('\n')
  let failed = false
  for (const m of info.messages) {
    const src = lines[m.lineNum - 1] ?? ''
    const msg = `[${name}] ${m.type} ${m.lineNum}:${m.linePos} ${m.message}\n  ${src.trim()}`
    if (m.type === 'error') {
      failed = true
      console.error(msg)
    } else console.warn(msg)
  }
  if (failed) throw new Error(`WGSL compile failed: ${name}`)
  return module
}
