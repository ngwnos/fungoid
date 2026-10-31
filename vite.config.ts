import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

// Dev only: the page encodes video on the GPU (WebCodecs) and streams the
// H.264 here; ffmpeg wraps it in renders/<name>.mp4 without re-encoding (see
// window.fungoid.record in src/dev.ts). Stills go to renders/<name>.png.
function recorder(): Plugin {
  let ff: ChildProcessWithoutNullStreams | null = null
  let done: Promise<number> | null = null
  const body = (req: NodeJS.ReadableStream) =>
    new Promise<Buffer>((ok, fail) => {
      const parts: Buffer[] = []
      req.on('data', (c: Buffer) => parts.push(c))
      req.on('end', () => ok(Buffer.concat(parts)))
      req.on('error', fail)
    })
  return {
    name: 'fungoid-recorder',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__record', async (req, res) => {
        const url = new URL(req.url ?? '/', 'http://local')
        try {
          if (url.pathname === '/png') {
            const name = (url.searchParams.get('name') ?? 'still').replace(/[^\w.-]/g, '_')
            const dir = resolve(__dirname, 'renders')
            mkdirSync(dir, { recursive: true })
            const out = resolve(dir, `${name}.png`)
            writeFileSync(out, await body(req))
            res.end(out)
          } else if (url.pathname === '/start') {
            const fps = Number(url.searchParams.get('fps'))
            const name = (url.searchParams.get('name') ?? 'render').replace(/[^\w.-]/g, '_')
            const dir = resolve(__dirname, 'renders')
            mkdirSync(dir, { recursive: true })
            const out = resolve(dir, `${name}.mp4`)
            ff = spawn('ffmpeg', [
              '-y', '-loglevel', 'error',
              // raw H.264 has no timestamps: number the frames at the recording rate
              '-fflags', '+genpts', '-r', String(fps), '-f', 'h264', '-i', '-',
              '-c:v', 'copy', '-movflags', '+faststart',
              out,
            ])
            ff.stderr.on('data', (d) => server.config.logger.error(`[ffmpeg] ${d}`))
            done = new Promise((ok) => ff!.on('close', (code) => ok(code ?? -1)))
            res.end(out)
          } else if (url.pathname === '/data') {
            const data = await body(req)
            if (!ff) throw new Error('not recording')
            if (!ff.stdin.write(data)) await new Promise((ok) => ff!.stdin.once('drain', ok))
            res.end('ok')
          } else if (url.pathname === '/end') {
            ff?.stdin.end()
            const code = await done
            ff = null
            res.end(String(code))
          } else {
            res.statusCode = 404
            res.end()
          }
        } catch (e) {
          res.statusCode = 500
          res.end(String(e))
        }
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), recorder()],
  server: { host: '127.0.0.1', port: 5244, strictPort: true },
})
