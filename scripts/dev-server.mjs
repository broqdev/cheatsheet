import { spawn } from 'node:child_process'
import { createServer } from 'node:net'
import { argv, exit, platform } from 'node:process'

const host = '127.0.0.1'

function findFreeLoopbackPort() {
  return new Promise((resolve, reject) => {
    const server = createServer()

    server.unref()
    server.once('error', reject)
    server.listen(0, host, () => {
      const address = server.address()

      server.close((error) => {
        if (error) {
          reject(error)
          return
        }

        if (!address || typeof address === 'string') {
          reject(new Error('Could not determine a free loopback port.'))
          return
        }

        resolve(address.port)
      })
    })
  })
}

function withoutControlledViteOptions(args) {
  const filtered = []

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index]

    if (arg === '--host' || arg === '--port') {
      index += 1
      continue
    }

    if (
      arg === '--strictPort' ||
      arg === '--no-strictPort' ||
      arg.startsWith('--host=') ||
      arg.startsWith('--port=')
    ) {
      continue
    }

    filtered.push(arg)
  }

  return filtered
}

const port = await findFreeLoopbackPort()
const viteArgs = [
  '--host',
  host,
  '--port',
  String(port),
  ...withoutControlledViteOptions(argv.slice(2)),
]

const vite = spawn('vite', viteArgs, {
  shell: platform === 'win32',
  stdio: 'inherit',
})

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    vite.kill(signal)
  })
}

vite.once('error', (error) => {
  console.error(`Failed to start Vite: ${error.message}`)
  exit(1)
})

vite.once('close', (code) => {
  exit(code ?? 0)
})
