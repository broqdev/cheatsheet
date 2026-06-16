import { existsSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { chromium } from 'playwright-core'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const input = resolve(root, process.argv[2] ?? 'cover/og-image.html')
const output = resolve(root, process.argv[3] ?? 'public/og-image.png')

const chromeCandidates = [
  process.env.CHROME_BIN,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
].filter(Boolean)

const chrome = chromeCandidates.find((candidate) => existsSync(candidate))

if (!chrome) {
  console.error('Could not find Chrome or Chromium. Set CHROME_BIN to a browser executable.')
  process.exit(1)
}

if (!existsSync(input)) {
  console.error(`Cover HTML not found: ${input}`)
  process.exit(1)
}

const browser = await chromium.launch({
  executablePath: chrome,
  headless: true,
  args: ['--disable-gpu', '--hide-scrollbars', '--no-sandbox'],
})

try {
  const context = await browser.newContext({
    deviceScaleFactor: 1,
    viewport: { width: 1200, height: 630 },
  })
  const page = await context.newPage()

  await page.goto(pathToFileURL(input).href, { waitUntil: 'load' })
  await page.evaluate(() => document.fonts?.ready)
  await page.screenshot({
    path: output,
    animations: 'disabled',
    fullPage: false,
    type: 'png',
  })

  console.log(`Rendered ${output}`)
} finally {
  await browser.close()
}
