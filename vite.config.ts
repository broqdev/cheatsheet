import { copyFileSync } from 'node:fs'
import { join } from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

const repoName = process.env.GITHUB_REPOSITORY?.split('/')[1]
const githubPagesBase =
  process.env.GITHUB_ACTIONS === 'true' && repoName ? `/${repoName}/` : '/'
const base = process.env.VITE_BASE_PATH ?? githubPagesBase

const siteTitle = 'Broq Cheatsheet'
const siteDescription =
  'Broq Cheatsheet maps naive attention, FlashAttention-1, FlashAttention-2, and FlashAttention-3 equations to the PyTorch and Triton-style code beside them.'

function withTrailingSlash(value: string) {
  return value.endsWith('/') ? value : `${value}/`
}

function siteUrl() {
  const configuredUrl = process.env.VITE_SITE_URL

  if (configuredUrl) {
    return withTrailingSlash(configuredUrl)
  }

  const owner = process.env.GITHUB_REPOSITORY_OWNER ?? 'broqdev'
  const repository = repoName ?? 'cheatsheet'

  return `https://${owner}.github.io/${repository}/`
}

function escapedJson(value: unknown) {
  return JSON.stringify(value)
    .replace(/</g, '\\u003c')
    .replace(/>/g, '\\u003e')
    .replace(/&/g, '\\u0026')
}

function seoPlugin(): Plugin {
  const canonicalUrl = siteUrl()
  const normalizedBase = base.endsWith('/') ? base : `${base}/`
  const structuredData = {
    '@context': 'https://schema.org',
    '@type': ['WebApplication', 'LearningResource'],
    name: siteTitle,
    alternateName: 'Attention Math and Code',
    description: siteDescription,
    url: canonicalUrl,
    applicationCategory: 'EducationalApplication',
    operatingSystem: 'Any',
    inLanguage: 'en',
    isAccessibleForFree: true,
    learningResourceType: 'Interactive cheatsheet',
    about: [
      { '@type': 'Thing', name: 'FlashAttention' },
      { '@type': 'Thing', name: 'FlashAttention-1' },
      { '@type': 'Thing', name: 'FlashAttention-2' },
      { '@type': 'Thing', name: 'FlashAttention-3' },
      { '@type': 'Thing', name: 'Naive attention' },
      { '@type': 'Thing', name: 'Attention mechanism' },
      { '@type': 'Thing', name: 'Online softmax' },
      { '@type': 'Thing', name: 'Triton kernels' },
    ],
    teaches: [
      'Naive attention forward and backward passes',
      'FlashAttention forward pass',
      'FlashAttention backward pass',
      'FlashAttention-2 work partitioning',
      'FlashAttention-3 warp specialization',
      'Masked attention',
      'Softmax backward pass',
    ],
  }

  return {
    name: 'flash-attention-seo',
    transformIndexHtml(html) {
      return html
        .replaceAll('%SITE_URL%', canonicalUrl)
        .replaceAll('%BASE_URL%', normalizedBase)
        .replace('%STRUCTURED_DATA%', escapedJson(structuredData))
    },
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'robots.txt',
        source: `User-agent: *\nAllow: /\nSitemap: ${canonicalUrl}sitemap.xml\n`,
      })
      this.emitFile({
        type: 'asset',
        fileName: 'sitemap.xml',
        source: `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n  <url>\n    <loc>${canonicalUrl}</loc>\n    <changefreq>weekly</changefreq>\n    <priority>1.0</priority>\n  </url>\n</urlset>\n`,
      })
    },
    writeBundle(options) {
      if (!options.dir) {
        return
      }

      copyFileSync(join(options.dir, 'index.html'), join(options.dir, '404.html'))
    },
  }
}

export default defineConfig({
  base,
  plugins: [tailwindcss(), react(), seoPlugin()],
})
