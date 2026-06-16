# Flash Attention Lab

Static Vite + React project site scaffolded for GitHub Pages.

## Local Development

```sh
npm install
npm run dev
```

## Build

```sh
npm run build
npm run preview
```

## Social Cover

The X/Open Graph cover image is rendered from editable HTML with Playwright + Chrome:

```sh
npm run render:cover
```

Edit `cover/og-image.html` to tweak layout, colors, and text. The render script opens that HTML in Chromium at a fixed 1200x630 viewport and writes `public/og-image.png`, which is the image referenced by the page metadata.

The Vite config automatically uses `/<repo-name>/` as the base path when building in GitHub Actions, which matches GitHub Pages project-site URLs like:

```text
https://<user>.github.io/pj_flash_attention/
```

For a custom domain, set `VITE_BASE_PATH=/` in the GitHub Actions build environment.
Set `VITE_SITE_URL=https://your-domain.example/` as well so canonical metadata, Open Graph links, `robots.txt`, and `sitemap.xml` point at the production URL.

## Deploy

The workflow at `.github/workflows/deploy.yml` builds `dist/` and publishes it with GitHub Pages Actions. In the repository settings, set:

```text
Settings -> Pages -> Build and deployment -> Source -> GitHub Actions
```
