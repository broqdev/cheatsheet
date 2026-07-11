import { useEffect, useMemo, useRef, useState } from 'react'
import { AlgorithmPapers } from '../features/cheat-sheet-view/components/AlgorithmPapers'
import { AttentionControls } from '../features/cheat-sheet-view/components/AttentionControls'
import { CatalogDropdown } from '../features/cheat-sheet-view/components/CatalogDropdown'
import { CheatsheetSummary } from '../features/cheat-sheet-view/components/CheatsheetSummary'
import { CodePanel } from '../features/cheat-sheet-view/components/CodePanel'
import { ExampleTabs } from '../features/cheat-sheet-view/components/ExampleTabs'
import {
  exampleFromTag,
  examples,
  examplesForExampleGroup,
  routeSegmentsForExample,
} from '../features/cheat-sheet-view/data/attentionExamples'
import { copyTextToClipboard } from '../features/cheat-sheet-view/lib/clipboard'
import { highlightedCodeLines } from '../features/cheat-sheet-view/lib/highlightCode'
import { latexDocument } from '../features/cheat-sheet-view/lib/equationPresentation'
import type {
  AlgorithmLine,
  AttentionExample,
  CopyTarget,
  VariantKey,
  VariantState,
} from '../features/cheat-sheet-view/model'
import {
  readStoredVariantState,
  resolveVariantView,
  searchForVariantState,
  setVariantEnabled,
  variantStateFromSearch,
  writeStoredVariantState,
} from '../features/cheat-sheet-view/variantSelection'

function siteBasePath() {
  const pathname = new URL(import.meta.env.BASE_URL, window.location.origin).pathname

  return pathname.endsWith('/') ? pathname : `${pathname}/`
}

function routeTagFromLocation() {
  const pathname = window.location.pathname
  const basePath = siteBasePath()
  const pathInsideBase =
    basePath === '/'
      ? pathname.replace(/^\/+/, '')
      : pathname.startsWith(basePath)
        ? pathname.slice(basePath.length)
        : pathname.replace(/^\/+/, '')

  return pathInsideBase.split('/').filter(Boolean).join('-')
}

function exampleFromLocation() {
  return (
    exampleFromTag(routeTagFromLocation()) ??
    exampleFromTag(window.location.hash) ??
    examples[0]
  )
}

function canonicalPathForExample(example: AttentionExample) {
  const routePath = routeSegmentsForExample(example).map(encodeURIComponent).join('/')

  return `${siteBasePath()}${routePath}`
}

function viewStateFromLocation() {
  const example = exampleFromLocation()
  const storedState = readStoredVariantState(example, window.localStorage)
  const view = resolveVariantView(
    example,
    variantStateFromSearch(window.location.search, storedState)
  )

  return { activeExampleId: example.id, variantState: view.state }
}

function CheatSheetViewPage() {
  const [{ activeExampleId, variantState }, setViewState] = useState(viewStateFromLocation)
  const [hoveredLineId, setHoveredLineId] = useState<string | null>(null)
  const [selectedLineId, setSelectedLineId] = useState<string | null>(null)
  const [copiedTarget, setCopiedTarget] = useState<CopyTarget | null>(null)
  const codeRegionRef = useRef<HTMLDivElement>(null)

  const activeExample = exampleFromTag(activeExampleId) ?? examples[0]
  const activeView = useMemo(
    () => resolveVariantView(activeExample, variantState),
    [activeExample, variantState]
  )
  const { content: activeContent, controls } = activeView
  const activeGroupExamples = useMemo(
    () => examplesForExampleGroup(activeExample.id),
    [activeExample.id]
  )
  const activeLineId = selectedLineId ?? hoveredLineId
  const selectableRows = useMemo(
    () => [
      ...(activeContent.prelude?.flatMap((note) => note.rows) ?? []),
      ...activeContent.rows,
      ...(activeContent.notes?.flatMap((note) => note.rows) ?? []),
    ],
    [activeContent]
  )
  const activeLine = selectableRows.find((line) => line.id === activeLineId)
  const activeCodeLines = new Set(activeLine?.codeLines ?? [])
  const codeLines = useMemo(() => highlightedCodeLines(activeContent.code), [activeContent.code])
  const blocks = activeContent.blocks
  const activeLatex = useMemo(
    () => latexDocument(activeExample, activeContent.prelude, blocks, activeContent.notes),
    [activeExample, activeContent.notes, activeContent.prelude, blocks]
  )

  useEffect(() => {
    function syncFromLocation() {
      const { activeExampleId: nextExampleId, variantState: nextVariantState } =
        viewStateFromLocation()
      const example = exampleFromTag(nextExampleId) ?? examples[0]

      setViewState({ activeExampleId: example.id, variantState: nextVariantState })
      setHoveredLineId(null)
      setSelectedLineId(null)

      const nextPath = routeTagFromLocation() || window.location.hash
        ? canonicalPathForExample(example)
        : window.location.pathname
      const nextSearch = searchForVariantState(
        example,
        nextVariantState,
        window.location.search
      )

      if (
        window.location.pathname !== nextPath ||
        window.location.search !== nextSearch ||
        window.location.hash
      ) {
        window.history.replaceState(null, '', `${nextPath}${nextSearch}`)
      }
    }

    syncFromLocation()
    window.addEventListener('hashchange', syncFromLocation)
    window.addEventListener('popstate', syncFromLocation)

    return () => {
      window.removeEventListener('hashchange', syncFromLocation)
      window.removeEventListener('popstate', syncFromLocation)
    }
  }, [])

  function replaceUrlForVariantState(example: AttentionExample, state: VariantState) {
    writeStoredVariantState(example, state, window.localStorage)
    window.history.replaceState(
      null,
      '',
      `${canonicalPathForExample(example)}${searchForVariantState(
        example,
        state,
        window.location.search
      )}`
    )
  }

  function resetExampleState() {
    setHoveredLineId(null)
    setSelectedLineId(null)
  }

  function switchExample(exampleId: string) {
    const nextExample = exampleFromTag(exampleId)

    if (!nextExample) {
      return
    }

    const nextVariantState = resolveVariantView(
      nextExample,
      readStoredVariantState(nextExample, window.localStorage)
    ).state

    setViewState({ activeExampleId: nextExample.id, variantState: nextVariantState })
    resetExampleState()

    const nextPath = canonicalPathForExample(nextExample)
    const nextSearch = searchForVariantState(
      nextExample,
      nextVariantState,
      window.location.search
    )
    const nextUrl = `${nextPath}${nextSearch}`

    if (
      window.location.pathname !== nextPath ||
      window.location.search !== nextSearch ||
      window.location.hash
    ) {
      window.history.pushState(null, '', nextUrl)
    }
  }

  function toggleVariant(key: VariantKey, enabled: boolean) {
    const nextVariantState = setVariantEnabled(activeExample, variantState, key, enabled)

    setViewState({ activeExampleId: activeExample.id, variantState: nextVariantState })
    resetExampleState()
    replaceUrlForVariantState(activeExample, nextVariantState)
  }

  function scrollCodeLineIntoView(codeLineNumbers: number[]) {
    const [firstCodeLine] = codeLineNumbers
    const lastCodeLine = codeLineNumbers.at(-1)
    const codeRegion = codeRegionRef.current

    if (!firstCodeLine || !lastCodeLine || !codeRegion) {
      return
    }

    const targetLine = codeRegion.querySelector<HTMLElement>(
      `[data-line="${firstCodeLine}"]`
    )
    const finalLine = codeRegion.querySelector<HTMLElement>(
      `[data-line="${lastCodeLine}"]`
    )

    if (!targetLine || !finalLine) {
      return
    }

    const gutter = 18
    const regionRect = codeRegion.getBoundingClientRect()
    const targetRect = targetLine.getBoundingClientRect()
    const finalRect = finalLine.getBoundingClientRect()
    const isVisible =
      targetRect.top >= regionRect.top + gutter &&
      finalRect.bottom <= regionRect.bottom - gutter

    if (isVisible) {
      return
    }

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const nextScrollTop = Math.max(
      0,
      codeRegion.scrollTop + targetRect.top - regionRect.top - codeRegion.clientHeight * 0.35
    )

    codeRegion.scrollTo({
      top: nextScrollTop,
      behavior: prefersReducedMotion ? 'auto' : 'smooth',
    })
  }

  function handleAlgorithmRowClick(line: AlgorithmLine) {
    setSelectedLineId(line.id)
    setHoveredLineId(line.id)
    scrollCodeLineIntoView(line.codeLines)
  }

  async function copyRegion(target: CopyTarget, value: string) {
    setCopiedTarget(target)
    window.setTimeout(() => {
      setCopiedTarget((current) => (current === target ? null : current))
    }, 1400)
    await copyTextToClipboard(value)
  }

  return (
    <main className="workspace" aria-label="Equation and code explorer">
      <header className="cheatsheet-header">
        <div className="cheatsheet-title">
          <h1>Broq Cheatsheet</h1>
        </div>
        <CatalogDropdown
          activeLabel={activeExample.label}
          activeExampleId={activeExample.id}
          onSelectExample={switchExample}
        />
      </header>

      <CheatsheetSummary description={activeExample.description} />

      <section
        className="main-panel"
        id="cheatsheet-panel"
        aria-label={`${activeExample.label} equations and code`}
        role="tabpanel"
      >
        <AlgorithmPapers
          activeExample={activeExample}
          activeLineId={activeLine?.id}
          blocks={blocks}
          latexCopied={copiedTarget === 'latex'}
          notes={activeContent.notes}
          prelude={activeContent.prelude}
          onCopyLatex={() => copyRegion('latex', activeLatex)}
          onLineActivate={handleAlgorithmRowClick}
          onLineFocus={setHoveredLineId}
          onLineLeave={() => setHoveredLineId(null)}
          selectedLineId={selectedLineId}
        />

        <CodePanel
          ref={codeRegionRef}
          activeCodeLines={activeCodeLines}
          codeCopied={copiedTarget === 'code'}
          codeLines={codeLines}
          label={activeExample.label}
          onCopyCode={() => copyRegion('code', activeContent.code)}
        />
      </section>

      <AttentionControls controls={controls} onToggle={toggleVariant} />

      <ExampleTabs
        activeExample={activeExample}
        examples={activeGroupExamples}
        onSwitchExample={switchExample}
      />
    </main>
  )
}

export default CheatSheetViewPage
