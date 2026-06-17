import { type CSSProperties, useEffect, useMemo, useRef, useState } from 'react'
import { AlgorithmPapers } from '../features/cheat-sheet-view/components/AlgorithmPapers'
import { AttentionControls } from '../features/cheat-sheet-view/components/AttentionControls'
import { CatalogDropdown } from '../features/cheat-sheet-view/components/CatalogDropdown'
import { CheatsheetSummary } from '../features/cheat-sheet-view/components/CheatsheetSummary'
import { CodePanel } from '../features/cheat-sheet-view/components/CodePanel'
import { ExampleTabs } from '../features/cheat-sheet-view/components/ExampleTabs'
import {
  examples,
  examplesForExampleGroup,
} from '../features/cheat-sheet-view/data/attentionExamples'
import { algorithmBlocks } from '../features/cheat-sheet-view/lib/algorithmBlocks'
import { copyTextToClipboard } from '../features/cheat-sheet-view/lib/clipboard'
import { highlightedCodeLines } from '../features/cheat-sheet-view/lib/highlightCode'
import { latexDocument } from '../features/cheat-sheet-view/lib/latexDocument'
import { renderInlineLatex } from '../features/cheat-sheet-view/lib/renderLatex'
import { toggleDeltaColors } from '../features/cheat-sheet-view/lib/toggleDeltas'
import type {
  AlgorithmLine,
  AttentionExample,
  AttentionMode,
  CopyTarget,
  Segment,
} from '../features/cheat-sheet-view/model'

function normalizeUrlTag(value: string) {
  const routeValue = value.replace(/^#/, '').replace(/^\/+|\/+$/g, '')
  let decodedValue = routeValue

  try {
    decodedValue = decodeURIComponent(routeValue)
  } catch {
    decodedValue = routeValue
  }

  return decodedValue
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function urlTagsForExample(example: AttentionExample) {
  return [example.id, example.urlTag, example.label].map(normalizeUrlTag)
}

function exampleFromUrlTag(value: string) {
  const tag = normalizeUrlTag(value)

  if (!tag) {
    return undefined
  }

  return examples.find((example) => urlTagsForExample(example).includes(tag))
}

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
    exampleFromUrlTag(routeTagFromLocation()) ??
    exampleFromUrlTag(window.location.hash) ??
    examples[0]
  )
}

function canonicalPathForExample(example: AttentionExample) {
  const routePath = example.urlTag
    .split('/')
    .map(normalizeUrlTag)
    .filter(Boolean)
    .map(encodeURIComponent)
    .join('/')

  return `${siteBasePath()}${routePath}`
}

type ToggleQueryState = {
  attentionMaskEnabled: boolean
  dropoutEnabled: boolean
  fp8Enabled: boolean
  weightDecayEnabled: boolean
  moonshotLrEnabled: boolean
  momentumEnabled: boolean
}

function queryToggleEnabled(params: URLSearchParams, key: string) {
  const value = params.get(key)?.toLowerCase()

  return ['1', 'true', 'yes', 'on'].includes(value ?? '')
}

function toggleStateFromSearch(search: string): ToggleQueryState {
  const params = new URLSearchParams(search)

  return {
    attentionMaskEnabled: queryToggleEnabled(params, 'mask'),
    dropoutEnabled: queryToggleEnabled(params, 'dropout'),
    fp8Enabled: queryToggleEnabled(params, 'fp8'),
    weightDecayEnabled:
      queryToggleEnabled(params, 'weightDecay') || queryToggleEnabled(params, 'wd'),
    moonshotLrEnabled:
      queryToggleEnabled(params, 'moonshotLr') ||
      queryToggleEnabled(params, 'moonshot') ||
      queryToggleEnabled(params, 'mslr'),
    momentumEnabled:
      queryToggleEnabled(params, 'momentum') || queryToggleEnabled(params, 'mom'),
  }
}

function attentionModeForState(toggleState: ToggleQueryState): AttentionMode {
  return toggleState.attentionMaskEnabled ? 'masked' : 'unmasked'
}

function hasAttentionMaskToggle(example: AttentionExample) {
  return example.content.masked !== example.content.unmasked
}

function availableToggleState(example: AttentionExample, toggleState: ToggleQueryState) {
  const attentionMaskEnabled = hasAttentionMaskToggle(example) && toggleState.attentionMaskEnabled
  const attentionMode = attentionModeForState({ ...toggleState, attentionMaskEnabled })

  return {
    attentionMaskEnabled,
    dropoutEnabled:
      toggleState.dropoutEnabled &&
      example.id === 'flash1' &&
      Boolean(example.dropoutContent?.[attentionMode]),
    fp8Enabled:
      toggleState.fp8Enabled &&
      example.id === 'flash3' &&
      Boolean(example.fp8Content?.[attentionMode]),
    weightDecayEnabled:
      toggleState.weightDecayEnabled &&
      Boolean(example.weightDecayContent?.[attentionMode]),
    moonshotLrEnabled:
      toggleState.moonshotLrEnabled &&
      Boolean(example.moonshotLrContent?.[attentionMode]),
    momentumEnabled:
      toggleState.momentumEnabled &&
      Boolean(example.momentumContent?.[attentionMode]),
  }
}

function viewStateFromLocation() {
  const example = exampleFromLocation()
  const toggleState = availableToggleState(example, toggleStateFromSearch(window.location.search))

  return { example, toggleState }
}

function searchForToggleState(example: AttentionExample, toggleState: ToggleQueryState) {
  const params = new URLSearchParams(window.location.search)
  const availableState = availableToggleState(example, toggleState)

  for (const key of [
    'sync',
    'mask',
    'dropout',
    'fp8',
    'weightDecay',
    'wd',
    'moonshotLr',
    'moonshot',
    'mslr',
    'momentum',
    'mom',
  ]) {
    params.delete(key)
  }

  if (availableState.attentionMaskEnabled) {
    params.set('mask', 'on')
  }

  if (availableState.dropoutEnabled) {
    params.set('dropout', 'on')
  }

  if (availableState.fp8Enabled) {
    params.set('fp8', 'on')
  }

  if (availableState.weightDecayEnabled) {
    params.set('weightDecay', 'on')
  }

  if (availableState.moonshotLrEnabled) {
    params.set('moonshotLr', 'on')
  }

  if (availableState.momentumEnabled) {
    params.set('momentum', 'on')
  }

  const nextSearch = params.toString()

  return nextSearch ? `?${nextSearch}` : ''
}

function CheatSheetViewPage() {
  const [activeExampleId, setActiveExampleId] = useState(
    () => viewStateFromLocation().example.id
  )
  const [hoveredLineId, setHoveredLineId] = useState<string | null>(null)
  const [selectedLineId, setSelectedLineId] = useState<string | null>(null)
  const [attentionMaskEnabled, setAttentionMaskEnabled] = useState(
    () => viewStateFromLocation().toggleState.attentionMaskEnabled
  )
  const [dropoutEnabled, setDropoutEnabled] = useState(
    () => viewStateFromLocation().toggleState.dropoutEnabled
  )
  const [fp8Enabled, setFp8Enabled] = useState(
    () => viewStateFromLocation().toggleState.fp8Enabled
  )
  const [weightDecayEnabled, setWeightDecayEnabled] = useState(
    () => viewStateFromLocation().toggleState.weightDecayEnabled
  )
  const [moonshotLrEnabled, setMoonshotLrEnabled] = useState(
    () => viewStateFromLocation().toggleState.moonshotLrEnabled
  )
  const [momentumEnabled, setMomentumEnabled] = useState(
    () => viewStateFromLocation().toggleState.momentumEnabled
  )
  const [copiedTarget, setCopiedTarget] = useState<CopyTarget | null>(null)
  const codeRegionRef = useRef<HTMLDivElement>(null)

  const activeExample = examples.find((example) => example.id === activeExampleId) ?? examples[0]
  const attentionMode: AttentionMode = attentionMaskEnabled ? 'masked' : 'unmasked'
  const dropoutContent = activeExample.dropoutContent?.[attentionMode]
  const fp8Content = activeExample.fp8Content?.[attentionMode]
  const weightDecayContent = activeExample.weightDecayContent?.[attentionMode]
  const moonshotLrContent = activeExample.moonshotLrContent?.[attentionMode]
  const moonshotLrWeightDecayContent =
    activeExample.moonshotLrWeightDecayContent?.[attentionMode]
  const momentumContent = activeExample.momentumContent?.[attentionMode]
  const momentumWeightDecayContent = activeExample.momentumWeightDecayContent?.[attentionMode]
  const attentionMaskAvailable = hasAttentionMaskToggle(activeExample)
  const dropoutAvailable = activeExample.id === 'flash1' && Boolean(dropoutContent)
  const fp8Available = activeExample.id === 'flash3' && Boolean(fp8Content)
  const weightDecayAvailable = Boolean(weightDecayContent)
  const moonshotLrAvailable = Boolean(moonshotLrContent)
  const momentumAvailable = Boolean(momentumContent)
  const activeGroupExamples = useMemo(
    () => examplesForExampleGroup(activeExample.id),
    [activeExample.id]
  )
  const activeContent =
    momentumEnabled &&
    momentumAvailable &&
    weightDecayEnabled &&
    weightDecayAvailable &&
    momentumWeightDecayContent
      ? momentumWeightDecayContent
      : moonshotLrEnabled &&
        moonshotLrAvailable &&
        weightDecayEnabled &&
        weightDecayAvailable &&
        moonshotLrWeightDecayContent
      ? moonshotLrWeightDecayContent
      : fp8Enabled && fp8Available && fp8Content
      ? fp8Content
      : dropoutEnabled && dropoutAvailable && dropoutContent
      ? dropoutContent
      : momentumEnabled && momentumAvailable && momentumContent
      ? momentumContent
      : moonshotLrEnabled && moonshotLrAvailable && moonshotLrContent
      ? moonshotLrContent
      : weightDecayEnabled && weightDecayAvailable && weightDecayContent
      ? weightDecayContent
      : activeExample.content[attentionMode]
  const activeLineId = selectedLineId ?? hoveredLineId
  const selectableRows = [
    ...(activeContent.prelude?.flatMap((note) => note.rows) ?? []),
    ...activeContent.rows,
    ...(activeContent.notes?.flatMap((note) => note.rows) ?? []),
  ]
  const activeLine = selectableRows.find((line) => line.id === activeLineId)
  const activeCodeLines = new Set(activeLine?.codeLines ?? [])
  const codeLines = useMemo(() => highlightedCodeLines(activeContent.code), [activeContent.code])
  const blocks = useMemo(
    () => algorithmBlocks(activeExample, attentionMode, activeContent),
    [activeExample, attentionMode, activeContent]
  )
  const activeLatex = useMemo(
    () => latexDocument(activeExample, activeContent.prelude, blocks, activeContent.notes),
    [activeExample, activeContent.notes, activeContent.prelude, blocks]
  )

  useEffect(() => {
    function syncFromLocation() {
      const { example, toggleState } = viewStateFromLocation()

      setActiveExampleId(example.id)
      applyToggleState(toggleState)
      setHoveredLineId(null)
      setSelectedLineId(null)

      const nextPath = routeTagFromLocation() || window.location.hash
        ? canonicalPathForExample(example)
        : window.location.pathname
      const nextSearch = searchForToggleState(example, toggleState)

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

  function currentToggleState(): ToggleQueryState {
    return {
      attentionMaskEnabled,
      dropoutEnabled,
      fp8Enabled,
      weightDecayEnabled,
      moonshotLrEnabled,
      momentumEnabled,
    }
  }

  function applyToggleState(toggleState: ToggleQueryState) {
    setAttentionMaskEnabled(toggleState.attentionMaskEnabled)
    setDropoutEnabled(toggleState.dropoutEnabled)
    setFp8Enabled(toggleState.fp8Enabled)
    setWeightDecayEnabled(toggleState.weightDecayEnabled)
    setMoonshotLrEnabled(toggleState.moonshotLrEnabled)
    setMomentumEnabled(toggleState.momentumEnabled)
  }

  function replaceUrlForToggleState(example: AttentionExample, toggleState: ToggleQueryState) {
    window.history.replaceState(
      null,
      '',
      `${canonicalPathForExample(example)}${searchForToggleState(example, toggleState)}`
    )
  }

  function resetExampleState() {
    setHoveredLineId(null)
    setSelectedLineId(null)
  }

  function switchExample(exampleId: string) {
    const nextExample = examples.find((example) => example.id === exampleId)

    if (!nextExample) {
      return
    }

    const nextToggleState = availableToggleState(nextExample, currentToggleState())

    setActiveExampleId(nextExample.id)
    applyToggleState(nextToggleState)
    resetExampleState()

    const nextPath = canonicalPathForExample(nextExample)
    const nextSearch = searchForToggleState(nextExample, nextToggleState)
    const nextUrl = `${nextPath}${nextSearch}`

    if (
      window.location.pathname !== nextPath ||
      window.location.search !== nextSearch ||
      window.location.hash
    ) {
      window.history.pushState(null, '', nextUrl)
    }
  }

  function toggleAttentionMask(enabled: boolean) {
    if (enabled && !attentionMaskAvailable) {
      return
    }

    const nextToggleState = availableToggleState(activeExample, {
      ...currentToggleState(),
      attentionMaskEnabled: enabled,
    })

    applyToggleState(nextToggleState)
    resetExampleState()
    replaceUrlForToggleState(activeExample, nextToggleState)
  }

  function toggleDropout(enabled: boolean) {
    if (enabled && !dropoutAvailable) {
      return
    }

    const nextToggleState = availableToggleState(activeExample, {
      ...currentToggleState(),
      dropoutEnabled: enabled,
    })

    applyToggleState(nextToggleState)
    resetExampleState()
    replaceUrlForToggleState(activeExample, nextToggleState)
  }

  function toggleFp8(enabled: boolean) {
    if (enabled && !fp8Available) {
      return
    }

    const nextToggleState = availableToggleState(activeExample, {
      ...currentToggleState(),
      fp8Enabled: enabled,
    })

    applyToggleState(nextToggleState)
    resetExampleState()
    replaceUrlForToggleState(activeExample, nextToggleState)
  }

  function toggleWeightDecay(enabled: boolean) {
    if (enabled && !weightDecayAvailable) {
      return
    }

    const nextToggleState = availableToggleState(activeExample, {
      ...currentToggleState(),
      weightDecayEnabled: enabled,
    })

    applyToggleState(nextToggleState)
    resetExampleState()
    replaceUrlForToggleState(activeExample, nextToggleState)
  }

  function toggleMoonshotLr(enabled: boolean) {
    if (enabled && !moonshotLrAvailable) {
      return
    }

    const nextToggleState = availableToggleState(activeExample, {
      ...currentToggleState(),
      moonshotLrEnabled: enabled,
    })

    applyToggleState(nextToggleState)
    resetExampleState()
    replaceUrlForToggleState(activeExample, nextToggleState)
  }

  function toggleMomentum(enabled: boolean) {
    if (enabled && !momentumAvailable) {
      return
    }

    const nextToggleState = availableToggleState(activeExample, {
      ...currentToggleState(),
      momentumEnabled: enabled,
    })

    applyToggleState(nextToggleState)
    resetExampleState()
    replaceUrlForToggleState(activeExample, nextToggleState)
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
    await copyTextToClipboard(value)
    setCopiedTarget(target)
    window.setTimeout(() => {
      setCopiedTarget((current) => (current === target ? null : current))
    }, 1400)
  }

  function renderSegment(segment: Segment, index: number) {
    const deltaClassName = segment.delta ? ' toggle-delta' : ''
    const deltaStyle = segment.delta
      ? ({ '--toggle-delta-color': toggleDeltaColors[segment.delta] } as CSSProperties)
      : undefined

    if (segment.kind === 'strong') {
      return (
        <strong className={segment.delta ? 'toggle-delta' : undefined} key={index} style={deltaStyle}>
          {segment.value}
        </strong>
      )
    }

    if (segment.kind === 'math') {
      return (
        <span
          className={`algorithm-math${deltaClassName}`}
          dangerouslySetInnerHTML={{ __html: renderInlineLatex(segment.value) }}
          key={index}
          style={deltaStyle}
        />
      )
    }

    return (
      <span className={segment.delta ? 'toggle-delta' : undefined} key={index} style={deltaStyle}>
        {segment.value}
      </span>
    )
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
          renderSegment={renderSegment}
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

      <AttentionControls
        attentionMaskAvailable={attentionMaskAvailable}
        attentionMaskEnabled={attentionMaskEnabled}
        attentionMaskLabel={activeExample.id === 'flash2' || activeExample.id === 'flash3' ? 'Causal Attention' : 'Attention mask'}
        dropoutAvailable={dropoutAvailable}
        dropoutEnabled={dropoutEnabled}
        fp8Available={fp8Available}
        fp8Enabled={fp8Enabled}
        weightDecayAvailable={weightDecayAvailable}
        weightDecayEnabled={weightDecayEnabled}
        moonshotLrAvailable={moonshotLrAvailable}
        moonshotLrEnabled={moonshotLrEnabled}
        momentumAvailable={momentumAvailable}
        momentumEnabled={momentumEnabled}
        onToggleDropout={toggleDropout}
        onToggleFp8={toggleFp8}
        onToggleAttentionMask={toggleAttentionMask}
        onToggleWeightDecay={toggleWeightDecay}
        onToggleMoonshotLr={toggleMoonshotLr}
        onToggleMomentum={toggleMomentum}
      />

      <ExampleTabs
        activeExample={activeExample}
        examples={activeGroupExamples}
        onSwitchExample={switchExample}
      />
    </main>
  )
}

export default CheatSheetViewPage
