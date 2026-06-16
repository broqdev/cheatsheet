import { type CSSProperties, useEffect, useMemo, useRef, useState } from 'react'
import { AlgorithmPapers } from '../features/cheat-sheet-view/components/AlgorithmPapers'
import { AttentionControls } from '../features/cheat-sheet-view/components/AttentionControls'
import { CatalogDropdown } from '../features/cheat-sheet-view/components/CatalogDropdown'
import { CheatsheetSummary } from '../features/cheat-sheet-view/components/CheatsheetSummary'
import { CodePanel } from '../features/cheat-sheet-view/components/CodePanel'
import { ExampleTabs } from '../features/cheat-sheet-view/components/ExampleTabs'
import { examples } from '../features/cheat-sheet-view/data/attentionExamples'
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
  const hashValue = value.startsWith('#') ? value.slice(1) : value
  let decodedValue = hashValue

  try {
    decodedValue = decodeURIComponent(hashValue)
  } catch {
    decodedValue = hashValue
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

function canonicalHashForExample(example: AttentionExample) {
  return `#${encodeURIComponent(normalizeUrlTag(example.urlTag))}`
}

function CheatSheetViewPage() {
  const [activeExampleId, setActiveExampleId] = useState(
    () => exampleFromUrlTag(window.location.hash)?.id ?? examples[0].id
  )
  const [hoveredLineId, setHoveredLineId] = useState<string | null>(null)
  const [selectedLineId, setSelectedLineId] = useState<string | null>(null)
  const [attentionMaskEnabled, setAttentionMaskEnabled] = useState(false)
  const [dropoutEnabled, setDropoutEnabled] = useState(false)
  const [fp8Enabled, setFp8Enabled] = useState(false)
  const [copiedTarget, setCopiedTarget] = useState<CopyTarget | null>(null)
  const codeRegionRef = useRef<HTMLDivElement>(null)

  const activeExample = examples.find((example) => example.id === activeExampleId) ?? examples[0]
  const attentionMode: AttentionMode = attentionMaskEnabled ? 'masked' : 'unmasked'
  const dropoutContent = activeExample.dropoutContent?.[attentionMode]
  const fp8Content = activeExample.fp8Content?.[attentionMode]
  const dropoutAvailable = activeExample.id === 'flash1' && Boolean(dropoutContent)
  const fp8Available = activeExample.id === 'flash3' && Boolean(fp8Content)
  const activeContent =
    fp8Enabled && fp8Available && fp8Content
      ? fp8Content
      : dropoutEnabled && dropoutAvailable && dropoutContent
      ? dropoutContent
      : activeExample.content[attentionMode]
  const activeLineId = selectedLineId ?? hoveredLineId
  const selectableRows = [
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
    () => latexDocument(activeExample, blocks, activeContent.notes),
    [activeExample, activeContent.notes, blocks]
  )

  useEffect(() => {
    function syncExampleFromUrlTag() {
      setActiveExampleId(exampleFromUrlTag(window.location.hash)?.id ?? examples[0].id)
      setDropoutEnabled(false)
      setFp8Enabled(false)
      setHoveredLineId(null)
      setSelectedLineId(null)
    }

    window.addEventListener('hashchange', syncExampleFromUrlTag)
    window.addEventListener('popstate', syncExampleFromUrlTag)

    return () => {
      window.removeEventListener('hashchange', syncExampleFromUrlTag)
      window.removeEventListener('popstate', syncExampleFromUrlTag)
    }
  }, [])

  function resetExampleState() {
    setDropoutEnabled(false)
    setFp8Enabled(false)
    setHoveredLineId(null)
    setSelectedLineId(null)
  }

  function switchExample(exampleId: string) {
    const nextExample = examples.find((example) => example.id === exampleId)

    if (!nextExample) {
      return
    }

    setActiveExampleId(nextExample.id)
    resetExampleState()

    const nextHash = canonicalHashForExample(nextExample)

    if (window.location.hash !== nextHash) {
      window.history.pushState(
        null,
        '',
        `${window.location.pathname}${window.location.search}${nextHash}`
      )
    }
  }

  function toggleAttentionMask(enabled: boolean) {
    setAttentionMaskEnabled(enabled)
    setHoveredLineId(null)
    setSelectedLineId(null)
  }

  function toggleDropout(enabled: boolean) {
    if (enabled && !dropoutAvailable) {
      return
    }

    setDropoutEnabled(enabled)
    setHoveredLineId(null)
    setSelectedLineId(null)
  }

  function toggleFp8(enabled: boolean) {
    if (enabled && !fp8Available) {
      return
    }

    setFp8Enabled(enabled)
    setHoveredLineId(null)
    setSelectedLineId(null)
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
    <main className="workspace" aria-label="Attention equation and code explorer">
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
        id="attention-panel"
        aria-label={`${activeExample.label} equations and code`}
        role="tabpanel"
      >
        <AlgorithmPapers
          activeExample={activeExample}
          activeLineId={activeLine?.id}
          blocks={blocks}
          latexCopied={copiedTarget === 'latex'}
          notes={activeContent.notes}
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
        attentionMaskEnabled={attentionMaskEnabled}
        attentionMaskLabel={activeExample.id === 'flash2' || activeExample.id === 'flash3' ? 'Causal Attention' : 'Attention mask'}
        dropoutAvailable={dropoutAvailable}
        dropoutEnabled={dropoutEnabled}
        fp8Available={fp8Available}
        fp8Enabled={fp8Enabled}
        onToggleDropout={toggleDropout}
        onToggleFp8={toggleFp8}
        onToggleAttentionMask={toggleAttentionMask}
      />

      <ExampleTabs
        activeExample={activeExample}
        examples={examples}
        onSwitchExample={switchExample}
      />
    </main>
  )
}

export default CheatSheetViewPage
