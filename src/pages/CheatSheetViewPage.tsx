import { useMemo, useRef, useState } from 'react'
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
import type {
  AlgorithmLine,
  AttentionMode,
  CopyTarget,
  Segment,
} from '../features/cheat-sheet-view/model'

function CheatSheetViewPage() {
  const [activeExampleId, setActiveExampleId] = useState(examples[0].id)
  const [hoveredLineId, setHoveredLineId] = useState<string | null>(null)
  const [selectedLineId, setSelectedLineId] = useState<string | null>(null)
  const [attentionMaskEnabled, setAttentionMaskEnabled] = useState(false)
  const [copiedTarget, setCopiedTarget] = useState<CopyTarget | null>(null)
  const codeRegionRef = useRef<HTMLDivElement>(null)

  const activeExample = examples.find((example) => example.id === activeExampleId) ?? examples[0]
  const attentionMode: AttentionMode = attentionMaskEnabled ? 'masked' : 'unmasked'
  const activeContent = activeExample.content[attentionMode]
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

  function switchExample(exampleId: string) {
    setActiveExampleId(exampleId)
    setHoveredLineId(null)
    setSelectedLineId(null)
  }

  function toggleAttentionMask(enabled: boolean) {
    setAttentionMaskEnabled(enabled)
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
    if (segment.kind === 'strong') {
      return <strong key={index}>{segment.value}</strong>
    }

    if (segment.kind === 'math') {
      return (
        <span
          className="algorithm-math"
          dangerouslySetInnerHTML={{ __html: renderInlineLatex(segment.value) }}
          key={index}
        />
      )
    }

    return <span key={index}>{segment.value}</span>
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

      <CheatsheetSummary />

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
