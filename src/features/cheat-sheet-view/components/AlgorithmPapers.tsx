import {
  type ClipboardEvent,
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
  useRef,
} from 'react'
import type { AlgorithmBlock, AlgorithmLine, AttentionExample, LatexBlock, Segment } from '../model'
import { CopyButton } from './CopyButton'

type AlgorithmPapersProps = {
  activeExample: AttentionExample
  activeLineId: string | undefined
  blocks: AlgorithmBlock[]
  latexCopied: boolean
  notes: LatexBlock[] | undefined
  prelude: LatexBlock[] | undefined
  onCopyLatex: () => void
  onLineActivate: (line: AlgorithmLine) => void
  onLineFocus: (lineId: string) => void
  onLineLeave: () => void
  renderSegment: (segment: Segment, index: number) => ReactNode
  selectedLineId: string | null
}

type AlgorithmRowProps = {
  activeLineId: string | undefined
  displayNumber: number | undefined
  line: AlgorithmLine
  onActivate: (line: AlgorithmLine) => void
  onFocus: (lineId: string) => void
  onLeave: () => void
  renderSegment: (segment: Segment, index: number) => ReactNode
  selectedLineId: string | null
}

type PointerStart = {
  x: number
  y: number
}

function closestKatex(node: Node) {
  const element = node instanceof Element ? node : node.parentElement

  return element?.closest('.katex') ?? null
}

function replaceKatexMathWithTex(fragment: DocumentFragment) {
  const renderedMath = fragment.querySelectorAll('.katex-mathml + .katex-html')

  renderedMath.forEach((node) => node.remove())

  const mathNodes = fragment.querySelectorAll('.katex-mathml')

  mathNodes.forEach((node) => {
    const texSource = node.querySelector('annotation')?.textContent

    if (texSource) {
      node.replaceWith(document.createTextNode(`$${texSource}$`))
    }
  })
}

function handleLatexSelectionCopy(event: ClipboardEvent<HTMLDivElement>) {
  const selection = window.getSelection()

  if (!selection || selection.isCollapsed || selection.rangeCount === 0 || !event.clipboardData) {
    return
  }

  const { anchorNode, focusNode } = selection
  const region = event.currentTarget

  if (!anchorNode || !focusNode || !region.contains(anchorNode) || !region.contains(focusNode)) {
    return
  }

  const range = selection.getRangeAt(0).cloneRange()
  const startKatex = closestKatex(range.startContainer)
  const endKatex = closestKatex(range.endContainer)

  if (startKatex && region.contains(startKatex)) {
    range.setStartBefore(startKatex)
  }

  if (endKatex && region.contains(endKatex)) {
    range.setEndAfter(endKatex)
  }

  const fragment = range.cloneContents()

  if (!fragment.querySelector('.katex-mathml')) {
    return
  }

  const html = Array.from(fragment.childNodes)
    .map((node) => (node instanceof Element ? node.outerHTML : node.textContent ?? ''))
    .join('')

  replaceKatexMathWithTex(fragment)

  const plainText = fragment.textContent

  if (!plainText) {
    return
  }

  event.clipboardData.setData('text/html', html)
  event.clipboardData.setData('text/plain', plainText)
  event.preventDefault()
}

function AlgorithmRow({
  activeLineId,
  displayNumber,
  line,
  onActivate,
  onFocus,
  onLeave,
  renderSegment,
  selectedLineId,
}: AlgorithmRowProps) {
  const isActive = activeLineId === line.id
  const isSelected = selectedLineId === line.id
  const pointerStart = useRef<PointerStart | null>(null)

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key !== 'Enter' && event.key !== ' ') {
      return
    }

    event.preventDefault()
    onActivate(line)
  }

  function handlePointerDown(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0) {
      return
    }

    pointerStart.current = {
      x: event.clientX,
      y: event.clientY,
    }
  }

  function handlePointerUp(event: PointerEvent<HTMLDivElement>) {
    const start = pointerStart.current
    pointerStart.current = null

    if (!start || event.button !== 0) {
      return
    }

    const pointerTravel = Math.hypot(event.clientX - start.x, event.clientY - start.y)

    if (pointerTravel <= 4) {
      onActivate(line)
    }
  }

  return (
    <div
      key={line.id}
      role="button"
      tabIndex={0}
      className={`algorithm-row${isActive ? ' active' : ''}${isSelected ? ' selected' : ''}`}
      onBlur={onLeave}
      onFocus={() => onFocus(line.id)}
      onKeyDown={handleKeyDown}
      onMouseEnter={() => onFocus(line.id)}
      onMouseLeave={onLeave}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
    >
      <span className="algorithm-row-number">{displayNumber ? `${displayNumber}:` : ''}</span>
      <span
        className="algorithm-row-body"
        style={{ '--indent': line.indent ?? 0 } as CSSProperties}
      >
        {line.parts.map(renderSegment)}
      </span>
    </div>
  )
}

export function AlgorithmPapers({
  activeExample,
  activeLineId,
  blocks,
  latexCopied,
  notes,
  prelude,
  onCopyLatex,
  onLineActivate,
  onLineFocus,
  onLineLeave,
  renderSegment,
  selectedLineId,
}: AlgorithmPapersProps) {
  function renderAlgorithmRow(line: AlgorithmLine, displayNumber = line.number) {
    return (
      <AlgorithmRow
        key={line.id}
        activeLineId={activeLineId}
        displayNumber={displayNumber}
        line={line}
        onActivate={onLineActivate}
        onFocus={onLineFocus}
        onLeave={onLineLeave}
        renderSegment={renderSegment}
        selectedLineId={selectedLineId}
      />
    )
  }

  return (
    <div className="region math-region" onCopy={handleLatexSelectionCopy}>
      <CopyButton copied={latexCopied} label="Copy LaTeX" onCopy={onCopyLatex} />

      {prelude?.map((part) => (
        <article
          className="algorithm-paper latex-block"
          key={part.id}
          aria-label={`${activeExample.label} ${part.title}`}
        >
          <header className="algorithm-header">
            <strong>{part.title}</strong>
            {part.require ? <h2>{part.require.map(renderSegment)}</h2> : null}
          </header>
          <div className="algorithm-lines">
            {part.rows.map((line, rowIndex) => renderAlgorithmRow(line, rowIndex + 1))}
          </div>
        </article>
      ))}

      {blocks.map((block, index) => (
        <article
          className="algorithm-paper"
          key={block.id}
          aria-label={`${activeExample.label} ${block.title}`}
        >
          <header className="algorithm-header">
            <strong>Algorithm {index + 1}</strong>
            <h2>
              {activeExample.algorithmTitle} {block.title}
            </h2>
          </header>
          <p className="algorithm-require">
            <strong>Require:</strong> {block.require.map(renderSegment)}
          </p>
          <div className="algorithm-lines">
            {block.rows.map((line, rowIndex) => renderAlgorithmRow(line, rowIndex + 1))}
          </div>
        </article>
      ))}

      {notes?.map((note, noteIndex) => (
        <article
          className="algorithm-paper latex-block"
          key={note.id}
          aria-label={`${activeExample.label} ${note.title}`}
        >
          <header className="algorithm-header">
            <strong>Algorithm {blocks.length + noteIndex + 1}</strong>
            <h2>{note.title}</h2>
          </header>
          {note.require ? (
            <p className="algorithm-require">
              <strong>{note.requireLabel ?? 'Given'}:</strong> {note.require.map(renderSegment)}
            </p>
          ) : null}
          <div className="algorithm-lines">
            {note.rows.map((line, rowIndex) => renderAlgorithmRow(line, rowIndex + 1))}
          </div>
        </article>
      ))}
    </div>
  )
}
