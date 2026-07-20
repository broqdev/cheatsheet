import {
  type ClipboardEvent,
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent,
  useRef,
} from 'react'
import { clipboardContentsForLatexSelection } from '../lib/equationPresentation'
import type { AlgorithmBlock, AlgorithmLine, AttentionExample, LatexBlock } from '../model'
import { CopyButton } from './CopyButton'
import { EquationSegment } from './EquationSegment'

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
  selectedLineId: string | null
}

type AlgorithmRowProps = {
  activeLineId: string | undefined
  displayNumber: number | undefined
  line: AlgorithmLine
  onActivate: (line: AlgorithmLine) => void
  onFocus: (lineId: string) => void
  onLeave: () => void
  selectedLineId: string | null
}

type PointerStart = {
  x: number
  y: number
}

function handleLatexSelectionCopy(event: ClipboardEvent<HTMLDivElement>) {
  const selection = window.getSelection()

  if (!selection || !event.clipboardData) {
    return
  }

  const contents = clipboardContentsForLatexSelection(event.currentTarget, selection)

  if (!contents) {
    return
  }

  event.clipboardData.setData('text/html', contents.html)
  event.clipboardData.setData('text/plain', contents.plainText)
  event.preventDefault()
}

function AlgorithmRow({
  activeLineId,
  displayNumber,
  line,
  onActivate,
  onFocus,
  onLeave,
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
        {line.parts.map((segment, index) => (
          <EquationSegment key={index} segment={segment} />
        ))}
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
  selectedLineId,
}: AlgorithmPapersProps) {
  function renderSegments(segments: LatexBlock['require']) {
    return segments?.map((segment, index) => (
      <EquationSegment key={index} segment={segment} />
    ))
  }

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
        selectedLineId={selectedLineId}
      />
    )
  }

  return (
    <div className="region math-region" onCopy={handleLatexSelectionCopy}>
      <CopyButton
        copied={latexCopied}
        label="Copy LaTeX"
        onCopy={onCopyLatex}
        showCopiedLabel
      />

      {prelude?.map((part) => (
        <article
          className="algorithm-paper latex-block"
          key={part.id}
          aria-label={`${activeExample.label} ${part.title}`}
        >
          <header className="algorithm-header">
            <strong>{part.title}</strong>
            {part.require ? <h2>{renderSegments(part.require)}</h2> : null}
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
            <strong>Require:</strong> {renderSegments(block.require)}
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
          {note.intro ? (
            <p className="algorithm-intro">{renderSegments(note.intro)}</p>
          ) : null}
          {note.require ? (
            <p className="algorithm-require">
              <strong>{note.requireLabel ?? 'Given'}:</strong> {renderSegments(note.require)}
            </p>
          ) : null}
          <div className="algorithm-lines">
            {note.rows.map((line) => renderAlgorithmRow(line))}
          </div>
        </article>
      ))}
    </div>
  )
}
