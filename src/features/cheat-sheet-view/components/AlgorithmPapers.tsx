import type { CSSProperties, ReactNode } from 'react'
import type { AlgorithmBlock, AlgorithmLine, AttentionExample, LatexBlock, Segment } from '../model'
import { CopyButton } from './CopyButton'

type AlgorithmPapersProps = {
  activeExample: AttentionExample
  activeLineId: string | undefined
  blocks: AlgorithmBlock[]
  latexCopied: boolean
  notes: LatexBlock[] | undefined
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

  return (
    <button
      key={line.id}
      type="button"
      className={`algorithm-row${isActive ? ' active' : ''}${isSelected ? ' selected' : ''}`}
      onBlur={onLeave}
      onClick={(event) => {
        if (event.detail === 0) {
          onActivate(line)
        }
      }}
      onFocus={() => onFocus(line.id)}
      onMouseEnter={() => onFocus(line.id)}
      onMouseLeave={onLeave}
      onPointerDown={() => onActivate(line)}
    >
      <span className="algorithm-row-number">{displayNumber ? `${displayNumber}:` : ''}</span>
      <span
        className="algorithm-row-body"
        style={{ '--indent': line.indent ?? 0 } as CSSProperties}
      >
        {line.parts.map(renderSegment)}
      </span>
    </button>
  )
}

export function AlgorithmPapers({
  activeExample,
  activeLineId,
  blocks,
  latexCopied,
  notes,
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
    <div className="region math-region">
      <CopyButton copied={latexCopied} label="Copy LaTeX" onCopy={onCopyLatex} />

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
