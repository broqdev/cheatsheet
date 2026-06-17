import { forwardRef } from 'react'
import type { HighlightedCodeLine } from '../lib/highlightCode'
import { CopyButton } from './CopyButton'

type CodePanelProps = {
  activeCodeLines: Set<number>
  codeCopied: boolean
  codeLines: HighlightedCodeLine[]
  label: string
  onCopyCode: () => void
}

export const CodePanel = forwardRef<HTMLDivElement, CodePanelProps>(function CodePanel(
  { activeCodeLines, codeCopied, codeLines, label, onCopyCode },
  ref
) {
  return (
    <div className="region code-region" aria-label={`${label} code`} ref={ref}>
      <CopyButton copied={codeCopied} label="Copy code" onCopy={onCopyCode} />

      <pre className="code-block">
        <code>
          {codeLines.map((line) => (
            <span
              className={`code-line${activeCodeLines.has(line.number) ? ' active' : ''}`}
              data-line={line.number}
              key={line.number}
            >
              <span className="line-number">{line.number}</span>
              <span className="line-code" dangerouslySetInnerHTML={{ __html: line.html }} />
            </span>
          ))}
        </code>
      </pre>
    </div>
  )
})
