import type { CSSProperties } from 'react'
import type { Segment } from '../model'
import { renderInlineLatex, variantColor } from '../lib/equationPresentation'

export function EquationSegment({ segment }: { segment: Segment }) {
  const deltaClassName = segment.delta ? ' toggle-delta' : ''
  const deltaStyle = segment.delta
    ? ({ '--toggle-delta-color': variantColor(segment.delta) } as CSSProperties)
    : undefined

  if (segment.kind === 'strong') {
    return (
      <strong className={segment.delta ? 'toggle-delta' : undefined} style={deltaStyle}>
        {segment.value}
      </strong>
    )
  }

  if (segment.kind === 'math') {
    return (
      <span
        className={`algorithm-math${deltaClassName}`}
        dangerouslySetInnerHTML={{ __html: renderInlineLatex(segment.value) }}
        style={deltaStyle}
      />
    )
  }

  return (
    <span className={segment.delta ? 'toggle-delta' : undefined} style={deltaStyle}>
      {segment.value}
    </span>
  )
}
