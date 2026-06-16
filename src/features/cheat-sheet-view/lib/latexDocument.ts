import type { AlgorithmBlock, AlgorithmLine, AttentionExample, LatexBlock, Segment } from '../model'

function segmentToLatex(segment: Segment) {
  if (segment.kind === 'math') {
    return `$${segment.value}$`
  }

  if (segment.kind === 'strong') {
    return `\\textbf{${segment.value}}`
  }

  return segment.value
}

function segmentsToLatex(segments: Segment[]) {
  return segments.map(segmentToLatex).join('')
}

function rowToLatex(row: AlgorithmLine) {
  const indent = row.indent ? `${'  '.repeat(row.indent)}` : ''
  const prefix = row.number ? `${row.number}. ` : ''

  return `${indent}${prefix}${segmentsToLatex(row.parts)}`
}

function blockToLatex(block: AlgorithmBlock, index: number, example: AttentionExample) {
  return [
    `\\textbf{Algorithm ${index + 1}} ${example.algorithmTitle} ${block.title}`,
    `\\textbf{Require:} ${segmentsToLatex(block.require)}`,
    ...block.rows.map(rowToLatex),
  ].join('\n')
}

function noteToLatex(note: LatexBlock) {
  const lines = [`\\textbf{LaTeX} ${note.title}`]

  if (note.require) {
    lines.push(`\\textbf{${note.requireLabel ?? 'Given'}:} ${segmentsToLatex(note.require)}`)
  }

  lines.push(...note.rows.map(rowToLatex))

  return lines.join('\n')
}

export function latexDocument(
  example: AttentionExample,
  blocks: AlgorithmBlock[],
  notes: LatexBlock[] | undefined
) {
  return [
    ...blocks.map((block, index) => blockToLatex(block, index, example)),
    ...(notes?.map(noteToLatex) ?? []),
  ].join('\n\n')
}

