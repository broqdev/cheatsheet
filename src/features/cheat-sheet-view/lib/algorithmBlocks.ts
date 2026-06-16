import type { AlgorithmBlock, AttentionContent, AttentionExample, AttentionMode, Segment } from '../model'
import { math, text } from './segments'

function labelFromParts(parts: Segment[]) {
  return parts.map((part) => part.value).join('').replace(/\.$/, '')
}

function blockRequire(
  example: AttentionExample,
  attentionMode: AttentionMode,
  content: AttentionContent,
  blockId: AlgorithmBlock['id']
) {
  if (blockId === 'forward') {
    if (example.id === 'naive' && attentionMode === 'masked') {
      return [
        text('Matrices '),
        math(String.raw`Q,K,V \in \mathbb{R}^{N \times d}`),
        text(' and additive mask '),
        math(String.raw`M \in \{0,-\infty\}^{N \times N}`),
        text('.'),
      ]
    }

    if (example.id === 'naive') {
      return [
        text('Matrices '),
        math(String.raw`Q,K,V \in \mathbb{R}^{N \times d}`),
        text('.'),
      ]
    }

    return content.require
  }

  if (example.id === 'naive') {
    return [
      text('Matrices '),
      math(String.raw`Q,K,V,O,P`),
      text(', output gradient '),
      math(String.raw`dO`),
      text('.'),
    ]
  }

  if (attentionMode === 'masked') {
    return [
      text('Forward outputs '),
      math(String.raw`O,L`),
      text(', matrices '),
      math(String.raw`Q,K,V`),
      text(', additive mask '),
      math(String.raw`A`),
      text(', output gradient '),
      math(String.raw`dO`),
      text('.'),
    ]
  }

  return [
    text('Forward output '),
    math(String.raw`(O,L)`),
    text(', input matrices, and output gradient '),
    math(String.raw`dO`),
    text('.'),
  ]
}

export function algorithmBlocks(
  example: AttentionExample,
  attentionMode: AttentionMode,
  content: AttentionContent
): AlgorithmBlock[] {
  const blocks: AlgorithmBlock[] = [
    {
      id: 'forward',
      title: 'Forward pass',
      require: blockRequire(example, attentionMode, content, 'forward'),
      rows: [],
    },
  ]

  for (const row of content.rows) {
    if (row.id.endsWith('forward-label')) {
      blocks[0].title = labelFromParts(row.parts)
      continue
    }

    if (row.id.endsWith('backward-label') || row.id.endsWith('bwd-label')) {
      blocks.push({
        id: 'backward',
        title: labelFromParts(row.parts),
        require: blockRequire(example, attentionMode, content, 'backward'),
        rows: [],
      })
      continue
    }

    blocks[blocks.length - 1].rows.push(row)
  }

  return blocks.filter((block) => block.rows.length > 0)
}

