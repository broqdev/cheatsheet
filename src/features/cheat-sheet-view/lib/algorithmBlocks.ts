import type { AlgorithmBlock, AttentionContent, AttentionExample, AttentionMode, Segment } from '../model'
import { math, text } from './segments'

type AlgorithmBlockRole = 'forward' | 'backward'

function labelFromParts(parts: Segment[]) {
  return parts.map((part) => part.value).join('').replace(/\.$/, '')
}

function contentUsesDropout(content: AttentionContent) {
  return content.require.some((part) => part.value.includes('p_{\\mathrm{drop}}'))
}

function qkScaleParts() {
  return [
    text(', QK scaling factor '),
    math(String.raw`\alpha \in \mathbb{R}`),
    text(' (usually '),
    math(String.raw`\alpha=1/\sqrt{d}`),
    text(')'),
  ]
}

function flash3BackwardRequire(causal: boolean) {
  return [
    text('Matrices '),
    math(String.raw`Q,K,V,O,dO\in\mathbb{R}^{N\times d}`),
    text(' in HBM, logsumexp vector '),
    math(String.raw`L\in\mathbb{R}^{N}`),
    text(' in HBM, block sizes '),
    math(String.raw`B_c,B_r`),
    text(', QK scaling factor '),
    math(String.raw`\alpha\in\mathbb{R}`),
    text(' (usually '),
    math(String.raw`\alpha=1/\sqrt{d}`),
    text(')'),
    ...(causal ? [text(', causal attention enabled', 'mask')] : []),
    text('.'),
  ]
}

function blockRequire(
  example: AttentionExample,
  attentionMode: AttentionMode,
  content: AttentionContent,
  blockRole: AlgorithmBlockRole
) {
  if (blockRole === 'forward') {
    if (example.id === 'naive' && attentionMode === 'masked') {
      return [
        text('Matrices '),
        math(String.raw`Q,K,V \in \mathbb{R}^{N \times d}`),
        ...qkScaleParts(),
        text(' and additive mask ', 'mask'),
        math(String.raw`M \in \{0,-\infty\}^{N \times N}`, 'mask'),
        text('.'),
      ]
    }

    if (example.id === 'naive') {
      return [
        text('Matrices '),
        math(String.raw`Q,K,V \in \mathbb{R}^{N \times d}`),
        ...qkScaleParts(),
        text('.'),
      ]
    }

    return content.require
  }

  if (example.id === 'naive') {
    return [
      text('Matrices '),
      math(String.raw`Q,K,V,O,P`),
      ...qkScaleParts(),
      text(', output gradient '),
      math(String.raw`dO`),
      text('.'),
    ]
  }

  if (example.id === 'flash1') {
    const dropoutParts = contentUsesDropout(content)
      ? [
          text(', dropout probability ', 'dropout'),
          math(String.raw`p_{\mathrm{drop}}`, 'dropout'),
          text(', pseudo-random number generator state ', 'dropout'),
          math(String.raw`\mathcal{R}`, 'dropout'),
          text(' from the forward pass.', 'dropout'),
        ]
      : [text('.')]
    const maskParts =
      attentionMode === 'masked'
        ? [text(', masking function ', 'mask'), math(String.raw`\operatorname{MASK}`, 'mask')]
        : []

    return [
      text('Matrices '),
      math(String.raw`Q,K,V,O,dO \in \mathbb{R}^{N \times d}`),
      text(' in HBM, vectors '),
      math(String.raw`\ell,m\in\mathbb{R}^{N}`),
      text(' in HBM, on-chip SRAM of size '),
      math(String.raw`M`),
      ...qkScaleParts(),
      ...maskParts,
      ...dropoutParts,
    ]
  }

  if (example.id === 'flash2') {
    const dropoutParts = contentUsesDropout(content)
      ? [
          text(', dropout probability ', 'dropout'),
          math(String.raw`p_{\mathrm{drop}}`, 'dropout'),
          text(', pseudo-random number generator state ', 'dropout'),
          math(String.raw`\mathcal{R}`, 'dropout'),
          text(' from the forward pass', 'dropout'),
        ]
      : []
    const maskParts =
      attentionMode === 'masked'
        ? [text(', causal attention enabled', 'mask')]
        : []
    return [
      text('Matrices '),
      math(String.raw`Q,K,V,O,dO \in \mathbb{R}^{N \times d}`),
      text(' in HBM, vector '),
      math(String.raw`L\in\mathbb{R}^{N}`),
      text(' in HBM, workspace '),
      math(String.raw`D\in\mathbb{R}^{N}`),
      text(' in HBM, on-chip SRAM of size '),
      math(String.raw`M`),
      ...qkScaleParts(),
      ...maskParts,
      ...dropoutParts,
      text('.'),
    ]
  }

  if (example.id === 'flash3') {
    return flash3BackwardRequire(attentionMode === 'masked')
  }

  const dropoutParts = contentUsesDropout(content)
    ? [
        text(', dropout probability ', 'dropout'),
        math(String.raw`p_{\mathrm{drop}}`, 'dropout'),
        text(', pseudo-random number generator state ', 'dropout'),
        math(String.raw`\mathcal{R}`, 'dropout'),
        text(' from the forward pass', 'dropout'),
      ]
    : []

  if (attentionMode === 'masked') {
    return [
      text('Forward outputs '),
      math(String.raw`O,L`),
      text(', matrices '),
      math(String.raw`Q,K,V`),
      text(', additive mask ', 'mask'),
      math(String.raw`A`, 'mask'),
      text(', output gradient '),
      math(String.raw`dO`),
      ...dropoutParts,
      text('.'),
    ]
  }

  return [
    text('Forward output '),
    math(String.raw`(O,L)`),
    text(', input matrices, and output gradient '),
    math(String.raw`dO`),
    ...dropoutParts,
    text('.'),
  ]
}

function blockRoleFromLabelId(rowId: string): AlgorithmBlockRole | null {
  if (rowId.endsWith('forward-label')) {
    return 'forward'
  }

  if (rowId.endsWith('backward-label') || rowId.endsWith('bwd-label')) {
    return 'backward'
  }

  return null
}

function blockIdFromLabelId(rowId: string) {
  return rowId.replace(/-label$/, '')
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
    const blockRole = blockRoleFromLabelId(row.id)

    if (blockRole) {
      const currentBlock = blocks[blocks.length - 1]
      const title = labelFromParts(row.parts)

      if (blocks.length === 1 && currentBlock.rows.length === 0) {
        currentBlock.id = blockIdFromLabelId(row.id)
        currentBlock.title = title
        currentBlock.require = blockRequire(example, attentionMode, content, blockRole)
        continue
      }

      blocks.push({
        id: blockIdFromLabelId(row.id),
        title,
        require: blockRequire(example, attentionMode, content, blockRole),
        rows: [],
      })
      continue
    }

    blocks[blocks.length - 1].rows.push(row)
  }

  return blocks.filter((block) => block.rows.length > 0)
}
