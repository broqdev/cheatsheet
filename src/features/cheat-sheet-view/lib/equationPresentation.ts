import katex from 'katex'
import 'katex/dist/katex.min.css'
import type {
  AlgorithmBlock,
  AlgorithmLine,
  AttentionExample,
  LatexBlock,
  Segment,
  VariantKey,
} from '../model'
import { variantColor } from '../variantSelection'

export { variantColor } from '../variantSelection'

export function renderInlineLatex(tex: string) {
  return katex.renderToString(tex, {
    displayMode: false,
    output: 'htmlAndMathml',
    throwOnError: false,
    trust: false,
  })
}

function colorizeLatex(segment: Segment, value: string) {
  return segment.delta ? `\\textcolor{${variantColor(segment.delta)}}{${value}}` : value
}

export function segmentToLatex(segment: Segment) {
  if (segment.kind === 'math') {
    return segment.delta
      ? `$\\textcolor{${variantColor(segment.delta)}}{${segment.value}}$`
      : `$${segment.value}$`
  }

  if (segment.kind === 'strong') {
    return colorizeLatex(segment, `\\textbf{${segment.value}}`)
  }

  return colorizeLatex(segment, segment.value)
}

function segmentsToLatex(segments: Segment[]) {
  return segments.map(segmentToLatex).join('')
}

function rowToLatex(row: AlgorithmLine) {
  const indent = row.indent ? '  '.repeat(row.indent) : ''
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

function noteToLatex(note: LatexBlock, index: number) {
  const lines = [`\\textbf{Algorithm ${index + 1}} ${note.title}`]

  if (note.intro) {
    lines.push(segmentsToLatex(note.intro))
  }

  if (note.require) {
    lines.push(`\\textbf{${note.requireLabel ?? 'Given'}:} ${segmentsToLatex(note.require)}`)
  }

  lines.push(...note.rows.map(rowToLatex))
  return lines.join('\n')
}

function preludeToLatex(prelude: LatexBlock) {
  const lines = [`\\textbf{${prelude.title}}`]

  if (prelude.require) {
    lines.push(segmentsToLatex(prelude.require))
  }

  lines.push(...prelude.rows.map(rowToLatex))
  return lines.join('\n')
}

export function latexDocument(
  example: AttentionExample,
  prelude: LatexBlock[] | undefined,
  blocks: AlgorithmBlock[],
  notes: LatexBlock[] | undefined
) {
  return [
    ...(prelude?.map(preludeToLatex) ?? []),
    ...blocks.map((block, index) => blockToLatex(block, index, example)),
    ...(notes?.map((note, noteIndex) => noteToLatex(note, blocks.length + noteIndex)) ?? []),
  ].join('\n\n')
}

export function latexDelta(delta: VariantKey, value: string) {
  return String.raw`\textcolor{${variantColor(delta)}}{${value}}`
}

function closestKatex(node: Node) {
  const element = node instanceof Element ? node : node.parentElement
  return element?.closest('.katex') ?? null
}

function replaceKatexMathWithTex(fragment: DocumentFragment) {
  fragment.querySelectorAll('.katex-mathml + .katex-html').forEach((node) => node.remove())

  fragment.querySelectorAll('.katex-mathml').forEach((node) => {
    const texSource = node.querySelector('annotation')?.textContent

    if (texSource) {
      node.replaceWith(document.createTextNode(`$${texSource}$`))
    }
  })
}

export function clipboardContentsForLatexSelection(region: HTMLElement, selection: Selection) {
  if (selection.isCollapsed || selection.rangeCount === 0) {
    return undefined
  }

  const { anchorNode, focusNode } = selection

  if (!anchorNode || !focusNode || !region.contains(anchorNode) || !region.contains(focusNode)) {
    return undefined
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
    return undefined
  }

  const html = Array.from(fragment.childNodes)
    .map((node) => (node instanceof Element ? node.outerHTML : node.textContent ?? ''))
    .join('')

  replaceKatexMathWithTex(fragment)
  const plainText = fragment.textContent

  return plainText ? { html, plainText } : undefined
}
