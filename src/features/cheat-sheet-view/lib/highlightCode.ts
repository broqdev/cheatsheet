import Prism from 'prismjs'
import 'prismjs/components/prism-python'

export type HighlightedCodeLine = { number: number; html: string }

export function highlightedCodeLines(code: string): HighlightedCodeLine[] {
  const grammar = Prism.languages.python

  if (!code) {
    return []
  }

  return code.split('\n').map((line, index) => ({
    number: index + 1,
    html: Prism.highlight(line || ' ', grammar, 'python'),
  }))
}
