import katex from 'katex'
import 'katex/dist/katex.min.css'

export function renderInlineLatex(tex: string) {
  return katex.renderToString(tex, {
    displayMode: false,
    output: 'htmlAndMathml',
    throwOnError: false,
    trust: false,
  })
}
