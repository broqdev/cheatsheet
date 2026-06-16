import type { ToggleDelta } from '../model'

export const toggleDeltaColors = {
  mask: '#9a3412',
  dropout: '#6d28d9',
  fp8: '#006d8f',
} satisfies Record<ToggleDelta, string>

export function latexDelta(delta: ToggleDelta, value: string) {
  return String.raw`\textcolor{${toggleDeltaColors[delta]}}{${value}}`
}
