import type { ToggleDelta } from '../model'

export const toggleDeltaColors = {
  mask: '#9a3412',
  dropout: '#6d28d9',
  fp8: '#006d8f',
  centered: '#0369a1',
  weightDecay: '#047857',
  momentum: '#b45309',
  nesterov: '#c026d3',
  moonshotLr: '#2563eb',
} satisfies Record<ToggleDelta, string>

export function latexDelta(delta: ToggleDelta, value: string) {
  return String.raw`\textcolor{${toggleDeltaColors[delta]}}{${value}}`
}
