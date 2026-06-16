import type { Segment, ToggleDelta } from '../model'

export const text = (value: string, delta?: ToggleDelta): Segment => ({
  kind: 'text',
  value,
  delta,
})
export const math = (value: string, delta?: ToggleDelta): Segment => ({
  kind: 'math',
  value,
  delta,
})
export const strong = (value: string, delta?: ToggleDelta): Segment => ({
  kind: 'strong',
  value,
  delta,
})
