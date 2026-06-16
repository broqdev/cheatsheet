import type { Segment } from '../model'

export const text = (value: string): Segment => ({ kind: 'text', value })
export const math = (value: string): Segment => ({ kind: 'math', value })
export const strong = (value: string): Segment => ({ kind: 'strong', value })
