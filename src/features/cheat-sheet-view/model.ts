export type ToggleDelta =
  | 'mask'
  | 'dropout'
  | 'fp8'
  | 'centered'
  | 'weightDecay'
  | 'momentum'
  | 'nesterov'
  | 'moonshotLr'

export type VariantKey = ToggleDelta

export type VariantState = Record<VariantKey, boolean>

export type AlgorithmBlockRole = 'forward' | 'backward'

export type AlgorithmBlockStart = {
  id: string
  role: AlgorithmBlockRole
}

export type Segment =
  | { kind: 'text'; value: string; delta?: ToggleDelta }
  | { kind: 'math'; value: string; delta?: ToggleDelta }
  | { kind: 'strong'; value: string; delta?: ToggleDelta }

export type AlgorithmLine = {
  id: string
  number?: number
  indent?: number
  parts: Segment[]
  codeLines: number[]
  startsBlock?: AlgorithmBlockStart
}

export type AttentionMode = 'unmasked' | 'masked'

export type LatexBlock = {
  id: string
  title: string
  require?: Segment[]
  requireLabel?: string
  rows: AlgorithmLine[]
}

export type AttentionContent = {
  require: Segment[]
  rows: AlgorithmLine[]
  code: string
  blocks: AlgorithmBlock[]
  blockRequires?: Partial<Record<AlgorithmBlockRole, Segment[]>>
  prelude?: LatexBlock[]
  notes?: LatexBlock[]
}

export type AttentionVariant = {
  enabled: VariantKey[]
  content: Partial<Record<AttentionMode, AttentionContent>>
}

export type AttentionExample = {
  id: string
  urlTag: string
  label: string
  description: string
  algorithmTitle: string
  content: Record<AttentionMode, AttentionContent>
  variants?: AttentionVariant[]
  variantLabels?: Partial<Record<VariantKey, string>>
}

export type CatalogItem = {
  id: string
  label: string
  exampleId: string
  hidden?: boolean
}

export type CatalogSection = {
  id: string
  label: string
  items: CatalogItem[]
}

export type AlgorithmBlock = {
  id: string
  title: string
  require: Segment[]
  rows: AlgorithmLine[]
}

export type CopyTarget = 'latex' | 'code'
