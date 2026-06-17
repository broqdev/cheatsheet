export type ToggleDelta =
  | 'mask'
  | 'dropout'
  | 'fp8'
  | 'weightDecay'
  | 'momentum'
  | 'moonshotLr'

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
  prelude?: LatexBlock[]
  notes?: LatexBlock[]
}

export type AttentionExample = {
  id: string
  urlTag: string
  label: string
  description: string
  algorithmTitle: string
  content: Record<AttentionMode, AttentionContent>
  dropoutContent?: Partial<Record<AttentionMode, AttentionContent>>
  fp8Content?: Partial<Record<AttentionMode, AttentionContent>>
  weightDecayContent?: Partial<Record<AttentionMode, AttentionContent>>
  moonshotLrContent?: Partial<Record<AttentionMode, AttentionContent>>
  moonshotLrWeightDecayContent?: Partial<Record<AttentionMode, AttentionContent>>
  momentumContent?: Partial<Record<AttentionMode, AttentionContent>>
  momentumWeightDecayContent?: Partial<Record<AttentionMode, AttentionContent>>
}

export type CatalogItem = {
  id: string
  label: string
  exampleId?: string
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
