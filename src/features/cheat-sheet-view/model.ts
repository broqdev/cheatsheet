export type Segment =
  | { kind: 'text'; value: string }
  | { kind: 'math'; value: string }
  | { kind: 'strong'; value: string }

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
  notes?: LatexBlock[]
}

export type AttentionExample = {
  id: string
  label: string
  algorithmTitle: string
  content: Record<AttentionMode, AttentionContent>
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
  id: 'forward' | 'backward'
  title: string
  require: Segment[]
  rows: AlgorithmLine[]
}

export type CopyTarget = 'latex' | 'code'

