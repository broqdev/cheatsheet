import type { AttentionExample, CatalogSection } from '../model'
import { flashAttention1Example } from './subPages/flashAttention1'
import { flashAttention2Example } from './subPages/flashAttention2'
import { flashAttention3Example } from './subPages/flashAttention3'
import { naiveAttentionExample } from './subPages/naiveAttention'

export const catalogSections: CatalogSection[] = [
  {
    id: 'attention',
    label: 'Attention',
    items: [
      { id: 'catalog-naive', label: 'Naive Attention', exampleId: 'naive' },
      { id: 'catalog-flash-1', label: 'FlashAttention-1', exampleId: 'flash1' },
      { id: 'catalog-flash-2', label: 'FlashAttention-2', exampleId: 'flash2' },
      { id: 'catalog-flash-3', label: 'FlashAttention-3', exampleId: 'flash3' },
    ],
  },
]

export const examples: AttentionExample[] = [
  naiveAttentionExample,
  flashAttention1Example,
  flashAttention2Example,
  flashAttention3Example,
]
