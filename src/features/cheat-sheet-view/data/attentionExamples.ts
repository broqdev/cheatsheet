import type { AttentionExample, CatalogSection } from '../model'
import { adagradExample } from './subPages/adagrad'
import { adamExample } from './subPages/adam'
import { adamWExample } from './subPages/adamw'
import { flashAttention1Example } from './subPages/flashAttention1'
import { flashAttention2Example } from './subPages/flashAttention2'
import { flashAttention3Example } from './subPages/flashAttention3'
import { flashAttention4Example } from './subPages/flashAttention4'
import { lbfgsExample } from './subPages/lbfgs'
import { muonExample } from './subPages/muon'
import { naiveAttentionExample } from './subPages/naiveAttention'
import { rmspropExample } from './subPages/rmsprop'
import { sgdExample } from './subPages/sgd'

export const catalogSections: CatalogSection[] = [
  {
    id: 'attention',
    label: 'Attention',
    items: [
      { id: 'catalog-naive', label: 'Naive Attention', exampleId: 'naive' },
      { id: 'catalog-flash-1', label: 'FlashAttention-1', exampleId: 'flash1' },
      { id: 'catalog-flash-2', label: 'FlashAttention-2', exampleId: 'flash2' },
      { id: 'catalog-flash-3', label: 'FlashAttention-3', exampleId: 'flash3' },
      { id: 'catalog-flash-4', label: 'FlashAttention-4', exampleId: 'flash4' },
    ],
  },
  {
    id: 'optimizer',
    label: 'Optimizer',
    items: [
      { id: 'catalog-sgd', label: 'SGD', exampleId: 'sgd' },
      { id: 'catalog-adagrad', label: 'AdaGrad', exampleId: 'adagrad' },
      { id: 'catalog-rmsprop', label: 'RMSProp', exampleId: 'rmsprop' },
      { id: 'catalog-adam', label: 'Adam', exampleId: 'adam' },
      { id: 'catalog-adamw', label: 'AdamW', exampleId: 'adamw' },
      { id: 'catalog-lbfgs', label: 'L-BFGS', exampleId: 'lbfgs', hidden: true },
      { id: 'catalog-muon', label: 'Muon', exampleId: 'muon' },
    ],
  },
]

export const navigationCatalogSections = catalogSections
  .map((section) => ({
    ...section,
    items: section.items.filter((item) => !item.hidden),
  }))
  .filter((section) => section.items.length > 0)

export const examples: AttentionExample[] = [
  naiveAttentionExample,
  flashAttention1Example,
  flashAttention2Example,
  flashAttention3Example,
  flashAttention4Example,
  sgdExample,
  adagradExample,
  rmspropExample,
  adamExample,
  adamWExample,
  lbfgsExample,
  muonExample,
]

function exampleIdsForSection(section: CatalogSection) {
  return section.items.flatMap((item) => (item.exampleId ? [item.exampleId] : []))
}

export function examplesForExampleGroup(exampleId: string) {
  const activeSection = catalogSections.find((section) =>
    exampleIdsForSection(section).includes(exampleId)
  )

  if (!activeSection) {
    return examples
  }

  const navigationSection = navigationCatalogSections.find(
    (section) => section.id === activeSection.id
  )
  const activeSectionExampleIds = new Set(
    navigationSection ? exampleIdsForSection(navigationSection) : []
  )

  return examples.filter((example) => activeSectionExampleIds.has(example.id))
}
