import type { AttentionExample, CatalogSection } from '../model'
import { adamExample } from './subPages/adam'
import { adamWExample } from './subPages/adamw'
import { flashAttention1Example } from './subPages/flashAttention1'
import { flashAttention2Example } from './subPages/flashAttention2'
import { flashAttention3Example } from './subPages/flashAttention3'
import { flashAttention4Example } from './subPages/flashAttention4'
import { muonExample } from './subPages/muon'
import { naiveAttentionExample } from './subPages/naiveAttention'
import { rmspropExample } from './subPages/rmsprop'
import { sgdExample } from './subPages/sgd'

type CatalogGroup = {
  id: string
  label: string
  examples: AttentionExample[]
}

const catalogGroups: CatalogGroup[] = [
  {
    id: 'attention',
    label: 'Attention',
    examples: [
      naiveAttentionExample,
      flashAttention1Example,
      flashAttention2Example,
      flashAttention3Example,
      flashAttention4Example,
    ],
  },
  {
    id: 'optimizer',
    label: 'Optimizer',
    examples: [sgdExample, rmspropExample, adamExample, adamWExample, muonExample],
  },
]

export function normalizeExampleTag(value: string) {
  const routeValue = value.replace(/^#/, '').replace(/^\/+|\/+$/g, '')
  let decodedValue = routeValue

  try {
    decodedValue = decodeURIComponent(routeValue)
  } catch {
    decodedValue = routeValue
  }

  return decodedValue
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function tagsForExample(example: AttentionExample) {
  return [example.id, example.urlTag, example.label].map(normalizeExampleTag)
}

function defineCatalog(groups: CatalogGroup[]) {
  const examples = groups.flatMap((group) => group.examples)
  const ids = new Set<string>()
  const routes = new Set<string>()

  for (const example of examples) {
    const route = normalizeExampleTag(example.urlTag)

    if (ids.has(example.id)) {
      throw new Error(`Duplicate cheatsheet example id "${example.id}".`)
    }

    if (routes.has(route)) {
      throw new Error(`Duplicate cheatsheet route "${example.urlTag}".`)
    }

    ids.add(example.id)
    routes.add(route)
  }

  const sections: CatalogSection[] = groups.map((group) => ({
    id: group.id,
    label: group.label,
    items: group.examples.map((example) => ({
      id: `catalog-${example.id}`,
      label: example.label,
      exampleId: example.id,
    })),
  }))

  return { examples, groups, sections }
}

export const cheatsheetCatalog = defineCatalog(catalogGroups)
export const examples = cheatsheetCatalog.examples
export const catalogSections = cheatsheetCatalog.sections

export function exampleFromTag(value: string) {
  const tag = normalizeExampleTag(value)

  return tag
    ? examples.find((example) => tagsForExample(example).includes(tag))
    : undefined
}

export function routeSegmentsForExample(example: AttentionExample) {
  return example.urlTag
    .split('/')
    .map(normalizeExampleTag)
    .filter(Boolean)
}

export function examplesForExampleGroup(exampleId: string) {
  const activeGroup = cheatsheetCatalog.groups.find((group) =>
    group.examples.some((example) => example.id === exampleId)
  )

  if (!activeGroup) {
    return examples
  }

  return activeGroup.examples
}
