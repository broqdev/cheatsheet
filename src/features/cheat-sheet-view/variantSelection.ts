import type {
  AttentionContent,
  AttentionExample,
  AttentionMode,
  VariantKey,
  VariantState,
} from './model'

type VariantDefinition = {
  aliases: string[]
  color: string
  label: string
  requires?: VariantKey[]
}

const variantDefinitions: Record<VariantKey, VariantDefinition> = {
  mask: { aliases: ['mask'], color: '#9a3412', label: 'Attention mask' },
  dropout: { aliases: ['dropout'], color: '#6d28d9', label: 'Dropout' },
  weightDecay: {
    aliases: ['weightDecay', 'wd'],
    color: '#047857',
    label: 'Weight Decay',
  },
  moonshotLr: {
    aliases: ['moonshotLr', 'moonshot', 'mslr'],
    color: '#2563eb',
    label: 'Moonshot lr',
  },
  momentum: { aliases: ['momentum', 'mom'], color: '#b45309', label: 'Momentum' },
  nesterov: {
    aliases: ['nesterov', 'nag'],
    color: '#c026d3',
    label: 'Nesterov',
    requires: ['momentum'],
  },
  fp8: { aliases: ['fp8'], color: '#006d8f', label: 'Hopper FP8' },
}

const variantKeys = Object.keys(variantDefinitions) as VariantKey[]
const storageKeyPrefix = 'broq-cheatsheet:toggle-state:'
const legacyStateKeys = {
  mask: 'attentionMaskEnabled',
  dropout: 'dropoutEnabled',
  fp8: 'fp8Enabled',
  weightDecay: 'weightDecayEnabled',
  moonshotLr: 'moonshotLrEnabled',
  momentum: 'momentumEnabled',
  nesterov: 'nesterovEnabled',
} satisfies Record<VariantKey, string>

export type VariantControl = {
  key: VariantKey
  label: string
  enabled: boolean
  color: string
}

type VariantStorage = Pick<Storage, 'getItem' | 'setItem'>

export function emptyVariantState(): VariantState {
  return Object.fromEntries(variantKeys.map((key) => [key, false])) as VariantState
}

export function variantColor(key: VariantKey) {
  return variantDefinitions[key].color
}

export function variantStateFromUnknown(value: unknown): VariantState {
  if (!value || typeof value !== 'object') {
    return emptyVariantState()
  }

  const record = value as Record<string, unknown>

  return Object.fromEntries(
    variantKeys.map((key) => [
      key,
      Object.hasOwn(record, key)
        ? record[key] === true
        : record[legacyStateKeys[key]] === true,
    ])
  ) as VariantState
}

function modeForState(state: VariantState): AttentionMode {
  return state.mask ? 'masked' : 'unmasked'
}

function hasMaskVariant(example: AttentionExample) {
  return example.content.masked !== example.content.unmasked
}

function availableVariantKeys(example: AttentionExample, state: VariantState) {
  const keys = new Set<VariantKey>()
  const mode = modeForState({ ...state, mask: hasMaskVariant(example) && state.mask })

  if (hasMaskVariant(example)) {
    keys.add('mask')
  }

  for (const variant of example.variants ?? []) {
    if (variant.content[mode]) {
      variant.enabled.forEach((key) => keys.add(key))
    }
  }

  return keys
}

export function normalizeVariantState(example: AttentionExample, requested: VariantState) {
  const available = availableVariantKeys(example, requested)
  const state = variantStateFromUnknown(requested)

  for (const key of variantKeys) {
    state[key] = available.has(key) && state[key]
  }

  for (const key of variantKeys) {
    if (!state[key]) {
      continue
    }

    for (const required of variantDefinitions[key].requires ?? []) {
      state[required] = available.has(required)
    }
  }

  for (const key of variantKeys) {
    const dependents = variantKeys.filter((candidate) =>
      variantDefinitions[candidate].requires?.includes(key)
    )

    if (!state[key]) {
      dependents.forEach((dependent) => {
        state[dependent] = false
      })
    }
  }

  return state
}

function resolvedContent(
  example: AttentionExample,
  mode: AttentionMode,
  state: VariantState
): AttentionContent {
  const match = (example.variants ?? [])
    .filter(
      (variant) =>
        variant.content[mode] && variant.enabled.every((key) => state[key])
    )
    .sort((left, right) => right.enabled.length - left.enabled.length)[0]

  return match?.content[mode] ?? example.content[mode]
}

export function resolveVariantView(example: AttentionExample, requested: VariantState) {
  const state = normalizeVariantState(example, requested)
  const mode = modeForState(state)
  const available = availableVariantKeys(example, state)
  const controls: VariantControl[] = variantKeys
    .filter((key) => available.has(key))
    .map((key) => ({
      key,
      label: example.variantLabels?.[key] ?? variantDefinitions[key].label,
      enabled: state[key],
      color: variantDefinitions[key].color,
    }))

  return {
    state,
    mode,
    content: resolvedContent(example, mode, state),
    controls,
  }
}

export function setVariantEnabled(
  example: AttentionExample,
  current: VariantState,
  key: VariantKey,
  enabled: boolean
) {
  const requested = { ...current, [key]: enabled }

  if (!enabled) {
    for (const candidate of variantKeys) {
      if (variantDefinitions[candidate].requires?.includes(key)) {
        requested[candidate] = false
      }
    }
  }

  if (enabled) {
    for (const required of variantDefinitions[key].requires ?? []) {
      requested[required] = true
    }
  }

  return normalizeVariantState(example, requested)
}

function queryEnabled(params: URLSearchParams, key: string) {
  return ['1', 'true', 'yes', 'on'].includes(params.get(key)?.toLowerCase() ?? '')
}

export function variantStateFromSearch(
  search: string,
  fallback: VariantState = emptyVariantState()
) {
  const params = new URLSearchParams(search)
  const hasVariantQuery = variantKeys.some((key) =>
    variantDefinitions[key].aliases.some((alias) => params.has(alias))
  )
  const state = hasVariantQuery ? emptyVariantState() : variantStateFromUnknown(fallback)

  for (const key of variantKeys) {
    const alias = variantDefinitions[key].aliases.find((candidate) => params.has(candidate))

    if (alias) {
      state[key] = queryEnabled(params, alias)
    }
  }

  return state
}

export function searchForVariantState(
  example: AttentionExample,
  requested: VariantState,
  currentSearch: string
) {
  const params = new URLSearchParams(currentSearch)
  const view = resolveVariantView(example, requested)
  const available = new Set(view.controls.map((control) => control.key))

  for (const key of available) {
    variantDefinitions[key].aliases.forEach((alias) => params.delete(alias))

    if (view.state[key]) {
      params.set(variantDefinitions[key].aliases[0], 'on')
    }
  }

  const search = params.toString()
  return search ? `?${search}` : ''
}

function storageKey(example: AttentionExample) {
  return `${storageKeyPrefix}${example.id}`
}

export function readStoredVariantState(example: AttentionExample, storage: VariantStorage) {
  try {
    const value = storage.getItem(storageKey(example))
    return value ? variantStateFromUnknown(JSON.parse(value)) : emptyVariantState()
  } catch {
    return emptyVariantState()
  }
}

export function writeStoredVariantState(
  example: AttentionExample,
  state: VariantState,
  storage: VariantStorage
) {
  try {
    storage.setItem(storageKey(example), JSON.stringify(state))
  } catch {
    // Storage can be unavailable in private browsing or after quota exhaustion.
  }
}
