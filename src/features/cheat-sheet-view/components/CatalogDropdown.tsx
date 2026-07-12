import { type KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { navigationCatalogSections } from '../data/attentionExamples'
import type { CatalogItem } from '../model'

function normalizeCatalogQuery(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]/g, '')
}

function fuzzyMatch(label: string, query: string) {
  const normalizedLabel = normalizeCatalogQuery(label)
  const normalizedQuery = normalizeCatalogQuery(query)

  if (!normalizedQuery) {
    return true
  }

  if (normalizedLabel.includes(normalizedQuery)) {
    return true
  }

  let queryIndex = 0

  for (const char of normalizedLabel) {
    if (char === normalizedQuery[queryIndex]) {
      queryIndex += 1
    }

    if (queryIndex === normalizedQuery.length) {
      return true
    }
  }

  return false
}

function filteredCatalogSections(query: string) {
  return navigationCatalogSections
    .map((section) => {
      const sectionMatches = fuzzyMatch(section.label, query)
      const items = section.items.filter((item) => sectionMatches || fuzzyMatch(item.label, query))

      return { ...section, items }
    })
    .filter((section) => section.items.length > 0)
}

function selectableCatalogItems(query: string) {
  return filteredCatalogSections(query).flatMap((section) => section.items)
}

function catalogOptionId(itemId: string) {
  return `catalog-option-${itemId}`
}

const editableTargetSelector =
  'input, textarea, select, [contenteditable="true"], [contenteditable="plaintext-only"]'

function isTextEntryTarget(target: EventTarget | null) {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable || Boolean(target.closest(editableTargetSelector)))
  )
}

type CatalogDropdownProps = {
  activeLabel: string
  activeExampleId: string
  onSelectExample: (exampleId: string) => void
}

export function CatalogDropdown({
  activeLabel,
  activeExampleId,
  onSelectExample,
}: CatalogDropdownProps) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [activeItemId, setActiveItemId] = useState<string | null>(null)
  const rootRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const itemButtonRefs = useRef(new Map<string, HTMLButtonElement>())
  const sections = useMemo(() => filteredCatalogSections(query), [query])
  const selectableItems = useMemo(() => selectableCatalogItems(query), [query])
  const firstSelectableItem = selectableItems[0]
  const activeItem =
    selectableItems.find((item) => item.id === activeItemId) ?? firstSelectableItem
  const activeOptionId = activeItem ? catalogOptionId(activeItem.id) : undefined

  function defaultActiveItemIdForQuery(nextQuery: string) {
    const nextItems = selectableCatalogItems(nextQuery)

    if (!nextItems.length) {
      return null
    }

    if (nextQuery.trim()) {
      return nextItems[0].id
    }

    return nextItems.find((item) => item.exampleId === activeExampleId)?.id ?? nextItems[0].id
  }

  const openSearch = useCallback((options?: { preserveQuery?: boolean }) => {
    setOpen(true)

    if (!options?.preserveQuery) {
      setQuery('')
      setActiveItemId(defaultActiveItemIdForQuery(''))
    }

    window.setTimeout(() => inputRef.current?.focus(), 0)
  }, [activeExampleId])

  useEffect(() => {
    if (!open) {
      return
    }

    function handlePointerDown(event: PointerEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false)
      }
    }

    document.addEventListener('pointerdown', handlePointerDown)

    return () => document.removeEventListener('pointerdown', handlePointerDown)
  }, [open])

  useEffect(() => {
    function handleCatalogHotkey(event: globalThis.KeyboardEvent) {
      if (
        event.defaultPrevented ||
        event.metaKey ||
        event.ctrlKey ||
        event.altKey ||
        event.shiftKey ||
        event.key.toLowerCase() !== 'f' ||
        isTextEntryTarget(event.target)
      ) {
        return
      }

      event.preventDefault()
      openSearch({ preserveQuery: open })
    }

    document.addEventListener('keydown', handleCatalogHotkey)

    return () => document.removeEventListener('keydown', handleCatalogHotkey)
  }, [open, openSearch])

  function closeSearch() {
    setOpen(false)
    setQuery('')
    setActiveItemId(null)
  }

  function selectItem(item: CatalogItem) {
    onSelectExample(item.exampleId)
    closeSearch()
  }

  function targetCatalogItem(index: number) {
    if (!selectableItems.length) {
      return
    }

    const nextIndex = (index + selectableItems.length) % selectableItems.length
    const nextItem = selectableItems[nextIndex]
    const nextButton = itemButtonRefs.current.get(nextItem.id)

    setActiveItemId(nextItem.id)
    nextButton?.scrollIntoView({ block: 'nearest' })
  }

  function activeItemIndex() {
    return selectableItems.findIndex((item) => item.id === activeItem?.id)
  }

  function moveActiveCatalogItem(offset: number) {
    const currentIndex = activeItemIndex()

    targetCatalogItem(currentIndex >= 0 ? currentIndex + offset : 0)
  }

  function updateQuery(nextQuery: string) {
    setQuery(nextQuery)
    setActiveItemId(defaultActiveItemIdForQuery(nextQuery))
  }

  function handleSearchKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'Escape') {
      event.preventDefault()
      closeSearch()
    }

    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      moveActiveCatalogItem(event.key === 'ArrowDown' ? 1 : -1)
    }

    if (event.key === 'Enter' && activeItem) {
      event.preventDefault()
      selectItem(activeItem)
    }
  }

  function handleItemKeyDown(event: KeyboardEvent<HTMLButtonElement>, item: CatalogItem) {
    if (event.key === 'Escape') {
      event.preventDefault()
      closeSearch()
      return
    }

    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      inputRef.current?.focus({ preventScroll: true })
      moveActiveCatalogItem(event.key === 'ArrowDown' ? 1 : -1)
      return
    }

    if (event.key === 'Home' || event.key === 'End') {
      event.preventDefault()
      inputRef.current?.focus({ preventScroll: true })
      targetCatalogItem(event.key === 'Home' ? 0 : selectableItems.length - 1)
      return
    }

    if (event.key === 'Enter') {
      event.preventDefault()
      selectItem(item)
    }
  }

  return (
    <div className="catalog-picker" ref={rootRef}>
      {open ? (
        <input
          ref={inputRef}
          className="catalog-search"
          value={query}
          role="combobox"
          aria-autocomplete="list"
          aria-activedescendant={activeOptionId}
          aria-controls="catalog-list"
          aria-expanded="true"
          aria-label="Search cheatsheet catalog"
          placeholder="Search catalog"
          onChange={(event) => updateQuery(event.currentTarget.value)}
          onKeyDown={handleSearchKeyDown}
        />
      ) : (
        <button
          type="button"
          className="catalog-trigger"
          aria-haspopup="listbox"
          aria-expanded="false"
          aria-keyshortcuts="F"
          onClick={() => openSearch()}
        >
          <span className="catalog-active-label">{activeLabel}</span>
          <kbd className="catalog-hotkey" aria-hidden="true">F</kbd>
          <span className="catalog-caret" aria-hidden="true" />
        </button>
      )}

      {open ? (
        <div className="catalog-menu" id="catalog-list" role="listbox" aria-label="Cheatsheet catalog">
          {sections.length ? (
            sections.map((section) => (
              <div className="catalog-section" key={section.id}>
                <div className="catalog-parent">{section.label}</div>
                <div className="catalog-children">
                  {section.items.map((item) => {
                    const selected = item.exampleId === activeExampleId
                    const active = item.id === activeItem?.id
                    return (
                      <button
                        key={item.id}
                        type="button"
                        className={`catalog-item${selected ? ' selected' : ''}${active ? ' active' : ''}`}
                        aria-selected={selected}
                        id={catalogOptionId(item.id)}
                        ref={(node) => {
                          if (node) {
                            itemButtonRefs.current.set(item.id, node)
                          } else {
                            itemButtonRefs.current.delete(item.id)
                          }
                        }}
                        role="option"
                        onClick={() => selectItem(item)}
                        onFocus={() => setActiveItemId(item.id)}
                        onKeyDown={(event) => handleItemKeyDown(event, item)}
                      >
                        {item.label}
                      </button>
                    )
                  })}
                </div>
              </div>
            ))
          ) : (
            <div className="catalog-empty">No matches</div>
          )}
        </div>
      ) : null}
    </div>
  )
}
