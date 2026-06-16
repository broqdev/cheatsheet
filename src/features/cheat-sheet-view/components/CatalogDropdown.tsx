import { type KeyboardEvent, useEffect, useMemo, useRef, useState } from 'react'
import { catalogSections } from '../data/attentionExamples'
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
  return catalogSections
    .map((section) => {
      const sectionMatches = fuzzyMatch(section.label, query)
      const items = section.items.filter((item) => sectionMatches || fuzzyMatch(item.label, query))

      return { ...section, items }
    })
    .filter((section) => section.items.length > 0)
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
  const rootRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const sections = useMemo(() => filteredCatalogSections(query), [query])
  const firstSelectableItem = sections.flatMap((section) => section.items).find((item) => item.exampleId)

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

  function openSearch() {
    setOpen(true)
    setQuery('')
    window.setTimeout(() => inputRef.current?.focus(), 0)
  }

  function closeSearch() {
    setOpen(false)
    setQuery('')
  }

  function selectItem(item: CatalogItem) {
    if (!item.exampleId) {
      return
    }

    onSelectExample(item.exampleId)
    closeSearch()
  }

  function handleSearchKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'Escape') {
      event.preventDefault()
      closeSearch()
    }

    if (event.key === 'Enter' && firstSelectableItem) {
      event.preventDefault()
      selectItem(firstSelectableItem)
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
          aria-controls="catalog-list"
          aria-expanded="true"
          aria-label="Search cheatsheet catalog"
          placeholder="Search catalog"
          onChange={(event) => setQuery(event.currentTarget.value)}
          onKeyDown={handleSearchKeyDown}
        />
      ) : (
        <button
          type="button"
          className="catalog-trigger"
          aria-haspopup="listbox"
          aria-expanded="false"
          onClick={openSearch}
        >
          <span>{activeLabel}</span>
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
                    const disabled = !item.exampleId

                    return (
                      <button
                        key={item.id}
                        type="button"
                        className={`catalog-item${selected ? ' selected' : ''}`}
                        aria-disabled={disabled}
                        aria-selected={selected}
                        disabled={disabled}
                        role="option"
                        onClick={() => selectItem(item)}
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

