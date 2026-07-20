import { forwardRef, useImperativeHandle, useRef, useState } from 'react'
import type {
  CSSProperties,
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent,
  ReactNode,
} from 'react'

type ResizableSplitPanelProps = {
  ariaLabel: string
  className: string
  defaultPrimaryPercent: number
  primary: ReactNode
  primaryLabel: string
  resizeLabel: string
  secondary: ReactNode
  secondaryLabel: string
}

type SplitPanelStorage = Pick<Storage, 'getItem' | 'setItem'>

const minimumPrimaryPercent = 30
const maximumPrimaryPercent = 70

export const splitPanelStorageKey = 'broq-cheatsheet:split-panel-primary-percent'

function clampPrimaryPercent(value: number) {
  return Math.min(maximumPrimaryPercent, Math.max(minimumPrimaryPercent, value))
}

export function readStoredPrimaryPercent(
  defaultPrimaryPercent: number,
  storage: Pick<SplitPanelStorage, 'getItem'> | undefined
) {
  const fallback = clampPrimaryPercent(defaultPrimaryPercent)

  try {
    const storedValue = storage?.getItem(splitPanelStorageKey)

    if (!storedValue?.trim()) {
      return fallback
    }

    const parsedValue = Number(storedValue)
    return Number.isFinite(parsedValue) ? clampPrimaryPercent(parsedValue) : fallback
  } catch {
    return fallback
  }
}

export function writeStoredPrimaryPercent(
  primaryPercent: number,
  storage: Pick<SplitPanelStorage, 'setItem'> | undefined
) {
  try {
    storage?.setItem(
      splitPanelStorageKey,
      String(clampPrimaryPercent(primaryPercent))
    )
  } catch {
    // Storage may be unavailable in embedded or restricted browser contexts.
  }
}

function browserStorage() {
  if (typeof window === 'undefined') {
    return undefined
  }

  try {
    return window.localStorage
  } catch {
    return undefined
  }
}

export const ResizableSplitPanel = forwardRef<HTMLElement, ResizableSplitPanelProps>(
  function ResizableSplitPanel(
    {
      ariaLabel,
      className,
      defaultPrimaryPercent,
      primary,
      primaryLabel,
      resizeLabel,
      secondary,
      secondaryLabel,
    },
    forwardedRef
  ) {
    const [primaryPercent, setPrimaryPercent] = useState(() =>
      readStoredPrimaryPercent(defaultPrimaryPercent, browserStorage())
    )
    const [isResizing, setIsResizing] = useState(false)
    const panelRef = useRef<HTMLElement>(null)
    const primaryPercentRef = useRef(primaryPercent)

    useImperativeHandle(forwardedRef, () => panelRef.current as HTMLElement, [])

    function updatePrimaryPercent(value: number, persist = false) {
      const nextPrimaryPercent = clampPrimaryPercent(value)
      primaryPercentRef.current = nextPrimaryPercent
      setPrimaryPercent(nextPrimaryPercent)

      if (persist) {
        writeStoredPrimaryPercent(nextPrimaryPercent, browserStorage())
      }
    }

    function persistPrimaryPercent() {
      writeStoredPrimaryPercent(primaryPercentRef.current, browserStorage())
    }

    function setPrimaryPercentFromClientX(clientX: number) {
      const panelRect = panelRef.current?.getBoundingClientRect()

      if (!panelRect || panelRect.width <= 0) {
        return
      }

      updatePrimaryPercent(((clientX - panelRect.left) / panelRect.width) * 100)
    }

    function handlePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
      if (event.button !== 0) {
        return
      }

      event.preventDefault()
      event.currentTarget.setPointerCapture(event.pointerId)
      setIsResizing(true)
      setPrimaryPercentFromClientX(event.clientX)
    }

    function handlePointerMove(event: ReactPointerEvent<HTMLDivElement>) {
      if (!event.currentTarget.hasPointerCapture(event.pointerId)) {
        return
      }

      setPrimaryPercentFromClientX(event.clientX)
    }

    function stopResizing(event: ReactPointerEvent<HTMLDivElement>) {
      persistPrimaryPercent()

      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId)
      }

      setIsResizing(false)
    }

    function handleKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
      const step = event.shiftKey ? 5 : 2
      let nextPrimaryPercent: number | undefined

      if (event.key === 'ArrowLeft') {
        nextPrimaryPercent = primaryPercent - step
      } else if (event.key === 'ArrowRight') {
        nextPrimaryPercent = primaryPercent + step
      } else if (event.key === 'Home') {
        nextPrimaryPercent = minimumPrimaryPercent
      } else if (event.key === 'End') {
        nextPrimaryPercent = maximumPrimaryPercent
      }

      if (nextPrimaryPercent === undefined) {
        return
      }

      event.preventDefault()
      updatePrimaryPercent(nextPrimaryPercent, true)
    }

    const roundedPrimaryPercent = Math.round(primaryPercent)

    return (
      <section
        className={`${className} split-panel-resizable${isResizing ? ' is-resizing' : ''}`}
        id="cheatsheet-panel"
        aria-label={ariaLabel}
        ref={panelRef}
        role="tabpanel"
        style={
          {
            '--split-panel-primary-width': `${primaryPercent}%`,
          } as CSSProperties
        }
      >
        {primary}

        <div
          className="split-panel-resize-handle"
          role="separator"
          aria-label={resizeLabel}
          aria-orientation="vertical"
          aria-valuemax={maximumPrimaryPercent}
          aria-valuemin={minimumPrimaryPercent}
          aria-valuenow={roundedPrimaryPercent}
          aria-valuetext={`${roundedPrimaryPercent}% ${primaryLabel}, ${100 - roundedPrimaryPercent}% ${secondaryLabel}`}
          tabIndex={0}
          onKeyDown={handleKeyDown}
          onLostPointerCapture={() => {
            persistPrimaryPercent()
            setIsResizing(false)
          }}
          onPointerCancel={stopResizing}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={stopResizing}
        />

        {secondary}
      </section>
    )
  }
)
