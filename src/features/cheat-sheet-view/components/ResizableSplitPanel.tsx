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

const minimumPrimaryPercent = 30
const maximumPrimaryPercent = 70

function clampPrimaryPercent(value: number) {
  return Math.min(maximumPrimaryPercent, Math.max(minimumPrimaryPercent, value))
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
    const [primaryPercent, setPrimaryPercent] = useState(defaultPrimaryPercent)
    const [isResizing, setIsResizing] = useState(false)
    const panelRef = useRef<HTMLElement>(null)

    useImperativeHandle(forwardedRef, () => panelRef.current as HTMLElement, [])

    function setPrimaryPercentFromClientX(clientX: number) {
      const panelRect = panelRef.current?.getBoundingClientRect()

      if (!panelRect || panelRect.width <= 0) {
        return
      }

      setPrimaryPercent(
        clampPrimaryPercent(((clientX - panelRect.left) / panelRect.width) * 100)
      )
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
      setPrimaryPercent(clampPrimaryPercent(nextPrimaryPercent))
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
          onLostPointerCapture={() => setIsResizing(false)}
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
