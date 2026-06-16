import type { CSSProperties, ReactNode } from 'react'
import type { ToggleDelta } from '../model'
import { toggleDeltaColors } from '../lib/toggleDeltas'

type AttentionControlsProps = {
  attentionMaskEnabled: boolean
  attentionMaskLabel: string
  dropoutAvailable: boolean
  dropoutEnabled: boolean
  onToggleDropout: (enabled: boolean) => void
  onToggleAttentionMask: (enabled: boolean) => void
}

function ToggleLabel({ children, delta }: { children: ReactNode; delta: ToggleDelta }) {
  return (
    <>
      <span className="toggle-copy">{children}</span>
      <span
        aria-hidden="true"
        className="toggle-swatch"
        style={{ '--toggle-delta-color': toggleDeltaColors[delta] } as CSSProperties}
      />
    </>
  )
}

export function AttentionControls({
  attentionMaskEnabled,
  attentionMaskLabel,
  dropoutAvailable,
  dropoutEnabled,
  onToggleDropout,
  onToggleAttentionMask,
}: AttentionControlsProps) {
  const dropoutActive = dropoutAvailable && dropoutEnabled

  return (
    <section className="control-strip" aria-label="Attention controls">
      <label className={`checkbox-control${attentionMaskEnabled ? ' active' : ''}`}>
        <input
          type="checkbox"
          checked={attentionMaskEnabled}
          onChange={(event) => onToggleAttentionMask(event.currentTarget.checked)}
        />
        <ToggleLabel delta="mask">{attentionMaskLabel}</ToggleLabel>
      </label>

      {dropoutAvailable ? (
        <label className={`checkbox-control${dropoutActive ? ' active' : ''}`}>
          <input
            type="checkbox"
            checked={dropoutActive}
            onChange={(event) => onToggleDropout(event.currentTarget.checked)}
          />
          <ToggleLabel delta="dropout">Dropout</ToggleLabel>
        </label>
      ) : null}
    </section>
  )
}
