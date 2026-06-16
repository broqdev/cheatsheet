import type { CSSProperties, ReactNode } from 'react'
import type { ToggleDelta } from '../model'
import { toggleDeltaColors } from '../lib/toggleDeltas'

type AttentionControlsProps = {
  attentionMaskEnabled: boolean
  attentionMaskLabel: string
  dropoutAvailable: boolean
  dropoutEnabled: boolean
  fp8Available: boolean
  fp8Enabled: boolean
  onToggleDropout: (enabled: boolean) => void
  onToggleFp8: (enabled: boolean) => void
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
  fp8Available,
  fp8Enabled,
  onToggleDropout,
  onToggleFp8,
  onToggleAttentionMask,
}: AttentionControlsProps) {
  const dropoutActive = dropoutAvailable && dropoutEnabled
  const fp8Active = fp8Available && fp8Enabled

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

      {fp8Available ? (
        <label className={`checkbox-control${fp8Active ? ' active' : ''}`}>
          <input
            type="checkbox"
            checked={fp8Active}
            onChange={(event) => onToggleFp8(event.currentTarget.checked)}
          />
          <ToggleLabel delta="fp8">Hopper FP8</ToggleLabel>
        </label>
      ) : null}
    </section>
  )
}
