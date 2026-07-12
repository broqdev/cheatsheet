import type { CSSProperties, ReactNode } from 'react'
import type { ToggleDelta } from '../model'
import { toggleDeltaColors } from '../lib/toggleDeltas'

type AttentionControlsProps = {
  attentionMaskAvailable: boolean
  attentionMaskEnabled: boolean
  attentionMaskLabel: string
  dropoutAvailable: boolean
  dropoutEnabled: boolean
  fp8Available: boolean
  fp8Enabled: boolean
  centeredAvailable: boolean
  centeredEnabled: boolean
  weightDecayAvailable: boolean
  weightDecayEnabled: boolean
  moonshotLrAvailable: boolean
  moonshotLrEnabled: boolean
  momentumAvailable: boolean
  momentumEnabled: boolean
  nesterovAvailable: boolean
  nesterovEnabled: boolean
  onToggleDropout: (enabled: boolean) => void
  onToggleFp8: (enabled: boolean) => void
  onToggleCentered: (enabled: boolean) => void
  onToggleAttentionMask: (enabled: boolean) => void
  onToggleWeightDecay: (enabled: boolean) => void
  onToggleMoonshotLr: (enabled: boolean) => void
  onToggleMomentum: (enabled: boolean) => void
  onToggleNesterov: (enabled: boolean) => void
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
  attentionMaskAvailable,
  attentionMaskEnabled,
  attentionMaskLabel,
  dropoutAvailable,
  dropoutEnabled,
  fp8Available,
  fp8Enabled,
  centeredAvailable,
  centeredEnabled,
  weightDecayAvailable,
  weightDecayEnabled,
  moonshotLrAvailable,
  moonshotLrEnabled,
  momentumAvailable,
  momentumEnabled,
  nesterovAvailable,
  nesterovEnabled,
  onToggleDropout,
  onToggleFp8,
  onToggleCentered,
  onToggleAttentionMask,
  onToggleWeightDecay,
  onToggleMoonshotLr,
  onToggleMomentum,
  onToggleNesterov,
}: AttentionControlsProps) {
  const dropoutActive = dropoutAvailable && dropoutEnabled
  const fp8Active = fp8Available && fp8Enabled
  const centeredActive = centeredAvailable && centeredEnabled
  const weightDecayActive = weightDecayAvailable && weightDecayEnabled
  const moonshotLrActive = moonshotLrAvailable && moonshotLrEnabled
  const momentumActive = momentumAvailable && momentumEnabled
  const nesterovActive = nesterovAvailable && nesterovEnabled
  const hasControls =
    attentionMaskAvailable ||
    dropoutAvailable ||
    fp8Available ||
    centeredAvailable ||
    weightDecayAvailable ||
    moonshotLrAvailable ||
    momentumAvailable ||
    nesterovAvailable

  if (!hasControls) {
    return (
      <section
        className="control-strip empty"
        aria-hidden="true"
        aria-label="Variant controls"
      />
    )
  }

  return (
    <section className="control-strip" aria-label="Variant controls">
      {attentionMaskAvailable ? (
        <label className={`checkbox-control${attentionMaskEnabled ? ' active' : ''}`}>
          <input
            type="checkbox"
            checked={attentionMaskEnabled}
            onChange={(event) => onToggleAttentionMask(event.currentTarget.checked)}
          />
          <ToggleLabel delta="mask">{attentionMaskLabel}</ToggleLabel>
        </label>
      ) : null}

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

      {centeredAvailable ? (
        <label className={`checkbox-control${centeredActive ? ' active' : ''}`}>
          <input
            type="checkbox"
            checked={centeredActive}
            onChange={(event) => onToggleCentered(event.currentTarget.checked)}
          />
          <ToggleLabel delta="centered">Centered Gradient Average</ToggleLabel>
        </label>
      ) : null}

      {weightDecayAvailable ? (
        <label className={`checkbox-control${weightDecayActive ? ' active' : ''}`}>
          <input
            type="checkbox"
            checked={weightDecayActive}
            onChange={(event) => onToggleWeightDecay(event.currentTarget.checked)}
          />
          <ToggleLabel delta="weightDecay">Weight Decay</ToggleLabel>
        </label>
      ) : null}

      {moonshotLrAvailable ? (
        <label className={`checkbox-control${moonshotLrActive ? ' active' : ''}`}>
          <input
            type="checkbox"
            checked={moonshotLrActive}
            onChange={(event) => onToggleMoonshotLr(event.currentTarget.checked)}
          />
          <ToggleLabel delta="moonshotLr">Moonshot lr</ToggleLabel>
        </label>
      ) : null}

      {momentumAvailable ? (
        <label className={`checkbox-control${momentumActive ? ' active' : ''}`}>
          <input
            type="checkbox"
            checked={momentumActive}
            onChange={(event) => onToggleMomentum(event.currentTarget.checked)}
          />
          <ToggleLabel delta="momentum">Momentum</ToggleLabel>
        </label>
      ) : null}

      {nesterovAvailable ? (
        <label className={`checkbox-control${nesterovActive ? ' active' : ''}`}>
          <input
            type="checkbox"
            checked={nesterovActive}
            onChange={(event) => onToggleNesterov(event.currentTarget.checked)}
          />
          <ToggleLabel delta="nesterov">Nesterov</ToggleLabel>
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
