import type { CSSProperties } from 'react'
import type { VariantKey } from '../model'
import type { VariantControl } from '../variantSelection'

type AttentionControlsProps = {
  controls: VariantControl[]
  onToggle: (key: VariantKey, enabled: boolean) => void
}

function ToggleLabel({ control }: { control: VariantControl }) {
  return (
    <>
      <span className="toggle-copy">{control.label}</span>
      <span
        aria-hidden="true"
        className="toggle-swatch"
        style={{ '--toggle-delta-color': control.color } as CSSProperties}
      />
    </>
  )
}

export function AttentionControls({ controls, onToggle }: AttentionControlsProps) {
  if (!controls.length) {
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
      {controls.map((control) => (
        <label
          className={`checkbox-control${control.enabled ? ' active' : ''}`}
          key={control.key}
        >
          <input
            type="checkbox"
            checked={control.enabled}
            onChange={(event) => onToggle(control.key, event.currentTarget.checked)}
          />
          <ToggleLabel control={control} />
        </label>
      ))}
    </section>
  )
}
