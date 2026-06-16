type AttentionControlsProps = {
  attentionMaskEnabled: boolean
  onToggleAttentionMask: (enabled: boolean) => void
}

export function AttentionControls({
  attentionMaskEnabled,
  onToggleAttentionMask,
}: AttentionControlsProps) {
  return (
    <section className="control-strip" aria-label="Attention controls">
      <label className={`checkbox-control${attentionMaskEnabled ? ' active' : ''}`}>
        <input
          type="checkbox"
          checked={attentionMaskEnabled}
          onChange={(event) => onToggleAttentionMask(event.currentTarget.checked)}
        />
        <span className="toggle-copy">Attention mask</span>
      </label>
    </section>
  )
}
