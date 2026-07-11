import { Copy, CopyCheck } from 'lucide-react'

type CopyButtonProps = {
  copied: boolean
  label: string
  onCopy: () => void
  showCopiedLabel?: boolean
}

export function CopyButton({
  copied,
  label,
  onCopy,
  showCopiedLabel = false,
}: CopyButtonProps) {
  const Icon = copied ? CopyCheck : Copy
  const statusLabel = copied ? 'Copied' : label
  const showConfirmation = copied && showCopiedLabel

  return (
    <button
      type="button"
      className={`region-copy-button${showConfirmation ? ' region-copy-button--confirmed' : ''}`}
      aria-label={copied ? `${label} copied` : label}
      title={statusLabel}
      onClick={onCopy}
      onPointerDown={(event) => event.stopPropagation()}
    >
      <Icon aria-hidden="true" size={17} strokeWidth={2.15} />
      {showConfirmation ? (
        <span className="region-copy-button-label" aria-live="polite">
          Copied
        </span>
      ) : null}
    </button>
  )
}
