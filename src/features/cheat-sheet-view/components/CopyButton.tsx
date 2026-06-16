import { Copy, CopyCheck } from 'lucide-react'

type CopyButtonProps = {
  copied: boolean
  label: string
  onCopy: () => void
}

export function CopyButton({ copied, label, onCopy }: CopyButtonProps) {
  const Icon = copied ? CopyCheck : Copy

  return (
    <button
      type="button"
      className="region-copy-button"
      aria-label={copied ? `${label} copied` : label}
      title={copied ? 'Copied' : label}
      onClick={onCopy}
      onPointerDown={(event) => event.stopPropagation()}
    >
      <Icon aria-hidden="true" size={17} strokeWidth={2.15} />
    </button>
  )
}

