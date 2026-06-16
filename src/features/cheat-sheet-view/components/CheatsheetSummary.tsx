type CheatsheetSummaryProps = {
  description: string
}

export function CheatsheetSummary({ description }: CheatsheetSummaryProps) {
  return (
    <p className="cheatsheet-summary" aria-live="polite">
      {description}
    </p>
  )
}
