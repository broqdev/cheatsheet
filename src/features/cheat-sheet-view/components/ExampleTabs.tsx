import type { AttentionExample } from '../model'

type ExampleTabsProps = {
  activeExample: AttentionExample
  examples: AttentionExample[]
  onSwitchExample: (exampleId: string) => void
}

export function ExampleTabs({ activeExample, examples, onSwitchExample }: ExampleTabsProps) {
  return (
    <nav className="example-nav" aria-label="Attention variant selector">
      <div className="example-nav-track" role="tablist">
        {examples.map((example) => {
          const isActive = example.id === activeExample.id

          return (
            <button
              key={example.id}
              type="button"
              className={isActive ? 'active' : ''}
              aria-controls="attention-panel"
              aria-current={isActive ? 'page' : undefined}
              aria-selected={isActive}
              onClick={() => onSwitchExample(example.id)}
              role="tab"
            >
              <span>{example.label}</span>
            </button>
          )
        })}
      </div>
    </nav>
  )
}
