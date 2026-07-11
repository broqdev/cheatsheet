import assert from 'node:assert/strict'
import { after, before, test } from 'node:test'
import { createServer } from 'vite'

let server

before(async () => {
  server = await createServer({
    appType: 'custom',
    server: { middlewareMode: true },
  })
})

after(async () => {
  await server?.close()
})

test('the catalog derives navigation and route lookup from one registration', async () => {
  const catalog = await server.ssrLoadModule(
    '/src/features/cheat-sheet-view/data/attentionExamples.ts'
  )

  assert.equal(catalog.examples.length, 10)
  assert.equal(catalog.exampleFromTag('optimizer/sgd')?.id, 'sgd')
  assert.equal(catalog.exampleFromTag('#FlashAttention 2')?.id, 'flash2')
  assert.deepEqual(
    catalog.examplesForExampleGroup('adam').map((example) => example.id),
    ['sgd', 'rmsprop', 'adam', 'adamw', 'muon']
  )
  assert.deepEqual(catalog.routeSegmentsForExample(catalog.exampleFromTag('adamw')), [
    'optimizer',
    'adamw',
  ])
  assert.equal(new Set(catalog.examples.map((example) => example.id)).size, 10)
})

test('variant selection owns dependencies, content resolution, and controls', async () => {
  const catalog = await server.ssrLoadModule(
    '/src/features/cheat-sheet-view/data/attentionExamples.ts'
  )
  const variants = await server.ssrLoadModule(
    '/src/features/cheat-sheet-view/variantSelection.ts'
  )
  const example = catalog.exampleFromTag('sgd')
  const enabled = variants.setVariantEnabled(
    example,
    variants.emptyVariantState(),
    'nesterov',
    true
  )
  const view = variants.resolveVariantView(example, enabled)

  assert.equal(view.state.momentum, true)
  assert.equal(view.state.nesterov, true)
  assert.deepEqual(
    view.controls.map((control) => control.key),
    ['weightDecay', 'momentum', 'nesterov']
  )
  assert.match(view.content.code, /nesterov=True/)

  const fromAliases = variants.variantStateFromSearch('?mom=on&nag=yes')
  assert.equal(fromAliases.momentum, true)
  assert.equal(fromAliases.nesterov, true)
  assert.equal(
    variants.searchForVariantState(example, fromAliases, '?mask=on&campaign=docs'),
    '?mask=on&campaign=docs&momentum=on&nesterov=on'
  )

  const legacyStoredState = variants.variantStateFromUnknown({
    momentumEnabled: true,
    nesterovEnabled: true,
  })
  assert.equal(legacyStoredState.momentum, true)
  assert.equal(legacyStoredState.nesterov, true)
})

test('the content compiler resolves refs and explicit algorithm blocks together', async () => {
  const compiler = await server.ssrLoadModule(
    '/src/features/cheat-sheet-view/lib/contentCompiler.ts'
  )
  const content = compiler.defineAttentionContent({
    rawCode: '# @ref step\nx = 1\n# @end',
    require: [{ kind: 'text', value: 'Input x.' }],
    rows: [
      {
        id: 'forward-label',
        startsBlock: { id: 'forward', role: 'forward' },
        parts: [{ kind: 'strong', value: 'Forward pass.' }],
        codeRefs: ['step'],
      },
      {
        id: 'step',
        parts: [{ kind: 'text', value: 'Set x.' }],
        codeRefs: ['step'],
      },
    ],
  })
  const blocks = compiler.compileAlgorithmBlocks(content)

  assert.equal(content.code, 'x = 1')
  assert.equal(content.blocks, blocks)
  assert.deepEqual(content.rows[1].codeLines, [1])
  assert.equal(blocks[0].id, 'forward')
  assert.equal(blocks[0].title, 'Forward pass')
  assert.deepEqual(blocks[0].rows.map((row) => row.id), ['step'])

  assert.throws(
    () =>
      compiler.defineAttentionContent({
        rawCode: 'x = 1',
        require: [],
        rows: [{ id: 'missing', parts: [], codeRefs: ['missing'] }],
      }),
    /Unknown code ref/
  )
  assert.throws(
    () =>
      compiler.defineAttentionContent({
        rawCode: '# @ref outer\n# @ref inner\nx = 1\n# @end\n# @end',
        require: [],
        rows: [],
      }),
    /Nested code refs/
  )
  assert.throws(
    () =>
      compiler.defineAttentionContent({
        rawCode: 'x = 1',
        require: [],
        rows: [
          {
            id: 'backward-label',
            startsBlock: { id: 'backward', role: 'backward' },
            parts: [{ kind: 'text', value: 'Backward pass.' }],
          },
          { id: 'backward-step', parts: [{ kind: 'text', value: 'Differentiate.' }] },
        ],
      }),
    /Missing backward requirement/
  )
})

test('registered examples compile explicit forward and backward blocks', async () => {
  const catalog = await server.ssrLoadModule(
    '/src/features/cheat-sheet-view/data/attentionExamples.ts'
  )
  const compiler = await server.ssrLoadModule(
    '/src/features/cheat-sheet-view/lib/contentCompiler.ts'
  )

  for (const id of ['naive', 'flash1', 'flash2']) {
    const example = catalog.examples.find((candidate) => candidate.id === id)
    const blocks = example.content.unmasked.blocks

    assert.deepEqual(
      blocks.map((block) => block.title),
      ['Forward pass', 'Backward pass']
    )
    assert.ok(blocks.every((block) => block.rows.length > 0))
  }
})

test('every registered variant resolves its declared document', async () => {
  const catalog = await server.ssrLoadModule(
    '/src/features/cheat-sheet-view/data/attentionExamples.ts'
  )
  const variants = await server.ssrLoadModule(
    '/src/features/cheat-sheet-view/variantSelection.ts'
  )

  for (const example of catalog.examples) {
    for (const variant of example.variants ?? []) {
      const requested = variants.emptyVariantState()
      variant.enabled.forEach((key) => {
        requested[key] = true
      })
      const view = variants.resolveVariantView(example, requested)

      assert.equal(view.content, variant.content[view.mode], `${example.id}: ${variant.enabled}`)
      assert.ok(view.content.blocks.length > 0, `${example.id}: ${variant.enabled} blocks`)
      assert.ok(
        variant.enabled.every((key) => view.controls.some((control) => control.key === key)),
        `${example.id}: missing control for ${variant.enabled}`
      )
    }
  }
})

test('equation presentation keeps HTML and LaTeX adapters semantically aligned', async () => {
  const presentation = await server.ssrLoadModule(
    '/src/features/cheat-sheet-view/lib/equationPresentation.ts'
  )
  const segment = { kind: 'math', value: 'x^2', delta: 'momentum' }

  assert.match(presentation.renderInlineLatex(segment.value), /annotation encoding="application\/x-tex"/)
  assert.equal(
    presentation.segmentToLatex(segment),
    `$\\textcolor{${presentation.variantColor('momentum')}}{x^2}$`
  )
})
