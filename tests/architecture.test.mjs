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

  assert.equal(catalog.examples.length, 14)
  assert.equal(catalog.exampleFromTag('optimizer/sgd')?.id, 'sgd')
  assert.equal(catalog.exampleFromTag('optimizer/lbfgs')?.id, 'lbfgs')
  assert.equal(catalog.exampleFromTag('optimizer/shampoo')?.id, 'shampoo')
  assert.equal(catalog.exampleFromTag('optimizer/soap')?.id, 'soap')
  assert.equal(catalog.exampleFromTag('#FlashAttention 2')?.id, 'flash2')
  assert.deepEqual(
    catalog.examplesForExampleGroup('adam').map((example) => example.id),
    [
      'sgd',
      'adagrad',
      'rmsprop',
      'adam',
      'adamw',
      'lbfgs',
      'shampoo',
      'soap',
      'muon',
    ]
  )
  assert.deepEqual(
    catalog.navigationCatalogSections
      .find((section) => section.id === 'optimizer')
      ?.items.map((item) => item.exampleId),
    [
      'sgd',
      'adagrad',
      'rmsprop',
      'adam',
      'adamw',
      'lbfgs',
      'shampoo',
      'soap',
      'muon',
    ]
  )
  assert.deepEqual(catalog.routeSegmentsForExample(catalog.exampleFromTag('adamw')), [
    'optimizer',
    'adamw',
  ])
  assert.equal(new Set(catalog.examples.map((example) => example.id)).size, 14)
})

test('optimizer summaries explain each method at a high level', async () => {
  const catalog = await server.ssrLoadModule(
    '/src/features/cheat-sheet-view/data/attentionExamples.ts'
  )
  const summaries = Object.fromEntries(
    catalog
      .examplesForExampleGroup('sgd')
      .map((example) => [example.id, example.description])
  )

  assert.deepEqual(summaries, {
    sgd: 'SGD moves parameters against the loss gradient, using a learning rate to set the step size.',
    adagrad:
      "AdaGrad shrinks each parameter's step as its accumulated gradient history grows.",
    rmsprop:
      'RMSProp adapts each step using a moving average of recent squared gradients, keeping updates steady as gradient scales change.',
    adam: 'Adam combines momentum with adaptive step sizes, using gradient averages to move quickly while remaining stable.',
    adamw:
      'AdamW pairs adaptive updates with separate weight decay, keeping optimization and regularization distinct.',
    lbfgs:
      'L-BFGS uses recent changes in parameters and gradients to estimate curvature and choose a better search direction.',
    shampoo:
      'Shampoo learns how gradients vary across tensor dimensions, then uses that structure to balance the update.',
    soap: 'SOAP finds a geometry-aware coordinate system for each tensor, then applies adaptive updates in that transformed space.',
    muon: 'Muon reshapes momentum into an approximately orthogonal matrix update, balancing movement across parameter directions.',
  })
})

test('split panel storage restores bounded layout state safely', async () => {
  const splitPanel = await server.ssrLoadModule(
    '/src/features/cheat-sheet-view/components/ResizableSplitPanel.tsx'
  )
  const values = new Map()
  const storage = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
  }

  assert.equal(splitPanel.readStoredPrimaryPercent(47, storage), 47)

  splitPanel.writeStoredPrimaryPercent(54.5, storage)
  assert.equal(values.get(splitPanel.splitPanelStorageKey), '54.5')
  assert.equal(splitPanel.readStoredPrimaryPercent(47, storage), 54.5)

  values.set(splitPanel.splitPanelStorageKey, '99')
  assert.equal(splitPanel.readStoredPrimaryPercent(47, storage), 70)

  values.set(splitPanel.splitPanelStorageKey, '-5')
  assert.equal(splitPanel.readStoredPrimaryPercent(47, storage), 30)

  values.set(splitPanel.splitPanelStorageKey, 'not-a-number')
  assert.equal(splitPanel.readStoredPrimaryPercent(47, storage), 47)

  const blockedStorage = {
    getItem: () => {
      throw new Error('blocked')
    },
    setItem: () => {
      throw new Error('blocked')
    },
  }
  assert.equal(splitPanel.readStoredPrimaryPercent(47, blockedStorage), 47)
  assert.doesNotThrow(() => splitPanel.writeStoredPrimaryPercent(52, blockedStorage))
})

test('LBFGS follows the optimizer LaTeX structure and timestep notation', async () => {
  const { lbfgsExample } = await server.ssrLoadModule(
    '/src/features/cheat-sheet-view/data/subPages/lbfgs.ts'
  )
  const content = lbfgsExample.content.unmasked
  const requirementMath = content.require
    .filter((part) => part.kind === 'math')
    .map((part) => part.value)
  const rowMath = content.rows
    .flatMap((row) => row.parts)
    .filter((part) => part.kind === 'math')
    .map((part) => part.value)

  assert.equal(content.blocks[0].title, 'full-memory inverse-Hessian update')
  assert.equal(content.notes[0].title, 'L-BFGS Optimization step')
  assert.ok(requirementMath.includes(String.raw`\theta_{t-1}\in\mathbb{R}^{n}`))
  assert.ok(rowMath.includes(String.raw`\theta_t=\theta_{t-1}+s_t`))
  assert.ok(!rowMath.some((value) => value.includes(String.raw`\theta_0`)))

  const lbfgsRows = content.notes[0].rows
  const loopStartIndex = lbfgsRows.findIndex(
    (row) => row.id === 'lbfgs-inner-loop'
  )
  const loopEndIndex = lbfgsRows.findIndex(
    (row) => row.id === 'lbfgs-inner-loop-end'
  )
  const returnIndex = lbfgsRows.findIndex((row) => row.id === 'lbfgs-return')

  assert.deepEqual(
    lbfgsRows[loopStartIndex]?.parts.map(({ kind, value }) => [kind, value]),
    [
      ['strong', 'for '],
      ['math', String.raw`k=0,\ldots,K-1`],
      ['strong', ' do'],
    ]
  )
  assert.ok(loopEndIndex > loopStartIndex)
  assert.ok(
    lbfgsRows
      .slice(loopStartIndex + 1, loopEndIndex)
      .every((row) => row.indent === 1)
  )
  assert.deepEqual(
    lbfgsRows[loopEndIndex]?.parts.map(({ kind, value }) => [kind, value]),
    [['strong', 'end for']]
  )
  assert.ok(returnIndex > loopEndIndex)

  const twoLoop = content.notes.find((note) => note.id === 'lbfgs-two-loop')
  assert.equal(twoLoop?.title, 'L-BFGS two-loop recursion')
  assert.equal(twoLoop?.intro, undefined)

  const twoLoopRequireMath = twoLoop?.require
    ?.filter((part) => part.kind === 'math')
    .map((part) => part.value)
  assert.ok(twoLoopRequireMath?.includes(String.raw`c_j=y_j^\top s_j`))
  assert.ok(
    twoLoopRequireMath?.includes(String.raw`V_j=I-\frac{s_jy_j^\top}{c_j}`)
  )
  assert.ok(!twoLoopRequireMath?.some((value) => value.includes(String.raw`\rho`)))

  const directionMath = lbfgsRows
    .find((row) => row.id === 'lbfgs-direction')
    ?.parts.filter((part) => part.kind === 'math')
    .map((part) => part.value)
  assert.ok(directionMath?.includes(String.raw`r_h^{(k)}=M^{(k)}g^{(k)}`))
  assert.ok(directionMath?.includes(String.raw`p^{(k)}=-r_h^{(k)}`))

  const twoLoopReturn = twoLoop?.rows.find(
    (row) => row.id === 'lbfgs-two-loop-return'
  )
  const twoLoopReturnText = twoLoopReturn?.parts
    .filter((part) => part.kind !== 'math')
    .map((part) => part.value)
    .join('')
  const twoLoopReturnMath = twoLoopReturn?.parts
    .filter((part) => part.kind === 'math')
    .map((part) => part.value)
  assert.match(twoLoopReturnText ?? '', /Return/)
  assert.ok(twoLoopReturnMath?.includes(String.raw`r_h=M^{(k)}g^{(k)}`))
  assert.ok(!twoLoopReturnMath?.some((value) => value.includes(String.raw`p^{(k)}`)))

  const backwardTransform = twoLoop?.rows.find(
    (row) => row.id === 'lbfgs-two-loop-backward-transform'
  )
  const backwardStart = twoLoop?.rows.find(
    (row) => row.id === 'lbfgs-two-loop-start'
  )
  const forwardStart = twoLoop?.rows.find(
    (row) => row.id === 'lbfgs-two-loop-scale'
  )
  const forwardTransform = twoLoop?.rows.find(
    (row) => row.id === 'lbfgs-two-loop-forward-transform'
  )
  const threePairBase = twoLoop?.rows.find(
    (row) => row.id === 'lbfgs-two-loop-three-pair-base'
  )
  const threePairFirst = twoLoop?.rows.find(
    (row) => row.id === 'lbfgs-two-loop-three-pair-first'
  )
  const twoLoopMath = (row) =>
    row?.parts.filter((part) => part.kind === 'math').map((part) => part.value)
  const allTwoLoopRowMath = twoLoop?.rows
    .flatMap((row) => row.parts)
    .filter((part) => part.kind === 'math')
    .map((part) => part.value)

  assert.ok(twoLoopMath(backwardStart)?.includes(String.raw`q_h\leftarrow g^{(k)}`))
  assert.ok(
    twoLoopMath(backwardTransform)?.includes(
      String.raw`q_{j-1}\leftarrow V_j^\top q_j`
    )
  )
  assert.ok(
    twoLoopMath(forwardStart)?.includes(String.raw`r_0\leftarrow M_0^{(k)}q_0`)
  )
  assert.ok(
    twoLoopMath(forwardTransform)?.includes(
      String.raw`r_j\leftarrow r_{j-1}+s_j(a_j-b_j)=V_jr_{j-1}+s_ja_j`
    )
  )
  assert.ok(
    twoLoopMath(threePairBase)?.includes(
      String.raw`V_3V_2V_1r_0=V_3V_2V_1M_0^{(3)}q_0=V_3V_2V_1M_0^{(3)}V_1^\top V_2^\top V_3^\top g^{(3)}`
    )
  )
  assert.ok(
    twoLoopMath(threePairFirst)?.includes(
      String.raw`V_3V_2s_1a_1=V_3V_2\frac{s_1s_1^\top}{c_1}V_2^\top V_3^\top g^{(3)}`
    )
  )
  assert.ok(!allTwoLoopRowMath?.some((value) => value.includes(String.raw`\begin`)))
  assert.ok(!allTwoLoopRowMath?.some((value) => value.includes(String.raw`\\`)))

  assert.match(content.code, /def _two_loop_recursion\(/)
  assert.match(content.code, /a = torch\.dot\(s, q\) \/ c/)
  assert.match(content.code, /b = torch\.dot\(y, r\) \/ c/)
  assert.match(content.code, /return r/)
  assert.match(content.code, /c_history = state\.setdefault\("c_history", \[\]\)/)
  assert.match(content.code, /r = _two_loop_recursion\(/)
  assert.match(content.code, /p = -r/)
  assert.doesNotMatch(content.code, /\brho(?:_history)?\b/)
})

test('LBFGS includes a full-memory BFGS derivation mapped to runnable code', async () => {
  const { lbfgsExample } = await server.ssrLoadModule(
    '/src/features/cheat-sheet-view/data/subPages/lbfgs.ts'
  )
  const content = lbfgsExample.content.unmasked
  const bfgs = content.blocks.find((block) => block.id === 'bfgs-full-memory')
  const bfgsMath = bfgs?.rows
    .flatMap((row) => row.parts)
    .filter((part) => part.kind === 'math')
    .map((part) => part.value)
  const bfgsRequireMath = bfgs?.require
    .filter((part) => part.kind === 'math')
    .map((part) => part.value)

  assert.equal(bfgs?.title, 'full-memory inverse-Hessian update')
  assert.ok(bfgsRequireMath?.includes(String.raw`\epsilon_c`))
  assert.ok(
    bfgsRequireMath?.includes(
      String.raw`M_{t-1}\approx\left(\nabla_\theta^2L(\theta_{t-1})\right)^{-1}`
    )
  )
  assert.ok(bfgsMath?.includes(String.raw`p_t=-M_{t-1}g_{t-1}`))
  assert.ok(
    bfgsMath?.includes(
      String.raw`s_t=\theta_t-\theta_{t-1}\approx\delta\theta`
    )
  )
  assert.ok(
    bfgsMath?.includes(
      String.raw`y_t=g_t-g_{t-1}\approx\delta\nabla_{\theta}L(\theta)`
    )
  )
  assert.ok(!bfgsMath?.some((value) => value.includes(String.raw`L^{\prime}`)))
  assert.ok(bfgsMath?.includes(String.raw`c_t\le \epsilon_c`))
  assert.ok(
    bfgsMath?.includes(
      String.raw`V_t=I_n-\frac{s_t y_t^\top}{c_t}`
    )
  )
  assert.ok(
    bfgsMath?.includes(
      String.raw`V_t^\top=I_n-\frac{y_t s_t^\top}{c_t}`
    )
  )
  assert.ok(
    bfgsMath?.includes(
      String.raw`V_t^\top y_t=y_t-\frac{y_t s_t^\top y_t}{c_t}=y_t-\frac{y_t c_t}{c_t}=0`
    )
  )
  assert.ok(
    bfgsMath?.includes(
      String.raw`\frac{s_t s_t^\top}{c_t}y_t=s_t\frac{s_t^\top y_t}{c_t}=s_t\frac{c_t}{c_t}=s_t`
    )
  )
  assert.ok(
    bfgsMath?.includes(
      String.raw`M_t=V_tM_{t-1}V_t^\top+\frac{s_t s_t^\top}{c_t}`
    )
  )
  assert.ok(
    bfgsMath?.includes(
      String.raw`M_t y_t=V_tM_{t-1}V_t^\top y_t+\frac{s_t s_t^\top}{c_t}y_t=s_t`
    )
  )
  assert.ok(!bfgsMath?.some((value) => value.includes(String.raw`=0+s_t`)))
  assert.ok(!bfgsMath?.some((value) => value.includes(String.raw`\begin{aligned}`)))
  assert.ok(!bfgsMath?.some((value) => value.includes(String.raw`\\`)))
  assert.deepEqual(
    bfgs?.rows.slice(6, 11).map((row) => row.id),
    [
      'bfgs-transform',
      'bfgs-transform-cancellation',
      'bfgs-rank-one-correction',
      'bfgs-hessian-update',
      'bfgs-secant-verification',
    ]
  )
  assert.equal(
    bfgs?.rows.find((row) => row.id === 'bfgs-state-store')?.number,
    12
  )
  assert.ok(!bfgsMath?.some((value) => value.includes(String.raw`\rho_t`)))
  assert.ok(!bfgsMath?.some((value) => value.includes(String.raw`c_t^{-1}`)))
  assert.match(content.code, /def bfgs_step\(/)
  const bfgsCode = content.code.slice(
    content.code.indexOf('def bfgs_step'),
    content.code.indexOf('def _two_loop_recursion')
  )
  assert.match(bfgsCode, /epsilon_c=1e-10/)
  assert.match(
    bfgsCode,
    /loss, g_next = _loss_and_grad\(params, loss_gen\)/
  )
  assert.match(bfgsCode, /if c > epsilon_c:/)
  assert.match(bfgsCode, /V = I - torch\.outer\(s, y\) \/ c/)
  assert.match(bfgsCode, /torch\.outer\(s, s\) \/ c/)
  assert.match(bfgsCode, /return params, loss, state/)
  assert.doesNotMatch(bfgsCode, /\brho\b/)
  const parameterUpdateLine =
    content.code.split('\n').findIndex((line) =>
      line.includes('_add_flat_update(params, s)')
    ) + 1
  assert.ok(
    bfgs?.rows
      .find((row) => row.id === 'bfgs-parameter-update')
      ?.codeLines.includes(parameterUpdateLine)
  )
  const bfgsReturnMath = bfgs?.rows
    .find((row) => row.id === 'bfgs-state-store')
    ?.parts.filter((part) => part.kind === 'math')
    .map((part) => part.value)
  assert.deepEqual(bfgsReturnMath, [String.raw`M_t`, String.raw`\theta_t`, String.raw`L_t`])
  assert.ok(
    bfgs?.rows.find((row) => row.id === 'bfgs-hessian-update')?.codeLines.length
  )

  const lbfgs = content.notes.find((note) => note.id === 'lbfgs-optimization')
  const contrastText = lbfgs?.intro
    ?.filter((part) => part.kind !== 'math')
    .map((part) => part.value)
    .join('')
  const contrastMath = lbfgs?.intro
    ?.filter((part) => part.kind === 'math')
    .map((part) => part.value)

  assert.match(contrastText ?? '', /never stores the dense/)
  assert.ok(contrastMath?.includes(String.raw`\mathcal{O}(n^2)`))
  assert.ok(contrastMath?.includes(String.raw`\mathcal{O}(mn)`))
  assert.ok(!lbfgs?.rows.some((row) => row.id === 'lbfgs-bfgs-contrast'))
})

test('LBFGS uses the optimizer objective notation without introducing f', async () => {
  const { lbfgsExample } = await server.ssrLoadModule(
    '/src/features/cheat-sheet-view/data/subPages/lbfgs.ts'
  )
  const content = lbfgsExample.content.unmasked
  const mathValues = [
    ...content.blocks.flatMap((block) => [
      ...block.require,
      ...block.rows.flatMap((row) => row.parts),
    ]),
    ...content.notes.flatMap((note) => [
      ...(note.require ?? []),
      ...note.rows.flatMap((row) => row.parts),
    ]),
  ]
    .filter((part) => part.kind === 'math')
    .map((part) => part.value)

  assert.ok(mathValues.includes(String.raw`L(\theta)`))
  assert.ok(
    mathValues.includes(String.raw`g_{t-1}=\nabla_{\theta}L(\theta_{t-1})`)
  )
  assert.ok(mathValues.includes(String.raw`L_t=L(\theta_t)`))
  assert.ok(
    mathValues.includes(String.raw`L^{(k+1)}=L(\theta^{(k+1)})`)
  )
  assert.ok(!mathValues.some((value) => /f(?:\(|_|\^)/.test(value)))
})

test('LBFGS evaluates loss and gradients without a side-effecting closure', async () => {
  const { lbfgsExample } = await server.ssrLoadModule(
    '/src/features/cheat-sheet-view/data/subPages/lbfgs.ts'
  )
  const content = lbfgsExample.content.unmasked
  const prose = [
    ...content.blocks.flatMap((block) => [
      ...block.require,
      ...block.rows.flatMap((row) => row.parts),
    ]),
    ...content.notes.flatMap((note) => [
      ...(note.require ?? []),
      ...note.rows.flatMap((row) => row.parts),
    ]),
  ]
    .filter((part) => part.kind === 'text')
    .map((part) => part.value)
    .join(' ')

  assert.doesNotMatch(prose, /\bclosure\b/i)
  assert.doesNotMatch(content.code, /\bclosure\b/)
  assert.match(content.code, /def _loss_and_grad\(params, loss_gen\):/)
  assert.match(content.code, /param\.grad = None/)
  assert.match(content.code, /loss = loss_gen\(\)/)
  assert.match(content.code, /loss\.backward\(\)/)
})

test('LBFGS defers the final fixed-step evaluation without losing its correction', async () => {
  const { lbfgsExample } = await server.ssrLoadModule(
    '/src/features/cheat-sheet-view/data/subPages/lbfgs.ts'
  )
  const code = lbfgsExample.content.unmasked.code
  const historyHelperCode = code.slice(
    code.indexOf('def _append_correction('),
    code.indexOf('def lbfgs_step(')
  )
  const stepCode = code.slice(code.indexOf('def lbfgs_step('))
  const initialEvaluationIndex = stepCode.indexOf(
    'loss, g = _loss_and_grad(params, loss_gen)'
  )
  const gradientToleranceIndex = stepCode.indexOf(
    'if g.abs().max() <= tolerance_grad:'
  )
  const maxIterationGuardIndex = stepCode.indexOf('if max_iter <= 0:')
  const pendingRefreshIndex = stepCode.indexOf(
    'pending_step = state.pop("pending_step", None)'
  )
  const updateIndex = stepCode.indexOf('_add_flat_update(params, s)')
  const finalIterationIndex = stepCode.indexOf('if k == max_iter - 1:')
  const reevaluateIndex = stepCode.indexOf(
    'loss, g_next = _loss_and_grad(params, loss_gen)'
  )

  assert.match(historyHelperCode, /if not \(c > 1e-10\):/)
  assert.match(stepCode, /pending_step = state\.pop\("pending_step", None\)/)
  assert.match(stepCode, /s_prev, g_prev = pending_step/)
  assert.match(stepCode, /s_prev, g - g_prev, history_size/)
  assert.match(stepCode, /state\["pending_step"\] = \(s, g\)/)
  assert.ok(initialEvaluationIndex < gradientToleranceIndex)
  assert.ok(gradientToleranceIndex < maxIterationGuardIndex)
  assert.ok(maxIterationGuardIndex < pendingRefreshIndex)
  assert.ok(updateIndex < finalIterationIndex)
  assert.ok(finalIterationIndex < reevaluateIndex)
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

  const shampoo = catalog.exampleFromTag('shampoo')
  const shampooBaseView = variants.resolveVariantView(
    shampoo,
    variants.emptyVariantState()
  )
  const shampooView = variants.resolveVariantView(
    shampoo,
    variants.setVariantEnabled(
      shampoo,
      variants.emptyVariantState(),
      'weightDecay',
      true
    )
  )
  assert.deepEqual(
    shampooView.controls.map((control) => control.key),
    ['weightDecay']
  )
  assert.match(
    shampooView.content.code,
    /matrix_grad = matrix_grad\.add\(param, alpha=weight_decay\)/
  )
  assert.doesNotMatch(shampooBaseView.content.code, /weight_decay/)
  assert.ok(
    shampooView.content.code.indexOf('matrix_grad = matrix_grad.add') <
      shampooView.content.code.indexOf('left.addmm_')
  )

  const soap = catalog.exampleFromTag('soap')
  const soapBaseView = variants.resolveVariantView(
    soap,
    variants.emptyVariantState()
  )
  const soapView = variants.resolveVariantView(
    soap,
    variants.setVariantEnabled(soap, variants.emptyVariantState(), 'weightDecay', true)
  )
  assert.deepEqual(
    soapView.controls.map((control) => control.key),
    ['weightDecay']
  )
  assert.match(soapView.content.code, /param\.mul_\(1\.0 - lr \* weight_decay\)/)
  assert.doesNotMatch(soapBaseView.content.code, /weight_decay/)
  assert.ok(
    soapView.content.code.indexOf('param.mul_') <
      soapView.content.code.indexOf('param.add_')
  )

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

  const noteLatex = presentation.latexDocument(
    { algorithmTitle: 'Test' },
    undefined,
    [],
    [
      {
        id: 'intro-order',
        title: 'Intro order',
        intro: [{ kind: 'text', value: 'Difference first.' }],
        requireLabel: 'Require',
        require: [{ kind: 'text', value: 'Inputs second.' }],
        rows: [],
      },
    ]
  )
  assert.ok(
    noteLatex.indexOf('Difference first.') <
      noteLatex.indexOf('\\textbf{Require:}')
  )
})
