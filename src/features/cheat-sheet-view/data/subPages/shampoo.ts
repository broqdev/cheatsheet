import type { AttentionExample } from '../../model'
import { defineAttentionContent, type AlgorithmLineSpec } from '../../lib/contentCompiler'
import { math, strong, text } from '../../lib/segments'
import shampooCode from './code/shampoo.py?raw'
import shampooWeightDecayCode from './code/shampooWeightDecay.py?raw'

type ShampooVariant = {
  weightDecay: boolean
}

function shampooRequire({ weightDecay }: ShampooVariant) {
  return [
    text('Matrix parameters '),
    math(String.raw`\theta_{t-1}\in\mathbb{R}^{M\times N}`),
    text(', gradients '),
    math(String.raw`G_t=\nabla_{\theta}L(\theta_{t-1})`),
    text(', learning rate '),
    math(String.raw`\gamma`),
    text(', numerical constant '),
    math(String.raw`\epsilon`),
    ...(weightDecay
      ? [
          text(', coupled weight decay ', 'weightDecay'),
          math(String.raw`\lambda`, 'weightDecay'),
        ]
      : []),
    text(', and optimizer state '),
    math(String.raw`\{L_{t-1},R_{t-1}\}`),
    text('.'),
  ]
}

function shampooRows({ weightDecay }: ShampooVariant): AlgorithmLineSpec[] {
  let number = 1
  const row = (line: Omit<AlgorithmLineSpec, 'number'>): AlgorithmLineSpec => ({
    ...line,
    number: number++,
  })
  const direction = weightDecay ? String.raw`D_t` : String.raw`G_t`
  const preconditionedDirection = weightDecay
    ? String.raw`\widetilde{D}_t`
    : String.raw`\widetilde{G}_t`

  const rows: AlgorithmLineSpec[] = [
    {
      id: 'shampoo-forward-label',
      startsBlock: { id: 'shampoo-forward', role: 'forward' },
      parts: [
        strong(
          weightDecay
            ? 'Matrix-preconditioned optimization step with coupled decay.'
            : 'Matrix-preconditioned optimization step.'
        ),
      ],
      codeRefs: ['step-signature', 'no-grad'],
    },
    row({
      id: 'shampoo-loop',
      parts: [
        text('For each matrix parameter '),
        math(String.raw`\theta_{t-1}`),
        text(' and matrix gradient '),
        math(String.raw`G_t`),
        text('.'),
      ],
      codeRefs: ['parameter-loop'],
    }),
    row({
      id: 'shampoo-state-init',
      parts: [
        text('Initialize the row and column preconditioners as '),
        math(String.raw`L_0=\epsilon I_M`),
        text(' and '),
        math(String.raw`R_0=\epsilon I_N`),
        text(' when missing.'),
      ],
      codeRefs: ['state-init'],
    }),
  ]

  if (weightDecay) {
    rows.push(
      row({
        id: 'shampoo-weight-decay',
        parts: [
          text('Add coupled L2 weight decay ', 'weightDecay'),
          math(
            String.raw`D_t=G_t+\lambda\theta_{t-1}`,
            'weightDecay'
          ),
          text(' before updating the preconditioners.', 'weightDecay'),
        ],
        codeRefs: ['weight-decay'],
      })
    )
  }

  rows.push(
    row({
      id: 'shampoo-preconditioner-update',
      parts: [
        text('Accumulate row and column direction covariances '),
        math(String.raw`L_t=L_{t-1}+${direction}${direction}^\top`),
        text(' and '),
        math(String.raw`R_t=R_{t-1}+${direction}^\top ${direction}`),
        text('.'),
      ],
      codeRefs: ['preconditioner-update'],
    }),
    row({
      id: 'shampoo-inverse-roots',
      parts: [
        text('Use symmetric eigendecomposition to form inverse fourth roots '),
        math(String.raw`P_t=L_t^{-1/4}`),
        text(' and '),
        math(String.raw`Q_t=R_t^{-1/4}`),
        text('.'),
      ],
      codeRefs: [
        'matrix-power-signature',
        'matrix-power-eigh',
        'matrix-power-reconstruct',
        'inverse-roots',
      ],
    }),
    row({
      id: 'shampoo-precondition-gradient',
      parts: [
        text('Precondition both matrix axes '),
        math(String.raw`${preconditionedDirection}=P_t${direction}Q_t`),
        text('.'),
      ],
      codeRefs: ['precondition-gradient'],
    }),
    row({
      id: 'shampoo-update',
      parts: [
        text('Update parameters '),
        math(String.raw`\theta_t=\theta_{t-1}-\gamma${preconditionedDirection}`),
        text('.'),
      ],
      codeRefs: ['update'],
    }),
    row({
      id: 'shampoo-return',
      parts: [
        text('Store '),
        math(String.raw`\{L_t,R_t\}`),
        text(' and return updated parameters and state.'),
      ],
      codeRefs: ['state-store', 'return-state'],
    })
  )

  return rows
}

const shampooContent = defineAttentionContent({
  rawCode: shampooCode,
  require: shampooRequire({ weightDecay: false }),
  rows: shampooRows({ weightDecay: false }),
})

const shampooWeightDecayContent = defineAttentionContent({
  rawCode: shampooWeightDecayCode,
  require: shampooRequire({ weightDecay: true }),
  rows: shampooRows({ weightDecay: true }),
})

export const shampooExample: AttentionExample = {
  id: 'shampoo',
  urlTag: 'optimizer/shampoo',
  label: 'Shampoo',
  description:
    'Shampoo learns how gradients vary across tensor dimensions, then uses that structure to balance the update.',
  algorithmTitle: 'Shampoo',
  content: {
    unmasked: shampooContent,
    masked: shampooContent,
  },
  variants: [
    {
      enabled: ['weightDecay'],
      content: {
        unmasked: shampooWeightDecayContent,
        masked: shampooWeightDecayContent,
      },
    },
  ],
}
