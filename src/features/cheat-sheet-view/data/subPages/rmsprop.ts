import type { AttentionExample } from '../../model'
import rmspropCode from './code/rmsprop.py?raw'
import rmspropMomentumCode from './code/rmspropMomentum.py?raw'
import rmspropMomentumWeightDecayCode from './code/rmspropMomentumWeightDecay.py?raw'
import rmspropWeightDecayCode from './code/rmspropWeightDecay.py?raw'
import { defineAttentionContent, type AlgorithmLineSpec } from '../../lib/contentCompiler'
import { math, strong, text } from '../../lib/segments'

type RmspropVariant = {
  momentum: boolean
  weightDecay: boolean
}

function rmspropRequire({ momentum, weightDecay }: RmspropVariant) {
  return [
    text('Parameters '),
    math(String.raw`\theta_t`),
    text(', gradients '),
    math(String.raw`g_t=\nabla_{\theta}L_t(\theta_t)`),
    text(', learning rate '),
    math(String.raw`\gamma`),
    text(', second-moment coefficient '),
    math(String.raw`\alpha`),
    text(', numerical constant '),
    math(String.raw`\epsilon`),
    ...(weightDecay
      ? [
          text(', coupled weight decay ', 'weightDecay'),
          math(String.raw`\lambda`, 'weightDecay'),
        ]
      : []),
    ...(momentum
      ? [
          text(', momentum coefficient ', 'momentum'),
          math(String.raw`\mu`, 'momentum'),
        ]
      : []),
    text(', centered flag '),
    math(String.raw`c`),
    text(', and optimizer state '),
    math(momentum ? String.raw`\{v_{t-1},\bar{g}_{t-1},b_{t-1}\}` : String.raw`\{v_{t-1},\bar{g}_{t-1}\}`),
    text('.'),
  ]
}

function rmspropRows({ momentum, weightDecay }: RmspropVariant): AlgorithmLineSpec[] {
  let number = 1
  const row = (line: Omit<AlgorithmLineSpec, 'number'>): AlgorithmLineSpec => ({
    ...line,
    number: number++,
  })

  const rows: AlgorithmLineSpec[] = [
    {
      id: 'rmsprop-forward-label',
      startsBlock: { id: 'rmsprop-forward', role: 'forward' },
      parts: [strong('Optimization step.')],
      codeRefs: ['step-signature', 'no-grad'],
    },
    row({
      id: 'rmsprop-loop',
      parts: [
        text('For each parameter tensor '),
        math(String.raw`\theta_t`),
        text(' and gradient tensor '),
        math(String.raw`g_t`),
        text('.'),
      ],
      codeRefs: ['parameter-loop'],
    }),
    row({
      id: 'rmsprop-state-init',
      parts: [
        text('Initialize second-moment state '),
        math(String.raw`v_{t-1}`),
        text('; if centered, initialize first-moment state '),
        math(String.raw`\bar{g}_{t-1}`),
        ...(momentum
          ? [
              text('; initialize momentum buffer ', 'momentum'),
              math(String.raw`b_{t-1}`, 'momentum'),
            ]
          : []),
        text('.'),
      ],
      codeRefs: [
        'state-init',
        'centered-state-init',
        ...(momentum ? ['momentum-state-init'] : []),
      ],
    }),
    row({
      id: 'rmsprop-gradient-direction',
      parts: [
        text('Use the gradient direction '),
        math(String.raw`d_t=g_t`),
        text('.'),
      ],
      codeRefs: ['gradient-direction'],
    }),
  ]

  if (weightDecay) {
    rows.push(
      row({
        id: 'rmsprop-weight-decay',
        parts: [
          text('Add coupled L2 weight decay ', 'weightDecay'),
          math(String.raw`d_t \leftarrow d_t+\lambda\theta_t`, 'weightDecay'),
          text(' before moment updates.', 'weightDecay'),
        ],
        codeRefs: ['weight-decay'],
      })
    )
  }

  rows.push(
    row({
      id: 'rmsprop-square-average',
      parts: [
        text('Update second moment '),
        math(String.raw`v_t=\alpha v_{t-1}+(1-\alpha)d_t\odot d_t`),
        text('.'),
      ],
      codeRefs: ['square-average'],
    }),
    row({
      id: 'rmsprop-centered',
      parts: [
        text('If centered, update first moment '),
        math(String.raw`\bar{g}_t=\alpha\bar{g}_{t-1}+(1-\alpha)d_t`),
        text(' and use '),
        math(String.raw`\tilde{v}_t=v_t-\bar{g}_t\odot\bar{g}_t`),
        text('; otherwise '),
        math(String.raw`\tilde{v}_t=v_t`),
        text('.'),
      ],
      codeRefs: ['centered-average'],
    })
  )

  if (momentum) {
    rows.push(
      row({
        id: 'rmsprop-momentum',
        parts: [
          text('Update momentum buffer ', 'momentum'),
          math(String.raw`b_t=\mu b_{t-1}+d_t/(\sqrt{\tilde{v}_t}+\epsilon)`, 'momentum'),
          text('.', 'momentum'),
        ],
        codeRefs: ['denominator', 'momentum'],
      }),
      row({
        id: 'rmsprop-update',
        parts: [
          text('Apply the buffered update '),
          math(String.raw`\theta_{t+1}=\theta_t-\gamma b_t`, 'momentum'),
          text('.'),
        ],
        codeRefs: ['update'],
      })
    )
  } else {
    rows.push(
      row({
        id: 'rmsprop-update',
        parts: [
          text('Apply the normalized update '),
          math(String.raw`\theta_{t+1}=\theta_t-\gamma d_t/(\sqrt{\tilde{v}_t}+\epsilon)`),
          text('.'),
        ],
        codeRefs: ['denominator', 'update'],
      })
    )
  }

  rows.push(
    row({
      id: 'rmsprop-return',
      parts: [text('Return updated parameters and optimizer state.')],
      codeRefs: ['return-state'],
    })
  )

  return rows
}

const rmspropContent = defineAttentionContent({
  rawCode: rmspropCode,
  require: rmspropRequire({ momentum: false, weightDecay: false }),
  rows: rmspropRows({ momentum: false, weightDecay: false }),
})

const rmspropWeightDecayContent = defineAttentionContent({
  rawCode: rmspropWeightDecayCode,
  require: rmspropRequire({ momentum: false, weightDecay: true }),
  rows: rmspropRows({ momentum: false, weightDecay: true }),
})

const rmspropMomentumContent = defineAttentionContent({
  rawCode: rmspropMomentumCode,
  require: rmspropRequire({ momentum: true, weightDecay: false }),
  rows: rmspropRows({ momentum: true, weightDecay: false }),
})

const rmspropMomentumWeightDecayContent = defineAttentionContent({
  rawCode: rmspropMomentumWeightDecayCode,
  require: rmspropRequire({ momentum: true, weightDecay: true }),
  rows: rmspropRows({ momentum: true, weightDecay: true }),
})

export const rmspropExample: AttentionExample = {
  id: 'rmsprop',
  urlTag: 'optimizer/rmsprop',
  label: 'RMSprop',
  description:
    'RMSprop scales gradients by a running square average, with optional centered variance correction, coupled weight decay, and momentum.',
  algorithmTitle: 'RMSprop',
  content: {
    unmasked: rmspropContent,
    masked: rmspropContent,
  },
  variants: [
    {
      enabled: ['weightDecay'],
      content: { unmasked: rmspropWeightDecayContent, masked: rmspropWeightDecayContent },
    },
    {
      enabled: ['momentum'],
      content: { unmasked: rmspropMomentumContent, masked: rmspropMomentumContent },
    },
    {
      enabled: ['momentum', 'weightDecay'],
      content: {
        unmasked: rmspropMomentumWeightDecayContent,
        masked: rmspropMomentumWeightDecayContent,
      },
    },
  ],
}
