import type { AttentionExample } from '../../model'
import rmspropCode from './code/rmsprop.py?raw'
import rmspropCenteredCode from './code/rmspropCentered.py?raw'
import rmspropCenteredMomentumCode from './code/rmspropCenteredMomentum.py?raw'
import rmspropCenteredMomentumWeightDecayCode from './code/rmspropCenteredMomentumWeightDecay.py?raw'
import rmspropCenteredWeightDecayCode from './code/rmspropCenteredWeightDecay.py?raw'
import rmspropMomentumCode from './code/rmspropMomentum.py?raw'
import rmspropMomentumWeightDecayCode from './code/rmspropMomentumWeightDecay.py?raw'
import rmspropWeightDecayCode from './code/rmspropWeightDecay.py?raw'
import { defineAttentionContent, type AlgorithmLineSpec } from '../../lib/contentCompiler'
import { math, strong, text } from '../../lib/segments'

type RmspropVariant = {
  centered: boolean
  momentum: boolean
  weightDecay: boolean
}

function rmspropStoredState({ centered, momentum }: RmspropVariant) {
  const values = [
    String.raw`v_t`,
    ...(centered ? [String.raw`\bar{g}_t`] : []),
    ...(momentum ? [String.raw`b_t`] : []),
  ]

  return values.length === 1 ? values[0] : String.raw`\{${values.join(',')}\}`
}

function rmspropRequire({ centered, momentum, weightDecay }: RmspropVariant) {
  return [
    text('Parameters '),
    math(String.raw`\theta_{t-1}`),
    text(', gradients '),
    math(String.raw`g_t=\nabla_{\theta}L_t(\theta_{t-1})`),
    text(', learning rate '),
    math(String.raw`\gamma`),
    text(', square-average coefficient '),
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
    text(', and optimizer state '),
    math(String.raw`v_{t-1}`),
    ...(centered
      ? [
          text(', centered gradient average ', 'centered'),
          math(String.raw`\bar{g}_{t-1}`, 'centered'),
        ]
      : []),
    ...(momentum
      ? [
          text(', momentum buffer ', 'momentum'),
          math(String.raw`b_{t-1}`, 'momentum'),
        ]
      : []),
    text('.'),
  ]
}

function rmspropRows({ centered, momentum, weightDecay }: RmspropVariant): AlgorithmLineSpec[] {
  let number = 1
  const row = (line: Omit<AlgorithmLineSpec, 'number'>): AlgorithmLineSpec => ({
    ...line,
    number: number++,
  })

  const rows: AlgorithmLineSpec[] = [
    {
      id: 'rmsprop-forward-label',
      parts: [strong('Optimization step.')],
      codeRefs: ['step-signature', 'no-grad'],
    },
    row({
      id: 'rmsprop-loop',
      parts: [
        text('For each parameter tensor '),
        math(String.raw`\theta_{t-1}`),
        text(' and gradient tensor '),
        math(String.raw`g_t`),
        text('.'),
      ],
      codeRefs: ['parameter-loop'],
    }),
    row({
      id: 'rmsprop-state-init',
      parts: [
        text('Initialize running square average '),
        math(String.raw`v_{t-1}`),
        ...(centered
          ? [
              text('; initialize centered gradient average ', 'centered'),
              math(String.raw`\bar{g}_{t-1}`, 'centered'),
            ]
          : []),
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
        ...(centered ? ['centered-state-init'] : []),
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
          math(String.raw`d_t \leftarrow d_t+\lambda\theta_{t-1}`, 'weightDecay'),
          text(' before square averaging.', 'weightDecay'),
        ],
        codeRefs: ['weight-decay'],
      })
    )
  }

  rows.push(
    row({
      id: 'rmsprop-square-average',
      parts: [
        text('Update running square average '),
        math(String.raw`v_t=\alpha v_{t-1}+(1-\alpha)d_t\odot d_t`),
        text('.'),
      ],
      codeRefs: ['square-average'],
    }),
  )

  if (centered) {
    rows.push(
      row({
        id: 'rmsprop-centered',
        parts: [
          text('Update centered gradient average ', 'centered'),
          math(
            String.raw`\bar{g}_t=\alpha\bar{g}_{t-1}+(1-\alpha)d_t`,
            'centered'
          ),
          text(' and use ', 'centered'),
          math(String.raw`\tilde{v}_t=v_t-\bar{g}_t\odot\bar{g}_t`, 'centered'),
          text('.', 'centered'),
        ],
        codeRefs: ['centered-average'],
      })
    )
  }

  rows.push(
    row({
      id: 'rmsprop-denominator',
      parts: [
        text('Compute RMS denominator '),
        math(
          centered
            ? String.raw`r_t=\sqrt{\tilde{v}_t}+\epsilon`
            : String.raw`r_t=\sqrt{v_t}+\epsilon`,
          centered ? 'centered' : undefined
        ),
        text('.'),
      ],
      codeRefs: ['denominator'],
    })
  )

  if (momentum) {
    rows.push(
      row({
        id: 'rmsprop-momentum',
        parts: [
          text('Update momentum buffer ', 'momentum'),
          math(String.raw`b_t=\mu b_{t-1}+d_t/r_t`, 'momentum'),
          text('.', 'momentum'),
        ],
        codeRefs: ['momentum'],
      }),
      row({
        id: 'rmsprop-update',
        parts: [
          text('Apply the buffered update '),
          math(String.raw`\theta_t=\theta_{t-1}-\gamma b_t`, 'momentum'),
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
          math(String.raw`\theta_t=\theta_{t-1}-\gamma d_t/r_t`),
          text('.'),
        ],
        codeRefs: ['update'],
      })
    )
  }

  rows.push(
    row({
      id: 'rmsprop-return',
      parts: [
        text('Store '),
        math(rmspropStoredState({ centered, momentum, weightDecay })),
        text(' and return updated parameters and state.'),
      ],
      codeRefs: ['return-state'],
    })
  )

  return rows
}

function defineRmspropContent(rawCode: string, variant: RmspropVariant) {
  return defineAttentionContent({
    rawCode,
    require: rmspropRequire(variant),
    rows: rmspropRows(variant),
  })
}

const rmspropContent = defineRmspropContent(rmspropCode, {
  centered: false,
  momentum: false,
  weightDecay: false,
})

const rmspropCenteredContent = defineRmspropContent(rmspropCenteredCode, {
  centered: true,
  momentum: false,
  weightDecay: false,
})

const rmspropWeightDecayContent = defineRmspropContent(rmspropWeightDecayCode, {
  centered: false,
  momentum: false,
  weightDecay: true,
})

const rmspropCenteredWeightDecayContent = defineRmspropContent(
  rmspropCenteredWeightDecayCode,
  {
    centered: true,
    momentum: false,
    weightDecay: true,
  }
)

const rmspropMomentumContent = defineRmspropContent(rmspropMomentumCode, {
  centered: false,
  momentum: true,
  weightDecay: false,
})

const rmspropCenteredMomentumContent = defineRmspropContent(rmspropCenteredMomentumCode, {
  centered: true,
  momentum: true,
  weightDecay: false,
})

const rmspropMomentumWeightDecayContent = defineRmspropContent(
  rmspropMomentumWeightDecayCode,
  {
    centered: false,
    momentum: true,
    weightDecay: true,
  }
)

const rmspropCenteredMomentumWeightDecayContent = defineRmspropContent(
  rmspropCenteredMomentumWeightDecayCode,
  {
    centered: true,
    momentum: true,
    weightDecay: true,
  }
)

export const rmspropExample: AttentionExample = {
  id: 'rmsprop',
  urlTag: 'optimizer/rmsprop',
  label: 'RMSProp',
  description:
    'RMSProp scales gradients by a running square average, with optional centered variance correction, coupled weight decay, and momentum.',
  algorithmTitle: 'RMSProp',
  content: {
    unmasked: rmspropContent,
    masked: rmspropContent,
  },
  variants: [
    {
      enabled: ['centered'],
      content: { unmasked: rmspropCenteredContent, masked: rmspropCenteredContent },
    },
    {
      enabled: ['weightDecay'],
      content: { unmasked: rmspropWeightDecayContent, masked: rmspropWeightDecayContent },
    },
    {
      enabled: ['momentum'],
      content: { unmasked: rmspropMomentumContent, masked: rmspropMomentumContent },
    },
    {
      enabled: ['centered', 'weightDecay'],
      content: {
        unmasked: rmspropCenteredWeightDecayContent,
        masked: rmspropCenteredWeightDecayContent,
      },
    },
    {
      enabled: ['centered', 'momentum'],
      content: {
        unmasked: rmspropCenteredMomentumContent,
        masked: rmspropCenteredMomentumContent,
      },
    },
    {
      enabled: ['momentum', 'weightDecay'],
      content: {
        unmasked: rmspropMomentumWeightDecayContent,
        masked: rmspropMomentumWeightDecayContent,
      },
    },
    {
      enabled: ['centered', 'momentum', 'weightDecay'],
      content: {
        unmasked: rmspropCenteredMomentumWeightDecayContent,
        masked: rmspropCenteredMomentumWeightDecayContent,
      },
    },
  ],
}
