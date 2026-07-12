import type { AttentionExample } from '../../model'
import sgdCode from './code/sgd.py?raw'
import sgdMomentumCode from './code/sgdMomentum.py?raw'
import sgdMomentumNesterovCode from './code/sgdMomentumNesterov.py?raw'
import sgdMomentumWeightDecayCode from './code/sgdMomentumWeightDecay.py?raw'
import sgdMomentumWeightDecayNesterovCode from './code/sgdMomentumWeightDecayNesterov.py?raw'
import sgdWeightDecayCode from './code/sgdWeightDecay.py?raw'
import { defineAttentionContent, type AlgorithmLineSpec, type LatexBlockSpec } from '../../lib/codeRefs'
import { math, strong, text } from '../../lib/segments'

type SgdVariant = {
  momentum: boolean
  nesterov: boolean
  weightDecay: boolean
}

const baseRequire = [
  text('Parameters '),
  math(String.raw`\theta_{t-1}`),
  text(', gradients '),
  math(String.raw`g_t=\nabla_{\theta}L_t(\theta_{t-1})`),
  text(', and learning rate '),
  math(String.raw`\gamma`),
]

function sgdRequire({ momentum, nesterov, weightDecay }: SgdVariant) {
  return [
    ...baseRequire,
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
          text(', dampening ', 'momentum'),
          math(String.raw`\tau`, 'momentum'),
          ...(nesterov
            ? [
                text(', Nesterov flag ', 'nesterov'),
                math(String.raw`\operatorname{nesterov}`, 'nesterov'),
              ]
            : []),
          text(', and optimizer state ', 'momentum'),
          math(String.raw`\{b_{t-1}\}`, 'momentum'),
        ]
      : []),
    text('.'),
  ]
}

function sgdRows({ momentum, nesterov, weightDecay }: SgdVariant): AlgorithmLineSpec[] {
  let number = 1
  const row = (line: Omit<AlgorithmLineSpec, 'number'>): AlgorithmLineSpec => ({
    ...line,
    number: number++,
  })

  const rows: AlgorithmLineSpec[] = [
    {
      id: 'sgd-forward-label',
      parts: [strong('Optimization step.')],
      codeRefs: ['step-signature', 'no-grad'],
    },
    row({
      id: 'sgd-loop',
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
      id: 'sgd-gradient-direction',
      parts: [
        text('Use the gradient as the descent direction '),
        math(String.raw`d_t=g_t`),
        text('.'),
      ],
      codeRefs: ['gradient-direction'],
    }),
  ]

  if (weightDecay) {
    rows.push(
      row({
        id: 'sgd-weight-decay',
        parts: [
          text('Add coupled L2 weight decay ', 'weightDecay'),
          math(String.raw`d_t \leftarrow d_t+\lambda\theta_{t-1}`, 'weightDecay'),
          text(momentum ? ' before momentum.' : '.', 'weightDecay'),
        ],
        codeRefs: ['weight-decay'],
      })
    )
  }

  if (momentum) {
    rows.push(
      row({
        id: 'sgd-momentum-state',
        parts: [
          text('Read the momentum buffer ', 'momentum'),
          math(String.raw`b_{t-1}`, 'momentum'),
          text(', initializing it from ', 'momentum'),
          math(String.raw`d_t`, 'momentum'),
          text(' on the first step.', 'momentum'),
        ],
        codeRefs: ['state-init'],
      }),
      row({
        id: 'sgd-momentum',
        parts: [
          text('Update momentum direction ', 'momentum'),
          math(String.raw`b_t=\mu b_{t-1}+(1-\tau)d_t`, 'momentum'),
          text(' after the first step.', 'momentum'),
        ],
        codeRefs: ['momentum'],
      })
    )

    if (nesterov) {
      rows.push(
        row({
          id: 'sgd-nesterov',
          parts: [
            text('Form the PyTorch-style Nesterov lookahead direction ', 'nesterov'),
            math(String.raw`\hat{b}_t=d_t+\mu b_t`, 'nesterov'),
            text('.', 'nesterov'),
          ],
          codeRefs: ['nesterov'],
        })
      )
    }

    rows.push(
      row({
        id: 'sgd-update',
        parts: [
          text('Apply the parameter update '),
          math(
            nesterov
              ? String.raw`\theta_t=\theta_{t-1}-\gamma\hat{b}_t`
              : String.raw`\theta_t=\theta_{t-1}-\gamma b_t`,
            nesterov ? 'nesterov' : 'momentum'
          ),
          text('.'),
        ],
        codeRefs: ['update'],
      }),
      row({
        id: 'sgd-return',
        parts: [
          text('Store ', 'momentum'),
          math(String.raw`b_t`, 'momentum'),
          text(' and return updated parameters and state.', 'momentum'),
        ],
        codeRefs: ['state-store', 'return-state'],
      })
    )

    return rows
  }

  rows.push(
    row({
      id: 'sgd-update',
      parts: [
        text('Apply the parameter update '),
        math(String.raw`\theta_t=\theta_{t-1}-\gamma d_t`),
        text('.'),
      ],
      codeRefs: ['update'],
    }),
    row({
      id: 'sgd-return',
      parts: [
        text('Return the updated parameters '),
        math(String.raw`\theta_t`),
        text('.'),
      ],
      codeRefs: ['return-params'],
    })
  )

  return rows
}

const pytorchNesterovNote: LatexBlockSpec = {
  id: 'sgd-pytorch-sutskever-nesterov',
  title: 'PyTorch vs. Sutskever Nesterov',
  require: [
    text('Descent direction '),
    math(String.raw`d_t`),
    text(', learning rate '),
    math(String.raw`\gamma`),
    text(', momentum '),
    math(String.raw`\mu`),
    text(', and momentum buffer '),
    math(String.raw`b_t`),
    text('.'),
  ],
  rows: [
    {
      id: 'sgd-pytorch-nesterov-label',
      number: 1,
      parts: [text('PyTorch keeps learning rate outside the buffer.')],
    },
    {
      id: 'sgd-pytorch-nesterov-buffer',
      indent: 1,
      parts: [math(String.raw`b_t=\mu b_{t-1}+d_t`)],
      codeRefs: ['momentum'],
    },
    {
      id: 'sgd-pytorch-nesterov-theta',
      indent: 1,
      parts: [
        math(String.raw`\theta_t=\theta_{t-1}-\gamma(d_t+\mu b_t)=\theta_{t-1}-\gamma\left(d_t+\mu(\mu b_{t-1}+d_t)\right)`),
      ],
      codeRefs: ['momentum', 'nesterov', 'update'],
    },
    {
      id: 'sgd-sutskever-nesterov-label',
      number: 2,
      parts: [text('Sutskever-style notation puts learning rate inside the buffer.')],
    },
    {
      id: 'sgd-sutskever-nesterov-buffer',
      indent: 1,
      parts: [math(String.raw`b_t=\mu b_{t-1}+\gamma d_t`)],
    },
    {
      id: 'sgd-sutskever-nesterov-theta',
      indent: 1,
      parts: [
        math(String.raw`\theta_t=\theta_{t-1}-(\gamma d_t+\mu b_t)=\theta_{t-1}-\left(\gamma d_t+\mu(\mu b_{t-1}+\gamma d_t)\right)`),
      ],
    },
    {
      id: 'sgd-nesterov-scaling-note',
      number: 3,
      parts: [
        text('Both conventions reuse '),
        math(String.raw`b_t`),
        text(' for convention-local buffers; the expanded '),
        math(String.raw`\theta`),
        text(' updates show the same Nesterov shape with different learning-rate placement. PyTorch also initializes '),
        math(String.raw`b_1=d_1`),
        text(' and applies dampening after the first step.'),
      ],
      codeRefs: ['state-init', 'momentum'],
    },
  ],
}

const sgdContent = defineAttentionContent({
  rawCode: sgdCode,
  require: sgdRequire({ momentum: false, nesterov: false, weightDecay: false }),
  rows: sgdRows({ momentum: false, nesterov: false, weightDecay: false }),
})

const sgdWeightDecayContent = defineAttentionContent({
  rawCode: sgdWeightDecayCode,
  require: sgdRequire({ momentum: false, nesterov: false, weightDecay: true }),
  rows: sgdRows({ momentum: false, nesterov: false, weightDecay: true }),
})

const sgdMomentumContent = defineAttentionContent({
  rawCode: sgdMomentumCode,
  require: sgdRequire({ momentum: true, nesterov: false, weightDecay: false }),
  rows: sgdRows({ momentum: true, nesterov: false, weightDecay: false }),
})

const sgdMomentumWeightDecayContent = defineAttentionContent({
  rawCode: sgdMomentumWeightDecayCode,
  require: sgdRequire({ momentum: true, nesterov: false, weightDecay: true }),
  rows: sgdRows({ momentum: true, nesterov: false, weightDecay: true }),
})

const sgdNesterovContent = defineAttentionContent({
  rawCode: sgdMomentumNesterovCode,
  require: sgdRequire({ momentum: true, nesterov: true, weightDecay: false }),
  rows: sgdRows({ momentum: true, nesterov: true, weightDecay: false }),
  notes: [pytorchNesterovNote],
})

const sgdNesterovWeightDecayContent = defineAttentionContent({
  rawCode: sgdMomentumWeightDecayNesterovCode,
  require: sgdRequire({ momentum: true, nesterov: true, weightDecay: true }),
  rows: sgdRows({ momentum: true, nesterov: true, weightDecay: true }),
  notes: [pytorchNesterovNote],
})

export const sgdExample: AttentionExample = {
  id: 'sgd',
  urlTag: 'optimizer/sgd',
  label: 'SGD',
  description:
    'Stochastic gradient descent applies a learning-rate-scaled gradient step, with optional coupled weight decay, momentum, and PyTorch-style Nesterov lookahead.',
  algorithmTitle: 'SGD',
  content: {
    unmasked: sgdContent,
    masked: sgdContent,
  },
  weightDecayContent: {
    unmasked: sgdWeightDecayContent,
    masked: sgdWeightDecayContent,
  },
  momentumContent: {
    unmasked: sgdMomentumContent,
    masked: sgdMomentumContent,
  },
  momentumWeightDecayContent: {
    unmasked: sgdMomentumWeightDecayContent,
    masked: sgdMomentumWeightDecayContent,
  },
  nesterovContent: {
    unmasked: sgdNesterovContent,
    masked: sgdNesterovContent,
  },
  nesterovWeightDecayContent: {
    unmasked: sgdNesterovWeightDecayContent,
    masked: sgdNesterovWeightDecayContent,
  },
}
