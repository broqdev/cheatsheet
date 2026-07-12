import type { AttentionExample } from '../../model'
import adagradCode from './code/adagrad.py?raw'
import adagradWeightDecayCode from './code/adagradWeightDecay.py?raw'
import { defineAttentionContent, type AlgorithmLineSpec } from '../../lib/contentCompiler'
import { math, strong, text } from '../../lib/segments'

type AdagradVariant = {
  weightDecay: boolean
}

function adagradRequire({ weightDecay }: AdagradVariant) {
  return [
    text('Parameters '),
    math(String.raw`\theta_{t-1}`),
    text(', gradients '),
    math(String.raw`g_t=\nabla_{\theta}L_t(\theta_{t-1})`),
    text(', learning rate '),
    math(String.raw`\gamma`),
    text(', learning-rate decay '),
    math(String.raw`\eta`),
    text(', initial accumulator value '),
    math(String.raw`\tau`),
    text(', numerical constant '),
    math(String.raw`\epsilon`),
    ...(weightDecay
      ? [
          text(', coupled weight decay ', 'weightDecay'),
          math(String.raw`\lambda`, 'weightDecay'),
        ]
      : []),
    text(', and optimizer state '),
    math(String.raw`\{t-1,s_{t-1}\}`),
    text('.'),
  ]
}

function adagradRows({ weightDecay }: AdagradVariant): AlgorithmLineSpec[] {
  let number = 1
  const row = (line: Omit<AlgorithmLineSpec, 'number'>): AlgorithmLineSpec => ({
    ...line,
    number: number++,
  })

  const rows: AlgorithmLineSpec[] = [
    {
      id: 'adagrad-forward-label',
      parts: [strong('Optimization step.')],
      codeRefs: ['step-signature', 'no-grad'],
    },
    row({
      id: 'adagrad-loop',
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
      id: 'adagrad-state-init',
      parts: [
        text('Advance the timestep from '),
        math(String.raw`t-1`),
        text(' to '),
        math(String.raw`t`),
        text(' and initialize accumulated squared-gradient state '),
        math(String.raw`s_0=\tau`),
        text(' when missing.'),
      ],
      codeRefs: ['state-init'],
    }),
    row({
      id: 'adagrad-gradient-direction',
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
        id: 'adagrad-weight-decay',
        parts: [
          text('Add coupled L2 weight decay ', 'weightDecay'),
          math(String.raw`d_t \leftarrow d_t+\lambda\theta_{t-1}`, 'weightDecay'),
          text(' before accumulating squared gradients.', 'weightDecay'),
        ],
        codeRefs: ['weight-decay'],
      })
    )
  }

  rows.push(
    row({
      id: 'adagrad-learning-rate-decay',
      parts: [
        text('Compute the decayed step size '),
        math(String.raw`\tilde{\gamma}_t=\gamma/(1+(t-1)\eta)`),
        text('.'),
      ],
      codeRefs: ['learning-rate-decay'],
    }),
    row({
      id: 'adagrad-accumulator',
      parts: [
        text('Accumulate elementwise squared gradients '),
        math(String.raw`s_t=s_{t-1}+d_t\odot d_t`),
        text('.'),
      ],
      codeRefs: ['accumulator'],
    }),
    row({
      id: 'adagrad-denominator',
      parts: [
        text('Compute the adaptive denominator '),
        math(String.raw`r_t=\sqrt{s_t}+\epsilon`),
        text('.'),
      ],
      codeRefs: ['denominator'],
    }),
    row({
      id: 'adagrad-update',
      parts: [
        text('Apply the coordinate-wise update '),
        math(String.raw`\theta_t=\theta_{t-1}-\tilde{\gamma}_t d_t/r_t`),
        text('.'),
      ],
      codeRefs: ['update'],
    }),
    row({
      id: 'adagrad-return',
      parts: [
        text('Store '),
        math(String.raw`\{t,s_t\}`),
        text(' and return updated parameters and state.'),
      ],
      codeRefs: ['state-store', 'return-state'],
    })
  )

  return rows
}

const adagradContent = defineAttentionContent({
  rawCode: adagradCode,
  require: adagradRequire({ weightDecay: false }),
  rows: adagradRows({ weightDecay: false }),
})

const adagradWeightDecayContent = defineAttentionContent({
  rawCode: adagradWeightDecayCode,
  require: adagradRequire({ weightDecay: true }),
  rows: adagradRows({ weightDecay: true }),
})

export const adagradExample: AttentionExample = {
  id: 'adagrad',
  urlTag: 'optimizer/adagrad',
  label: 'AdaGrad',
  description: "AdaGrad adapts each parameter's step using accumulated squared gradients.",
  algorithmTitle: 'AdaGrad',
  content: {
    unmasked: adagradContent,
    masked: adagradContent,
  },
  variants: [
    {
      enabled: ['weightDecay'],
      content: {
        unmasked: adagradWeightDecayContent,
        masked: adagradWeightDecayContent,
      },
    },
  ],
}
