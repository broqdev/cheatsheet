import type { AttentionExample } from '../../model'
import sgdCode from './code/sgd.py?raw'
import sgdMomentumCode from './code/sgdMomentum.py?raw'
import sgdMomentumWeightDecayCode from './code/sgdMomentumWeightDecay.py?raw'
import sgdWeightDecayCode from './code/sgdWeightDecay.py?raw'
import { defineAttentionContent, type AlgorithmLineSpec } from '../../lib/codeRefs'
import { math, strong, text } from '../../lib/segments'

const baseRequire = [
  text('Parameters '),
  math(String.raw`\theta_t`),
  text(', gradients '),
  math(String.raw`g_t=\nabla_{\theta}L_t(\theta_t)`),
  text(', and learning rate '),
  math(String.raw`\gamma`),
]

const weightDecayRequireParts = [
  text(', coupled weight decay ', 'weightDecay'),
  math(String.raw`\lambda`, 'weightDecay'),
]

const momentumRequireParts = [
  text(', momentum coefficient ', 'momentum'),
  math(String.raw`\mu`, 'momentum'),
  text(', dampening ', 'momentum'),
  math(String.raw`\tau`, 'momentum'),
  text(', and optimizer state ', 'momentum'),
  math(String.raw`\{b_{t-1}\}`, 'momentum'),
]

const sgdRequire = [...baseRequire, text('.')]
const sgdWeightDecayRequire = [...baseRequire, ...weightDecayRequireParts, text('.', 'weightDecay')]
const sgdMomentumRequire = [...baseRequire, ...momentumRequireParts, text('.', 'momentum')]
const sgdMomentumWeightDecayRequire = [
  ...baseRequire,
  ...weightDecayRequireParts,
  ...momentumRequireParts,
  text('.', 'momentum'),
]

const sgdRows: AlgorithmLineSpec[] = [
  {
    id: 'sgd-forward-label',
    parts: [strong('Optimization step.')],
    codeRefs: ['step-signature', 'no-grad'],
  },
  {
    id: 'sgd-loop',
    number: 1,
    parts: [
      text('For each parameter tensor '),
      math(String.raw`\theta_t`),
      text(' and gradient tensor '),
      math(String.raw`g_t`),
      text('.'),
    ],
    codeRefs: ['parameter-loop'],
  },
  {
    id: 'sgd-gradient-direction',
    number: 2,
    parts: [
      text('Use the gradient as the descent direction '),
      math(String.raw`d_t=g_t`),
      text('.'),
    ],
    codeRefs: ['gradient-direction'],
  },
  {
    id: 'sgd-update',
    number: 3,
    parts: [
      text('Apply the parameter update '),
      math(String.raw`\theta_{t+1}=\theta_t-\gamma d_t`),
      text('.'),
    ],
    codeRefs: ['update'],
  },
  {
    id: 'sgd-return',
    number: 4,
    parts: [
      text('Return the updated parameters '),
      math(String.raw`\theta_{t+1}`),
      text('.'),
    ],
    codeRefs: ['return-params'],
  },
]

const sgdWeightDecayRows: AlgorithmLineSpec[] = [
  {
    id: 'sgd-forward-label',
    parts: [strong('Optimization step.')],
    codeRefs: ['step-signature', 'no-grad'],
  },
  {
    id: 'sgd-loop',
    number: 1,
    parts: [
      text('For each parameter tensor '),
      math(String.raw`\theta_t`),
      text(' and gradient tensor '),
      math(String.raw`g_t`),
      text('.'),
    ],
    codeRefs: ['parameter-loop'],
  },
  {
    id: 'sgd-gradient-direction',
    number: 2,
    parts: [
      text('Use the gradient as the descent direction '),
      math(String.raw`d_t=g_t`),
      text('.'),
    ],
    codeRefs: ['gradient-direction'],
  },
  {
    id: 'sgd-weight-decay',
    number: 3,
    parts: [
      text('Add coupled L2 weight decay ', 'weightDecay'),
      math(String.raw`d_t \leftarrow d_t+\lambda\theta_t`, 'weightDecay'),
      text('.', 'weightDecay'),
    ],
    codeRefs: ['weight-decay'],
  },
  {
    id: 'sgd-update',
    number: 4,
    parts: [
      text('Apply the parameter update '),
      math(String.raw`\theta_{t+1}=\theta_t-\gamma d_t`),
      text('.'),
    ],
    codeRefs: ['update'],
  },
  {
    id: 'sgd-return',
    number: 5,
    parts: [
      text('Return the updated parameters '),
      math(String.raw`\theta_{t+1}`),
      text('.'),
    ],
    codeRefs: ['return-params'],
  },
]

const sgdMomentumRows: AlgorithmLineSpec[] = [
  {
    id: 'sgd-forward-label',
    parts: [strong('Optimization step.')],
    codeRefs: ['step-signature', 'no-grad'],
  },
  {
    id: 'sgd-loop',
    number: 1,
    parts: [
      text('For each parameter tensor '),
      math(String.raw`\theta_t`),
      text(' and gradient tensor '),
      math(String.raw`g_t`),
      text('.'),
    ],
    codeRefs: ['parameter-loop'],
  },
  {
    id: 'sgd-gradient-direction',
    number: 2,
    parts: [
      text('Use the gradient as the descent direction '),
      math(String.raw`d_t=g_t`),
      text('.'),
    ],
    codeRefs: ['gradient-direction'],
  },
  {
    id: 'sgd-momentum-state',
    number: 3,
    parts: [
      text('Read the momentum buffer ', 'momentum'),
      math(String.raw`b_{t-1}`, 'momentum'),
      text(', initializing it from ', 'momentum'),
      math(String.raw`d_t`, 'momentum'),
      text(' on the first step.', 'momentum'),
    ],
    codeRefs: ['state-init'],
  },
  {
    id: 'sgd-momentum',
    number: 4,
    parts: [
      text('Use momentum direction ', 'momentum'),
      math(String.raw`b_t=\mu b_{t-1}+(1-\tau)d_t`, 'momentum'),
      text(' after the first step.', 'momentum'),
    ],
    codeRefs: ['momentum'],
  },
  {
    id: 'sgd-update',
    number: 5,
    parts: [
      text('Apply the parameter update '),
      math(String.raw`\theta_{t+1}=\theta_t-\gamma b_t`, 'momentum'),
      text('.'),
    ],
    codeRefs: ['update'],
  },
  {
    id: 'sgd-return',
    number: 6,
    parts: [
      text('Store ', 'momentum'),
      math(String.raw`b_t`, 'momentum'),
      text(' and return updated parameters and state.', 'momentum'),
    ],
    codeRefs: ['state-store', 'return-state'],
  },
]

const sgdMomentumWeightDecayRows: AlgorithmLineSpec[] = [
  {
    id: 'sgd-forward-label',
    parts: [strong('Optimization step.')],
    codeRefs: ['step-signature', 'no-grad'],
  },
  {
    id: 'sgd-loop',
    number: 1,
    parts: [
      text('For each parameter tensor '),
      math(String.raw`\theta_t`),
      text(' and gradient tensor '),
      math(String.raw`g_t`),
      text('.'),
    ],
    codeRefs: ['parameter-loop'],
  },
  {
    id: 'sgd-gradient-direction',
    number: 2,
    parts: [
      text('Use the gradient as the descent direction '),
      math(String.raw`d_t=g_t`),
      text('.'),
    ],
    codeRefs: ['gradient-direction'],
  },
  {
    id: 'sgd-weight-decay',
    number: 3,
    parts: [
      text('Add coupled L2 weight decay ', 'weightDecay'),
      math(String.raw`d_t \leftarrow d_t+\lambda\theta_t`, 'weightDecay'),
      text(' before momentum.', 'weightDecay'),
    ],
    codeRefs: ['weight-decay'],
  },
  {
    id: 'sgd-momentum-state',
    number: 4,
    parts: [
      text('Read the momentum buffer ', 'momentum'),
      math(String.raw`b_{t-1}`, 'momentum'),
      text(', initializing it from ', 'momentum'),
      math(String.raw`d_t`, 'momentum'),
      text(' on the first step.', 'momentum'),
    ],
    codeRefs: ['state-init'],
  },
  {
    id: 'sgd-momentum',
    number: 5,
    parts: [
      text('Use momentum direction ', 'momentum'),
      math(String.raw`b_t=\mu b_{t-1}+(1-\tau)d_t`, 'momentum'),
      text(' after the first step.', 'momentum'),
    ],
    codeRefs: ['momentum'],
  },
  {
    id: 'sgd-update',
    number: 6,
    parts: [
      text('Apply the parameter update '),
      math(String.raw`\theta_{t+1}=\theta_t-\gamma b_t`, 'momentum'),
      text('.'),
    ],
    codeRefs: ['update'],
  },
  {
    id: 'sgd-return',
    number: 7,
    parts: [
      text('Store ', 'momentum'),
      math(String.raw`b_t`, 'momentum'),
      text(' and return updated parameters and state.', 'momentum'),
    ],
    codeRefs: ['state-store', 'return-state'],
  },
]

const sgdContent = defineAttentionContent({
  rawCode: sgdCode,
  require: sgdRequire,
  rows: sgdRows,
})

const sgdWeightDecayContent = defineAttentionContent({
  rawCode: sgdWeightDecayCode,
  require: sgdWeightDecayRequire,
  rows: sgdWeightDecayRows,
})

const sgdMomentumContent = defineAttentionContent({
  rawCode: sgdMomentumCode,
  require: sgdMomentumRequire,
  rows: sgdMomentumRows,
})

const sgdMomentumWeightDecayContent = defineAttentionContent({
  rawCode: sgdMomentumWeightDecayCode,
  require: sgdMomentumWeightDecayRequire,
  rows: sgdMomentumWeightDecayRows,
})

export const sgdExample: AttentionExample = {
  id: 'sgd',
  urlTag: 'optimizer/sgd',
  label: 'SGD',
  description:
    'Stochastic gradient descent applies a learning-rate-scaled gradient step, with optional coupled weight decay and momentum.',
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
}
