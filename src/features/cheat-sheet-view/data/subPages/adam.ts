import type { AttentionExample } from '../../model'
import adamCode from './code/adam.py?raw'
import adamWeightDecayCode from './code/adamWeightDecay.py?raw'
import { defineAttentionContent, type AlgorithmLineSpec } from '../../lib/contentCompiler'
import { math, strong, text } from '../../lib/segments'

const adamRequire = [
  text('Parameters '),
  math(String.raw`\theta_{t-1}`),
  text(', gradients '),
  math(String.raw`g_t=\nabla_{\theta}L_t(\theta_{t-1})`),
  text(', learning rate '),
  math(String.raw`\gamma`),
  text(', moment coefficients '),
  math(String.raw`\beta_1,\beta_2`),
  text(', numerical constant '),
  math(String.raw`\epsilon`),
  text(', and optimizer state '),
  math(String.raw`\{t-1,m_{t-1},v_{t-1}\}`),
  text('.'),
]

const adamWeightDecayRequire = [
  text('Parameters '),
  math(String.raw`\theta_{t-1}`),
  text(', gradients '),
  math(String.raw`g_t=\nabla_{\theta}L_t(\theta_{t-1})`),
  text(', learning rate '),
  math(String.raw`\gamma`),
  text(', moment coefficients '),
  math(String.raw`\beta_1,\beta_2`),
  text(', numerical constant '),
  math(String.raw`\epsilon`),
  text(', coupled weight decay ', 'weightDecay'),
  math(String.raw`\lambda`, 'weightDecay'),
  text(', and optimizer state '),
  math(String.raw`\{t-1,m_{t-1},v_{t-1}\}`),
  text('.'),
]

const adamRows: AlgorithmLineSpec[] = [
  {
    id: 'adam-forward-label',
    startsBlock: { id: 'adam-forward', role: 'forward' },
    parts: [strong('Optimization step.')],
    codeRefs: ['step-signature', 'no-grad'],
  },
  {
    id: 'adam-loop',
    number: 1,
    parts: [
      text('For each parameter tensor '),
      math(String.raw`\theta_{t-1}`),
      text(' and gradient tensor '),
      math(String.raw`g_t`),
      text('.'),
    ],
    codeRefs: ['parameter-loop'],
  },
  {
    id: 'adam-state-init',
    number: 2,
    parts: [
      text('Advance the timestep from '),
      math(String.raw`t-1`),
      text(' to '),
      math(String.raw`t`),
      text(' and initialize moment state '),
      math(String.raw`m_{t-1},v_{t-1}`),
      text(' as zeros when missing.'),
    ],
    codeRefs: ['state-init'],
  },
  {
    id: 'adam-gradient-direction',
    number: 3,
    parts: [
      text('Use the gradient direction '),
      math(String.raw`d_t=g_t`),
      text('.'),
    ],
    codeRefs: ['gradient-direction'],
  },
  {
    id: 'adam-first-moment',
    number: 4,
    parts: [
      text('Update first moment '),
      math(String.raw`m_t=\beta_1m_{t-1}+(1-\beta_1)d_t`),
      text('.'),
    ],
    codeRefs: ['first-moment'],
  },
  {
    id: 'adam-second-moment',
    number: 5,
    parts: [
      text('Update second moment '),
      math(String.raw`v_t=\beta_2v_{t-1}+(1-\beta_2)d_t\odot d_t`),
      text('.'),
    ],
    codeRefs: ['second-moment'],
  },
  {
    id: 'adam-bias-correction',
    number: 6,
    parts: [
      text('Bias-correct moments '),
      math(String.raw`\hat{m}_t=m_t/(1-\operatorname{pow}(\beta_1,t))`),
      text(' and '),
      math(String.raw`\hat{v}_t=v_t/(1-\operatorname{pow}(\beta_2,t))`),
      text('.'),
    ],
    codeRefs: ['bias-correction'],
  },
  {
    id: 'adam-update',
    number: 7,
    parts: [
      text('Apply the parameter update '),
      math(String.raw`\theta_t=\theta_{t-1}-\gamma\hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)`),
      text(' with no separate parameter shrink.'),
    ],
    codeRefs: ['update'],
  },
  {
    id: 'adam-return',
    number: 8,
    parts: [
      text('Store '),
      math(String.raw`\{t,m_t,v_t\}`),
      text(' and return updated parameters and state.'),
    ],
    codeRefs: ['state-store', 'return-state'],
  },
]

const adamWeightDecayRows: AlgorithmLineSpec[] = [
  {
    id: 'adam-forward-label',
    startsBlock: { id: 'adam-forward', role: 'forward' },
    parts: [strong('Optimization step with coupled L2.')],
    codeRefs: ['step-signature', 'no-grad'],
  },
  {
    id: 'adam-loop',
    number: 1,
    parts: [
      text('For each parameter tensor '),
      math(String.raw`\theta_{t-1}`),
      text(' and gradient tensor '),
      math(String.raw`g_t`),
      text('.'),
    ],
    codeRefs: ['parameter-loop'],
  },
  {
    id: 'adam-state-init',
    number: 2,
    parts: [
      text('Advance the timestep from '),
      math(String.raw`t-1`),
      text(' to '),
      math(String.raw`t`),
      text(' and initialize moment state '),
      math(String.raw`m_{t-1},v_{t-1}`),
      text(' as zeros when missing.'),
    ],
    codeRefs: ['state-init'],
  },
  {
    id: 'adam-gradient-direction',
    number: 3,
    parts: [
      text('Use the gradient direction '),
      math(String.raw`d_t=g_t`),
      text('.'),
    ],
    codeRefs: ['gradient-direction'],
  },
  {
    id: 'adam-weight-decay',
    number: 4,
    parts: [
      text('Add coupled L2 weight decay ', 'weightDecay'),
      math(String.raw`d_t \leftarrow d_t+\lambda\theta_{t-1}`, 'weightDecay'),
      text(' before moment updates, so both moments include the penalty.', 'weightDecay'),
    ],
    codeRefs: ['weight-decay'],
  },
  {
    id: 'adam-first-moment',
    number: 5,
    parts: [
      text('Update first moment '),
      math(String.raw`m_t=\beta_1m_{t-1}+(1-\beta_1)d_t`),
      text('.'),
    ],
    codeRefs: ['first-moment'],
  },
  {
    id: 'adam-second-moment',
    number: 6,
    parts: [
      text('Update second moment '),
      math(String.raw`v_t=\beta_2v_{t-1}+(1-\beta_2)d_t\odot d_t`),
      text('.'),
    ],
    codeRefs: ['second-moment'],
  },
  {
    id: 'adam-bias-correction',
    number: 7,
    parts: [
      text('Bias-correct moments '),
      math(String.raw`\hat{m}_t=m_t/(1-\operatorname{pow}(\beta_1,t))`),
      text(' and '),
      math(String.raw`\hat{v}_t=v_t/(1-\operatorname{pow}(\beta_2,t))`),
      text('.'),
    ],
    codeRefs: ['bias-correction'],
  },
  {
    id: 'adam-update',
    number: 8,
    parts: [
      text('Apply the parameter update '),
      math(String.raw`\theta_t=\theta_{t-1}-\gamma\hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)`),
      text(' using moments of the regularized gradient.'),
    ],
    codeRefs: ['update'],
  },
  {
    id: 'adam-return',
    number: 9,
    parts: [
      text('Store '),
      math(String.raw`\{t,m_t,v_t\}`),
      text(' and return updated parameters and state.'),
    ],
    codeRefs: ['state-store', 'return-state'],
  },
]

const adamContent = defineAttentionContent({
  rawCode: adamCode,
  require: adamRequire,
  rows: adamRows,
})

const adamWeightDecayContent = defineAttentionContent({
  rawCode: adamWeightDecayCode,
  require: adamWeightDecayRequire,
  rows: adamWeightDecayRows,
})

export const adamExample: AttentionExample = {
  id: 'adam',
  urlTag: 'optimizer/adam',
  label: 'Adam',
  description:
    'Adam builds adaptive moments from the gradient; coupled L2, when enabled, is folded into that gradient before the moments.',
  algorithmTitle: 'Adam',
  content: {
    unmasked: adamContent,
    masked: adamContent,
  },
  variants: [
    {
      enabled: ['weightDecay'],
      content: { unmasked: adamWeightDecayContent, masked: adamWeightDecayContent },
    },
  ],
}
