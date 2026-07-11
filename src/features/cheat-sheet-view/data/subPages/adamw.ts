import type { AttentionExample } from '../../model'
import adamWCode from './code/adamw.py?raw'
import adamWWeightDecayCode from './code/adamwWeightDecay.py?raw'
import { defineAttentionContent, type AlgorithmLineSpec } from '../../lib/contentCompiler'
import { math, strong, text } from '../../lib/segments'

const adamWRequire = [
  text('Parameters '),
  math(String.raw`\theta_t`),
  text(', gradients '),
  math(String.raw`g_t=\nabla_{\theta}L_t(\theta_t)`),
  text(', learning rate '),
  math(String.raw`\gamma`),
  text(', moment coefficients '),
  math(String.raw`\beta_1,\beta_2`),
  text(', numerical constant '),
  math(String.raw`\epsilon`),
  text(', and optimizer state '),
  math(String.raw`\{t,m_{t-1},v_{t-1}\}`),
  text('.'),
]

const adamWWeightDecayRequire = [
  text('Parameters '),
  math(String.raw`\theta_t`),
  text(', gradients '),
  math(String.raw`g_t=\nabla_{\theta}L_t(\theta_t)`),
  text(', learning rate '),
  math(String.raw`\gamma`),
  text(', moment coefficients '),
  math(String.raw`\beta_1,\beta_2`),
  text(', numerical constant '),
  math(String.raw`\epsilon`),
  text(', decoupled weight decay ', 'weightDecay'),
  math(String.raw`\lambda`, 'weightDecay'),
  text(', and optimizer state '),
  math(String.raw`\{t,m_{t-1},v_{t-1}\}`),
  text('.'),
]

const adamWRows: AlgorithmLineSpec[] = [
  {
    id: 'adamw-forward-label',
    startsBlock: { id: 'adamw-forward', role: 'forward' },
    parts: [strong('Optimization step with no decay.')],
    codeRefs: ['step-signature', 'no-grad'],
  },
  {
    id: 'adamw-loop',
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
    id: 'adamw-state-init',
    number: 2,
    parts: [
      text('Increment timestep '),
      math(String.raw`t`),
      text(' and initialize moment state '),
      math(String.raw`m_{t-1},v_{t-1}`),
      text(' as zeros when missing.'),
    ],
    codeRefs: ['state-init'],
  },
  {
    id: 'adamw-gradient-direction',
    number: 3,
    parts: [
      text('Use the gradient direction '),
      math(String.raw`d_t=g_t`),
      text('.'),
    ],
    codeRefs: ['gradient-direction'],
  },
  {
    id: 'adamw-first-moment',
    number: 4,
    parts: [
      text('Update first moment '),
      math(String.raw`m_t=\beta_1m_{t-1}+(1-\beta_1)d_t`),
      text('.'),
    ],
    codeRefs: ['first-moment'],
  },
  {
    id: 'adamw-second-moment',
    number: 5,
    parts: [
      text('Update second moment '),
      math(String.raw`v_t=\beta_2v_{t-1}+(1-\beta_2)d_t\odot d_t`),
      text('.'),
    ],
    codeRefs: ['second-moment'],
  },
  {
    id: 'adamw-bias-correction',
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
    id: 'adamw-update',
    number: 7,
    parts: [
      text('Apply the adaptive update '),
      math(String.raw`\theta_{t+1}=\theta_t-\gamma\hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)`),
      text('; with '),
      math(String.raw`\lambda=0`),
      text(', this matches Adam.'),
    ],
    codeRefs: ['update'],
  },
  {
    id: 'adamw-return',
    number: 8,
    parts: [
      text('Store '),
      math(String.raw`\{t,m_t,v_t\}`),
      text(' and return updated parameters and state.'),
    ],
    codeRefs: ['state-store', 'return-state'],
  },
]

const adamWWeightDecayRows: AlgorithmLineSpec[] = [
  {
    id: 'adamw-forward-label',
    startsBlock: { id: 'adamw-forward', role: 'forward' },
    parts: [strong('Optimization step with decoupled decay.')],
    codeRefs: ['step-signature', 'no-grad'],
  },
  {
    id: 'adamw-loop',
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
    id: 'adamw-state-init',
    number: 2,
    parts: [
      text('Increment timestep '),
      math(String.raw`t`),
      text(' and initialize moment state '),
      math(String.raw`m_{t-1},v_{t-1}`),
      text(' as zeros when missing.'),
    ],
    codeRefs: ['state-init'],
  },
  {
    id: 'adamw-gradient-direction',
    number: 3,
    parts: [
      text('Use the gradient direction '),
      math(String.raw`d_t=g_t`),
      text('.'),
    ],
    codeRefs: ['gradient-direction'],
  },
  {
    id: 'adamw-weight-decay',
    number: 4,
    parts: [
      text('Apply decoupled weight decay ', 'weightDecay'),
      math(String.raw`\theta_t \leftarrow \theta_t-\gamma\lambda\theta_t`, 'weightDecay'),
      text(' outside the gradient path; the moments still use ', 'weightDecay'),
      math(String.raw`d_t=g_t`, 'weightDecay'),
      text('.', 'weightDecay'),
    ],
    codeRefs: ['weight-decay'],
  },
  {
    id: 'adamw-first-moment',
    number: 5,
    parts: [
      text('Update first moment '),
      math(String.raw`m_t=\beta_1m_{t-1}+(1-\beta_1)d_t`),
      text('.'),
    ],
    codeRefs: ['first-moment'],
  },
  {
    id: 'adamw-second-moment',
    number: 6,
    parts: [
      text('Update second moment '),
      math(String.raw`v_t=\beta_2v_{t-1}+(1-\beta_2)d_t\odot d_t`),
      text('.'),
    ],
    codeRefs: ['second-moment'],
  },
  {
    id: 'adamw-bias-correction',
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
    id: 'adamw-update',
    number: 8,
    parts: [
      text('Apply the adaptive update '),
      math(String.raw`\theta_{t+1}=\theta_t-\gamma\hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)`),
      text(' after the decoupled shrink.'),
    ],
    codeRefs: ['update'],
  },
  {
    id: 'adamw-return',
    number: 9,
    parts: [
      text('Store '),
      math(String.raw`\{t,m_t,v_t\}`),
      text(' and return updated parameters and state.'),
    ],
    codeRefs: ['state-store', 'return-state'],
  },
]

const adamWContent = defineAttentionContent({
  rawCode: adamWCode,
  require: adamWRequire,
  rows: adamWRows,
})

const adamWWeightDecayContent = defineAttentionContent({
  rawCode: adamWWeightDecayCode,
  require: adamWWeightDecayRequire,
  rows: adamWWeightDecayRows,
})

export const adamWExample: AttentionExample = {
  id: 'adamw',
  urlTag: 'optimizer/adamw',
  label: 'AdamW',
  description:
    'AdamW keeps raw-gradient Adam moments while applying weight decay as a separate parameter shrink.',
  algorithmTitle: 'AdamW',
  content: {
    unmasked: adamWContent,
    masked: adamWContent,
  },
  variants: [
    {
      enabled: ['weightDecay'],
      content: { unmasked: adamWWeightDecayContent, masked: adamWWeightDecayContent },
    },
  ],
}
