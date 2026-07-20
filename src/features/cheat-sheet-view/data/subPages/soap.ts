import type { AttentionExample } from '../../model'
import { defineAttentionContent, type AlgorithmLineSpec } from '../../lib/contentCompiler'
import { math, strong, text } from '../../lib/segments'
import soapCode from './code/soap.py?raw'
import soapWeightDecayCode from './code/soapWeightDecay.py?raw'

type SoapVariant = {
  weightDecay: boolean
}

function soapRequire({ weightDecay }: SoapVariant) {
  return [
    text('Matrix parameters '),
    math(String.raw`\theta_{t-1}\in\mathbb{R}^{M\times N}`),
    text(', gradients '),
    math(String.raw`G_t=\nabla_{\theta}L(\theta_{t-1})`),
    text(', learning rate '),
    math(String.raw`\gamma`),
    text(', Adam coefficients '),
    math(String.raw`\beta_1,\beta_2`),
    text(', numerical constant '),
    math(String.raw`\epsilon`),
    text(', preconditioning frequency '),
    math(String.raw`f`),
    ...(weightDecay
      ? [
          text(', decoupled weight decay ', 'weightDecay'),
          math(String.raw`\lambda`, 'weightDecay'),
        ]
      : []),
    text(', and optimizer state '),
    math(String.raw`\{t-1,M_{t-1},V_{t-1},L_{t-1},R_{t-1},Q_{L,t-1},Q_{R,t-1}\}`),
    text('.'),
  ]
}

function soapRows({ weightDecay }: SoapVariant): AlgorithmLineSpec[] {
  let number = 1
  const row = (line: Omit<AlgorithmLineSpec, 'number'>): AlgorithmLineSpec => ({
    ...line,
    number: number++,
  })

  const rows: AlgorithmLineSpec[] = [
    {
      id: 'soap-forward-label',
      startsBlock: { id: 'soap-forward', role: 'forward' },
      parts: [
        strong(
          weightDecay
            ? 'Adam in Shampoo’s eigenbasis with decoupled decay.'
            : 'Adam in Shampoo’s eigenbasis.'
        ),
      ],
      codeRefs: ['step-signature', 'no-grad'],
    },
    row({
      id: 'soap-loop',
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
      id: 'soap-state-init',
      parts: [
        text('Advance '),
        math(String.raw`t-1`),
        text(' to '),
        math(String.raw`t`),
        text(' and initialize moments, covariance matrices, and orthonormal bases when missing.'),
      ],
      codeRefs: ['eigenvectors-signature', 'eigendecomposition', 'state-init'],
    }),
    row({
      id: 'soap-project-gradient',
      parts: [
        text('Project the gradient into the current Shampoo eigenbasis '),
        math(String.raw`G'_t=Q_{L,t-1}^\top G_tQ_{R,t-1}`),
        text('.'),
      ],
      codeRefs: ['project-gradient'],
    }),
    row({
      id: 'soap-first-moment',
      parts: [
        text('Update the first moment in the original coordinates '),
        math(String.raw`M_t=\beta_1M_{t-1}+(1-\beta_1)G_t`),
        text(', then project '),
        math(String.raw`M'_t=Q_{L,t-1}^\top M_tQ_{R,t-1}`),
        text('.'),
      ],
      codeRefs: ['first-moment'],
    }),
    row({
      id: 'soap-second-moment',
      parts: [
        text('Update the elementwise second moment in the rotated coordinates '),
        math(String.raw`V_t=\beta_2V_{t-1}+(1-\beta_2)(G'_t\odot G'_t)`),
        text('.'),
      ],
      codeRefs: ['second-moment'],
    }),
    row({
      id: 'soap-normalize',
      parts: [
        text('Bias-correct the moments and apply Adam’s diagonal scaling '),
        math(
          String.raw`U'_t=\frac{M'_t/(1-\operatorname{pow}(\beta_1,t))}{\sqrt{V_t/(1-\operatorname{pow}(\beta_2,t))}+\epsilon}`
        ),
        text('.'),
      ],
      codeRefs: ['bias-correction', 'normalize'],
    }),
    row({
      id: 'soap-project-back',
      parts: [
        text('Rotate the normalized update back '),
        math(String.raw`U_t=Q_{L,t-1}U'_tQ_{R,t-1}^\top`),
        text('.'),
      ],
      codeRefs: ['project-back'],
    }),
  ]

  if (weightDecay) {
    rows.push(
      row({
        id: 'soap-weight-decay',
        parts: [
          text('Apply decoupled weight decay ', 'weightDecay'),
          math(
            String.raw`\theta_t^{\mathrm{decay}}=(1-\gamma\lambda)\theta_{t-1}`,
            'weightDecay'
          ),
          text(' outside SOAP’s moments and preconditioners.', 'weightDecay'),
        ],
        codeRefs: ['weight-decay'],
      })
    )
  }

  rows.push(
    row({
      id: 'soap-update',
      parts: [
        text('Update parameters '),
        math(
          weightDecay
            ? String.raw`\theta_t=\theta_t^{\mathrm{decay}}-\gamma U_t`
            : String.raw`\theta_t=\theta_{t-1}-\gamma U_t`
        ),
        text('.'),
      ],
      codeRefs: ['update'],
    }),
    row({
      id: 'soap-preconditioner-update',
      parts: [
        text('Update Shampoo’s row and column covariance estimates '),
        math(String.raw`L_t=\beta_2L_{t-1}+(1-\beta_2)G_tG_t^\top`),
        text(' and '),
        math(String.raw`R_t=\beta_2R_{t-1}+(1-\beta_2)G_t^\top G_t`),
        text('.'),
      ],
      codeRefs: ['preconditioner-update'],
    }),
    row({
      id: 'soap-basis-refresh',
      parts: [
        text('Every '),
        math(String.raw`f`),
        text(' steps, refresh each basis with one power iteration and QR: '),
        math(String.raw`Q_{L,t}=\operatorname{qr}(L_tQ_{L,t-1})`),
        text(' and '),
        math(String.raw`Q_{R,t}=\operatorname{qr}(R_tQ_{R,t-1})`),
        text('; otherwise retain the current bases.'),
      ],
      codeRefs: ['basis-refresh-signature', 'power-iteration', 'basis-refresh'],
    }),
    row({
      id: 'soap-return',
      parts: [
        text('Store the moments, covariance matrices, and bases, then return updated parameters and state.'),
      ],
      codeRefs: ['state-store', 'return-state'],
    })
  )

  return rows
}

const soapContent = defineAttentionContent({
  rawCode: soapCode,
  require: soapRequire({ weightDecay: false }),
  rows: soapRows({ weightDecay: false }),
})

const soapWeightDecayContent = defineAttentionContent({
  rawCode: soapWeightDecayCode,
  require: soapRequire({ weightDecay: true }),
  rows: soapRows({ weightDecay: true }),
})

export const soapExample: AttentionExample = {
  id: 'soap',
  urlTag: 'optimizer/soap',
  label: 'SOAP',
  description:
    'SOAP finds a geometry-aware coordinate system for each tensor, then applies adaptive updates in that transformed space.',
  algorithmTitle: 'SOAP',
  content: {
    unmasked: soapContent,
    masked: soapContent,
  },
  variants: [
    {
      enabled: ['weightDecay'],
      content: {
        unmasked: soapWeightDecayContent,
        masked: soapWeightDecayContent,
      },
    },
  ],
}
