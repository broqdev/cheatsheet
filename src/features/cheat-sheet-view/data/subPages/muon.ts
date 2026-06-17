import type { AttentionExample } from '../../model'
import muonCode from './code/muon.py?raw'
import muonMoonshotLrCode from './code/muonMoonshotLr.py?raw'
import muonMoonshotLrWeightDecayCode from './code/muonMoonshotLrWeightDecay.py?raw'
import muonWeightDecayCode from './code/muonWeightDecay.py?raw'
import { defineAttentionContent, type AlgorithmLineSpec, type LatexBlockSpec } from '../../lib/codeRefs'
import { math, strong, text } from '../../lib/segments'

const muonRequire = [
  text('Hidden 2D matrix parameters '),
  math(String.raw`\theta_t`),
  text(', gradients '),
  math(String.raw`g_t=\nabla_{\theta}L_t(\theta_t)`),
  text(', learning rate '),
  math(String.raw`\gamma`),
  text(', momentum coefficient '),
  math(String.raw`\mu`),
  text(', Newton-Schulz steps '),
  math(String.raw`K`),
  text(', and optimizer state '),
  math(String.raw`\{B_{t-1}\}`),
  text('.'),
]

const muonWeightDecayRequire = [
  text('Hidden 2D matrix parameters '),
  math(String.raw`\theta_t`),
  text(', gradients '),
  math(String.raw`g_t=\nabla_{\theta}L_t(\theta_t)`),
  text(', learning rate '),
  math(String.raw`\gamma`),
  text(', momentum coefficient '),
  math(String.raw`\mu`),
  text(', Newton-Schulz steps '),
  math(String.raw`K`),
  text(', decoupled weight decay ', 'weightDecay'),
  math(String.raw`\lambda`, 'weightDecay'),
  text(', and optimizer state '),
  math(String.raw`\{B_{t-1}\}`),
  text('.'),
]

function withMoonshotLrRequire(require: typeof muonRequire) {
  return [
    ...require.slice(0, -1),
    text(', Moonshot LR adjustment ', 'moonshotLr'),
    math(String.raw`\alpha_t=0.2\gamma\max(A,B)`, 'moonshotLr'),
    text('.', 'moonshotLr'),
  ]
}

const muonRows: AlgorithmLineSpec[] = [
  {
    id: 'muon-forward-label',
    parts: [strong('Optimization step.')],
    codeRefs: ['step-signature', 'no-grad'],
  },
  {
    id: 'muon-loop',
    number: 1,
    parts: [
      text('For each hidden matrix parameter '),
      math(String.raw`\theta_t`),
      text(' with '),
      math(String.raw`\theta_t\in\mathbb{R}^{A\times B}`),
      text(' and gradient '),
      math(String.raw`g_t`),
      text('.'),
    ],
    codeRefs: ['parameter-loop'],
  },
  {
    id: 'muon-state-init',
    number: 2,
    parts: [
      text('Initialize momentum buffer '),
      math(String.raw`B_{t-1}`),
      text(' as zeros when missing.'),
    ],
    codeRefs: ['state-init'],
  },
  {
    id: 'muon-momentum',
    number: 3,
    parts: [
      text('Update momentum '),
      math(String.raw`B_t=\mu B_{t-1}+g_t`),
      text('.'),
    ],
    codeRefs: ['momentum'],
  },
  {
    id: 'muon-direction',
    number: 4,
    parts: [
      text('Use the momentum buffer as the matrix update '),
      math(String.raw`U_t=B_t`),
      text('.'),
    ],
    codeRefs: ['direction'],
  },
  {
    id: 'muon-orthogonalize',
    number: 5,
    parts: [
      text('Approximate the polar factor '),
      math(String.raw`O_t\approx\operatorname{polar}(U_t)=\operatorname{NS}_K^{(a,b,c)}(U_t;\epsilon)`),
      text(' with Newton-Schulz5 orthogonalization.'),
    ],
    codeRefs: ['orthogonalize'],
  },
  {
    id: 'muon-adjust-lr',
    number: 6,
    parts: [
      text('Adjust the learning rate for matrix shape '),
      math(String.raw`\alpha_t=\gamma\sqrt{\max(1,A/B)}`),
      text('.'),
    ],
    codeRefs: ['adjust-lr'],
  },
  {
    id: 'muon-update',
    number: 7,
    parts: [
      text('Update parameters '),
      math(String.raw`\theta_{t+1}=\theta_t-\alpha_t O_t`),
      text('.'),
    ],
    codeRefs: ['update'],
  },
  {
    id: 'muon-return',
    number: 8,
    parts: [
      text('Store '),
      math(String.raw`B_t`),
      text(' and return updated parameters and state.'),
    ],
    codeRefs: ['state-store', 'return-state'],
  },
]

const muonWeightDecayRows: AlgorithmLineSpec[] = [
  {
    id: 'muon-forward-label',
    parts: [strong('Optimization step with decoupled decay.')],
    codeRefs: ['step-signature', 'no-grad'],
  },
  {
    id: 'muon-loop',
    number: 1,
    parts: [
      text('For each hidden matrix parameter '),
      math(String.raw`\theta_t`),
      text(' with '),
      math(String.raw`\theta_t\in\mathbb{R}^{A\times B}`),
      text(' and gradient '),
      math(String.raw`g_t`),
      text('.'),
    ],
    codeRefs: ['parameter-loop'],
  },
  {
    id: 'muon-state-init',
    number: 2,
    parts: [
      text('Initialize momentum buffer '),
      math(String.raw`B_{t-1}`),
      text(' as zeros when missing.'),
    ],
    codeRefs: ['state-init'],
  },
  {
    id: 'muon-momentum',
    number: 3,
    parts: [
      text('Update momentum '),
      math(String.raw`B_t=\mu B_{t-1}+g_t`),
      text('.'),
    ],
    codeRefs: ['momentum'],
  },
  {
    id: 'muon-direction',
    number: 4,
    parts: [
      text('Use the momentum buffer as the matrix update '),
      math(String.raw`U_t=B_t`),
      text('.'),
    ],
    codeRefs: ['direction'],
  },
  {
    id: 'muon-orthogonalize',
    number: 5,
    parts: [
      text('Approximate the polar factor '),
      math(String.raw`O_t\approx\operatorname{polar}(U_t)=\operatorname{NS}_K^{(a,b,c)}(U_t;\epsilon)`),
      text(' with Newton-Schulz5 orthogonalization.'),
    ],
    codeRefs: ['orthogonalize'],
  },
  {
    id: 'muon-weight-decay',
    number: 6,
    parts: [
      text('Apply decoupled weight decay ', 'weightDecay'),
      math(String.raw`\theta_t\leftarrow(1-\gamma\lambda)\theta_t`, 'weightDecay'),
      text(' before the Muon update.', 'weightDecay'),
    ],
    codeRefs: ['weight-decay'],
  },
  {
    id: 'muon-adjust-lr',
    number: 7,
    parts: [
      text('Adjust the learning rate for matrix shape '),
      math(String.raw`\alpha_t=\gamma\sqrt{\max(1,A/B)}`),
      text('.'),
    ],
    codeRefs: ['adjust-lr'],
  },
  {
    id: 'muon-update',
    number: 8,
    parts: [
      text('Update parameters '),
      math(String.raw`\theta_{t+1}=\theta_t-\alpha_t O_t`),
      text('.'),
    ],
    codeRefs: ['update'],
  },
  {
    id: 'muon-return',
    number: 9,
    parts: [
      text('Store '),
      math(String.raw`B_t`),
      text(' and return updated parameters and state.'),
    ],
    codeRefs: ['state-store', 'return-state'],
  },
]

function withMoonshotLrRows(rows: AlgorithmLineSpec[]) {
  return rows.map((row) => {
    if (row.id !== 'muon-adjust-lr') {
      return row
    }

    return {
      ...row,
      parts: [
        text('Use Moonshot LR adjustment to match AdamW RMS ', 'moonshotLr'),
        math(String.raw`\alpha_t=0.2\gamma\max(A,B)`, 'moonshotLr'),
        text('.', 'moonshotLr'),
      ],
    }
  })
}

const newtonSchulzNote: LatexBlockSpec = {
  id: 'muon-newton-schulz',
  title: 'Newton-Schulz5 orthogonalization',
  require: [
    text('Matrix update '),
    math(String.raw`U_t`),
    text(', coefficients '),
    math(String.raw`a=3.4445,b=-4.7750,c=2.0315`),
    text(', and iteration count '),
    math(String.raw`K`),
    text('.'),
  ],
  rows: [
    {
      id: 'muon-ns-polar-target',
      number: 1,
      parts: [
        text('Target the polar factor '),
        math(String.raw`\operatorname{polar}(U_t)=QR^\top`),
        text(' where '),
        math(String.raw`U_t=Q\Sigma R^\top`),
        text(' is a thin SVD.'),
      ],
      codeRefs: ['ns-signature'],
    },
    {
      id: 'muon-ns-shape',
      number: 2,
      parts: [
        text('Require a matrix input and transpose tall matrices for stable products.'),
      ],
      codeRefs: ['ns-assert', 'ns-transpose'],
    },
    {
      id: 'muon-ns-normalize',
      number: 3,
      parts: [
        text('Normalize '),
        math(String.raw`X_0=U_t/\max(\lVert U_t\rVert_F,\epsilon)`),
        text(' after casting to bfloat16.'),
      ],
      codeRefs: ['ns-cast', 'ns-normalize'],
    },
    {
      id: 'muon-ns-iteration-explanation',
      number: 4,
      parts: [
        text('Newton-Schulz5 uses the quintic matrix iteration '),
        math(String.raw`A_i=X_iX_i^\top,\quad X_{i+1}=aX_i+(bA_i+cA_i^2)X_i`),
        text('.'),
      ],
      codeRefs: ['ns-coefficients', 'ns-iteration'],
    },
    {
      id: 'muon-ns-iterate',
      number: 5,
      parts: [
        text('Repeat '),
        math(String.raw`A_i=X_iX_i^\top`),
        text(', '),
        math(String.raw`B_i=bA_i+cA_i^2`),
        text(', and '),
        math(String.raw`X_{i+1}=aX_i+B_iX_i`),
        text('.'),
      ],
      codeRefs: ['ns-coefficients', 'ns-iteration'],
    },
    {
      id: 'muon-ns-return',
      number: 6,
      parts: [
        text('Undo the transpose and return '),
        math(String.raw`X_K\approx\operatorname{polar}(U_t)`),
        text(', the approximate polar update.'),
      ],
      codeRefs: ['ns-restore'],
    },
  ],
}

const newtonSchulzSingularValueNote: LatexBlockSpec = {
  id: 'muon-newton-schulz-singular-values',
  title: 'Newton-Schulz5 singular-value dynamics',
  requireLabel: 'Idea',
  require: [
    text('Thin SVD '),
    math(String.raw`X_i=Q_iS_iR_i^\top`),
    text(', nonzero singular values '),
    math(String.raw`s_i`),
    text(', and polar target '),
    math(String.raw`Q_iR_i^\top`),
    text('.'),
  ],
  rows: [
    {
      id: 'muon-ns-svd-decouples',
      number: 1,
      parts: [
        text('Because '),
        math(String.raw`X_iX_i^\top=Q_iS_i^2Q_i^\top`),
        text(', the Newton-Schulz5 update keeps the singular vectors aligned.'),
      ],
      codeRefs: ['ns-iteration'],
    },
    {
      id: 'muon-ns-scalar-recurrence',
      number: 2,
      parts: [
        text('Only the diagonal singular values change, each by the same scalar recurrence '),
        math(String.raw`s_{i+1}=as_i+bs_i^3+cs_i^5`),
        text('.'),
      ],
      codeRefs: ['ns-coefficients', 'ns-iteration'],
    },
    {
      id: 'muon-ns-attracting-one',
      number: 3,
      parts: [
        text('For an exact polar Newton-Schulz iteration, the scalar map has '),
        math(String.raw`s=1`),
        text(' as an attracting target: small singular values grow, oversized singular values shrink, and repeated steps move nonzero singular values toward one.'),
      ],
      codeRefs: ['ns-iteration'],
    },
    {
      id: 'muon-ns-polar-result',
      number: 4,
      parts: [
        text('When the singular values approach one, '),
        math(String.raw`X_i=Q_iS_iR_i^\top`),
        text(' approaches '),
        math(String.raw`Q_iR_i^\top=\operatorname{polar}(U_t)`),
        text(', the polar factor used as the Muon update direction.'),
      ],
      codeRefs: ['ns-restore'],
    },
    {
      id: 'muon-ns-five-step-approximation',
      number: 5,
      parts: [
        text('Muon uses five Newton-Schulz5 steps as a fast polar approximation, pushing singular values toward an order-one band near '),
        math(String.raw`1`),
        text(' rather than solving the exact polar factor.'),
      ],
      codeRefs: ['ns-iteration'],
    },
  ],
}

const muonNotes = [newtonSchulzNote, newtonSchulzSingularValueNote]

const muonContent = defineAttentionContent({
  rawCode: muonCode,
  require: muonRequire,
  rows: muonRows,
  notes: muonNotes,
})

const muonWeightDecayContent = defineAttentionContent({
  rawCode: muonWeightDecayCode,
  require: muonWeightDecayRequire,
  rows: muonWeightDecayRows,
  notes: muonNotes,
})

const muonMoonshotLrContent = defineAttentionContent({
  rawCode: muonMoonshotLrCode,
  require: withMoonshotLrRequire(muonRequire),
  rows: withMoonshotLrRows(muonRows),
  notes: muonNotes,
})

const muonMoonshotLrWeightDecayContent = defineAttentionContent({
  rawCode: muonMoonshotLrWeightDecayCode,
  require: withMoonshotLrRequire(muonWeightDecayRequire),
  rows: withMoonshotLrRows(muonWeightDecayRows),
  notes: muonNotes,
})

export const muonExample: AttentionExample = {
  id: 'muon',
  urlTag: 'optimizer/muon',
  label: 'Muon',
  description:
    'Muon applies momentum to hidden matrix gradients, approximates their polar factor with Newton-Schulz5, then steps in spectral-norm units.',
  algorithmTitle: 'Muon',
  content: {
    unmasked: muonContent,
    masked: muonContent,
  },
  weightDecayContent: {
    unmasked: muonWeightDecayContent,
    masked: muonWeightDecayContent,
  },
  moonshotLrContent: {
    unmasked: muonMoonshotLrContent,
    masked: muonMoonshotLrContent,
  },
  moonshotLrWeightDecayContent: {
    unmasked: muonMoonshotLrWeightDecayContent,
    masked: muonMoonshotLrWeightDecayContent,
  },
}
