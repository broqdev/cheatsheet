import type { AttentionExample } from '../../model'
import muonCode from './code/muon.py?raw'
import muonMoonshotLrCode from './code/muonMoonshotLr.py?raw'
import muonMoonshotLrWeightDecayCode from './code/muonMoonshotLrWeightDecay.py?raw'
import muonWeightDecayCode from './code/muonWeightDecay.py?raw'
import { defineAttentionContent, type AlgorithmLineSpec, type LatexBlockSpec } from '../../lib/contentCompiler'
import { math, strong, text } from '../../lib/segments'

const muonRequire = [
  text('Hidden 2D matrix parameters '),
  math(String.raw`\theta_{t-1}`),
  text(', gradients '),
  math(String.raw`g_t=\nabla_{\theta}L(\theta_{t-1})`),
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
  math(String.raw`\theta_{t-1}`),
  text(', gradients '),
  math(String.raw`g_t=\nabla_{\theta}L(\theta_{t-1})`),
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
    math(String.raw`\alpha_t=0.2\gamma\sqrt{\max(M,N)}`, 'moonshotLr'),
    text('.', 'moonshotLr'),
  ]
}

const muonRows: AlgorithmLineSpec[] = [
  {
    id: 'muon-forward-label',
    startsBlock: { id: 'muon-forward', role: 'forward' },
    parts: [strong('Optimization step.')],
    codeRefs: ['step-signature', 'no-grad'],
  },
  {
    id: 'muon-loop',
    number: 1,
    parts: [
      text('For each hidden matrix parameter '),
      math(String.raw`\theta_{t-1}`),
      text(' with '),
      math(String.raw`\theta_{t-1}\in\mathbb{R}^{M\times N}`),
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
      math(String.raw`\alpha_t=\gamma\sqrt{\max(1,M/N)}`),
      text('.'),
    ],
    codeRefs: ['adjust-lr'],
  },
  {
    id: 'muon-update',
    number: 7,
    parts: [
      text('Update parameters '),
      math(String.raw`\theta_t=\theta_{t-1}-\alpha_t O_t`),
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
    startsBlock: { id: 'muon-forward', role: 'forward' },
    parts: [strong('Optimization step with decoupled decay.')],
    codeRefs: ['step-signature', 'no-grad'],
  },
  {
    id: 'muon-loop',
    number: 1,
    parts: [
      text('For each hidden matrix parameter '),
      math(String.raw`\theta_{t-1}`),
      text(' with '),
      math(String.raw`\theta_{t-1}\in\mathbb{R}^{M\times N}`),
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
      math(
        String.raw`\theta_t^{\mathrm{decay}}=(1-\gamma\lambda)\theta_{t-1}`,
        'weightDecay'
      ),
      text(' before the Muon update.', 'weightDecay'),
    ],
    codeRefs: ['weight-decay'],
  },
  {
    id: 'muon-adjust-lr',
    number: 7,
    parts: [
      text('Adjust the learning rate for matrix shape '),
      math(String.raw`\alpha_t=\gamma\sqrt{\max(1,M/N)}`),
      text('.'),
    ],
    codeRefs: ['adjust-lr'],
  },
  {
    id: 'muon-update',
    number: 8,
    parts: [
      text('Update parameters '),
      math(String.raw`\theta_t=\theta_t^{\mathrm{decay}}-\alpha_t O_t`),
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
        math(String.raw`\alpha_t=0.2\gamma\sqrt{\max(M,N)}`, 'moonshotLr'),
        text('.', 'moonshotLr'),
      ],
    }
  })
}

const newtonSchulzNote: LatexBlockSpec = {
  id: 'muon-newton-schulz',
  title: 'Newton-Schulz5 polar approximation',
  require: [
    text('Matrix update '),
    math(String.raw`U_t`),
    text(', coefficients '),
    math(String.raw`a=3.4445,b=-4.7750,c=2.0315`),
    text(', and iteration count '),
    math(String.raw`K`),
    text('; for intuition, thin SVD '),
    math(String.raw`X_i=Q_iS_iR_i^\top`),
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
      id: 'muon-ns-iteration',
      number: 4,
      parts: [
        text('Newton-Schulz5 uses the quintic matrix iteration '),
        math(String.raw`A_i=X_iX_i^\top,\quad X_{i+1}=aX_i+(bA_i+cA_i^2)X_i`),
        text('.'),
      ],
      codeRefs: ['ns-coefficients', 'ns-iteration'],
    },
    {
      id: 'muon-ns-svd-decouples',
      number: 5,
      parts: [
        text('In SVD view, the update is a matrix polynomial in '),
        math(String.raw`A_i=X_iX_i^\top`),
        text('. Since '),
        math(String.raw`A_i=Q_iS_i^2Q_i^\top`),
        text(', the singular-vector directions stay aligned.'),
      ],
      codeRefs: ['ns-iteration'],
    },
    {
      id: 'muon-ns-scalar-recurrence',
      number: 6,
      parts: [
        text('Only the diagonal singular values move, each independently following '),
        math(String.raw`s_{i+1}=as_i+bs_i^3+cs_i^5`),
        text('.'),
      ],
      codeRefs: ['ns-coefficients', 'ns-iteration'],
    },
    {
      id: 'muon-ns-attracting-one',
      number: 7,
      parts: [
        text('The target value is '),
        math(String.raw`s=1`),
        text(': small singular values are pushed upward, large ones are pushed downward, and repeated polar iterations move nonzero values toward one.'),
      ],
      codeRefs: ['ns-iteration'],
    },
    {
      id: 'muon-ns-polar-result',
      number: 8,
      parts: [
        text('When '),
        math(String.raw`S_i`),
        text(' is close to identity on the nonzero subspace, '),
        math(String.raw`X_i=Q_iS_iR_i^\top`),
        text(' is close to '),
        math(String.raw`Q_iR_i^\top=\operatorname{polar}(X_i)`),
        text(', the direction Muon wants to step along.'),
      ],
      codeRefs: ['ns-restore'],
    },
    {
      id: 'muon-ns-five-step-approximation',
      number: 9,
      parts: [
        text('Muon stops after five Newton-Schulz5 steps. It wants a fast, nearly polar update with singular values near '),
        math(String.raw`1`),
        text(', not an exact polar decomposition.'),
      ],
      codeRefs: ['ns-iteration'],
    },
    {
      id: 'muon-ns-return',
      number: 10,
      parts: [
        text('Undo the transpose and return '),
        math(String.raw`X_K\approx\operatorname{polar}(U_t)`),
        text(', the approximate polar update.'),
      ],
      codeRefs: ['ns-restore'],
    },
  ],
}

const muonLearningRateNote: LatexBlockSpec = {
  id: 'muon-learning-rate-shape-scaling',
  title: 'Learning-rate shape scaling',
  require: [
    text('Matrix parameter '),
    math(String.raw`\theta_{t-1}\in\mathbb{R}^{M\times N}`),
    text(', polar update '),
    math(String.raw`O_t`),
    text(', and base learning rate '),
    math(String.raw`\gamma`),
    text('.'),
  ],
  rows: [
    {
      id: 'muon-lr-frobenius-derivation',
      number: 1,
      parts: [
        text('A full-rank polar update has '),
        math(String.raw`\min(M,N)`),
        text(' singular values equal to one, so its squared Frobenius norm is '),
        math(String.raw`\lVert O_t\rVert_F^2=\min(M,N)`),
        text('.'),
      ],
      codeRefs: ['orthogonalize'],
    },
    {
      id: 'muon-lr-rms-derivation',
      number: 2,
      parts: [
        text('RMS divides this energy over all '),
        math(String.raw`MN`),
        text(' entries: '),
        math(String.raw`\operatorname{RMS}(O_t)=\sqrt{\min(M,N)/(MN)}=1/\sqrt{\max(M,N)}`),
        text('.'),
      ],
      codeRefs: ['orthogonalize'],
    },
    {
      id: 'muon-lr-why-needed',
      number: 3,
      parts: [
        text('Consider applying Muon to two matrices with different dimensions: the one with larger '),
        math(String.raw`\max(M,N)`),
        text(' has smaller '),
        math(String.raw`\operatorname{RMS}(O_t)`),
        text(', so it gets a smaller per-entry step unless the LR factor cancels this shape effect.'),
      ],
      codeRefs: ['adjust-lr'],
    },
    {
      id: 'muon-lr-keller-original',
      number: 4,
      parts: [
        text('Keller original Muon uses '),
        math(String.raw`\alpha_t=\gamma\sqrt{\max(1,M/N)}`),
        text('. Multiplying by '),
        math(String.raw`\operatorname{RMS}(O_t)`),
        text(' gives roughly '),
        math(String.raw`\gamma\sqrt{\max(1,M/N)}/\sqrt{\max(M,N)}=\gamma/\sqrt{N}`),
        text(' when '),
        math(String.raw`M\ge N`),
        text(', so it is equivalent to Moonshot up to a global scale when matrices share the same second dimension.'),
      ],
      codeRefs: ['adjust-lr'],
    },
    {
      id: 'muon-lr-moonshot-rms',
      number: 5,
      parts: [
        text('Moonshot RMS matching uses '),
        math(String.raw`\alpha_t=0.2\gamma\sqrt{\max(M,N)}`),
        text('. The ratio cancels exactly: '),
        math(String.raw`0.2\sqrt{\max(M,N)}\operatorname{RMS}(O_t)=0.2`),
        text(', close to the AdamW update RMS range reported in the paper before applying the base '),
        math(String.raw`\gamma`),
        text('.'),
      ],
      codeRefs: ['adjust-lr'],
    },
    {
      id: 'muon-lr-difference',
      number: 6,
      parts: [
        text('So the original rule is orientation-sensitive through '),
        math(String.raw`N`),
        text(', while Moonshot is shape-symmetric and targets consistent RMS across matrix shapes.'),
      ],
      codeRefs: ['adjust-lr'],
    },
  ],
}

const muonNotes = [newtonSchulzNote, muonLearningRateNote]

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
    'Muon reshapes momentum into an approximately orthogonal matrix update, balancing movement across parameter directions.',
  algorithmTitle: 'Muon',
  content: {
    unmasked: muonContent,
    masked: muonContent,
  },
  variants: [
    {
      enabled: ['weightDecay'],
      content: { unmasked: muonWeightDecayContent, masked: muonWeightDecayContent },
    },
    {
      enabled: ['moonshotLr'],
      content: { unmasked: muonMoonshotLrContent, masked: muonMoonshotLrContent },
    },
    {
      enabled: ['moonshotLr', 'weightDecay'],
      content: {
        unmasked: muonMoonshotLrWeightDecayContent,
        masked: muonMoonshotLrWeightDecayContent,
      },
    },
  ],
}
