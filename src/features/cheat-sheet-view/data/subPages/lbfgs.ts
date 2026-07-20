import type { AttentionExample } from '../../model'
import {
  defineAttentionContent,
  type AlgorithmLineSpec,
  type LatexBlockSpec,
} from '../../lib/contentCompiler'
import { math, strong, text } from '../../lib/segments'
import lbfgsCode from './code/lbfgs.py?raw'

const lbfgsRequire = [
  text('Real parameter tensors '),
  math(String.raw`\theta_{t-1}`),
  text(' on one device, flattened into one vector; loss '),
  math(String.raw`L(\theta)`),
  text('; learning rate '),
  math(String.raw`\gamma`),
  text(', inner-iteration limit '),
  math(String.raw`K`),
  text(', history size '),
  math(String.raw`m\ge 1`),
  text(', tolerances '),
  math(String.raw`\epsilon_g,\epsilon_x`),
  text(', and optimizer state '),
  math(String.raw`\mathcal{C}_{t-1}=\{(s_j,y_j,c_j)\}_{j=1}^{h}`),
  text(' with '),
  math(String.raw`0\le h\le m`),
  text('.'),
]

const lbfgsRows: AlgorithmLineSpec[] = [
  {
    id: 'lbfgs-forward-label',
    startsBlock: { id: 'lbfgs-forward', role: 'forward' },
    parts: [strong('Optimization step.')],
    codeRefs: ['step-signature', 'state-init'],
  },
  {
    id: 'lbfgs-initial-evaluation',
    number: 1,
    parts: [
      text('Set '),
      math(String.raw`\theta^{(0)}=\theta_{t-1}`),
      text('; evaluate '),
      math(String.raw`L^{(0)}=L(\theta^{(0)})`),
      text(' and compute '),
      math(String.raw`g^{(0)}=\nabla_{\theta}L(\theta^{(0)})`),
      text('; stop when '),
      math(String.raw`\lVert g^{(0)}\rVert_\infty\le\epsilon_g`),
      text('.'),
    ],
    codeRefs: [
      'step-signature',
      'state-init',
      'loss-gradient-helper',
      'initial-evaluation',
    ],
  },
  {
    id: 'lbfgs-inner-loop',
    number: 2,
    parts: [
      strong('for '),
      math(String.raw`k=0,\ldots,K-1`),
      strong(' do'),
    ],
    codeRefs: ['inner-loop'],
  },
  {
    id: 'lbfgs-direction',
    number: 3,
    indent: 1,
    parts: [
      text('Use the L-BFGS two-loop recursion to compute '),
      math(String.raw`r_h^{(k)}=M^{(k)}g^{(k)}`),
      text(', then form the search direction '),
      math(String.raw`p^{(k)}=-r_h^{(k)}`),
      text('.'),
    ],
    codeRefs: ['direction'],
  },
  {
    id: 'lbfgs-step-size',
    number: 4,
    indent: 1,
    parts: [
      text('Choose step length '),
      math(String.raw`\alpha^{(k)}`),
      text('; when '),
      math(String.raw`k=0`),
      text(' and the history is empty, use '),
      math(String.raw`\alpha^{(0)}=\min(1,\lVert g^{(0)}\rVert_1^{-1})\gamma`),
      text(', otherwise use '),
      math(String.raw`\alpha^{(k)}=\gamma`),
      text('. Stop if '),
      math(String.raw`(g^{(k)})^\top p^{(k)}> -\epsilon_x`),
      text('.'),
    ],
    codeRefs: ['step-size'],
  },
  {
    id: 'lbfgs-parameter-update',
    number: 5,
    indent: 1,
    parts: [
      text('Form '),
      math(String.raw`s^{(k)}=\alpha^{(k)}p^{(k)}`),
      text(' and update '),
      math(String.raw`\theta^{(k+1)}=\theta^{(k)}+s^{(k)}`),
      text('.'),
    ],
    codeRefs: ['save-iterate', 'parameter-update'],
  },
  {
    id: 'lbfgs-reevaluate',
    number: 6,
    indent: 1,
    parts: [
      text('Unless '),
      math(String.raw`k=K-1`),
      text(', evaluate '),
      math(String.raw`L^{(k+1)}=L(\theta^{(k+1)})`),
      text(' and compute '),
      math(String.raw`g^{(k+1)}=\nabla_{\theta}L(\theta^{(k+1)})`),
      text('; defer the final reevaluation to the next outer optimization step.'),
    ],
    codeRefs: [
      'loss-gradient-helper',
      'defer-final-evaluation',
      'reevaluate',
    ],
  },
  {
    id: 'lbfgs-history-update',
    number: 7,
    indent: 1,
    parts: [
      text('Form '),
      math(String.raw`y^{(k)}=g^{(k+1)}-g^{(k)}`),
      text('. If '),
      math(String.raw`(y^{(k)})^\top s^{(k)}>10^{-10}`),
      text(', append '),
      math(String.raw`(s^{(k)},y^{(k)},c^{(k)})`),
      text(' with '),
      math(String.raw`c^{(k)}=(y^{(k)})^\top s^{(k)}`),
      text(' and discard the oldest pair past '),
      math(String.raw`m`),
      text('.'),
    ],
    codeRefs: ['history-helper', 'pending-history', 'history-update'],
  },
  {
    id: 'lbfgs-convergence',
    number: 8,
    indent: 1,
    parts: [
      text('Stop when '),
      math(String.raw`\lVert g^{(k+1)}\rVert_\infty\le\epsilon_g`),
      text(', '),
      math(String.raw`\lVert s^{(k)}\rVert_\infty\le\epsilon_x`),
      text(', or '),
      math(String.raw`|L^{(k+1)}-L^{(k)}|<\epsilon_x`),
      text('; otherwise continue the inner loop.'),
    ],
    codeRefs: ['convergence'],
  },
  {
    id: 'lbfgs-inner-loop-end',
    number: 9,
    parts: [strong('end for')],
  },
  {
    id: 'lbfgs-return',
    number: 10,
    parts: [
      text('After '),
      math(String.raw`J\le K`),
      text(' completed inner updates, identify the final parameters as '),
      math(String.raw`\theta_t=\theta^{(J)}`),
      text(', store the retained history as '),
      math(String.raw`\mathcal{C}_t`),
      text(', and return '),
      math(String.raw`L^{(0)}`),
      text(' with optimizer state.'),
    ],
    codeRefs: ['state-store'],
  },
]

const bfgsRequire = [
  text('Parameters '),
  math(String.raw`\theta_{t-1}\in\mathbb{R}^{n}`),
  text(', loss '),
  math(String.raw`L(\theta)`),
  text(', learning rate '),
  math(String.raw`\gamma`),
  text(', curvature tolerance '),
  math(String.raw`\epsilon_c`),
  text(', and dense inverse-Hessian preconditioner '),
  math(
    String.raw`M_{t-1}\approx\left(\nabla_\theta^2L(\theta_{t-1})\right)^{-1}`
  ),
  text('.'),
]

const bfgsRows: AlgorithmLineSpec[] = [
  {
    id: 'bfgs-state-init',
    number: 1,
    parts: [
      text('If the optimizer state has no inverse-Hessian matrix, initialize '),
      math(String.raw`M_{t-1}\leftarrow I_n`),
      text('.'),
    ],
    codeRefs: ['bfgs-signature', 'bfgs-state-init'],
  },
  {
    id: 'bfgs-initial-evaluation',
    number: 2,
    parts: [
      text('Evaluate '),
      math(String.raw`L_{t-1}=L(\theta_{t-1})`),
      text(' and compute '),
      math(String.raw`g_{t-1}=\nabla_{\theta}L(\theta_{t-1})`),
      text('.'),
    ],
    codeRefs: ['loss-gradient-helper', 'bfgs-initial-evaluation'],
  },
  {
    id: 'bfgs-direction',
    number: 3,
    parts: [
      text('Compute the quasi-Newton search direction '),
      math(String.raw`p_t=-M_{t-1}g_{t-1}`),
      text('.'),
    ],
    codeRefs: ['bfgs-direction'],
  },
  {
    id: 'bfgs-parameter-update',
    number: 4,
    parts: [
      text('Form '),
      math(String.raw`s_t=\gamma p_t`),
      text(' and update '),
      math(String.raw`\theta_t=\theta_{t-1}+s_t`),
      text(', so '),
      math(String.raw`s_t=\theta_t-\theta_{t-1}\approx\delta\theta`),
      text(' for a small parameter displacement.'),
    ],
    codeRefs: ['bfgs-parameter-update'],
  },
  {
    id: 'bfgs-reevaluate',
    number: 5,
    parts: [
      text('Evaluate '),
      math(String.raw`L_t=L(\theta_t)`),
      text(', compute '),
      math(String.raw`g_t=\nabla_{\theta}L(\theta_t)`),
      text(', and set '),
      math(String.raw`y_t=g_t-g_{t-1}\approx\delta\nabla_{\theta}L(\theta)`),
      text('.'),
    ],
    codeRefs: ['loss-gradient-helper', 'bfgs-reevaluate'],
  },
  {
    id: 'bfgs-curvature',
    number: 6,
    parts: [
      text('Compute the scalar curvature '),
      math(String.raw`c_t=y_t^\top s_t`),
      text('. If '),
      math(String.raw`c_t\le \epsilon_c`),
      text(', keep '),
      math(String.raw`M_t=M_{t-1}`),
      text(' and skip the update.'),
    ],
    codeRefs: ['bfgs-curvature'],
  },
  {
    id: 'bfgs-transform',
    number: 7,
    parts: [
      text('Otherwise form the paired matrices '),
      math(String.raw`V_t=I_n-\frac{s_t y_t^\top}{c_t}`),
      text(' and '),
      math(String.raw`V_t^\top=I_n-\frac{y_t s_t^\top}{c_t}`),
      text('.'),
    ],
    codeRefs: ['bfgs-hessian-update'],
  },
  {
    id: 'bfgs-transform-cancellation',
    number: 8,
    parts: [
      text('The transpose cancels the new gradient-change direction: '),
      math(
        String.raw`V_t^\top y_t=y_t-\frac{y_t s_t^\top y_t}{c_t}=y_t-\frac{y_t c_t}{c_t}=0`
      ),
      text('.'),
    ],
    codeRefs: ['bfgs-hessian-update'],
  },
  {
    id: 'bfgs-rank-one-correction',
    number: 9,
    parts: [
      text(
        'The rank-one correction maps that gradient change back to the observed parameter step: '
      ),
      math(
        String.raw`\frac{s_t s_t^\top}{c_t}y_t=s_t\frac{s_t^\top y_t}{c_t}=s_t\frac{c_t}{c_t}=s_t`
      ),
      text('.'),
    ],
    codeRefs: ['bfgs-hessian-update'],
  },
  {
    id: 'bfgs-hessian-update',
    number: 10,
    parts: [
      text('Combine both pieces in the symmetric BFGS update '),
      math(String.raw`M_t=V_tM_{t-1}V_t^\top+\frac{s_t s_t^\top}{c_t}`),
      text('.'),
    ],
    codeRefs: ['bfgs-hessian-update'],
  },
  {
    id: 'bfgs-secant-verification',
    number: 11,
    parts: [
      text('Multiplying by '),
      math(String.raw`y_t`),
      text(' verifies the secant condition: '),
      math(
        String.raw`M_t y_t=V_tM_{t-1}V_t^\top y_t+\frac{s_t s_t^\top}{c_t}y_t=s_t`
      ),
      text('.'),
    ],
    codeRefs: ['bfgs-hessian-update'],
  },
  {
    id: 'bfgs-state-store',
    number: 12,
    parts: [
      text('Store '),
      math(String.raw`M_t`),
      text(', then return the updated parameters '),
      math(String.raw`\theta_t`),
      text(', loss '),
      math(String.raw`L_t`),
      text(', and optimizer state.'),
    ],
    codeRefs: ['bfgs-state-store'],
  },
]

const bfgsAlgorithmRows: AlgorithmLineSpec[] = [
  {
    id: 'bfgs-forward-label',
    startsBlock: { id: 'bfgs-full-memory', role: 'forward' },
    parts: [strong('full-memory inverse-Hessian update.')],
  },
  ...bfgsRows,
]

const lbfgsNote: LatexBlockSpec = {
  id: 'lbfgs-optimization',
  title: 'L-BFGS Optimization step',
  requireLabel: 'Require',
  intro: [
    strong('Difference from BFGS.'),
    text(' BFGS stores the dense '),
    math(String.raw`M_t\in\mathbb{R}^{n\times n}`),
    text(', requiring '),
    math(String.raw`\mathcal{O}(n^2)`),
    text(' memory. L-BFGS never stores the dense matrix; it keeps only the latest '),
    math(String.raw`m`),
    text(' correction pairs '),
    math(String.raw`(s_j,y_j)`),
    text(
      ' and applies their implicit inverse-Hessian approximation to the gradient with the two-loop recursion, reducing memory to '
    ),
    math(String.raw`\mathcal{O}(mn)`),
    text('.'),
  ],
  require: lbfgsRequire,
  rows: lbfgsRows.slice(1),
}

const twoLoopNote: LatexBlockSpec = {
  id: 'lbfgs-two-loop',
  title: 'L-BFGS two-loop recursion',
  require: [
    text('Gradient '),
    math(String.raw`g^{(k)}`),
    text(' and ordered correction history '),
    math(String.raw`\mathcal{C}=\{(s_j,y_j,c_j)\}_{j=1}^{h}`),
    text(' with '),
    math(String.raw`0\le h\le m`),
    text(' and '),
    math(String.raw`c_j=y_j^\top s_j`),
    text('; define '),
    math(String.raw`V_j=I-\frac{s_jy_j^\top}{c_j}`),
    text('. Compute '),
    math(String.raw`r_h=M^{(k)}g^{(k)}`),
    text(' without forming the dense '),
    math(String.raw`M^{(k)}`),
    text('.'),
  ],
  rows: [
    {
      id: 'lbfgs-two-loop-start',
      number: 1,
      parts: [
        text('Set the backward-pass work vector '),
        math(String.raw`q_h\leftarrow g^{(k)}`),
        text('.'),
      ],
      codeRefs: ['two-loop-signature', 'two-loop-start'],
    },
    {
      id: 'lbfgs-two-loop-backward',
      number: 2,
      parts: [
        strong('for '),
        math(String.raw`j=h,\ldots,1`),
        strong(' do'),
      ],
      codeRefs: ['two-loop-backward'],
    },
    {
      id: 'lbfgs-two-loop-backward-coefficient',
      number: 3,
      indent: 1,
      parts: [
        text('Compute '),
        math(String.raw`a_j=\frac{s_j^\top q_j}{c_j}`),
        text('.'),
      ],
      codeRefs: ['two-loop-backward'],
    },
    {
      id: 'lbfgs-two-loop-backward-transform',
      number: 4,
      indent: 1,
      parts: [
        text('Apply the transpose factor '),
        math(String.raw`q_{j-1}\leftarrow q_j-a_j y_j`),
        text(', equivalently '),
        math(String.raw`q_{j-1}\leftarrow V_j^\top q_j`),
        text('.'),
      ],
      codeRefs: ['two-loop-backward'],
    },
    {
      id: 'lbfgs-two-loop-backward-end',
      number: 5,
      parts: [strong('end for')],
    },
    {
      id: 'lbfgs-two-loop-scale',
      number: 6,
      parts: [
        text('Start the forward-pass result with '),
        math(String.raw`r_0\leftarrow M_0^{(k)}q_0`),
        text(', where '),
        math(String.raw`M_0^{(k)}=\frac{c_h}{y_h^\top y_h}I`),
        text(' when '),
        math(String.raw`h>0`),
        text(', and '),
        math(String.raw`M_0^{(k)}=I`),
        text(' when the history is empty.'),
      ],
      codeRefs: ['initial-hessian-scale'],
    },
    {
      id: 'lbfgs-two-loop-forward',
      number: 7,
      parts: [
        strong('for '),
        math(String.raw`j=1,\ldots,h`),
        strong(' do'),
      ],
      codeRefs: ['two-loop-forward'],
    },
    {
      id: 'lbfgs-two-loop-forward-coefficient',
      number: 8,
      indent: 1,
      parts: [
        text('Compute '),
        math(String.raw`b_j=\frac{y_j^\top r_{j-1}}{c_j}`),
        text('.'),
      ],
      codeRefs: ['two-loop-forward'],
    },
    {
      id: 'lbfgs-two-loop-forward-transform',
      number: 9,
      indent: 1,
      parts: [
        text('Apply the forward factor and correction: '),
        math(
          String.raw`r_j\leftarrow r_{j-1}+s_j(a_j-b_j)=V_jr_{j-1}+s_ja_j`
        ),
        text('.'),
      ],
      codeRefs: ['two-loop-forward'],
    },
    {
      id: 'lbfgs-two-loop-forward-end',
      number: 10,
      parts: [strong('end for')],
    },
    {
      id: 'lbfgs-two-loop-return',
      number: 11,
      parts: [
        text('Return '),
        math(String.raw`r_h=M^{(k)}g^{(k)}`),
        text(', the inverse-Hessian–gradient product.'),
      ],
      codeRefs: ['two-loop-forward'],
    },
    {
      id: 'lbfgs-two-loop-three-pair-label',
      number: 12,
      parts: [
        strong('Three-pair expansion.'),
        text(' For '),
        math(String.raw`h=3`),
        text(', the symbols above map directly to the expanded product.'),
      ],
    },
    {
      id: 'lbfgs-two-loop-three-pair-backward',
      number: 13,
      parts: [
        text('The backward work vector is '),
        math(String.raw`q_0=V_1^\top V_2^\top V_3^\top g^{(3)}`),
        text('.'),
      ],
      codeRefs: ['two-loop-backward'],
    },
    {
      id: 'lbfgs-two-loop-three-pair-base',
      number: 14,
      parts: [
        text('The scaled base carried through the forward loop is '),
        math(
          String.raw`V_3V_2V_1r_0=V_3V_2V_1M_0^{(3)}q_0=V_3V_2V_1M_0^{(3)}V_1^\top V_2^\top V_3^\top g^{(3)}`
        ),
        text('.'),
      ],
      codeRefs: ['initial-hessian-scale', 'two-loop-forward'],
    },
    {
      id: 'lbfgs-two-loop-three-pair-first',
      number: 15,
      parts: [
        text('The forward addition from '),
        math(String.raw`a_1`),
        text(' becomes '),
        math(
          String.raw`V_3V_2s_1a_1=V_3V_2\frac{s_1s_1^\top}{c_1}V_2^\top V_3^\top g^{(3)}`
        ),
        text('.'),
      ],
      codeRefs: ['two-loop-backward', 'two-loop-forward'],
    },
    {
      id: 'lbfgs-two-loop-three-pair-second',
      number: 16,
      parts: [
        text('The forward addition from '),
        math(String.raw`a_2`),
        text(' becomes '),
        math(
          String.raw`V_3s_2a_2=V_3\frac{s_2s_2^\top}{c_2}V_3^\top g^{(3)}`
        ),
        text('.'),
      ],
      codeRefs: ['two-loop-backward', 'two-loop-forward'],
    },
    {
      id: 'lbfgs-two-loop-three-pair-third',
      number: 17,
      parts: [
        text('The forward addition from '),
        math(String.raw`a_3`),
        text(' becomes '),
        math(String.raw`s_3a_3=\frac{s_3s_3^\top}{c_3}g^{(3)}`),
        text('.'),
      ],
      codeRefs: ['two-loop-backward', 'two-loop-forward'],
    },
    {
      id: 'lbfgs-two-loop-three-pair-result',
      number: 18,
      parts: [
        text('Adding the base and three correction terms gives '),
        math(String.raw`r_3=M^{(3)}g^{(3)}`),
        text('.'),
      ],
    },
  ],
}

const lbfgsContent = defineAttentionContent({
  rawCode: lbfgsCode,
  require: bfgsRequire,
  rows: bfgsAlgorithmRows,
  notes: [lbfgsNote, twoLoopNote],
  ignoredUnusedRefs: ['flatten-helpers'],
})

export const lbfgsExample: AttentionExample = {
  id: 'lbfgs',
  urlTag: 'optimizer/lbfgs',
  label: 'L-BFGS',
  description:
    'L-BFGS uses recent changes in parameters and gradients to estimate curvature and choose a better search direction.',
  algorithmTitle: 'BFGS',
  content: {
    unmasked: lbfgsContent,
    masked: lbfgsContent,
  },
}
