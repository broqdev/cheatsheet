import type { AttentionExample } from '../../model'
import { defineAttentionContent, type AlgorithmLineSpec, type LatexBlockSpec } from '../../lib/codeRefs'
import { math, strong, text } from '../../lib/segments'
import lbfgsCode from './code/lbfgs.py?raw'

const lbfgsRequire = [
  text('Real parameter tensors on one device, flattened as '),
  math(String.raw`\theta`),
  text('; closure '),
  math(String.raw`f(\theta)`),
  text(' that clears gradients, recomputes the loss, and backpropagates; learning rate '),
  math(String.raw`\gamma`),
  text(', inner-iteration limit '),
  math(String.raw`K`),
  text(', history size '),
  math(String.raw`m\ge 1`),
  text(', tolerances '),
  math(String.raw`\epsilon_g,\epsilon_x`),
  text(', and correction history '),
  math(String.raw`\mathcal{H}=\{(s_i,y_i,\rho_i)\}`),
  text('.'),
]

const lbfgsRows: AlgorithmLineSpec[] = [
  {
    id: 'lbfgs-forward-label',
    parts: [strong('Limited-memory quasi-Newton step.')],
    codeRefs: ['step-signature', 'state-init'],
  },
  {
    id: 'lbfgs-initial-evaluation',
    number: 1,
    parts: [
      text('Evaluate the closure to obtain '),
      math(String.raw`f_0=f(\theta_0)`),
      text(' and '),
      math(String.raw`g_0=\nabla f(\theta_0)`),
      text('; stop when '),
      math(String.raw`\lVert g_0\rVert_\infty\le\epsilon_g`),
      text('.'),
    ],
    codeRefs: ['initial-evaluation'],
  },
  {
    id: 'lbfgs-inner-loop',
    number: 2,
    parts: [
      text('For '),
      math(String.raw`k=0,\ldots,K-1`),
      text(', compute search direction '),
      math(String.raw`p_k=-H_k g_k`),
      text(' with the two-loop recursion over at most '),
      math(String.raw`m`),
      text(' correction pairs.'),
    ],
    codeRefs: ['inner-loop', 'direction'],
  },
  {
    id: 'lbfgs-step-size',
    number: 3,
    parts: [
      text('Choose step length '),
      math(String.raw`\alpha_k`),
      text('; this fixed-step version uses a guarded first step and then '),
      math(String.raw`\gamma`),
      text('.'),
    ],
    codeRefs: ['step-size'],
  },
  {
    id: 'lbfgs-parameter-update',
    number: 4,
    parts: [
      text('Form '),
      math(String.raw`s_k=\alpha_kp_k`),
      text(' and update '),
      math(String.raw`\theta_{k+1}=\theta_k+s_k=\theta_k+\alpha_kp_k`),
      text('.'),
    ],
    codeRefs: ['save-iterate', 'parameter-update'],
  },
  {
    id: 'lbfgs-reevaluate',
    number: 5,
    parts: [
      text('Reevaluate the closure at '),
      math(String.raw`\theta_{k+1}`),
      text(' to obtain '),
      math(String.raw`f_{k+1}`),
      text(' and '),
      math(String.raw`g_{k+1}`),
      text('.'),
    ],
    codeRefs: ['reevaluate'],
  },
  {
    id: 'lbfgs-history-update',
    number: 6,
    parts: [
      text('Form '),
      math(String.raw`y_k=g_{k+1}-g_k`),
      text('. If '),
      math(String.raw`y_k^\top s_k>10^{-10}`),
      text(', append '),
      math(String.raw`(s_k,y_k,\rho_k)`),
      text(' with '),
      math(String.raw`\rho_k=(y_k^\top s_k)^{-1}`),
      text(' and discard the oldest pair past '),
      math(String.raw`m`),
      text('.'),
    ],
    codeRefs: ['history-update'],
  },
  {
    id: 'lbfgs-convergence',
    number: 7,
    parts: [
      text('Stop for a small gradient, small parameter change, or small loss change; otherwise continue the inner loop.'),
    ],
    codeRefs: ['convergence'],
  },
  {
    id: 'lbfgs-return',
    number: 8,
    parts: [
      text('Store the limited correction history and return the loss from the first closure evaluation.'),
    ],
    codeRefs: ['state-store'],
  },
]

const twoLoopNote: LatexBlockSpec = {
  id: 'lbfgs-two-loop',
  title: 'Two-loop inverse-Hessian product',
  require: [
    text('Gradient '),
    math(String.raw`g_k`),
    text(' and ordered correction history '),
    math(String.raw`\{(s_i,y_i,\rho_i)\}_{i=k-r}^{k-1}`),
    text(' with '),
    math(String.raw`r\le m`),
    text('.'),
  ],
  rows: [
    {
      id: 'lbfgs-two-loop-start',
      number: 1,
      parts: [
        text('Set '),
        math(String.raw`q\leftarrow g_k`),
        text('.'),
      ],
      codeRefs: ['two-loop-signature', 'two-loop-start'],
    },
    {
      id: 'lbfgs-two-loop-backward',
      number: 2,
      parts: [
        text('For stored pairs from newest to oldest, compute '),
        math(String.raw`a_i=\rho_i s_i^\top q`),
        text(' and update '),
        math(String.raw`q\leftarrow q-a_i y_i`),
        text('.'),
      ],
      codeRefs: ['two-loop-backward'],
    },
    {
      id: 'lbfgs-two-loop-scale',
      number: 3,
      parts: [
        text('Scale by the newest curvature pair: '),
        math(String.raw`r\leftarrow H_k^{(0)}q`),
        text(', where '),
        math(String.raw`H_k^{(0)}=(s_{k-1}^\top y_{k-1})/(y_{k-1}^\top y_{k-1})`),
        text(' or '),
        math(String.raw`1`),
        text(' before any pair exists.'),
      ],
      codeRefs: ['initial-hessian-scale'],
    },
    {
      id: 'lbfgs-two-loop-forward',
      number: 4,
      parts: [
        text('For pairs from oldest to newest, compute '),
        math(String.raw`b_i=\rho_i y_i^\top r`),
        text(' and update '),
        math(String.raw`r\leftarrow r+s_i(a_i-b_i)`),
        text('; return '),
        math(String.raw`p_k=-r`),
        text('.'),
      ],
      codeRefs: ['two-loop-forward'],
    },
  ],
}

const lbfgsContent = defineAttentionContent({
  rawCode: lbfgsCode,
  require: lbfgsRequire,
  rows: lbfgsRows,
  notes: [twoLoopNote],
  ignoredUnusedRefs: ['flatten-helpers'],
})

export const lbfgsExample: AttentionExample = {
  id: 'lbfgs',
  urlTag: 'optimizer/lbfgs',
  label: 'L-BFGS',
  description:
    'L-BFGS estimates inverse-Hessian directions from recent parameter and gradient changes.',
  algorithmTitle: 'L-BFGS',
  content: {
    unmasked: lbfgsContent,
    masked: lbfgsContent,
  },
}
