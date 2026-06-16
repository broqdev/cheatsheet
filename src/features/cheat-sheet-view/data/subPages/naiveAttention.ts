import type { AttentionExample } from '../../model'
import naiveAttentionCode from './code/naiveAttention.py?raw'
import maskedNaiveAttentionCode from './code/maskedNaiveAttention.py?raw'
import { defineAttentionContent, type AlgorithmLineSpec, type LatexBlockSpec } from '../../lib/codeRefs'
import { math, strong, text } from '../../lib/segments'
import { latexDelta } from '../../lib/toggleDeltas'

const naiveRequire = [
  text('Matrices '),
  math(String.raw`Q,K,V \in \mathbb{R}^{N \times d}`),
  text(', QK scaling factor '),
  math(String.raw`\alpha \in \mathbb{R}`),
  text(' (usually '),
  math(String.raw`\alpha=1/\sqrt{d}`),
  text(')'),
  text(', output gradient '),
  math(String.raw`dO`),
  text('.'),
]

const maskedNaiveRequire = [
  text('Matrices '),
  math(String.raw`Q,K,V \in \mathbb{R}^{N \times d}`),
  text(', QK scaling factor '),
  math(String.raw`\alpha \in \mathbb{R}`),
  text(' (usually '),
  math(String.raw`\alpha=1/\sqrt{d}`),
  text(')'),
  text(', additive mask ', 'mask'),
  math(String.raw`M \in \{0,-\infty\}^{N \times N}`, 'mask'),
  text(', output gradient '),
  math(String.raw`dO`),
  text('.'),
]

const naiveRows: AlgorithmLineSpec[] = [
  {
    id: 'naive-forward-label',
    parts: [strong('Forward pass.')],
    codeRefs: ['forward-signature'],
  },
  {
    id: 'naive-score',
    number: 1,
    parts: [
      text('Compute scores '),
      math(String.raw`S = \alpha QK^\top`),
      text(', where '),
      math(String.raw`\alpha`),
      text(' scales QK logits before softmax.'),
    ],
    codeRefs: ['score'],
  },
  {
    id: 'naive-probability',
    number: 2,
    parts: [
      text('Normalize each row '),
      math(String.raw`P_{ij}=\exp(S_{ij}) / \sum_k \exp(S_{ik})`),
      text('.'),
    ],
    codeRefs: ['probability'],
  },
  {
    id: 'naive-output',
    number: 3,
    parts: [
      text('Return weighted values '),
      math(String.raw`O = PV`),
      text('.'),
    ],
    codeRefs: ['output'],
  },
  {
    id: 'naive-backward-label',
    parts: [strong('Backward pass.')],
    codeRefs: ['backward-signature'],
  },
  {
    id: 'naive-dv-dp',
    number: 4,
    parts: [
      text('Compute '),
      math(String.raw`dV=P^\top dO`),
      text(' and '),
      math(String.raw`dP=dOV^\top`),
      text('.'),
    ],
    codeRefs: ['dv-dp'],
  },
  {
    id: 'naive-ds',
    number: 5,
    parts: [
      text('Backprop through softmax '),
      math(String.raw`D=\operatorname{rowsum}(P\odot dP)`),
      text(', '),
      math(String.raw`dS=P\odot(dP-D)`),
      text('.'),
    ],
    codeRefs: ['softmax-call', 'softmax-body'],
  },
  {
    id: 'naive-dq-dk',
    number: 6,
    parts: [
      text('Return '),
      math(String.raw`dQ=\alpha dSK`),
      text(', '),
      math(String.raw`dK=\alpha dS^\top Q`),
      text(', and '),
      math(String.raw`dV`),
      text('.'),
    ],
    codeRefs: ['dq-dk'],
  },
]

const maskedNaiveRows: AlgorithmLineSpec[] = [
  {
    id: 'naive-forward-label',
    parts: [strong('Forward pass.')],
    codeRefs: ['forward-signature'],
  },
  {
    id: 'naive-score',
    number: 1,
    parts: [
      text('Compute '),
      text('masked ', 'mask'),
      text('scores '),
      math(String.raw`S = \alpha QK^\top ${latexDelta('mask', '+ M')}`),
      text(', where '),
      math(String.raw`\alpha`),
      text(' scales QK logits and '),
      math(String.raw`M_{ij}=0`, 'mask'),
      text(' or '),
      math(String.raw`-\infty`, 'mask'),
      text('.'),
    ],
    codeRefs: ['score'],
  },
  {
    id: 'naive-probability',
    number: 2,
    parts: [
      text('Normalize '),
      text('visible ', 'mask'),
      text('keys '),
      math(String.raw`P_{ij}=\exp(S_{ij}) / \sum_k \exp(S_{ik})`),
      text(', giving '),
      math(String.raw`P_{ij}=0`, 'mask'),
      text(' where masked.', 'mask'),
    ],
    codeRefs: ['probability'],
  },
  {
    id: 'naive-output',
    number: 3,
    parts: [
      text('Return weighted values '),
      math(String.raw`O = PV`),
      text('.'),
    ],
    codeRefs: ['output'],
  },
  {
    id: 'naive-backward-label',
    parts: [strong('Backward pass.')],
    codeRefs: ['backward-signature'],
  },
  {
    id: 'naive-dv-dp',
    number: 4,
    parts: [
      text('Compute '),
      math(String.raw`dV=P^\top dO`),
      text(' and '),
      math(String.raw`dP=dOV^\top`),
      text('.'),
    ],
    codeRefs: ['dv-dp'],
  },
  {
    id: 'naive-ds',
    number: 5,
    parts: [
      text('Backprop through '),
      text('masked ', 'mask'),
      text('softmax '),
      math(String.raw`D=\operatorname{rowsum}(P\odot dP)`),
      text(', '),
      math(String.raw`dS=P\odot(dP-D)`),
      text('; masked entries stay zero because ', 'mask'),
      math(String.raw`P_{ij}=0`, 'mask'),
      text('.'),
    ],
    codeRefs: ['softmax-call', 'softmax-body'],
  },
  {
    id: 'naive-dq-dk',
    number: 6,
    parts: [
      text('Return '),
      math(String.raw`dQ=\alpha dSK`),
      text(', '),
      math(String.raw`dK=\alpha dS^\top Q`),
      text(', and '),
      math(String.raw`dV`),
      text('.'),
    ],
    codeRefs: ['dq-dk'],
  },
]

const softmaxBackwardNotes = (masked: boolean): LatexBlockSpec[] => [
  {
    id: 'naive-softmax-backward',
    title: 'Softmax backward',
    requireLabel: 'Require',
    require: [
      text('One probability row '),
      math(String.raw`P_i=\operatorname{softmax}(S_i)`),
      text(' and upstream gradient '),
      math(String.raw`dP_i=\partial L/\partial P_i`),
      text('.'),
    ],
    rows: [
      {
        id: 'softmax-output-target',
        number: 1,
        parts: [
          text('Return score gradient '),
          math(String.raw`dS_i=\partial L/\partial S_i`),
          text(' for the same query row.'),
        ],
        codeRefs: ['softmax-call-result', 'softmax-def'],
      },
      {
        id: 'softmax-chain-rule',
        number: 2,
        parts: [
          text('Chain rule for one score entry: '),
          math(String.raw`dS_{ij}=\sum_k dP_{ik}\frac{\partial P_{ik}}{\partial S_{ij}}`),
          text('.'),
        ],
        codeRefs: ['softmax-call-result', 'softmax-body'],
      },
      {
        id: 'softmax-jacobian',
        number: 3,
        parts: [
          math(String.raw`\frac{\partial P_{ik}}{\partial S_{ij}}=P_{ik}(\delta_{kj}-P_{ij})`),
          text(', where '),
          math(String.raw`\delta_{kj}=1`),
          text(' if '),
          math(String.raw`k=j`),
          text(' and '),
          math(String.raw`0`),
          text(' otherwise.'),
        ],
        codeRefs: ['softmax-body'],
      },
      {
        id: 'softmax-substitute',
        number: 4,
        parts: [
          math(String.raw`dS_{ij}=P_{ij}dP_{ij}-P_{ij}\sum_k dP_{ik}P_{ik}`),
          text('.'),
        ],
        codeRefs: ['softmax-comment', 'softmax-correction', 'softmax-return'],
      },
      {
        id: 'softmax-row-correction',
        number: 5,
        parts: [
          text('Define row correction '),
          math(String.raw`D_i=\sum_k dP_{ik}P_{ik}`),
          text('.'),
        ],
        codeRefs: ['softmax-comment', 'softmax-correction'],
      },
      {
        id: 'softmax-vectorized',
        number: 6,
        parts: [
          text('Vector form: '),
          math(String.raw`D=\operatorname{rowsum}(P\odot dP)`),
          text(', '),
          math(String.raw`dS=P\odot(dP-D)`),
          text('.'),
        ],
        codeRefs: ['softmax-body'],
      },
      {
        id: 'softmax-masked-zero',
        number: 7,
        parts: [
          text(
            masked
              ? 'Masked logits get zero gradient because '
              : 'Unmasked rows apply the same row formula to every key because ',
            'mask'
          ),
          math(masked ? String.raw`p_i=0 \Rightarrow dS_i=0` : String.raw`p_i>0`, 'mask'),
          text('.'),
        ],
        codeRefs: ['softmax-call-result', 'softmax-return'],
      },
    ],
  },
]

const naiveCostNotes: LatexBlockSpec[] = [
  {
    id: 'naive-flops-space',
    title: 'Naive FLOPs and space',
    rows: [
      {
        id: 'cost-naive-flops-theorem',
        number: 1,
        parts: [
          strong('Theorem 1 (FLOPs). '),
          text('Naive attention forward costs '),
          math(String.raw`\Theta(N^2d)`),
          text(' FLOPs.'),
        ],
      },
      {
        id: 'cost-naive-score',
        number: 2,
        parts: [
          text('Scores: '),
          math(String.raw`S=\alpha QK^\top\in\mathbb{R}^{N\times N}`),
          text(' has '),
          math(String.raw`N^2`),
          text(' length-'),
          math(String.raw`d`),
          text(' dot products; '),
          math(String.raw`\alpha`),
          text(' is one scalar multiply per score, so the dominant work is still '),
          math(String.raw`\Theta(N^2d)`),
          text(' FLOPs.'),
        ],
      },
      {
        id: 'cost-naive-softmax',
        number: 3,
        parts: [
          text('Softmax normalizes '),
          math(String.raw`N`),
          text(' rows of length '),
          math(String.raw`N`),
          text(', adding '),
          math(String.raw`\Theta(N^2)`),
          text(' elementwise/reduction work.'),
        ],
      },
      {
        id: 'cost-naive-output',
        number: 4,
        parts: [
          text('Output: '),
          math(String.raw`O=PV`),
          text(' multiplies '),
          math(String.raw`N\times N`),
          text(' by '),
          math(String.raw`N\times d`),
          text(', another '),
          math(String.raw`\Theta(N^2d)`),
          text('.'),
        ],
      },
      {
        id: 'cost-naive-total',
        number: 5,
        parts: [
          text('Total: '),
          math(String.raw`\Theta(N^2d)+\Theta(N^2)+\Theta(N^2d)=\Theta(N^2d)`),
          text(' for '),
          math(String.raw`d\ge 1`),
          text('.'),
        ],
      },
      {
        id: 'cost-naive-space-theorem',
        number: 6,
        parts: [
          strong('Theorem 2 (Space). '),
          text('Naive attention stores '),
          math(String.raw`\Theta(N^2)`),
          text(' extra attention state.'),
        ],
      },
      {
        id: 'cost-naive-space',
        number: 7,
        parts: [
          text('Extra space: materialize '),
          math(String.raw`S`),
          text(' and '),
          math(String.raw`P`),
          text(', each '),
          math(String.raw`N\times N`),
          text(', so memory beyond inputs/output is '),
          math(String.raw`\Theta(N^2)`),
          text('.'),
        ],
      },
      {
        id: 'cost-naive-backward-space',
        number: 8,
        parts: [
          text('Training keeps '),
          math(String.raw`P`),
          text(' for backward and forms '),
          math(String.raw`dP`),
          text('; the dominant saved state remains attention-sized, '),
          math(String.raw`\Theta(N^2)`),
          text('.'),
        ],
      },
    ],
  },
]

export const naiveAttentionExample: AttentionExample = {
  id: 'naive',
  urlTag: 'naive-attention',
  label: 'Naive Attention',
  description: 'Baseline attention materializes scores and probabilities, making each equation map directly to the PyTorch code.',
  algorithmTitle: 'NaiveAttention',
  content: {
    unmasked: defineAttentionContent({
      rawCode: naiveAttentionCode,
      require: naiveRequire,
      rows: naiveRows,
      notes: [...softmaxBackwardNotes(false), ...naiveCostNotes],
    }),
    masked: defineAttentionContent({
      rawCode: maskedNaiveAttentionCode,
      require: maskedNaiveRequire,
      rows: maskedNaiveRows,
      notes: [...softmaxBackwardNotes(true), ...naiveCostNotes],
    }),
  },
}
