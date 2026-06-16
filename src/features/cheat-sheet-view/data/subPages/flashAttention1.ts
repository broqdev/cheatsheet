import type { AttentionExample } from '../../model'
import flashAttentionCode from './code/flashAttention1.py?raw'
import maskedFlashAttentionCode from './code/maskedFlashAttention1.py?raw'
import flashAttentionDropoutCode from './code/flashAttention1Dropout.py?raw'
import maskedFlashAttentionDropoutCode from './code/maskedFlashAttention1Dropout.py?raw'
import { defineAttentionContent, type AlgorithmLineSpec, type LatexBlockSpec } from '../../lib/codeRefs'
import { math, strong, text } from '../../lib/segments'
import { latexDelta } from '../../lib/toggleDeltas'

const flashCostNotes: LatexBlockSpec[] = [
  {
    id: 'flash-flops-space',
    title: 'FlashAttention-1 FLOPs, space, and HBM accesses',
    rows: [
      {
        id: 'cost-flash-flops-theorem',
        number: 1,
        parts: [
          strong('Theorem 1 (FLOPs). '),
          text('FlashAttention-1 forward costs '),
          math(String.raw`\Theta(N^2d)`),
          text(' FLOPs.'),
        ],
      },
      {
        id: 'cost-flash-tile-count',
        number: 2,
        parts: [
          text('Tile count: '),
          math(String.raw`T_rT_c=\lceil N/B_r\rceil\lceil N/B_c\rceil`),
          text(' query/key block pairs.'),
        ],
      },
      {
        id: 'cost-flash-score-tile',
        number: 3,
        parts: [
          text('Each score tile computes '),
          math(String.raw`\alpha Q_iK_j^\top`),
          text('; '),
          math(String.raw`\alpha`),
          text(' is applied before online softmax and the tile still has '),
          math(String.raw`\Theta(B_rB_cd)`),
          text(' work.'),
        ],
      },
      {
        id: 'cost-flash-output-tile',
        number: 4,
        parts: [
          text('Each output update multiplies tile probabilities by '),
          math(String.raw`V_j`),
          text(', another '),
          math(String.raw`\Theta(B_rB_cd)`),
          text('.'),
        ],
      },
      {
        id: 'cost-flash-total',
        number: 5,
        parts: [
          text('Total: '),
          math(String.raw`T_rT_c\Theta(B_rB_cd)=\Theta(N^2d)`),
          text('; online softmax adds lower-order '),
          math(String.raw`\Theta(N^2)`),
          text(' work.'),
        ],
      },
      {
        id: 'cost-flash-space-theorem',
        number: 6,
        parts: [
          strong('Theorem 2 (Space). '),
          text('FlashAttention-1 stores '),
          math(String.raw`\Theta(N)`),
          text(' extra HBM state beyond inputs/output.'),
        ],
      },
      {
        id: 'cost-flash-row-state',
        number: 7,
        parts: [
          text('Saved row state is '),
          math(String.raw`m,\ell\in\mathbb{R}^{N}`),
          text('; the full '),
          math(String.raw`S,P\in\mathbb{R}^{N\times N}`),
          text(' matrices are not materialized.'),
        ],
      },
      {
        id: 'cost-flash-sram',
        number: 8,
        parts: [
          text('SRAM holds only current tiles such as '),
          math(String.raw`Q_i,K_j,V_j,S_{ij},\tilde{P}_{ij}`),
          text(', bounded by the tile budget '),
          math(String.raw`M`),
          text('.'),
        ],
      },
      {
        id: 'cost-flash-backward-space',
        number: 9,
        parts: [
          text('Backward recomputes tile probabilities from '),
          math(String.raw`Q,K`),
          text(' and saved normalizers, avoiding saved '),
          math(String.raw`\Theta(N^2)`),
          text(' attention state.'),
        ],
      },
      {
        id: 'cost-flash-hbm-theorem',
        number: 10,
        parts: [
          strong('Theorem 3 (HBM accesses). '),
          text('For '),
          math(String.raw`d\le M\le Nd`),
          text(', FlashAttention-1 uses '),
          math(String.raw`\Theta(N^2d^2M^{-1})`),
          text(' HBM accesses; standard attention uses '),
          math(String.raw`\Theta(Nd+N^2)`),
          text('.'),
        ],
      },
      {
        id: 'cost-flash-hbm-passes',
        number: 11,
        parts: [
          text('A '),
          math(String.raw`K,V`),
          text(' tile stays in SRAM while the algorithm makes '),
          math(String.raw`\Theta(NdM^{-1})`),
          text(' passes over '),
          math(String.raw`\Theta(Nd)`),
          text(' query data.'),
        ],
      },
      {
        id: 'cost-flash-hbm-intuition',
        number: 12,
        parts: [
          text('When '),
          math(String.raw`d^2\ll M`),
          text(', this avoids the dominant '),
          math(String.raw`N\times N`),
          text(' score/probability traffic through HBM.'),
        ],
      },
    ],
  },
]

const flashRequire = [
  text('Matrices '),
  math(String.raw`Q,K,V \in \mathbb{R}^{N \times d}`),
  text(' in HBM, on-chip SRAM of size '),
  math(String.raw`M`),
  text(', QK scaling factor '),
  math(String.raw`\alpha \in \mathbb{R}`),
  text(' (usually '),
  math(String.raw`\alpha=1/\sqrt{d}`),
  text(')'),
  text('.'),
]

const maskedFlashRequire = [
  text('Matrices '),
  math(String.raw`Q,K,V \in \mathbb{R}^{N \times d}`),
  text(' in HBM, on-chip SRAM of size '),
  math(String.raw`M`),
  text(', QK scaling factor '),
  math(String.raw`\alpha \in \mathbb{R}`),
  text(' (usually '),
  math(String.raw`\alpha=1/\sqrt{d}`),
  text(')'),
  text(', masking function ', 'mask'),
  math(String.raw`\operatorname{MASK}`, 'mask'),
  text('.'),
]

const flashDropoutRequire = [
  text('Matrices '),
  math(String.raw`Q,K,V \in \mathbb{R}^{N \times d}`),
  text(' in HBM, on-chip SRAM of size '),
  math(String.raw`M`),
  text(', QK scaling factor '),
  math(String.raw`\alpha \in \mathbb{R}`),
  text(' (usually '),
  math(String.raw`\alpha=1/\sqrt{d}`),
  text(')'),
  text(', dropout probability ', 'dropout'),
  math(String.raw`p_{\mathrm{drop}}`, 'dropout'),
  text('.'),
]

const maskedFlashDropoutRequire = [
  text('Matrices '),
  math(String.raw`Q,K,V \in \mathbb{R}^{N \times d}`),
  text(' in HBM, on-chip SRAM of size '),
  math(String.raw`M`),
  text(', QK scaling factor '),
  math(String.raw`\alpha \in \mathbb{R}`),
  text(' (usually '),
  math(String.raw`\alpha=1/\sqrt{d}`),
  text(')'),
  text(', masking function ', 'mask'),
  math(String.raw`\operatorname{MASK}`, 'mask'),
  text(', dropout probability ', 'dropout'),
  math(String.raw`p_{\mathrm{drop}}`, 'dropout'),
  text('.'),
]

const flashDropoutRows: AlgorithmLineSpec[] = [
  {
    id: 'flash-rng',
    number: 1,
    parts: [
      text('Initialize the pseudo-random number generator state ', 'dropout'),
      math(String.raw`\mathcal{R}`, 'dropout'),
      text(' and save to HBM.', 'dropout'),
    ],
    codeRefs: ['fwd-rng'],
  },
  {
    id: 'flash-blocks',
    number: 2,
    parts: [
      text('Set block sizes '),
      math(String.raw`B_c=\lceil M/(4d)\rceil`),
      text(', '),
      math(String.raw`B_r=\min(\lceil M/(4d)\rceil,d)`),
      text('.'),
    ],
    codeRefs: ['fwd-blocks'],
  },
  {
    id: 'flash-init',
    number: 3,
    parts: [
      text('Initialize '),
      math(String.raw`O=(0)_{N\times d}`),
      text(', '),
      math(String.raw`\ell=(0)_N`),
      text(', '),
      math(String.raw`m=(-\infty)_N`),
      text(' in HBM.'),
    ],
    codeRefs: ['fwd-init'],
  },
  {
    id: 'flash-divide-qkv',
    number: 4,
    parts: [
      text('Divide '),
      math(String.raw`Q`),
      text(' into '),
      math(String.raw`T_r=\lceil N/B_r\rceil`),
      text(' blocks and divide '),
      math(String.raw`K,V`),
      text(' into '),
      math(String.raw`T_c=\lceil N/B_c\rceil`),
      text(' blocks.'),
    ],
    codeRefs: ['fwd-loop-kv', 'fwd-load-kv', 'fwd-loop-q'],
  },
  {
    id: 'flash-divide-state',
    number: 5,
    parts: [
      text('Divide '),
      math(String.raw`O`),
      text(' into '),
      math(String.raw`T_r`),
      text(' blocks and divide '),
      math(String.raw`\ell,m`),
      text(' into '),
      math(String.raw`T_r`),
      text(' row-state blocks.'),
    ],
    codeRefs: ['fwd-load-q-state', 'fwd-store-output', 'fwd-write-state'],
  },
  {
    id: 'flash-loop-kv',
    number: 6,
    parts: [
      strong('for '),
      math(String.raw`1 \le j \le T_c`),
      strong(' do'),
    ],
    codeRefs: ['fwd-loop-kv'],
  },
  {
    id: 'flash-load-kv',
    number: 7,
    indent: 1,
    parts: [
      text('Load '),
      math(String.raw`K_j,V_j`),
      text(' from HBM to on-chip SRAM.'),
    ],
    codeRefs: ['fwd-load-kv'],
  },
  {
    id: 'flash-loop-q',
    number: 8,
    indent: 1,
    parts: [
      strong('for '),
      math(String.raw`1 \le i \le T_r`),
      strong(' do'),
    ],
    codeRefs: ['fwd-loop-q'],
  },
  {
    id: 'flash-load-q',
    number: 9,
    indent: 2,
    parts: [
      text('Load '),
      math(String.raw`Q_i,O_i,\ell_i,m_i`),
      text(' from HBM to on-chip SRAM.'),
    ],
    codeRefs: ['fwd-load-q-state'],
  },
  {
    id: 'flash-score',
    number: 10,
    indent: 2,
    parts: [
      text('On chip, compute '),
      math(String.raw`S_{ij}=\alpha Q_iK_j^\top`),
      text(', where '),
      math(String.raw`\alpha`),
      text(' scales QK logits before softmax.'),
    ],
    codeRefs: ['fwd-score'],
  },
  {
    id: 'flash-mask',
    number: 11,
    indent: 2,
    parts: [
      text('On chip, compute '),
      math(String.raw`S_{ij}^{\mathrm{masked}}=\operatorname{MASK}(S_{ij})`, 'mask'),
      text('.'),
    ],
    codeRefs: ['fwd-mask'],
  },
  {
    id: 'flash-rowmax',
    number: 12,
    indent: 2,
    parts: [
      text('On chip, compute '),
      math(String.raw`\tilde{m}_{ij}=\operatorname{rowmax}(${latexDelta('mask', String.raw`S_{ij}^{\mathrm{masked}}`)})`),
      text(', '),
      math(String.raw`\tilde{P}_{ij}=\exp(${latexDelta('mask', String.raw`S_{ij}^{\mathrm{masked}}`)}-\tilde{m}_{ij})`),
      text(', '),
      math(String.raw`\tilde{\ell}_{ij}=\operatorname{rowsum}(\tilde{P}_{ij})`),
      text('.'),
    ],
    codeRefs: ['fwd-rowmax'],
  },
  {
    id: 'flash-update',
    number: 13,
    indent: 2,
    parts: [
      text('On chip, compute '),
      math(String.raw`m_i^{new}=\max(m_i,\tilde{m}_{ij})`),
      text(', '),
      math(String.raw`\ell_i^{new}=e^{m_i-m_i^{new}}\ell_i+e^{\tilde{m}_{ij}-m_i^{new}}\tilde{\ell}_{ij}`),
      text('.'),
    ],
    codeRefs: ['fwd-update'],
  },
  {
    id: 'flash-dropout',
    number: 14,
    indent: 2,
    parts: [
      text('On chip, compute '),
      math(String.raw`\tilde{P}_{ij}^{\mathrm{dropped}}=\operatorname{dropout}(\tilde{P}_{ij},p_{\mathrm{drop}})`, 'dropout'),
      text('.'),
    ],
    codeRefs: ['fwd-dropout'],
  },
  {
    id: 'flash-write-forward',
    number: 15,
    indent: 2,
    parts: [
      text('Write '),
      math(String.raw`O_i\leftarrow \operatorname{diag}(\ell_i^{new})^{-1}(\operatorname{diag}(\ell_i)e^{m_i-m_i^{new}}O_i+e^{\tilde{m}_{ij}-m_i^{new}}${latexDelta('dropout', String.raw`\tilde{P}_{ij}^{\mathrm{dropped}}`)}V_j)`),
      text(' to HBM.'),
    ],
    codeRefs: ['fwd-output-numer', 'fwd-store-output'],
  },
  {
    id: 'flash-write-state',
    number: 16,
    indent: 2,
    parts: [
      text('Write '),
      math(String.raw`\ell_i\leftarrow\ell_i^{new},\ m_i\leftarrow m_i^{new}`),
      text(' to HBM.'),
    ],
    codeRefs: ['fwd-write-state'],
  },
  {
    id: 'flash-end-inner',
    number: 17,
    indent: 1,
    parts: [strong('end for')],
  },
  {
    id: 'flash-end-outer',
    number: 18,
    parts: [strong('end for')],
  },
  {
    id: 'flash-return',
    number: 19,
    parts: [
      text('Return '),
      math(String.raw`O,\ell,m,${latexDelta('dropout', String.raw`\mathcal{R}`)}`),
      text('.'),
    ],
    codeRefs: ['fwd-return'],
  },
  {
    id: 'flash-bwd-label',
    parts: [text('Backward pass')],
    codeRefs: ['bwd-signature'],
  },
  {
    id: 'flash-bwd-rng',
    number: 1,
    parts: [
      text('Set the pseudo-random number generator state to ', 'dropout'),
      math(String.raw`\mathcal{R}`, 'dropout'),
      text('.', 'dropout'),
    ],
    codeRefs: ['bwd-rng'],
  },
  {
    id: 'flash-bwd-blocks',
    number: 2,
    parts: [
      text('Set block sizes '),
      math(String.raw`B_c=\lceil M/(4d)\rceil`),
      text(', '),
      math(String.raw`B_r=\min(\lceil M/(4d)\rceil,d)`),
      text('.'),
    ],
    codeRefs: ['bwd-blocks'],
  },
  {
    id: 'flash-bwd-divide-qkv',
    number: 3,
    parts: [
      text('Divide '),
      math(String.raw`Q`),
      text(' into '),
      math(String.raw`T_r`),
      text(' blocks and divide '),
      math(String.raw`K,V`),
      text(' into '),
      math(String.raw`T_c`),
      text(' blocks.'),
    ],
    codeRefs: ['bwd-loop-kv', 'bwd-load-kv', 'bwd-loop-q'],
  },
  {
    id: 'flash-bwd-divide-state',
    number: 4,
    parts: [
      text('Divide '),
      math(String.raw`O,dO,\ell,m`),
      text(' into '),
      math(String.raw`T_r`),
      text(' blocks.'),
    ],
    codeRefs: ['bwd-load-divide-state'],
  },
  {
    id: 'flash-bwd-init-grads',
    number: 5,
    parts: [
      text('Initialize '),
      math(String.raw`dQ,dK,dV`),
      text(' to zero in HBM and divide them into matching blocks.'),
    ],
    codeRefs: ['bwd-init-grads'],
  },
  {
    id: 'flash-bwd-loop-kv',
    number: 6,
    parts: [
      strong('for '),
      math(String.raw`1 \le j \le T_c`),
      strong(' do'),
    ],
    codeRefs: ['bwd-loop-kv'],
  },
  {
    id: 'flash-bwd-load-kv',
    number: 7,
    indent: 1,
    parts: [
      text('Load '),
      math(String.raw`K_j,V_j`),
      text(' from HBM to on-chip SRAM.'),
    ],
    codeRefs: ['bwd-load-kv'],
  },
  {
    id: 'flash-bwd-init-kv-grads',
    number: 8,
    indent: 1,
    parts: [
      text('Initialize '),
      math(String.raw`\tilde{dK}_j,\tilde{dV}_j`),
      text(' to zero on SRAM.'),
    ],
    codeRefs: ['bwd-init-kv-grads'],
  },
  {
    id: 'flash-bwd-loop-q',
    number: 9,
    indent: 1,
    parts: [
      strong('for '),
      math(String.raw`1 \le i \le T_r`),
      strong(' do'),
    ],
    codeRefs: ['bwd-loop-q'],
  },
  {
    id: 'flash-bwd-load-q',
    number: 10,
    indent: 2,
    parts: [
      text('Load '),
      math(String.raw`Q_i,O_i,dO_i,dQ_i,\ell_i,m_i`),
      text(' from HBM to on-chip SRAM.'),
    ],
    codeRefs: ['bwd-load-q-state'],
  },
  {
    id: 'flash-bwd-score',
    number: 11,
    indent: 2,
    parts: [
      text('On chip, compute '),
      math(String.raw`S_{ij}=\alpha Q_iK_j^\top`),
      text(', using the same QK scaling factor as the forward pass.'),
    ],
    codeRefs: ['bwd-score'],
  },
  {
    id: 'flash-bwd-mask',
    number: 12,
    indent: 2,
    parts: [
      text('On chip, compute '),
      math(String.raw`S_{ij}^{\mathrm{masked}}=\operatorname{MASK}(S_{ij})`, 'mask'),
      text('.'),
    ],
    codeRefs: ['bwd-mask'],
  },
  {
    id: 'flash-bwd-prob',
    number: 13,
    indent: 2,
    parts: [
      text('On chip, compute '),
      math(String.raw`P_{ij}=\operatorname{diag}(\ell_i)^{-1}\exp(${latexDelta('mask', String.raw`S_{ij}^{\mathrm{masked}}`)}-m_i)`),
      text('.'),
    ],
    codeRefs: ['bwd-prob'],
  },
  {
    id: 'flash-bwd-dropout-mask',
    number: 14,
    indent: 2,
    parts: [
      text('On chip, compute dropout mask ', 'dropout'),
      math(String.raw`Z_{ij}\in\mathbb{R}^{B_r\times B_c}`, 'dropout'),
      text(', where each entry is '),
      math(String.raw`\frac{1}{1-p_{\mathrm{drop}}}`, 'dropout'),
      text(' with probability '),
      math(String.raw`1-p_{\mathrm{drop}}`, 'dropout'),
      text(' and zero otherwise.'),
    ],
    codeRefs: ['bwd-dropout-mask'],
  },
  {
    id: 'flash-bwd-dropped-prob',
    number: 15,
    indent: 2,
    parts: [
      text('On chip, compute '),
      math(String.raw`P_{ij}^{\mathrm{dropped}}=P_{ij}\odot Z_{ij}`, 'dropout'),
      text('.'),
    ],
    codeRefs: ['bwd-dropped-prob'],
  },
  {
    id: 'flash-bwd-dv',
    number: 16,
    indent: 2,
    parts: [
      text('On chip, compute '),
      math(String.raw`\tilde{dV}_j\leftarrow\tilde{dV}_j+(${latexDelta('dropout', String.raw`P_{ij}^{\mathrm{dropped}}`)})^\top dO_i`),
      text('.'),
    ],
    codeRefs: ['bwd-dv'],
  },
  {
    id: 'flash-bwd-dp-dropped',
    number: 17,
    indent: 2,
    parts: [
      text('On chip, compute '),
      math(String.raw`dP_{ij}^{\mathrm{dropped}}=dO_iV_j^\top`, 'dropout'),
      text('.'),
    ],
    codeRefs: ['bwd-dp'],
  },
  {
    id: 'flash-bwd-dp',
    number: 18,
    indent: 2,
    parts: [
      text('On chip, compute '),
      math(String.raw`dP_{ij}=${latexDelta('dropout', String.raw`dP_{ij}^{\mathrm{dropped}}\odot Z_{ij}`)}`),
      text('.'),
    ],
    codeRefs: ['bwd-dp-after-dropout'],
  },
  {
    id: 'flash-bwd-d',
    number: 19,
    indent: 2,
    parts: [
      text('On chip, compute '),
      math(String.raw`D_i=\operatorname{rowsum}(dO_i\odot O_i)`),
      text('.'),
    ],
    codeRefs: ['bwd-d'],
  },
  {
    id: 'flash-bwd-ds',
    number: 20,
    indent: 2,
    parts: [
      text('On chip, compute '),
      math(String.raw`dS_{ij}=P_{ij}\odot(dP_{ij}-D_i)`),
      text('.'),
    ],
    codeRefs: ['bwd-ds'],
  },
  {
    id: 'flash-bwd-dq',
    number: 21,
    indent: 2,
    parts: [
      text('Write '),
      math(String.raw`dQ_i\leftarrow dQ_i+\alpha dS_{ij}K_j`),
      text(' to HBM.'),
    ],
    codeRefs: ['bwd-dq'],
  },
  {
    id: 'flash-bwd-dk',
    number: 22,
    indent: 2,
    parts: [
      text('On chip, compute '),
      math(String.raw`\tilde{dK}_j\leftarrow\tilde{dK}_j+\alpha dS_{ij}^\top Q_i`),
      text('.'),
    ],
    codeRefs: ['bwd-dk'],
  },
  {
    id: 'flash-bwd-end-inner',
    number: 23,
    indent: 1,
    parts: [strong('end for')],
  },
  {
    id: 'flash-bwd-write-kv',
    number: 24,
    indent: 1,
    parts: [
      text('Write '),
      math(String.raw`dK_j\leftarrow\tilde{dK}_j,\ dV_j\leftarrow\tilde{dV}_j`),
      text(' to HBM.'),
    ],
    codeRefs: ['bwd-write-kv'],
  },
  {
    id: 'flash-bwd-end-outer',
    number: 25,
    parts: [strong('end for')],
  },
  {
    id: 'flash-bwd-return',
    number: 26,
    parts: [
      text('Return '),
      math(String.raw`dQ,dK,dV`),
      text('.'),
    ],
    codeRefs: ['bwd-return'],
  },
]

const flashNoDropoutOmitRows = new Set([
  'flash-rng',
  'flash-dropout',
  'flash-bwd-rng',
  'flash-bwd-dropout-mask',
  'flash-bwd-dropped-prob',
  'flash-bwd-dp',
])

const flashNoMaskOmitRows = new Set(['flash-mask', 'flash-bwd-mask'])

function renumberAlgorithmRows(rows: AlgorithmLineSpec[]) {
  let nextNumber = 1

  return rows.map((row) => {
    if (row.id.endsWith('bwd-label')) {
      nextNumber = 1
      return row
    }

    if (row.number === undefined) {
      return row
    }

    return { ...row, number: nextNumber++ }
  })
}

function withNoDropoutRows(row: AlgorithmLineSpec): AlgorithmLineSpec {
  if (row.id === 'flash-write-forward') {
    return {
      ...row,
      parts: [
        text('Write '),
        math(String.raw`O_i\leftarrow \operatorname{diag}(\ell_i^{new})^{-1}(\operatorname{diag}(\ell_i)e^{m_i-m_i^{new}}O_i+e^{\tilde{m}_{ij}-m_i^{new}}\tilde{P}_{ij}V_j)`),
        text(' to HBM.'),
      ],
    }
  }

  if (row.id === 'flash-return') {
    return {
      ...row,
      parts: [
        text('Return '),
        math(String.raw`O,\ell,m`),
        text('.'),
      ],
    }
  }

  if (row.id === 'flash-bwd-dv') {
    return {
      ...row,
      parts: [
        text('On chip, compute '),
        math(String.raw`\tilde{dV}_j\leftarrow\tilde{dV}_j+P_{ij}^\top dO_i`),
        text('.'),
      ],
    }
  }

  if (row.id === 'flash-bwd-dp-dropped') {
    return {
      ...row,
      id: 'flash-bwd-dp-direct',
      parts: [
        text('On chip, compute '),
        math(String.raw`dP_{ij}=dO_iV_j^\top`),
        text('.'),
      ],
    }
  }

  return row
}

function withNoMaskRows(row: AlgorithmLineSpec): AlgorithmLineSpec {
  if (row.id === 'flash-rowmax') {
    return {
      ...row,
      parts: [
        text('On chip, compute '),
        math(String.raw`\tilde{m}_{ij}=\operatorname{rowmax}(S_{ij})`),
        text(', '),
        math(String.raw`\tilde{P}_{ij}=\exp(S_{ij}-\tilde{m}_{ij})`),
        text(', '),
        math(String.raw`\tilde{\ell}_{ij}=\operatorname{rowsum}(\tilde{P}_{ij})`),
        text('.'),
      ],
    }
  }

  if (row.id === 'flash-bwd-prob') {
    return {
      ...row,
      parts: [
        text('On chip, compute '),
        math(String.raw`P_{ij}=\operatorname{diag}(\ell_i)^{-1}\exp(S_{ij}-m_i)`),
        text('.'),
      ],
    }
  }

  return row
}

const flashRows: AlgorithmLineSpec[] = renumberAlgorithmRows(
  flashDropoutRows
    .filter((row) => !flashNoDropoutOmitRows.has(row.id))
    .filter((row) => !flashNoMaskOmitRows.has(row.id))
    .map(withNoDropoutRows)
    .map(withNoMaskRows)
)

const maskedFlashRows: AlgorithmLineSpec[] = renumberAlgorithmRows(
  flashDropoutRows.filter((row) => !flashNoDropoutOmitRows.has(row.id)).map(withNoDropoutRows)
)

const flashDropoutUnmaskedRows: AlgorithmLineSpec[] = renumberAlgorithmRows(
  flashDropoutRows.filter((row) => !flashNoMaskOmitRows.has(row.id)).map(withNoMaskRows)
)

export const flashAttention1Example: AttentionExample = {
  id: 'flash1',
  urlTag: 'flashattention-1',
  label: 'FlashAttention-1',
  description: 'FlashAttention-1 tiles Q, K, and V, applies online softmax, and avoids materializing the full attention matrix in HBM.',
  algorithmTitle: 'FlashAttention-1',
  content: {
    unmasked: defineAttentionContent({
      rawCode: flashAttentionCode,
      require: flashRequire,
      rows: flashRows,
      notes: flashCostNotes,
    }),
    masked: defineAttentionContent({
      rawCode: maskedFlashAttentionCode,
      require: maskedFlashRequire,
      rows: maskedFlashRows,
      notes: flashCostNotes,
    }),
  },
  dropoutContent: {
    unmasked: defineAttentionContent({
      rawCode: flashAttentionDropoutCode,
      require: flashDropoutRequire,
      rows: flashDropoutUnmaskedRows,
      notes: flashCostNotes,
    }),
    masked: defineAttentionContent({
      rawCode: maskedFlashAttentionDropoutCode,
      require: maskedFlashDropoutRequire,
      rows: flashDropoutRows,
      notes: flashCostNotes,
    }),
  },
}
