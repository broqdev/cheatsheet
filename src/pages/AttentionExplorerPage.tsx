import {
  type CSSProperties,
  type KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import katex from 'katex'
import Prism from 'prismjs'
import 'prismjs/components/prism-python'
import 'katex/dist/katex.min.css'

type Segment =
  | { kind: 'text'; value: string }
  | { kind: 'math'; value: string }
  | { kind: 'strong'; value: string }

type AlgorithmLine = {
  id: string
  number?: number
  indent?: number
  parts: Segment[]
  codeLines: number[]
}

type AttentionMode = 'unmasked' | 'masked'

type LatexBlock = {
  id: string
  title: string
  require?: Segment[]
  requireLabel?: string
  rows: AlgorithmLine[]
}

type AttentionContent = {
  require: Segment[]
  rows: AlgorithmLine[]
  code: string
  notes?: LatexBlock[]
}

type AttentionExample = {
  id: string
  label: string
  algorithmTitle: string
  content: Record<AttentionMode, AttentionContent>
}

type CatalogItem = {
  id: string
  label: string
  exampleId?: string
}

type CatalogSection = {
  id: string
  label: string
  items: CatalogItem[]
}

type AlgorithmBlock = {
  id: 'forward' | 'backward'
  title: string
  require: Segment[]
  rows: AlgorithmLine[]
}

const text = (value: string): Segment => ({ kind: 'text', value })
const math = (value: string): Segment => ({ kind: 'math', value })
const strong = (value: string): Segment => ({ kind: 'strong', value })

const catalogSections: CatalogSection[] = [
  {
    id: 'flash-attention',
    label: 'FlashAttention',
    items: [
      { id: 'catalog-flash-1', label: 'FlashAttention-1', exampleId: 'flash1' },
      { id: 'catalog-flash-2', label: 'FlashAttention-2' },
    ],
  },
]

const naiveAttentionCode = `import torch

def attention_forward(Q, K, V):
    # Scale logits before softmax to keep variance stable.
    scale = 1.0 / torch.sqrt(torch.tensor(Q.shape[-1], device=Q.device))
    S = Q @ K.transpose(-2, -1) * scale
    P = torch.softmax(S, dim=-1)
    O = P @ V
    return O, P

def attention_backward(Q, K, V, P, O, dO):
    # Value gradient is the easy matrix multiply.
    dV = P.transpose(-2, -1) @ dO
    dP = dO @ V.transpose(-2, -1)

    # Rowwise softmax backward: dS = P * (dP - rowsum(P * dP)).
    dS = softmax_backward(P, dP)

    scale = 1.0 / torch.sqrt(torch.tensor(Q.shape[-1], device=Q.device))
    dQ = dS @ K * scale
    dK = dS.transpose(-2, -1) @ Q * scale
    return dQ, dK, dV

def softmax_backward(P, dP):
    # Subtract one correction scalar per row before scaling by P.
    D = (P * dP).sum(dim=-1, keepdim=True)
    return P * (dP - D)`

const maskedNaiveAttentionCode = `import torch

def attention_forward(Q, K, V, mask):
    # Additive mask is 0 for visible keys and -inf for blocked keys.
    scale = 1.0 / torch.sqrt(torch.tensor(Q.shape[-1], device=Q.device))
    S = Q @ K.transpose(-2, -1) * scale
    S = S + mask
    P = torch.softmax(S, dim=-1)
    O = P @ V
    return O, P

def attention_backward(Q, K, V, P, O, dO):
    # Masked positions already have P=0, so dS is zero there.
    dV = P.transpose(-2, -1) @ dO
    dP = dO @ V.transpose(-2, -1)

    # Rowwise softmax backward: dS = P * (dP - rowsum(P * dP)).
    dS = softmax_backward(P, dP)

    scale = 1.0 / torch.sqrt(torch.tensor(Q.shape[-1], device=Q.device))
    dQ = dS @ K * scale
    dK = dS.transpose(-2, -1) @ Q * scale
    return dQ, dK, dV

def softmax_backward(P, dP):
    # Subtract one correction scalar per row before scaling by P.
    D = (P * dP).sum(dim=-1, keepdim=True)
    return P * (dP - D)`

const flashAttentionCode = `@triton.jit
def flash_fwd(Q, K, V, O, L, N_CTX: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    q = tl.load(Q + offs_m[:, None] * STRIDE_QM + tl.arange(0, BLOCK_D)[None, :])
    # Online softmax state for one query tile.
    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K + (start_n + offs_n)[None, :] * STRIDE_KN + tl.arange(0, BLOCK_D)[:, None])
        v = tl.load(V + (start_n + offs_n)[:, None] * STRIDE_VN + tl.arange(0, BLOCK_D)[None, :])
        s = tl.dot(q, k) * SCALE
        # Update row max and denominator without materializing P.
        m_new = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m = m_new

    # Store normalized output and log-sum-exp normalizer for backward.
    tl.store(O + offs_m[:, None] * STRIDE_OM + tl.arange(0, BLOCK_D)[None, :], acc / l[:, None])
    tl.store(L + offs_m, m + tl.log(l))

@triton.jit
def flash_bwd(Q, K, V, O, dO, L, dQ, dK, dV):
    q, k, v, do = load_tiles(Q, K, V, dO)
    # Recompute probabilities from Q,K and saved log-sum-exp.
    s = tl.dot(q, tl.trans(k)) * SCALE
    p = tl.exp(s - L[:, None])
    dp = tl.dot(do, tl.trans(v))
    delta = tl.sum(do * O, axis=1)
    ds = p * (dp - delta[:, None]) * SCALE
    tl.store(dQ, tl.dot(ds, k))
    tl.store(dK, tl.dot(tl.trans(ds), q))
    tl.store(dV, tl.dot(tl.trans(p), do))`

const maskedFlashAttentionCode = `@triton.jit
def flash_fwd(Q, K, V, M, O, L, N_CTX: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    q = tl.load(Q + offs_m[:, None] * STRIDE_QM + tl.arange(0, BLOCK_D)[None, :])
    # Online softmax state for one query tile.
    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K + (start_n + offs_n)[None, :] * STRIDE_KN + tl.arange(0, BLOCK_D)[:, None])
        v = tl.load(V + (start_n + offs_n)[:, None] * STRIDE_VN + tl.arange(0, BLOCK_D)[None, :])
        s = tl.dot(q, k) * SCALE
        # Additive mask is 0 for visible keys and -inf for blocked keys.
        mask_tile = tl.load(M + offs_m[:, None] * STRIDE_MM + (start_n + offs_n)[None, :])
        s = s + mask_tile
        # Update row max and denominator without materializing P.
        m_new = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m = m_new

    # Store normalized output and log-sum-exp normalizer for backward.
    tl.store(O + offs_m[:, None] * STRIDE_OM + tl.arange(0, BLOCK_D)[None, :], acc / l[:, None])
    tl.store(L + offs_m, m + tl.log(l))

@triton.jit
def flash_bwd(Q, K, V, M, O, dO, L, dQ, dK, dV):
    q, k, v, do = load_tiles(Q, K, V, dO)
    # Recompute masked probabilities from Q,K and saved log-sum-exp.
    s = tl.dot(q, tl.trans(k)) * SCALE
    mask_tile = tl.load(M + tl.arange(0, BLOCK_M)[:, None] * STRIDE_MM + tl.arange(0, BLOCK_N)[None, :])
    s = s + mask_tile
    p = tl.exp(s - L[:, None])
    dp = tl.dot(do, tl.trans(v))
    delta = tl.sum(do * O, axis=1)
    ds = p * (dp - delta[:, None]) * SCALE
    tl.store(dQ, tl.dot(ds, k))
    tl.store(dK, tl.dot(tl.trans(ds), q))
    tl.store(dV, tl.dot(tl.trans(p), do))`

const naiveRequire = [
  text('Matrices '),
  math(String.raw`Q,K,V \in \mathbb{R}^{N \times d}`),
  text(', output gradient '),
  math(String.raw`dO`),
  text('.'),
]

const maskedNaiveRequire = [
  text('Matrices '),
  math(String.raw`Q,K,V \in \mathbb{R}^{N \times d}`),
  text(', additive mask '),
  math(String.raw`M \in \{0,-\infty\}^{N \times N}`),
  text(', output gradient '),
  math(String.raw`dO`),
  text('.'),
]

const naiveRows: AlgorithmLine[] = [
  {
    id: 'naive-forward-label',
    parts: [strong('Forward pass.')],
    codeLines: [3],
  },
  {
    id: 'naive-score',
    number: 1,
    parts: [
      text('Compute scores '),
      math(String.raw`S = QK^\top / \sqrt{d}`),
      text('.'),
    ],
    codeLines: [4, 5, 6],
  },
  {
    id: 'naive-probability',
    number: 2,
    parts: [
      text('Normalize each row '),
      math(String.raw`P_{ij}=\exp(S_{ij}) / \sum_k \exp(S_{ik})`),
      text('.'),
    ],
    codeLines: [7],
  },
  {
    id: 'naive-output',
    number: 3,
    parts: [
      text('Return weighted values '),
      math(String.raw`O = PV`),
      text('.'),
    ],
    codeLines: [8, 9],
  },
  {
    id: 'naive-backward-label',
    parts: [strong('Backward pass.')],
    codeLines: [11],
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
    codeLines: [12, 13, 14],
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
    codeLines: [16, 17, 24, 25, 26, 27],
  },
  {
    id: 'naive-dq-dk',
    number: 6,
    parts: [
      text('Return '),
      math(String.raw`dQ=dSK/\sqrt{d}`),
      text(', '),
      math(String.raw`dK=dS^\top Q/\sqrt{d}`),
      text(', and '),
      math(String.raw`dV`),
      text('.'),
    ],
    codeLines: [19, 20, 21, 22],
  },
]

const maskedNaiveRows: AlgorithmLine[] = [
  {
    id: 'naive-forward-label',
    parts: [strong('Forward pass.')],
    codeLines: [3],
  },
  {
    id: 'naive-score',
    number: 1,
    parts: [
      text('Compute masked scores '),
      math(String.raw`S = QK^\top / \sqrt{d} + M`),
      text(', where '),
      math(String.raw`M_{ij}=0`),
      text(' or '),
      math(String.raw`-\infty`),
      text('.'),
    ],
    codeLines: [4, 5, 6, 7],
  },
  {
    id: 'naive-probability',
    number: 2,
    parts: [
      text('Normalize visible keys '),
      math(String.raw`P_{ij}=\exp(S_{ij}) / \sum_k \exp(S_{ik})`),
      text(', giving '),
      math(String.raw`P_{ij}=0`),
      text(' where masked.'),
    ],
    codeLines: [8],
  },
  {
    id: 'naive-output',
    number: 3,
    parts: [
      text('Return weighted values '),
      math(String.raw`O = PV`),
      text('.'),
    ],
    codeLines: [9, 10],
  },
  {
    id: 'naive-backward-label',
    parts: [strong('Backward pass.')],
    codeLines: [12],
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
    codeLines: [13, 14, 15],
  },
  {
    id: 'naive-ds',
    number: 5,
    parts: [
      text('Backprop through masked softmax '),
      math(String.raw`D=\operatorname{rowsum}(P\odot dP)`),
      text(', '),
      math(String.raw`dS=P\odot(dP-D)`),
      text('; masked entries stay zero because '),
      math(String.raw`P_{ij}=0`),
      text('.'),
    ],
    codeLines: [17, 18, 25, 26, 27, 28],
  },
  {
    id: 'naive-dq-dk',
    number: 6,
    parts: [
      text('Return '),
      math(String.raw`dQ=dSK/\sqrt{d}`),
      text(', '),
      math(String.raw`dK=dS^\top Q/\sqrt{d}`),
      text(', and '),
      math(String.raw`dV`),
      text('.'),
    ],
    codeLines: [20, 21, 22, 23],
  },
]

const softmaxBackwardNotes = (masked: boolean): LatexBlock[] => [
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
        codeLines: masked ? [18, 25] : [17, 24],
      },
      {
        id: 'softmax-chain-rule',
        number: 2,
        parts: [
          text('Chain rule for one score entry: '),
          math(String.raw`dS_{ij}=\sum_k dP_{ik}\frac{\partial P_{ik}}{\partial S_{ij}}`),
          text('.'),
        ],
        codeLines: masked ? [18, 25, 26, 27, 28] : [17, 24, 25, 26, 27],
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
        codeLines: masked ? [25, 26, 27, 28] : [24, 25, 26, 27],
      },
      {
        id: 'softmax-substitute',
        number: 4,
        parts: [
          math(String.raw`dS_{ij}=P_{ij}dP_{ij}-P_{ij}\sum_k dP_{ik}P_{ik}`),
          text('.'),
        ],
        codeLines: masked ? [26, 27, 28] : [25, 26, 27],
      },
      {
        id: 'softmax-row-correction',
        number: 5,
        parts: [
          text('Define row correction '),
          math(String.raw`D_i=\sum_k dP_{ik}P_{ik}`),
          text('.'),
        ],
        codeLines: masked ? [26, 27] : [25, 26],
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
        codeLines: masked ? [25, 26, 27, 28] : [24, 25, 26, 27],
      },
      {
        id: 'softmax-masked-zero',
        number: 7,
        parts: [
          text(
            masked
              ? 'Masked logits get zero gradient because '
              : 'Unmasked rows apply the same row formula to every key because '
          ),
          math(masked ? String.raw`p_i=0 \Rightarrow dS_i=0` : String.raw`p_i>0`),
          text('.'),
        ],
        codeLines: masked ? [18, 28] : [17, 27],
      },
    ],
  },
]

const naiveCostNotes: LatexBlock[] = [
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
        codeLines: [],
      },
      {
        id: 'cost-naive-score',
        number: 2,
        parts: [
          text('Scores: '),
          math(String.raw`S=QK^\top\in\mathbb{R}^{N\times N}`),
          text(' has '),
          math(String.raw`N^2`),
          text(' length-'),
          math(String.raw`d`),
          text(' dot products, so '),
          math(String.raw`\Theta(N^2d)`),
          text(' FLOPs.'),
        ],
        codeLines: [],
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
        codeLines: [],
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
        codeLines: [],
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
        codeLines: [],
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
        codeLines: [],
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
        codeLines: [],
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
        codeLines: [],
      },
    ],
  },
]

const flashCostNotes: LatexBlock[] = [
  {
    id: 'flash-flops-space',
    title: 'FlashAttention-1 FLOPs and space',
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
        codeLines: [],
      },
      {
        id: 'cost-flash-tile-count',
        number: 2,
        parts: [
          text('Tile count: '),
          math(String.raw`T_rT_c=\lceil N/B_r\rceil\lceil N/B_c\rceil`),
          text(' query/key block pairs.'),
        ],
        codeLines: [],
      },
      {
        id: 'cost-flash-score-tile',
        number: 3,
        parts: [
          text('Each score tile computes '),
          math(String.raw`Q_iK_j^\top`),
          text(' with '),
          math(String.raw`\Theta(B_rB_cd)`),
          text(' work.'),
        ],
        codeLines: [],
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
        codeLines: [],
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
        codeLines: [],
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
        codeLines: [],
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
        codeLines: [],
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
        codeLines: [],
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
        codeLines: [],
      },
    ],
  },
]

const flashRequire = [
  text('Matrices '),
  math(String.raw`Q,K,V \in \mathbb{R}^{N \times d}`),
  text(' in HBM, SRAM tile budget '),
  math(String.raw`M`),
  text(', no attention mask.'),
]

const maskedFlashRequire = [
  text('Matrices '),
  math(String.raw`Q,K,V \in \mathbb{R}^{N \times d}`),
  text(' in HBM, additive mask '),
  math(String.raw`A \in \{0,-\infty\}^{N \times N}`),
  text(', SRAM tile budget '),
  math(String.raw`M`),
  text('.'),
]

const flashRows: AlgorithmLine[] = [
  {
    id: 'flash-blocks',
    number: 1,
    parts: [
      text('Set block sizes '),
      math(String.raw`B_c=\lceil M/(4d)\rceil`),
      text(', '),
      math(String.raw`B_r=\min(\lceil M/(4d)\rceil,d)`),
      text('.'),
    ],
    codeLines: [2, 3, 4],
  },
  {
    id: 'flash-init',
    number: 2,
    parts: [
      text('Initialize '),
      math(String.raw`O=(0)_{N\times d}`),
      text(', '),
      math(String.raw`\ell=(0)_N`),
      text(', '),
      math(String.raw`m=(-\infty)_N`),
      text('.'),
    ],
    codeLines: [6, 7, 8, 9],
  },
  {
    id: 'flash-loop-kv',
    number: 3,
    parts: [
      strong('for '),
      math(String.raw`1 \le j \le T_c`),
      strong(' do'),
    ],
    codeLines: [11],
  },
  {
    id: 'flash-load-kv',
    number: 4,
    indent: 1,
    parts: [
      text('Load '),
      math(String.raw`K_j,V_j`),
      text(' from HBM to on-chip SRAM.'),
    ],
    codeLines: [12, 13],
  },
  {
    id: 'flash-loop-q',
    number: 5,
    indent: 1,
    parts: [
      strong('for '),
      math(String.raw`1 \le i \le T_r`),
      strong(' do'),
    ],
    codeLines: [5],
  },
  {
    id: 'flash-load-q',
    number: 6,
    indent: 2,
    parts: [
      text('Load '),
      math(String.raw`Q_i,O_i,\ell_i,m_i`),
      text(' from HBM to SRAM.'),
    ],
    codeLines: [5, 7, 8, 9],
  },
  {
    id: 'flash-score',
    number: 7,
    indent: 2,
    parts: [
      text('On chip, compute '),
      math(String.raw`S_{ij}=Q_iK_j^\top/\sqrt{d}`),
      text('.'),
    ],
    codeLines: [14],
  },
  {
    id: 'flash-rowmax',
    number: 8,
    indent: 2,
    parts: [
      text('Compute '),
      math(String.raw`\tilde{m}_{ij}=\operatorname{rowmax}(S_{ij})`),
      text(', '),
      math(String.raw`\tilde{P}_{ij}=\exp(S_{ij}-\tilde{m}_{ij})`),
      text(', '),
      math(String.raw`\tilde{\ell}_{ij}=\operatorname{rowsum}(\tilde{P}_{ij})`),
      text('.'),
    ],
    codeLines: [15, 16, 17, 19],
  },
  {
    id: 'flash-update',
    number: 9,
    indent: 2,
    parts: [
      text('Update '),
      math(String.raw`m_i^{new}=\max(m_i,\tilde{m}_{ij})`),
      text(', '),
      math(String.raw`\ell_i^{new}=e^{m_i-m_i^{new}}\ell_i+e^{\tilde{m}_{ij}-m_i^{new}}\tilde{\ell}_{ij}`),
      text(', and accumulator.'),
    ],
    codeLines: [16, 18, 19, 20, 21],
  },
  {
    id: 'flash-write-forward',
    number: 10,
    indent: 2,
    parts: [
      text('Write '),
      math(String.raw`O_i\leftarrow \operatorname{diag}(\ell_i^{new})^{-1}acc`),
      text(' and '),
      math(String.raw`L_i=m_i+\log\ell_i`),
      text(' to HBM.'),
    ],
    codeLines: [23, 24, 25],
  },
  {
    id: 'flash-bwd-label',
    parts: [strong('Backward pass recomputes probabilities tilewise.')],
    codeLines: [27, 28],
  },
  {
    id: 'flash-bwd-recompute',
    number: 11,
    parts: [
      text('Recompute '),
      math(String.raw`S=QK^\top/\sqrt{d}`),
      text(', '),
      math(String.raw`P=\exp(S-L)`),
      text(', and '),
      math(String.raw`dP=dOV^\top`),
      text('.'),
    ],
    codeLines: [29, 30, 31, 32, 33],
  },
  {
    id: 'flash-bwd-ds',
    number: 12,
    parts: [
      text('Compute '),
      math(String.raw`\Delta_i=\sum_r dO_{ir}O_{ir}`),
      text(' and '),
      math(String.raw`dS=P\odot(dP-\Delta)/\sqrt{d}`),
      text('.'),
    ],
    codeLines: [34, 35],
  },
  {
    id: 'flash-bwd-write',
    number: 13,
    parts: [
      text('Write '),
      math(String.raw`dQ=dSK`),
      text(', '),
      math(String.raw`dK=dS^\top Q`),
      text(', '),
      math(String.raw`dV=P^\top dO`),
      text('.'),
    ],
    codeLines: [36, 37, 38],
  },
]

const maskedFlashRows: AlgorithmLine[] = [
  {
    id: 'flash-blocks',
    number: 1,
    parts: [
      text('Set block sizes '),
      math(String.raw`B_c=\lceil M/(4d)\rceil`),
      text(', '),
      math(String.raw`B_r=\min(\lceil M/(4d)\rceil,d)`),
      text('.'),
    ],
    codeLines: [2, 3, 4],
  },
  {
    id: 'flash-init',
    number: 2,
    parts: [
      text('Initialize '),
      math(String.raw`O=(0)_{N\times d}`),
      text(', '),
      math(String.raw`\ell=(0)_N`),
      text(', '),
      math(String.raw`m=(-\infty)_N`),
      text('.'),
    ],
    codeLines: [6, 7, 8, 9],
  },
  {
    id: 'flash-loop-kv',
    number: 3,
    parts: [
      strong('for '),
      math(String.raw`1 \le j \le T_c`),
      strong(' do'),
    ],
    codeLines: [11],
  },
  {
    id: 'flash-load-kv',
    number: 4,
    indent: 1,
    parts: [
      text('Load '),
      math(String.raw`K_j,V_j`),
      text(' and matching mask tile from HBM to SRAM.'),
    ],
    codeLines: [12, 13, 15, 16],
  },
  {
    id: 'flash-loop-q',
    number: 5,
    indent: 1,
    parts: [
      strong('for '),
      math(String.raw`1 \le i \le T_r`),
      strong(' do'),
    ],
    codeLines: [5],
  },
  {
    id: 'flash-load-q',
    number: 6,
    indent: 2,
    parts: [
      text('Load '),
      math(String.raw`Q_i,O_i,\ell_i,m_i`),
      text(' from HBM to SRAM.'),
    ],
    codeLines: [5, 7, 8, 9],
  },
  {
    id: 'flash-score',
    number: 7,
    indent: 2,
    parts: [
      text('On chip, compute masked scores '),
      math(String.raw`S_{ij}=Q_iK_j^\top/\sqrt{d}+A_{ij}`),
      text('.'),
    ],
    codeLines: [14, 15, 16, 17],
  },
  {
    id: 'flash-rowmax',
    number: 8,
    indent: 2,
    parts: [
      text('Compute '),
      math(String.raw`\tilde{m}_{ij}=\operatorname{rowmax}(S_{ij})`),
      text(', '),
      math(String.raw`\tilde{P}_{ij}=\exp(S_{ij}-\tilde{m}_{ij})`),
      text(', '),
      math(String.raw`\tilde{\ell}_{ij}=\operatorname{rowsum}(\tilde{P}_{ij})`),
      text('.'),
    ],
    codeLines: [18, 19, 20, 22],
  },
  {
    id: 'flash-update',
    number: 9,
    indent: 2,
    parts: [
      text('Update '),
      math(String.raw`m_i^{new}=\max(m_i,\tilde{m}_{ij})`),
      text(', '),
      math(String.raw`\ell_i^{new}=e^{m_i-m_i^{new}}\ell_i+e^{\tilde{m}_{ij}-m_i^{new}}\tilde{\ell}_{ij}`),
      text(', and accumulator.'),
    ],
    codeLines: [19, 21, 22, 23, 24],
  },
  {
    id: 'flash-write-forward',
    number: 10,
    indent: 2,
    parts: [
      text('Write '),
      math(String.raw`O_i\leftarrow \operatorname{diag}(\ell_i^{new})^{-1}acc`),
      text(' and '),
      math(String.raw`L_i=m_i+\log\ell_i`),
      text(' to HBM.'),
    ],
    codeLines: [26, 27, 28],
  },
  {
    id: 'flash-bwd-label',
    parts: [strong('Backward pass recomputes masked probabilities tilewise.')],
    codeLines: [30, 31],
  },
  {
    id: 'flash-bwd-recompute',
    number: 11,
    parts: [
      text('Recompute '),
      math(String.raw`S=QK^\top/\sqrt{d}+A`),
      text(', '),
      math(String.raw`P=\exp(S-L)`),
      text(', and '),
      math(String.raw`dP=dOV^\top`),
      text('.'),
    ],
    codeLines: [32, 33, 34, 35, 36, 37, 38],
  },
  {
    id: 'flash-bwd-ds',
    number: 12,
    parts: [
      text('Compute '),
      math(String.raw`\Delta_i=\sum_r dO_{ir}O_{ir}`),
      text(' and '),
      math(String.raw`dS=P\odot(dP-\Delta)/\sqrt{d}`),
      text(', so masked entries remain zero.'),
    ],
    codeLines: [39, 40],
  },
  {
    id: 'flash-bwd-write',
    number: 13,
    parts: [
      text('Write '),
      math(String.raw`dQ=dSK`),
      text(', '),
      math(String.raw`dK=dS^\top Q`),
      text(', '),
      math(String.raw`dV=P^\top dO`),
      text('.'),
    ],
    codeLines: [41, 42, 43],
  },
]

const examples: AttentionExample[] = [
  {
    id: 'naive',
    label: 'Naive attention',
    algorithmTitle: 'NaiveAttention',
    content: {
      unmasked: {
        code: naiveAttentionCode,
        require: naiveRequire,
        rows: naiveRows,
        notes: [...softmaxBackwardNotes(false), ...naiveCostNotes],
      },
      masked: {
        code: maskedNaiveAttentionCode,
        require: maskedNaiveRequire,
        rows: maskedNaiveRows,
        notes: [...softmaxBackwardNotes(true), ...naiveCostNotes],
      },
    },
  },
  {
    id: 'flash1',
    label: 'FlashAttention-1',
    algorithmTitle: 'FlashAttention-1',
    content: {
      unmasked: {
        code: flashAttentionCode,
        require: flashRequire,
        rows: flashRows,
        notes: flashCostNotes,
      },
      masked: {
        code: maskedFlashAttentionCode,
        require: maskedFlashRequire,
        rows: maskedFlashRows,
        notes: flashCostNotes,
      },
    },
  },
]

function renderInlineLatex(tex: string) {
  return katex.renderToString(tex, {
    displayMode: false,
    output: 'html',
    throwOnError: false,
    trust: false,
  })
}

function highlightedCodeLines(code: string) {
  const grammar = Prism.languages.python

  return code.split('\n').map((line, index) => ({
    number: index + 1,
    html: Prism.highlight(line || ' ', grammar, 'python'),
  }))
}

function labelFromParts(parts: Segment[]) {
  return parts.map((part) => part.value).join('').replace(/\.$/, '')
}

function blockRequire(
  example: AttentionExample,
  attentionMode: AttentionMode,
  content: AttentionContent,
  blockId: AlgorithmBlock['id']
) {
  if (blockId === 'forward') {
    if (example.id === 'naive' && attentionMode === 'masked') {
      return [
        text('Matrices '),
        math(String.raw`Q,K,V \in \mathbb{R}^{N \times d}`),
        text(' and additive mask '),
        math(String.raw`M \in \{0,-\infty\}^{N \times N}`),
        text('.'),
      ]
    }

    if (example.id === 'naive') {
      return [
        text('Matrices '),
        math(String.raw`Q,K,V \in \mathbb{R}^{N \times d}`),
        text('.'),
      ]
    }

    return content.require
  }

  if (example.id === 'naive') {
    return [
      text('Matrices '),
      math(String.raw`Q,K,V,O,P`),
      text(', output gradient '),
      math(String.raw`dO`),
      text('.'),
    ]
  }

  if (attentionMode === 'masked') {
    return [
      text('Forward outputs '),
      math(String.raw`O,L`),
      text(', matrices '),
      math(String.raw`Q,K,V`),
      text(', additive mask '),
      math(String.raw`A`),
      text(', output gradient '),
      math(String.raw`dO`),
      text('.'),
    ]
  }

  return [
    text('Forward output '),
    math(String.raw`(O,L)`),
    text(', input matrices, and output gradient '),
    math(String.raw`dO`),
    text('.'),
  ]
}

function algorithmBlocks(
  example: AttentionExample,
  attentionMode: AttentionMode,
  content: AttentionContent
): AlgorithmBlock[] {
  const blocks: AlgorithmBlock[] = [
    {
      id: 'forward',
      title: 'Forward pass',
      require: blockRequire(example, attentionMode, content, 'forward'),
      rows: [],
    },
  ]

  for (const row of content.rows) {
    if (row.id.endsWith('forward-label')) {
      blocks[0].title = labelFromParts(row.parts)
      continue
    }

    if (row.id.endsWith('backward-label') || row.id.endsWith('bwd-label')) {
      blocks.push({
        id: 'backward',
        title: labelFromParts(row.parts),
        require: blockRequire(example, attentionMode, content, 'backward'),
        rows: [],
      })
      continue
    }

    blocks[blocks.length - 1].rows.push(row)
  }

  return blocks.filter((block) => block.rows.length > 0)
}

function normalizeCatalogQuery(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]/g, '')
}

function fuzzyMatch(label: string, query: string) {
  const normalizedLabel = normalizeCatalogQuery(label)
  const normalizedQuery = normalizeCatalogQuery(query)

  if (!normalizedQuery) {
    return true
  }

  if (normalizedLabel.includes(normalizedQuery)) {
    return true
  }

  let queryIndex = 0

  for (const char of normalizedLabel) {
    if (char === normalizedQuery[queryIndex]) {
      queryIndex += 1
    }

    if (queryIndex === normalizedQuery.length) {
      return true
    }
  }

  return false
}

function filteredCatalogSections(query: string) {
  return catalogSections
    .map((section) => {
      const sectionMatches = fuzzyMatch(section.label, query)
      const items = section.items.filter((item) => sectionMatches || fuzzyMatch(item.label, query))

      return { ...section, items }
    })
    .filter((section) => section.items.length > 0)
}

type CatalogDropdownProps = {
  activeLabel: string
  activeExampleId: string
  onSelectExample: (exampleId: string) => void
}

function CatalogDropdown({
  activeLabel,
  activeExampleId,
  onSelectExample,
}: CatalogDropdownProps) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const rootRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const sections = useMemo(() => filteredCatalogSections(query), [query])
  const firstSelectableItem = sections.flatMap((section) => section.items).find((item) => item.exampleId)

  useEffect(() => {
    if (!open) {
      return
    }

    function handlePointerDown(event: PointerEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false)
      }
    }

    document.addEventListener('pointerdown', handlePointerDown)

    return () => document.removeEventListener('pointerdown', handlePointerDown)
  }, [open])

  function openSearch() {
    setOpen(true)
    setQuery('')
    window.setTimeout(() => inputRef.current?.focus(), 0)
  }

  function closeSearch() {
    setOpen(false)
    setQuery('')
  }

  function selectItem(item: CatalogItem) {
    if (!item.exampleId) {
      return
    }

    onSelectExample(item.exampleId)
    closeSearch()
  }

  function handleSearchKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'Escape') {
      event.preventDefault()
      closeSearch()
    }

    if (event.key === 'Enter' && firstSelectableItem) {
      event.preventDefault()
      selectItem(firstSelectableItem)
    }
  }

  return (
    <div className="catalog-picker" ref={rootRef}>
      {open ? (
        <input
          ref={inputRef}
          className="catalog-search"
          value={query}
          role="combobox"
          aria-autocomplete="list"
          aria-controls="catalog-list"
          aria-expanded="true"
          aria-label="Search cheatsheet catalog"
          placeholder="Search catalog"
          onChange={(event) => setQuery(event.currentTarget.value)}
          onKeyDown={handleSearchKeyDown}
        />
      ) : (
        <button
          type="button"
          className="catalog-trigger"
          aria-haspopup="listbox"
          aria-expanded="false"
          onClick={openSearch}
        >
          <span>{activeLabel}</span>
          <span className="catalog-caret" aria-hidden="true" />
        </button>
      )}

      {open ? (
        <div className="catalog-menu" id="catalog-list" role="listbox" aria-label="Cheatsheet catalog">
          {sections.length ? (
            sections.map((section) => (
              <div className="catalog-section" key={section.id}>
                <div className="catalog-parent">{section.label}</div>
                <div className="catalog-children">
                  {section.items.map((item) => {
                    const selected = item.exampleId === activeExampleId
                    const disabled = !item.exampleId

                    return (
                      <button
                        key={item.id}
                        type="button"
                        className={`catalog-item${selected ? ' selected' : ''}`}
                        aria-disabled={disabled}
                        aria-selected={selected}
                        disabled={disabled}
                        role="option"
                        onClick={() => selectItem(item)}
                      >
                        {item.label}
                      </button>
                    )
                  })}
                </div>
              </div>
            ))
          ) : (
            <div className="catalog-empty">No matches</div>
          )}
        </div>
      ) : null}
    </div>
  )
}

function CheatsheetSummary() {
  return (
    <p className="cheatsheet-summary">
      FlashAttention equations, masked attention, and code side by side.
    </p>
  )
}

function AttentionExplorerPage() {
  const [activeExampleId, setActiveExampleId] = useState(examples[0].id)
  const [hoveredLineId, setHoveredLineId] = useState<string | null>(null)
  const [pinnedLineId, setPinnedLineId] = useState<string | null>(null)
  const [attentionMaskEnabled, setAttentionMaskEnabled] = useState(false)

  const activeExample = examples.find((example) => example.id === activeExampleId) ?? examples[0]
  const attentionMode: AttentionMode = attentionMaskEnabled ? 'masked' : 'unmasked'
  const activeContent = activeExample.content[attentionMode]
  const activeLineId = pinnedLineId ?? hoveredLineId
  const selectableRows = [
    ...activeContent.rows,
    ...(activeContent.notes?.flatMap((note) => note.rows) ?? []),
  ]
  const activeLine = selectableRows.find((line) => line.id === activeLineId)
  const activeCodeLines = new Set(activeLine?.codeLines ?? [])
  const codeLines = useMemo(() => highlightedCodeLines(activeContent.code), [activeContent.code])
  const blocks = useMemo(
    () => algorithmBlocks(activeExample, attentionMode, activeContent),
    [activeExample, attentionMode, activeContent]
  )

  function switchExample(exampleId: string) {
    setActiveExampleId(exampleId)
    setHoveredLineId(null)
    setPinnedLineId(null)
  }

  function toggleAttentionMask(enabled: boolean) {
    setAttentionMaskEnabled(enabled)
    setHoveredLineId(null)
    setPinnedLineId(null)
  }

  function togglePinnedLine(lineId: string) {
    setPinnedLineId((current) => (current === lineId ? null : lineId))
  }

  function renderSegment(segment: Segment, index: number) {
    if (segment.kind === 'strong') {
      return <strong key={index}>{segment.value}</strong>
    }

    if (segment.kind === 'math') {
      return (
        <span
          className="algorithm-math"
          dangerouslySetInnerHTML={{ __html: renderInlineLatex(segment.value) }}
          key={index}
        />
      )
    }

    return <span key={index}>{segment.value}</span>
  }

  function renderAlgorithmRow(line: AlgorithmLine, displayNumber = line.number) {
    const isActive = activeLine?.id === line.id
    const isPinned = pinnedLineId === line.id

    return (
      <button
        key={line.id}
        type="button"
        className={`algorithm-row${isActive ? ' active' : ''}${isPinned ? ' pinned' : ''}`}
        onBlur={() => setHoveredLineId(null)}
        onClick={() => togglePinnedLine(line.id)}
        onFocus={() => setHoveredLineId(line.id)}
        onMouseEnter={() => setHoveredLineId(line.id)}
        onMouseLeave={() => setHoveredLineId(null)}
      >
        <span className="algorithm-row-number">{displayNumber ? `${displayNumber}:` : ''}</span>
        <span
          className="algorithm-row-body"
          style={{ '--indent': line.indent ?? 0 } as CSSProperties}
        >
          {line.parts.map(renderSegment)}
        </span>
      </button>
    )
  }

  return (
    <main className="workspace" aria-label="Attention equation and code explorer">
      <header className="cheatsheet-header">
        <div className="cheatsheet-title">
          <h1>Broq Cheatsheet</h1>
        </div>
        <CatalogDropdown
          activeLabel={activeExample.label}
          activeExampleId={activeExample.id}
          onSelectExample={switchExample}
        />
      </header>

      <CheatsheetSummary />

      <section className="main-panel" aria-label={`${activeExample.label} equations and code`}>
        <div className="region math-region">
          {blocks.map((block, index) => (
            <article
              className="algorithm-paper"
              key={block.id}
              aria-label={`${activeExample.label} ${block.title}`}
            >
              <header className="algorithm-header">
                <strong>Algorithm {index + 1}</strong>
                <h2>
                  {activeExample.algorithmTitle} {block.title}
                </h2>
              </header>
              <p className="algorithm-require">
                <strong>Require:</strong> {block.require.map(renderSegment)}
              </p>
              <div className="algorithm-lines">
                {block.rows.map((line, rowIndex) => renderAlgorithmRow(line, rowIndex + 1))}
              </div>
            </article>
          ))}

          {activeContent.notes?.map((note) => (
            <article
              className="algorithm-paper latex-block"
              key={note.id}
              aria-label={`${activeExample.label} ${note.title}`}
            >
              <header className="algorithm-header">
                <strong>LaTeX</strong>
                <h2>{note.title}</h2>
              </header>
              {note.require ? (
                <p className="algorithm-require">
                  <strong>{note.requireLabel ?? 'Given'}:</strong> {note.require.map(renderSegment)}
                </p>
              ) : null}
              <div className="algorithm-lines">
                {note.rows.map((line, rowIndex) => renderAlgorithmRow(line, rowIndex + 1))}
              </div>
            </article>
          ))}
        </div>

        <pre className="region code-region" aria-label={`${activeExample.label} code`}>
          <code>
            {codeLines.map((line) => (
              <span
                className={`code-line${activeCodeLines.has(line.number) ? ' active' : ''}`}
                data-line={line.number}
                key={line.number}
              >
                <span className="line-number">{line.number}</span>
                <span
                  className="line-code"
                  dangerouslySetInnerHTML={{ __html: line.html }}
                />
              </span>
            ))}
          </code>
        </pre>
      </section>

      <section className="control-strip" aria-label="Attention controls">
        <label className={`checkbox-control${attentionMaskEnabled ? ' active' : ''}`}>
          <input
            type="checkbox"
            checked={attentionMaskEnabled}
            onChange={(event) => toggleAttentionMask(event.currentTarget.checked)}
          />
          <span className="toggle-copy">Attention mask</span>
        </label>
      </section>

      <nav className="example-nav" aria-label="Attention variant selector" role="tablist">
        {examples.map((example) => (
          <button
            key={example.id}
            type="button"
            className={example.id === activeExample.id ? 'active' : ''}
            aria-current={example.id === activeExample.id ? 'page' : undefined}
            aria-selected={example.id === activeExample.id}
            onClick={() => switchExample(example.id)}
            role="tab"
          >
            {example.label}
          </button>
        ))}
      </nav>
    </main>
  )
}

export default AttentionExplorerPage
