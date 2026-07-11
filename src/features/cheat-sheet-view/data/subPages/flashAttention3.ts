import type { AttentionExample, Segment } from '../../model'
import flashAttention3Code from './code/flashAttention3Gluon.py?raw'
import causalFlashAttention3Code from './code/causalFlashAttention3Gluon.py?raw'
import flashAttention3HopperFp8Code from './code/flashAttention3GluonHopperFp8.py?raw'
import causalFlashAttention3HopperFp8Code from './code/causalFlashAttention3GluonHopperFp8.py?raw'
import { defineAttentionContent, type AlgorithmLineSpec, type LatexBlockSpec } from '../../lib/contentCompiler'
import { math, strong, text } from '../../lib/segments'

const mappedCodeRefIds = new Set([
  'flash3-cta-forward-label',
  'flash3-cta-pipeline',
  'flash3-cta-producer-registers',
  'flash3-cta-load-q',
  'flash3-cta-commit-q',
  'flash3-cta-producer-loop',
  'flash3-cta-wait-stage',
  'flash3-cta-load-kv',
  'flash3-cta-commit-kv',
  'flash3-cta-init-state',
  'flash3-cta-wait-q',
  'flash3-cta-consumer-loop',
  'flash3-cta-consumer-registers',
  'flash3-cta-wait-k',
  'flash3-cta-score',
  'flash3-cta-rowmax',
  'flash3-cta-prob-l',
  'flash3-cta-wait-v',
  'flash3-cta-output',
  'flash3-cta-consumer-end',
  'flash3-cta-normalize',
  'flash3-cta-write',
  'flash3-consumer-forward-label',
  'flash3-consumer-registers',
  'flash3-consumer-init',
  'flash3-consumer-wait-qk0',
  'flash3-consumer-score-cur',
  'flash3-consumer-release-k0',
  'flash3-consumer-online-cur',
  'flash3-consumer-loop',
  'flash3-consumer-wait-kj',
  'flash3-consumer-score-next',
  'flash3-consumer-wait-vprev',
  'flash3-consumer-output-prev',
  'flash3-consumer-wait-score-next',
  'flash3-consumer-online-next',
  'flash3-consumer-rescale-output',
  'flash3-consumer-release-buffer',
  'flash3-consumer-copy-next',
  'flash3-consumer-wait-vlast',
  'flash3-consumer-output-last',
  'flash3-consumer-epilogue',
  'flash3-backward-label',
  'flash3-bwd-preprocess',
  'flash3-bwd-partition-qkv',
  'flash3-bwd-partition-do-l',
  'flash3-bwd-load-kv',
  'flash3-bwd-commit-kv',
  'flash3-bwd-producer-loop',
  'flash3-bwd-load-q-do',
  'flash3-bwd-commit-q-do',
  'flash3-bwd-init-dk-dv',
  'flash3-bwd-wait-kv',
  'flash3-bwd-consumer-loop',
  'flash3-bwd-wait-qi',
  'flash3-bwd-load-li-di',
  'flash3-bwd-score',
  'flash3-bwd-wait-do',
  'flash3-bwd-dp',
  'flash3-bwd-prob',
  'flash3-bwd-ds',
  'flash3-bwd-dv',
  'flash3-bwd-dk',
  'flash3-bwd-dq-local',
  'flash3-bwd-dq-writer-else-if',
  'flash3-bwd-dq-writer-loop',
  'flash3-bwd-dq-ready',
  'flash3-bwd-dq-atomic',
  'flash3-bwd-dq-writer-end',
])

const causalCodeRefIds = [
  'flash3-fwd-causal-mask',
  'flash3-bwd-causal-prob',
  'flash3-bwd-dq-causal-mask',
]

const fp8CodeRefIds = [
  'flash3-fp8-dtype',
  'flash3-fp8-descale-load',
  'flash3-fp8-qk-descale',
  'flash3-fp8-v-desc',
  'flash3-fp8-v-load',
  'flash3-fp8-p-cast',
  'flash3-fp8-v-descale',
  'flash3-fp8-output-store',
]

function addCodeRefs(row: AlgorithmLineSpec, ...codeRefs: string[]): AlgorithmLineSpec {
  return {
    ...row,
    codeRefs: [...(row.codeRefs ?? []), ...codeRefs],
  }
}

const row = (id: string, parts: Segment[], indent = 0): AlgorithmLineSpec => ({
  id,
  indent,
  parts,
  codeRefs: mappedCodeRefIds.has(id) ? [id] : [],
})

const flash3ForwardRequire = [
  text('Matrices '),
  math(String.raw`Q_i \in \mathbb{R}^{B_r\times d}`),
  text(' and '),
  math(String.raw`K,V\in\mathbb{R}^{N\times d}`),
  text(' in HBM, key block size '),
  math(String.raw`B_c`),
  text(' with '),
  math(String.raw`T_c=\lceil N/B_c\rceil`),
  text(', QK scaling factor '),
  math(String.raw`\alpha\in\mathbb{R}`),
  text(' (usually '),
  math(String.raw`\alpha=1/\sqrt{d}`),
  text(').'),
]

const causalFlash3ForwardRequire = [
  text('Matrices '),
  math(String.raw`Q_i \in \mathbb{R}^{B_r\times d}`),
  text(' and '),
  math(String.raw`K,V\in\mathbb{R}^{N\times d}`),
  text(' in HBM, causal attention enabled, key block size ', 'mask'),
  math(String.raw`B_c`, 'mask'),
  text(' with '),
  math(String.raw`T_c=\lceil N/B_c\rceil`),
  text(', QK scaling factor '),
  math(String.raw`\alpha\in\mathbb{R}`),
  text(' (usually '),
  math(String.raw`\alpha=1/\sqrt{d}`),
  text(').'),
]

function withHopperFp8Require(require: Segment[]) {
  return [
    ...require.slice(0, -1),
    text(', '),
    text('Hopper FP8 inputs ', 'fp8'),
    math(String.raw`\widehat Q,\widehat K,\widehat V`, 'fp8'),
    text(' with descales ', 'fp8'),
    math(String.raw`d_Q,d_K,d_V`, 'fp8'),
    text('; scores use ', 'fp8'),
    math(String.raw`\alpha d_Q d_K`, 'fp8'),
    text(', the output epilogue uses ', 'fp8'),
    math(String.raw`d_V`, 'fp8'),
    text(', and ', 'fp8'),
    math(String.raw`\widehat V`, 'fp8'),
    text(' uses a transposed descriptor.', 'fp8'),
  ]
}

const hopperFp8Flash3ForwardRequire = withHopperFp8Require(flash3ForwardRequire)
const causalHopperFp8Flash3ForwardRequire = withHopperFp8Require(causalFlash3ForwardRequire)

const flash3HardwarePrelude: LatexBlockSpec[] = [
  {
    id: 'flash3-hardware',
    title: 'Hardware',
    require: [
      text('Hopper hardware features'),
    ],
    rows: [
      addCodeRefs(
        row('flash3-hardware-tma', [
          strong('Direct-to-SMEM transfers (TMA). '),
          text('The Tensor Memory Accelerator lets producer warp groups issue asynchronous loads that move multidimensional Q/K/V tiles from HBM directly into shared memory, bypassing the register file and avoiding scalarized memory copies.'),
        ]),
        'flash3-cta-pipeline',
        'flash3-cta-load-q',
        'flash3-cta-load-kv'
      ),
      addCodeRefs(
        row('flash3-hardware-mma', [
          strong('Tensor-core tile math (MMA). '),
          text('Matrix multiply-accumulate is the tensor-core operation that computes '),
          math(String.raw`D=A B + C`),
          text(' for matrix fragments. In FlashAttention-3, the QK score product and PV output product are the MMA-shaped tile computations that Hopper WGMMA executes.'),
        ]),
        'flash3-cta-score',
        'flash3-cta-output',
        'flash3-consumer-score-cur',
        'flash3-consumer-output-prev'
      ),
      addCodeRefs(
        row('flash3-hardware-wgmma', [
          strong('Direct SMEM consumption (WGMMA). '),
          text('Warp-Group MMA instructions let tensor cores compute matrix multiplications by reading operands straight from staged shared-memory tiles, instead of requiring warps to load those operands into registers first.'),
        ]),
        'flash3-cta-score',
        'flash3-cta-output',
        'flash3-consumer-score-cur',
        'flash3-consumer-output-prev'
      ),
      addCodeRefs(
        row('flash3-hardware-circular-buffers', [
          strong('Asynchronous circular buffers. '),
          text('The code sets up an '),
          math(String.raw`s`),
          text('-stage circular pipeline in shared memory via '),
          math(String.raw`\mathrm{SmemChannel}`),
          text(' and '),
          math(String.raw`\mathrm{gl.allocate\_shared\_memory}`),
          text(', so TMA can fetch future tiles in the background while WGMMA computes on the current tiles.'),
        ]),
        'flash3-cta-pipeline',
        'flash3-cta-producer-loop',
        'flash3-cta-wait-stage',
        'flash3-cta-consumer-loop'
      ),
    ],
  },
]

const flash3Rows: AlgorithmLineSpec[] = [
  row('flash3-cta-forward-label', [
    strong('forward pass without intra-consumer overlapping - CTA view'),
  ]),
  row('flash3-cta-pipeline', [
    text('Initialize pipeline object to manage barrier synchronization with '),
    math(String.raw`s`),
    text('-stage circular SMEM buffer.'),
  ]),
  row('flash3-cta-producer-if', [strong('if '), text('in producer warpgroup '), strong('then')]),
  row('flash3-cta-producer-registers', [
    text('Deallocate predetermined number of registers.'),
  ], 1),
  row('flash3-cta-load-q', [
    text('Issue load '),
    math(String.raw`Q_i`),
    text(' from HBM to shared memory.'),
  ], 1),
  row('flash3-cta-commit-q', [
    text('Upon completion, commit to notify consumer of the load of '),
    math(String.raw`Q_i`),
    text('.'),
  ], 1),
  row('flash3-cta-producer-loop', [
    strong('for '),
    math(String.raw`0\le j<T_c`),
    strong(' do'),
  ], 1),
  row('flash3-cta-wait-stage', [
    text('Wait for the '),
    math(String.raw`(j\bmod s)`),
    text('th stage of the buffer to be consumed.'),
  ], 2),
  row('flash3-cta-load-kv', [
    text('Issue loads of '),
    math(String.raw`K_j,V_j`),
    text(' from HBM to shared memory at the '),
    math(String.raw`(j\bmod s)`),
    text('th stage of the buffer.'),
  ], 2),
  row('flash3-cta-commit-kv', [
    text('Upon completion, commit to notify consumers of the loads of '),
    math(String.raw`K_j,V_j`),
    text('.'),
  ], 2),
  row('flash3-cta-producer-end', [strong('end for')], 1),
  row('flash3-cta-consumer-else', [strong('else')]),
  row('flash3-cta-consumer-registers', [
    text('Reallocate predetermined number of registers as function of number of consumer warps.'),
  ], 1),
  row('flash3-cta-init-state', [
    text('On-chip, initialize '),
    math(String.raw`O_i=(0)\in\mathbb{R}^{B_r\times d}`),
    text(' and '),
    math(String.raw`\ell_i,m_i=(0),(-\infty)\in\mathbb{R}^{B_r}`),
    text('.'),
  ], 1),
  row('flash3-cta-wait-q', [
    text('Wait for '),
    math(String.raw`Q_i`),
    text(' to be loaded in shared memory.'),
  ], 1),
  row('flash3-cta-consumer-loop', [
    strong('for '),
    math(String.raw`0\le j<T_c`),
    strong(' do'),
  ], 1),
  row('flash3-cta-wait-k', [
    text('Wait for '),
    math(String.raw`K_j`),
    text(' to be loaded in shared memory.'),
  ], 2),
  row('flash3-cta-score', [
    text('Compute '),
    math(String.raw`S_i^{(j)}=\alpha Q_iK_j^\top`),
    text(' (SS-GEMM). Commit and wait.'),
  ], 2),
  row('flash3-cta-rowmax', [
    text('Store '),
    math(String.raw`m_i^{\mathrm{old}}=m_i`),
    text(' and compute '),
    math(String.raw`m_i=\max(m_i^{\mathrm{old}},\operatorname{rowmax}(S_i^{(j)}))`),
    text('.'),
  ], 2),
  row('flash3-cta-prob-l', [
    text('Compute '),
    math(String.raw`\tilde P_i^{(j)}=\exp(S_i^{(j)}-m_i)`),
    text(' and '),
    math(String.raw`\ell_i=\exp(m_i^{\mathrm{old}}-m_i)\ell_i+\operatorname{rowsum}(\tilde P_i^{(j)})`),
    text('.'),
  ], 2),
  row('flash3-cta-wait-v', [
    text('Wait for '),
    math(String.raw`V_j`),
    text(' to be loaded in shared memory.'),
  ], 2),
  row('flash3-cta-output', [
    text('Compute '),
    math(String.raw`O_i=\operatorname{diag}(\exp(m_i^{\mathrm{old}}-m_i))O_i+\tilde P_i^{(j)}V_j`),
    text(' (RS-GEMM). Commit and wait.'),
  ], 2),
  row('flash3-cta-release', [
    text('Release the '),
    math(String.raw`(j\bmod s)`),
    text('th stage of the buffer for the producer.'),
  ], 2),
  row('flash3-cta-consumer-end', [strong('end for')], 1),
  row('flash3-cta-normalize', [
    text('Compute '),
    math(String.raw`O_i=\operatorname{diag}(\ell_i)^{-1}O_i`),
    text(' and '),
    math(String.raw`L_i=m_i+\log(\ell_i)`),
    text('.'),
  ], 1),
  row('flash3-cta-write', [
    text('Write '),
    math(String.raw`O_i`),
    text(' and '),
    math(String.raw`L_i`),
    text(' to HBM as the '),
    math(String.raw`i`),
    text('th block of '),
    math(String.raw`O`),
    text(' and '),
    math(String.raw`L`),
    text('.'),
  ], 1),
  row('flash3-cta-end-if', [strong('end if')]),

  row('flash3-consumer-forward-label', [strong('consumer warpgroup forward pass')]),
  row('flash3-consumer-registers', [
    text('Reallocate predetermined number of registers as function of number of consumer warps.'),
  ]),
  row('flash3-consumer-init', [
    text('On-chip, initialize '),
    math(String.raw`O_i=(0)\in\mathbb{R}^{B_r\times d}`),
    text(' and '),
    math(String.raw`\ell_i,m_i=(0),(-\infty)\in\mathbb{R}^{B_r}`),
    text('.'),
  ]),
  row('flash3-consumer-wait-qk0', [
    text('Wait for '),
    math(String.raw`Q_i`),
    text(' and '),
    math(String.raw`K_0`),
    text(' to be loaded in shared memory.'),
  ]),
  row('flash3-consumer-score-cur', [
    text('Compute '),
    math(String.raw`S_{\mathrm{cur}}=\alpha Q_iK_0^\top`),
    text(' using WGMMA. Commit and wait.'),
  ]),
  row('flash3-consumer-release-k0', [
    text('Release the 0th stage of the buffer for '),
    math(String.raw`K`),
    text('.'),
  ]),
  row('flash3-consumer-online-cur', [
    text('Compute '),
    math(String.raw`m_i,\tilde P_{\mathrm{cur}},\ell_i`),
    text(' based on '),
    math(String.raw`S_{\mathrm{cur}}`),
    text(', and rescale '),
    math(String.raw`O_i`),
    text('.'),
  ]),
  row('flash3-consumer-loop', [
    strong('for '),
    math(String.raw`1\le j<T_c-1`),
    strong(' do'),
  ]),
  row('flash3-consumer-wait-kj', [
    text('Wait for '),
    math(String.raw`K_j`),
    text(' to be loaded in shared memory.'),
  ], 1),
  row('flash3-consumer-score-next', [
    text('Compute '),
    math(String.raw`S_{\mathrm{next}}=\alpha Q_iK_j^\top`),
    text(' using WGMMA. Commit but do not wait.'),
  ], 1),
  row('flash3-consumer-wait-vprev', [
    text('Wait for '),
    math(String.raw`V_{j-1}`),
    text(' to be loaded in shared memory.'),
  ], 1),
  row('flash3-consumer-output-prev', [
    text('Compute '),
    math(String.raw`O_i=O_i+\tilde P_{\mathrm{cur}}V_{j-1}`),
    text(' using WGMMA. Commit but do not wait.'),
  ], 1),
  row('flash3-consumer-wait-score-next', [
    text('Wait for the WGMMA '),
    math(String.raw`\alpha Q_iK_j^\top`),
    text('.'),
  ], 1),
  row('flash3-consumer-online-next', [
    text('Compute '),
    math(String.raw`m_i,\tilde P_{\mathrm{next}},\ell_i`),
    text(' based on '),
    math(String.raw`S_{\mathrm{next}}`),
    text('.'),
  ], 1),
  row('flash3-consumer-rescale-output', [
    text('Wait for the WGMMA '),
    math(String.raw`\tilde P_{\mathrm{cur}}V_{j-1}`),
    text(' and then rescale '),
    math(String.raw`O_i`),
    text('.'),
  ], 1),
  row('flash3-consumer-release-buffer', [
    text('Release the '),
    math(String.raw`(j\bmod s)`),
    text('th, resp. '),
    math(String.raw`(j-1\bmod s)`),
    text('th stage of the buffer for '),
    math(String.raw`K`),
    text(', resp. '),
    math(String.raw`V`),
    text('.'),
  ], 1),
  row('flash3-consumer-copy-next', [
    text('Copy '),
    math(String.raw`S_{\mathrm{next}}`),
    text(' to '),
    math(String.raw`S_{\mathrm{cur}}`),
    text('.'),
  ], 1),
  row('flash3-consumer-end-loop', [strong('end for')]),
  row('flash3-consumer-wait-vlast', [
    text('Wait for '),
    math(String.raw`V_{T_c-1}`),
    text(' to be loaded in shared memory.'),
  ]),
  row('flash3-consumer-output-last', [
    text('Compute '),
    math(String.raw`O_i=O_i+\tilde P_{\mathrm{last}}V_{T_c-1}`),
    text(' using WGMMA. Commit and wait.'),
  ]),
  row('flash3-consumer-epilogue', [
    text('Epilogue: rescale '),
    math(String.raw`O_i`),
    text(' based on '),
    math(String.raw`m_i`),
    text('. Compute '),
    math(String.raw`L_i`),
    text(' based on '),
    math(String.raw`m_i`),
    text(' and '),
    math(String.raw`\ell_i`),
    text('. Write '),
    math(String.raw`O_i`),
    text(' and '),
    math(String.raw`L_i`),
    text(' to HBM as the '),
    math(String.raw`i`),
    text('-th block of '),
    math(String.raw`O`),
    text(' and '),
    math(String.raw`L`),
    text('.'),
  ]),

  row('flash3-backward-label', [strong('backward pass with warp specialization')]),
  row('flash3-bwd-preprocess', [
    text('In a preprocessing kernel, compute '),
    math(String.raw`D=\operatorname{rowsum}(dO\odot O)\in\mathbb{R}^{N}`),
    text(' pointwise, write '),
    math(String.raw`D`),
    text(' to HBM, and divide it into '),
    math(String.raw`T_r`),
    text(' blocks '),
    math(String.raw`D_1,\ldots,D_{T_r}`),
    text(' of size '),
    math(String.raw`B_r`),
    text(' each.'),
  ]),
  row('flash3-bwd-partition-qkv', [
    text('Divide '),
    math(String.raw`Q`),
    text(' into '),
    math(String.raw`T_r=\lceil N/B_r\rceil`),
    text(' blocks '),
    math(String.raw`Q_1,\ldots,Q_{T_r}`),
    text(' of size '),
    math(String.raw`B_r\times d`),
    text(', and divide '),
    math(String.raw`K,V`),
    text(' into '),
    math(String.raw`T_c=\lceil N/B_c\rceil`),
    text(' blocks of size '),
    math(String.raw`B_c\times d`),
    text('.'),
  ]),
  row('flash3-bwd-partition-do-l', [
    text('Divide '),
    math(String.raw`dO`),
    text(' into '),
    math(String.raw`T_r`),
    text(' blocks '),
    math(String.raw`dO_1,\ldots,dO_{T_r}`),
    text(' of size '),
    math(String.raw`B_r\times d`),
    text(', and divide '),
    math(String.raw`L`),
    text(' into '),
    math(String.raw`T_r`),
    text(' blocks '),
    math(String.raw`L_1,\ldots,L_{T_r}`),
    text(' of size '),
    math(String.raw`B_r`),
    text(' each.'),
  ]),
  row('flash3-bwd-pipeline', [
    text('Initialize pipeline object to manage barrier synchronization with '),
    math(String.raw`s`),
    text('-stage circular SMEM buffer.'),
  ]),
  row('flash3-bwd-producer-if', [strong('if '), text('in producer warpgroup '), strong('then')]),
  row('flash3-bwd-producer-registers', [
    text('Deallocate predetermined number of registers.'),
  ], 1),
  row('flash3-bwd-load-kv', [
    text('Issue load '),
    math(String.raw`K_j`),
    text(' and '),
    math(String.raw`V_j`),
    text(' from HBM to shared memory.'),
  ], 1),
  row('flash3-bwd-commit-kv', [
    text('Upon completion, commit to notify consumer of the load of '),
    math(String.raw`K_j`),
    text(' and '),
    math(String.raw`V_j`),
    text('.'),
  ], 1),
  row('flash3-bwd-producer-loop', [
    strong('for '),
    math(String.raw`1\le i\le T_r`),
    strong(' do'),
  ], 1),
  row('flash3-bwd-wait-stage', [
    text('Wait for the '),
    math(String.raw`(i\bmod s)`),
    text('th stage of the buffer to be consumed.'),
  ], 2),
  row('flash3-bwd-load-q-do', [
    text('Issue loads of '),
    math(String.raw`Q_i,dO_i`),
    text(' from HBM to shared memory at the '),
    math(String.raw`(i\bmod s)`),
    text('th stage of the buffer.'),
  ], 2),
  row('flash3-bwd-commit-q-do', [
    text('Upon completion, commit to notify consumers of the loads of '),
    math(String.raw`Q_i,dO_i`),
    text('.'),
  ], 2),
  row('flash3-bwd-producer-end', [strong('end for')], 1),
  row('flash3-bwd-consumer-else-if', [
    strong('else if '),
    text('in consumer warpgroups '),
    strong('then'),
  ]),
  row('flash3-bwd-consumer-registers', [
    text('Reallocate predetermined number of registers as function of number of consumer warps.'),
  ], 1),
  row('flash3-bwd-init-dk-dv', [
    text('On-chip, initialize '),
    math(String.raw`dK_j=(0)_{B_c\times d}`),
    text(', '),
    math(String.raw`dV_j=(0)_{B_c\times d}`),
    text('.'),
  ], 1),
  row('flash3-bwd-wait-kv', [
    text('Wait for '),
    math(String.raw`K_j`),
    text(' and '),
    math(String.raw`V_j`),
    text(' to be loaded in shared memory.'),
  ], 1),
  row('flash3-bwd-consumer-loop', [
    strong('for '),
    math(String.raw`1\le i\le T_r`),
    strong(' do'),
  ], 1),
  row('flash3-bwd-wait-qi', [
    text('Wait for '),
    math(String.raw`Q_i`),
    text(' to be loaded in shared memory.'),
  ], 2),
  row('flash3-bwd-load-li-di', [
    text('Load '),
    math(String.raw`L_i,D_i`),
    text(' from HBM to on-chip SRAM.'),
  ], 2),
  row('flash3-bwd-score', [
    text('On chip, compute '),
    math(String.raw`S_i^{(j)}=\alpha Q_iK_j^\top\in\mathbb{R}^{B_r\times B_c}`),
    text(' (SS-GEMM). Commit.'),
  ], 2),
  row('flash3-bwd-wait-do', [
    text('Wait for '),
    math(String.raw`dO_i`),
    text(' to be loaded in shared memory.'),
  ], 2),
  row('flash3-bwd-dp', [
    text('On chip, compute '),
    math(String.raw`dP_i^{(j)}=dO_iV_j^\top\in\mathbb{R}^{B_r\times B_c}`),
    text(' (SS-GEMM). Commit.'),
  ], 2),
  row('flash3-bwd-prob', [
    text('On chip, wait for '),
    math(String.raw`S_i^{(j)}`),
    text(', then compute '),
    math(String.raw`P_i^{(j)}=\exp(S_i^{(j)}-L_i)\in\mathbb{R}^{B_r\times B_c}`),
    text('.'),
  ], 2),
  row('flash3-bwd-ds', [
    text('On chip, wait for '),
    math(String.raw`dP_i^{(j)}`),
    text(', then compute '),
    math(String.raw`dS_i^{(j)}=P_i^{(j)}\odot(dP_i^{(j)}-D_i)\in\mathbb{R}^{B_r\times B_c}`),
    text('.'),
  ], 2),
  row('flash3-bwd-dv', [
    text('On chip, compute '),
    math(String.raw`dV_j\leftarrow dV_j+(P_i^{(j)})^\top dO_i\in\mathbb{R}^{B_c\times d}`),
    text(' (RS-GEMM). Commit.'),
  ], 2),
  row('flash3-bwd-dk', [
    text('On chip, compute '),
    math(String.raw`dK_j\leftarrow dK_j+\alpha(dS_i^{(j)})^\top Q_i\in\mathbb{R}^{B_c\times d}`),
    text(' (RS-GEMM). Commit and wait for both '),
    math(String.raw`dV_j`),
    text(' and '),
    math(String.raw`dK_j`),
    text('.'),
  ], 2),
  row('flash3-bwd-dq-local', [
    text('On chip, compute '),
    math(String.raw`dQ_i^{(\mathrm{local})}=\alpha dS_i^{(j)}K_j\in\mathbb{R}^{B_r\times d}`),
    text(' (SS-GEMM), write '),
    math(String.raw`dQ_i^{(\mathrm{local})}`),
    text(' to SMEM, and notify the '),
    math(String.raw`dQ`),
    text('-writer.'),
  ], 2),
  row('flash3-bwd-consumer-end', [strong('end for')], 1),
  row('flash3-bwd-dq-writer-else-if', [
    strong('else if '),
    text('in '),
    math(String.raw`dQ`),
    text('-writer warp '),
    strong('then'),
  ]),
  row('flash3-bwd-dq-writer-loop', [
    strong('for '),
    math(String.raw`1\le i\le T_r`),
    strong(' do'),
  ], 1),
  row('flash3-bwd-dq-ready', [
    text('Wait for '),
    math(String.raw`dQ_i^{(\mathrm{local})}`),
    text(' to be ready in SMEM.'),
  ], 2),
  row('flash3-bwd-dq-atomic', [
    text('The '),
    math(String.raw`dQ`),
    text('-writer stores the first partial for each '),
    math(String.raw`dQ_i`),
    text(' and atomically accumulates later key-block partials.'),
  ], 2),
  row('flash3-bwd-dq-writer-end', [strong('end for')], 1),
  row('flash3-bwd-end-if', [strong('end if')]),
]

function withCausalRows(row: AlgorithmLineSpec): AlgorithmLineSpec {
  if (row.id === 'flash3-cta-score') {
    return {
      ...row,
      parts: [
        text('Compute '),
        math(String.raw`S_i^{(j)}=\alpha Q_iK_j^\top+M_{ij}^{\mathrm{causal}}`, 'mask'),
        text(' (SS-GEMM), where future keys receive ', 'mask'),
        math(String.raw`-\infty`, 'mask'),
        text('. Commit and wait.'),
      ],
      codeRefs: ['flash3-cta-score', 'flash3-fwd-causal-mask'],
    }
  }

  if (row.id === 'flash3-consumer-score-cur') {
    return {
      ...row,
      parts: [
        text('Compute '),
        math(String.raw`S_{\mathrm{cur}}=\alpha Q_iK_0^\top+M_{i0}^{\mathrm{causal}}`, 'mask'),
        text(' using WGMMA. Commit and wait.'),
      ],
      codeRefs: ['flash3-consumer-score-cur', 'flash3-fwd-causal-mask'],
    }
  }

  if (row.id === 'flash3-consumer-score-next') {
    return {
      ...row,
      parts: [
        text('Compute '),
        math(String.raw`S_{\mathrm{next}}=\alpha Q_iK_j^\top+M_{ij}^{\mathrm{causal}}`, 'mask'),
        text(' using WGMMA. Commit but do not wait.'),
      ],
      codeRefs: ['flash3-consumer-score-next', 'flash3-fwd-causal-mask'],
    }
  }

  if (row.id === 'flash3-consumer-wait-score-next') {
    return {
      ...row,
      parts: [
        text('Wait for the WGMMA '),
        math(String.raw`\alpha Q_iK_j^\top+M_{ij}^{\mathrm{causal}}`, 'mask'),
        text('.'),
      ],
      codeRefs: ['flash3-consumer-wait-score-next', 'flash3-fwd-causal-mask'],
    }
  }

  if (row.id === 'flash3-bwd-score') {
    return {
      ...row,
      parts: [
        text('On chip, compute '),
        math(String.raw`S_i^{(j)}=\alpha Q_iK_j^\top+M_{ij}^{\mathrm{causal}}\in\mathbb{R}^{B_r\times B_c}`, 'mask'),
        text(' (SS-GEMM). Commit.'),
      ],
      codeRefs: ['flash3-bwd-score', 'flash3-bwd-causal-prob', 'flash3-bwd-dq-causal-mask'],
    }
  }

  if (row.id === 'flash3-bwd-prob') {
    return {
      ...row,
      parts: [
        text('On chip, wait for '),
        math(String.raw`S_i^{(j)}`),
        text(', then compute '),
        math(String.raw`P_i^{(j)}=\exp(S_i^{(j)}-L_i)`, 'mask'),
        text('; masked future positions stay zero.', 'mask'),
      ],
      codeRefs: ['flash3-bwd-prob', 'flash3-bwd-causal-prob', 'flash3-bwd-dq-causal-mask'],
    }
  }

  if (row.id === 'flash3-bwd-dq-local') {
    return addCodeRefs(row, 'flash3-bwd-dq-causal-mask')
  }

  return row
}

const causalFlash3Rows = flash3Rows.map(withCausalRows)

function hasCausalMask(row: AlgorithmLineSpec) {
  return row.codeRefs?.includes('flash3-fwd-causal-mask') ?? false
}

function fp8ScoreFormula(rowId: string, causal: boolean) {
  const mask = causal
    ? rowId === 'flash3-consumer-score-cur'
      ? String.raw`+M_{i0}^{\mathrm{causal}}`
      : String.raw`+M_{ij}^{\mathrm{causal}}`
    : ''

  if (rowId === 'flash3-consumer-score-cur') {
    return String.raw`S_{\mathrm{cur}}=\alpha d_Q d_K\widehat Q_i\widehat K_0^\top${mask}`
  }

  if (rowId === 'flash3-consumer-score-next' || rowId === 'flash3-consumer-wait-score-next') {
    return String.raw`S_{\mathrm{next}}=\alpha d_Q d_K\widehat Q_i\widehat K_j^\top${mask}`
  }

  return String.raw`S_i^{(j)}=\alpha d_Q d_K\widehat Q_i\widehat K_j^\top${mask}`
}

function withHopperFp8Rows(row: AlgorithmLineSpec): AlgorithmLineSpec {
  if (row.id === 'flash3-cta-init-state' || row.id === 'flash3-consumer-init') {
    return addCodeRefs(
      {
        ...row,
        parts: [
          ...row.parts,
          text(' Load FP8 descales ', 'fp8'),
          math(String.raw`d_Q,d_K,d_V`, 'fp8'),
          text(' and set the effective score scale to ', 'fp8'),
          math(String.raw`\alpha d_Q d_K`, 'fp8'),
          text('.', 'fp8'),
        ],
      },
      'flash3-fp8-descale-load',
      'flash3-fp8-qk-descale'
    )
  }

  if (row.id === 'flash3-cta-load-kv') {
    return addCodeRefs(
      {
        ...row,
        parts: [
          text('Issue loads of '),
          math(String.raw`\widehat K_j`, 'fp8'),
          text(' and '),
          text('Hopper FP8 ', 'fp8'),
          math(String.raw`\widehat V_j`, 'fp8'),
          text(' using a transposed ', 'fp8'),
          math(String.raw`[d,N]`, 'fp8'),
          text(' descriptor at the '),
          math(String.raw`(j\bmod s)`),
          text('th stage of the buffer.'),
        ],
      },
      'flash3-fp8-v-desc',
      'flash3-fp8-v-load'
    )
  }

  if (
    row.id === 'flash3-cta-score'
    || row.id === 'flash3-consumer-score-cur'
    || row.id === 'flash3-consumer-score-next'
  ) {
    return addCodeRefs(
      {
        ...row,
        parts: [
          text('Compute '),
          math(fp8ScoreFormula(row.id, hasCausalMask(row)), 'fp8'),
          text(row.id === 'flash3-cta-score' ? ' (SS-GEMM). Commit and wait.' : ' using WGMMA. Commit'),
          ...(row.id === 'flash3-consumer-score-next' ? [text(' but do not wait.')] : row.id === 'flash3-consumer-score-cur' ? [text(' and wait.')] : []),
        ],
      },
      'flash3-fp8-qk-descale'
    )
  }

  if (row.id === 'flash3-consumer-wait-score-next') {
    return addCodeRefs(
      {
        ...row,
        parts: [
          text('Wait for the WGMMA '),
          math(fp8ScoreFormula(row.id, hasCausalMask(row)), 'fp8'),
          text('.'),
        ],
      },
      'flash3-fp8-qk-descale'
    )
  }

  if (row.id === 'flash3-cta-output') {
    return addCodeRefs(
      {
        ...row,
        parts: [
          text('Cast '),
          math(String.raw`\tilde P_i^{(j)}`, 'fp8'),
          text(' to ', 'fp8'),
          math(String.raw`\mathrm{gl.float8e5}`, 'fp8'),
          text(' and compute '),
          math(String.raw`O_i=\operatorname{diag}(\exp(m_i^{\mathrm{old}}-m_i))O_i+\tilde P_i^{(j)}\widehat V_j`, 'fp8'),
          text(' (RS-GEMM). Commit and wait; ', 'fp8'),
          math(String.raw`d_V`, 'fp8'),
          text(' is applied in the epilogue.', 'fp8'),
        ],
      },
      'flash3-fp8-dtype',
      'flash3-fp8-p-cast'
    )
  }

  if (row.id === 'flash3-cta-normalize') {
    return addCodeRefs(
      {
        ...row,
        parts: [
          text('Compute '),
          math(String.raw`O_i=d_V\operatorname{diag}(\ell_i)^{-1}O_i`, 'fp8'),
          text(' and '),
          math(String.raw`L_i=m_i+\log(\ell_i)`),
          text('.'),
        ],
      },
      'flash3-fp8-v-descale'
    )
  }

  if (row.id === 'flash3-cta-write') {
    return addCodeRefs(
      {
        ...row,
        parts: [
          text('Write '),
          math(String.raw`O_i`, 'fp8'),
          text(' from the Hopper FP8 path and write ', 'fp8'),
          math(String.raw`L_i`),
          text(' to HBM as the '),
          math(String.raw`i`),
          text('th block of '),
          math(String.raw`O`),
          text(' and '),
          math(String.raw`L`),
          text('.'),
        ],
      },
      'flash3-fp8-output-store'
    )
  }

  if (row.id === 'flash3-consumer-wait-vprev' || row.id === 'flash3-consumer-wait-vlast') {
    return addCodeRefs(row, 'flash3-fp8-v-desc', 'flash3-fp8-v-load')
  }

  if (row.id === 'flash3-consumer-output-prev' || row.id === 'flash3-consumer-output-last') {
    return addCodeRefs(
      {
        ...row,
        parts: [
          text('Cast the probability tile to ', 'fp8'),
          math(String.raw`\mathrm{gl.float8e5}`, 'fp8'),
          text(' and compute the FP8 output update with ', 'fp8'),
          math(row.id === 'flash3-consumer-output-prev'
            ? String.raw`\tilde P_{\mathrm{cur}}\widehat V_{j-1}`
            : String.raw`\tilde P_{\mathrm{last}}\widehat V_{T_c-1}`, 'fp8'),
          text('; ', 'fp8'),
          math(String.raw`d_V`, 'fp8'),
          text(' waits for the epilogue.', 'fp8'),
        ],
      },
      'flash3-fp8-dtype',
      'flash3-fp8-p-cast'
    )
  }

  if (row.id === 'flash3-consumer-epilogue') {
    return addCodeRefs(
      {
        ...row,
        parts: [
          text('Epilogue: rescale '),
          math(String.raw`O_i`),
          text(', apply ', 'fp8'),
          math(String.raw`d_V`, 'fp8'),
          text(', compute '),
          math(String.raw`L_i`),
          text(', and write '),
          math(String.raw`O_i`, 'fp8'),
          text(' from the Hopper FP8 path while ', 'fp8'),
          math(String.raw`L_i`),
          text(' stays in HBM.'),
        ],
      },
      'flash3-fp8-v-descale',
      'flash3-fp8-output-store'
    )
  }

  return row
}

const hopperFp8Flash3Rows = flash3Rows.map(withHopperFp8Rows)
const causalHopperFp8Flash3Rows = causalFlash3Rows.map(withHopperFp8Rows)

export const flashAttention3Example: AttentionExample = {
  id: 'flash3',
  urlTag: 'flash-attention-3',
  label: 'FlashAttention-3',
  description:
    'FlashAttention-3 refines exact tiled attention for Hopper with stronger overlap and faster on-chip execution.',
  algorithmTitle: 'FlashAttention-3',
  content: {
    unmasked: defineAttentionContent({
      rawCode: flashAttention3Code,
      require: flash3ForwardRequire,
      prelude: flash3HardwarePrelude,
      rows: flash3Rows,
      ignoredUnusedRefs: [...causalCodeRefIds, ...fp8CodeRefIds],
    }),
    masked: defineAttentionContent({
      rawCode: causalFlashAttention3Code,
      require: causalFlash3ForwardRequire,
      prelude: flash3HardwarePrelude,
      rows: causalFlash3Rows,
      ignoredUnusedRefs: fp8CodeRefIds,
    }),
  },
  variants: [
    {
      enabled: ['fp8'],
      content: {
        unmasked: defineAttentionContent({
          rawCode: flashAttention3HopperFp8Code,
          require: hopperFp8Flash3ForwardRequire,
          prelude: flash3HardwarePrelude,
          rows: hopperFp8Flash3Rows,
          ignoredUnusedRefs: causalCodeRefIds,
        }),
        masked: defineAttentionContent({
          rawCode: causalFlashAttention3HopperFp8Code,
          require: causalHopperFp8Flash3ForwardRequire,
          prelude: flash3HardwarePrelude,
          rows: causalHopperFp8Flash3Rows,
        }),
      },
    },
  ],
  variantLabels: { mask: 'Causal Attention' },
}
