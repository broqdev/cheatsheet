import type { AlgorithmLine, AttentionExample, Segment } from '../../model'
import { math, strong, text } from '../../lib/segments'

const flashAttention3Code = ''

const row = (id: string, parts: Segment[], indent = 0): AlgorithmLine => ({
  id,
  indent,
  parts,
  codeLines: [],
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

const flash3Rows: AlgorithmLine[] = [
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
    math(String.raw`O_i=\operatorname{diag}(\exp(m_i^{\mathrm{old}}-m_i))^{-1}O_i+\tilde P_i^{(j)}V_j`),
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
    text('Using a semaphore, atomically add '),
    math(String.raw`dQ_i^{(\mathrm{local})}`),
    text(' to '),
    math(String.raw`dQ_i`),
    text(' in global memory.'),
  ], 2),
  row('flash3-bwd-dq-writer-end', [strong('end for')], 1),
  row('flash3-bwd-end-if', [strong('end if')]),
]

function withCausalRows(row: AlgorithmLine): AlgorithmLine {
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
    }
  }

  return row
}

const causalFlash3Rows = flash3Rows.map(withCausalRows)

export const flashAttention3Example: AttentionExample = {
  id: 'flash3',
  urlTag: 'flashattention-3',
  label: 'FlashAttention-3',
  description:
    'FlashAttention-3 adds warp specialization and asynchronous producer, consumer, and dQ-writer roles around exact tiled attention.',
  algorithmTitle: 'FlashAttention-3',
  content: {
    unmasked: {
      code: flashAttention3Code,
      require: flash3ForwardRequire,
      rows: flash3Rows,
    },
    masked: {
      code: flashAttention3Code,
      require: causalFlash3ForwardRequire,
      rows: causalFlash3Rows,
    },
  },
}
