import type { AttentionExample, Segment } from '../../model'
import { defineAttentionContent, type AlgorithmLineSpec, type LatexBlockSpec } from '../../lib/codeRefs'
import { math, strong, text } from '../../lib/segments'
import flashAttention4CuteDslCode from './code/flashAttention4CuteDsl.py?raw'

const row = (
  id: string,
  parts: Segment[],
  codeRefs: string[],
  indent = 0
): AlgorithmLineSpec => ({
  id,
  indent,
  parts,
  codeRefs,
})

const causalCodeRefIds = ['flash4-fwd-causal-mask']

function addCodeRefs(row: AlgorithmLineSpec, ...codeRefs: string[]): AlgorithmLineSpec {
  return {
    ...row,
    codeRefs: [...(row.codeRefs ?? []), ...codeRefs],
  }
}

const flash4MmaRequire = [
  text('FP8 Q/K/V/O tensors, transposed V, SM100 CuTe DSL TMA descriptors, '),
  math(String.raw`tcgen05`),
  text(' MMA atoms, two-CTA cluster layout, tensor-memory score/probability/output tiles, and specialized load/MMA/softmax/correction warp groups.'),
]

const causalFlash4MmaRequire = [
  text('FP8 Q/K/V/O tensors, causal attention enabled, transposed V, SM100 CuTe DSL TMA descriptors, '),
  math(String.raw`tcgen05`),
  text(' MMA atoms, two-CTA cluster layout, tensor-memory score/probability/output tiles, and specialized load/MMA/softmax/correction warp groups.'),
]

const flash4MmaRows: AlgorithmLineSpec[] = [
  row('flash4-mma-forward-label', [strong('MMA ops in forward')], [
    'flash4-mma-atoms',
    'flash4-mma-operands',
  ]),
  row('flash4-mma-meaning-row', [
    strong('MMA means matrix multiply-accumulate. '),
    text('At the hardware level, it is a tensor-core tile operation that computes '),
    math(String.raw`D=A B + C`),
    text(' for small matrix fragments. In this code, '),
    math(String.raw`cute.gemm`),
    text(' is the CuTe DSL call that drives those tiled MMA atoms.'),
  ], ['flash4-mma-meaning', 'flash4-fwd-qk-mma', 'flash4-fwd-pv-mma'], 1),
  row('flash4-wgmma-meaning-row', [
    strong('WGMMA means warp-group MMA. '),
    text('It is Hopper\'s form of MMA where a group of warps cooperates on one tensor-core matrix instruction. FlashAttention-3 targets Hopper WGMMA; this FlashAttention-4 sketch targets Blackwell '),
    math(String.raw`tcgen05`),
    text(' / UMMA-style atoms instead, with tensor-memory accumulators and two-CTA support.'),
  ], ['flash4-wgmma-vs-umma', 'flash4-mma-atoms', 'flash4-2cta-cgroup'], 1),
  row('flash4-mma-contract', [
    strong('SM100 MMA replaces the older GEMM center. '),
    text('Compared with FA2 block matmuls and FA3 Hopper WGMMA, this sketch names Blackwell '),
    math(String.raw`tcgen05`),
    text(' MMA atoms, operand major modes, warp-group ownership, and tensor-memory accumulator placement.'),
  ], ['flash4-mma-atoms'], 1),
  row('flash4-mma-operands', [
    strong('Forward is two explicit MMA products. '),
    text('QK consumes shared-memory Q/K operands; PV consumes probabilities from tensor memory plus a transposed V operand laid out for MN-major access.'),
  ], ['flash4-mma-operands', 'flash4-fp8-only-assert', 'flash4-fp8-mma-atoms'], 1),
  row('flash4-mma-fp8-path', [
    strong('The code path is FP8-first. '),
    text('Q/K/V/O are checked as 8-bit operands, MMA atoms inherit FP8 element types, and the probability tile is cast back to FP8 before PV.'),
  ], ['flash4-fp8-only-assert', 'flash4-fp8-mma-atoms', 'flash4-fp8-p-cast'], 1),
  row('flash4-mma-schedule', [
    strong('The pipeline is Blackwell-specialized. '),
    text('TMA feeds Q/K/V, the MMA warp consumes staged tiles, softmax rewrites score tiles in tensor memory, and correction warps keep the online output accumulator coherent.'),
  ], ['flash4-pipelines', 'flash4-tmem', 'flash4-warp-specialization'], 1),
  row('flash4-mma-qk', [
    strong('QK MMA writes scores to tensor memory. '),
    text('The first '),
    math(String.raw`cute.gemm`),
    text(' drives Q/K fragments through the QK tiled MMA and lands '),
    math(String.raw`S`),
    text(' where softmax warps can read it.'),
  ], ['flash4-fwd-qk-mma'], 1),
  row('flash4-mma-softmax-bridge', [
    strong('Softmax is the tensor-memory bridge. '),
    text('Softmax warps load '),
    math(String.raw`S`),
    text(' from tensor memory, apply scale/mask/online normalization, and store FP8 probabilities back in the PV operand layout.'),
  ], ['flash4-softmax-bridge'], 1),
  row('flash4-mma-pv', [
    strong('PV MMA accumulates O in tensor memory. '),
    text('The second '),
    math(String.raw`cute.gemm`),
    text(' treats '),
    math(String.raw`P`),
    text(' as the A operand from tensor memory and transposed V as the B operand from shared memory.'),
  ], ['flash4-fwd-pv-mma', 'flash4-correction-bridge'], 1),

  row('flash4-mma-backward-label', [strong('MMA ops in backward')], [
    'flash4-bwd-mma-atoms',
  ]),
  row('flash4-bwd-contract', [
    strong('Backward is decomposed into MMA products. '),
    text('Separate tiled MMA atoms rebuild scores, form dP, accumulate dK/dV, and accumulate dQ without materializing the full attention matrix.'),
  ], ['flash4-bwd-mma-atoms'], 1),
  row('flash4-bwd-recompute', [
    strong('Recompute scores on chip. '),
    text('A QK-style MMA rebuilds score tiles in the backward path, preserving the FlashAttention recompute strategy while moving the math to SM100 MMA atoms.'),
  ], ['flash4-bwd-score-mma'], 1),
  row('flash4-bwd-dp', [
    strong('dP uses the same matrix engine. '),
    text('The dO/V product lands in tensor memory beside the score tile, ready for the softmax derivative.'),
  ], ['flash4-bwd-dp-mma'], 1),
  row('flash4-bwd-dsoftmax', [
    strong('dS is the scalar bridge between MMAs. '),
    text('Score, dP, and row statistics combine into a dS tile before the next tensor-core products consume it.'),
  ], ['flash4-bwd-dsoftmax'], 1),
  row('flash4-bwd-dkv', [
    strong('dK and dV reuse tensor-memory operands. '),
    text('One MMA accumulates dV from P and dO; another accumulates dK from dS and Q, then the epilogue drains those accumulators.'),
  ], ['flash4-bwd-dkv-mma', 'flash4-bwd-epilogue'], 1),
  row('flash4-bwd-dq', [
    strong('dQ is the final query-side MMA. '),
    text('A dS/K product accumulates dQ, and the epilogue stores the tensor-memory result back to global memory.'),
  ], ['flash4-bwd-dq-mma', 'flash4-bwd-epilogue'], 1),

  row('flash4-2cta-forward-label', [strong('2-CTA cluster forward pass')], [
    'flash4-2cta-shape',
    'flash4-2cta-cgroup',
  ]),
  row('flash4-2cta-define-row', [
    strong('CTA means cooperative thread array. '),
    text('In CUDA terms, it is one thread block: a fixed group of warps with CTA-local shared memory, named barriers, and pipeline state. This sketch assigns softmax, correction, MMA, load, and empty warps inside each CTA.'),
  ], ['flash4-cta-thread-block'], 1),
  row('flash4-2cta-shape-row', [
    strong('Forward binds two CTAs as one M tile. '),
    text('The cluster shape is '),
    math(String.raw`(2,1,1)`),
    text(', so two CTA ranks cooperate along the M dimension, and '),
    math(String.raw`\mathrm{tcgen05.CtaGroup.TWO}`),
    text(' lets their MMA slices act as one larger logical tile.'),
  ], ['flash4-2cta-shape', 'flash4-2cta-cgroup', 'flash4-schedule-create'], 1),
  row('flash4-2cta-tma-row', [
    strong('TMA becomes cluster-aware. '),
    text('The copy atoms receive the cluster layout, so Q/K/V land in the shared-memory partitions expected by both CTAs; byte counts follow the MMA thread-id shape.'),
  ], ['flash4-2cta-cluster-layout', 'flash4-2cta-tma', 'flash4-2cta-copy-bytes'], 1),
  row('flash4-2cta-q-stage-row', [
    strong('HD256 stages the head dimension explicitly. '),
    text('For the dedicated 2-CTA HD256 path, '),
    math(String.raw`\mathrm{iterations}_{QK}=2`),
    text(' and the load warp issues matching Q/K and V slice handles. The compact HD128 branch is intentionally single-Q-stage, rather than the generic upstream two-M-stage pipeline.'),
  ], ['flash4-q-stage-policy', 'flash4-q-staging-hd256', 'flash4-kv-staging-hd256'], 1),
  row('flash4-2cta-pipeline-row', [
    strong('Cluster barriers define ownership. '),
    text('Producer/consumer barriers pass '),
    math(String.raw`\mathrm{cta\_layout\_vmnk}`),
    text(' so load, MMA, softmax, and correction warp groups agree when the peer CTA can consume data.'),
  ], ['flash4-2cta-pipelines'], 1),
  row('flash4-2cta-rank-row', [
    strong('CTA rank chooses the slice. '),
    text('The kernel reads '),
    math(String.raw`\mathrm{block\_idx\_in\_cluster}`),
    text(' and derives '),
    math(String.raw`\mathrm{mma\_tile\_coord\_v}`),
    text(' so the two CTAs partition QK and PV fragments instead of redundantly computing the same slice.'),
  ], ['flash4-2cta-rank'], 1),
  row('flash4-2cta-tmem-row', [
    strong('Tensor memory lifetime is cluster-scoped. '),
    text('The allocator is opened with '),
    math(String.raw`\mathrm{is\_two\_cta}=\mathrm{True}`),
    text(' and freed only after cluster-wide arrive/wait confirms both CTAs have finished with the shared tensor-memory allocation.'),
  ], ['flash4-2cta-tmem', 'flash4-2cta-free'], 1),

  row('flash4-2cta-backward-label', [strong('2-CTA cluster backward pass')], [
    'flash4-2cta-bwd-traffic',
    'flash4-2cta-bwd-split-kernels',
    'flash4-2cta-bwd-dq-store',
  ]),
  row('flash4-2cta-bwd-traffic-row', [
    strong('Backward is shared-memory-bound. '),
    text('The FA4 paper notes that, even after TMEM staging, most backward GEMM operands still come from shared memory; 2-CTA MMA targets that bottleneck directly.'),
  ], ['flash4-2cta-bwd-traffic', 'flash4-bwd-mma-atoms'], 1),
  row('flash4-2cta-bwd-operand-b-row', [
    strong('Operand-B traffic is cut by the CTA pair. '),
    text('With an M=256, N=K=128 MMA tile, the two CTAs behave as one larger tile: each CTA stages half of operand B and keeps its own accumulator slice.'),
  ], ['flash4-2cta-bwd-traffic', 'flash4-bwd-mma-atoms'], 1),
  row('flash4-2cta-bwd-dq-reduction-row', [
    strong('dQ differs by implementation path. '),
    text('The paper/generic 2-CTA idea repacks dS through DSMEM for the query-side reduction. Current upstream HD256 code instead launches a dedicated dQ kernel and a separate dK/dV kernel.'),
  ], ['flash4-2cta-bwd-dsmem', 'flash4-bwd-dq-mma', 'flash4-2cta-bwd-split-kernels'], 1),
  row('flash4-2cta-bwd-pipeline-row', [
    strong('The inner dQ pipeline still overlaps work. '),
    text('Inside the dedicated dQ kernel, dP and dQ MMA work are ordered to reuse tensor-memory space, but the HD256 wrapper is not one monolithic backward kernel.'),
  ], ['flash4-2cta-bwd-pipeline', 'flash4-bwd-dsoftmax', 'flash4-bwd-dq-mma', 'flash4-2cta-bwd-split-kernels'], 1),
  row('flash4-2cta-bwd-dq-store-row', [
    strong('Dedicated HD256 dQ is a direct store path. '),
    text('The wrapper asserts that dQ/dK/dV semaphores are absent, treats the dQ_accum slot as the dQ output, launches dQ first, and lets the dQ epilogue store the result directly.'),
  ], ['flash4-2cta-bwd-split-kernels', 'flash4-2cta-bwd-dq-store', 'flash4-bwd-epilogue'], 1),
]

const flash4IdeaNotes: LatexBlockSpec[] = [
  {
    id: 'flash4-exp-emulation-note',
    title: 'Exponential function emulation, partial emulation, MUFU.EX2',
    require: [
      text('Online softmax in base-2 form, FP8 probabilities, and SM100 special-function hardware.'),
    ],
    rows: [
      row('flash4-exp-base2-row', [
        strong('Rewrite the softmax exponential as base-2 work. '),
        text('The natural exponential is represented as '),
        math(String.raw`e^x=2^{x\log_2 e}`),
        text(', so '),
        math(String.raw`\mathrm{scale\_softmax\_log2}`),
        text(' carries the conversion before the row fragment reaches '),
        math(String.raw`\operatorname{exp2}`),
        text('.'),
      ], ['flash4-exp2-log2-scale'], 1),
      row('flash4-exp-mufu-row', [
        strong('The normal path is not emulation. '),
        text('Most entries still call fast '),
        math(String.raw`\operatorname{exp2}`),
        text(', which is the path expected to lower to the hardware special-function exponential, '),
        math(String.raw`\mathrm{MUFU.EX2}`),
        text('.'),
      ], ['flash4-exp2-mufu-path'], 1),
      row('flash4-exp-partial-emulation-row', [
        strong('Partial emulation means selected pairs bypass MUFU.EX2. '),
        text('The conversion loop groups a row into 32-value fragments; only configured residue lanes after the start fragment route to '),
        math(String.raw`\mathrm{ex2\_emulation\_2}`),
        text(', while the other lanes remain on the fast hardware path.'),
      ], ['flash4-exp2-convert-setup', 'flash4-exp2-emulation'], 1),
      row('flash4-exp-emu-range-row', [
        strong('The emulation first clamps and floors the pair. '),
        text('Each input is clamped at -127, then a packed round-down add separates the integer exponent from the fractional interval '),
        math(String.raw`[0,1)`),
        text('.'),
      ], ['flash4-ex2-clamp-round', 'flash4-ex2-fraction'], 1),
      row('flash4-exp-emu-poly-row', [
        strong('The fractional part is approximated by a polynomial. '),
        text('The code evaluates the degree-3 approximation for both lanes as a packed pair, giving an approximation to '),
        math(String.raw`2^{\operatorname{frac}(x)}`),
        text('.'),
      ], ['flash4-ex2-poly'], 1),
      row('flash4-exp-emu-combine-row', [
        strong('The emulated pair reconstructs '),
        math(String.raw`2^x`),
        strong(' by bit composition. '),
        text('The rounded integer is shifted into exponent bits, then added to the polynomial mantissa bits to form the final FP32 outputs.'),
      ], ['flash4-ex2-combine', 'flash4-ex2-output'], 1),
      row('flash4-exp-convert-row', [
        strong('The result is conversion, not just approximation. '),
        text('After either path, the row fragment is stored into the converted FP8 probability tile that PV MMA consumes from tensor memory.'),
      ], ['flash4-exp2-apply', 'flash4-exp2-convert-store', 'flash4-fp8-p-cast'], 1),
    ],
  },
  {
    id: 'flash4-skip-rescale-note',
    title: 'Skipping online softmax rescaling',
    require: [
      text('Online softmax row state, a dtype-dependent rescale threshold, and FP8 probability conversion.'),
    ],
    rows: [
      row('flash4-skip-threshold-row', [
        strong('Conditional rescaling starts with a threshold. '),
        text('For the FP8 SM100 path, upstream uses a positive threshold, so this sketch carries '),
        math(String.raw`\mathrm{rescale\_threshold}=4.0`),
        text(' with the softmax state.'),
      ], ['flash4-skip-rescale-threshold'], 1),
      row('flash4-skip-candidate-row', [
        strong('Compute the usual online-softmax candidate first. '),
        text('The kernel forms '),
        math(String.raw`m_{\mathrm{new}}=\max(m_{\mathrm{old}}, \max S)`),
        text(' and the would-be accumulator scale '),
        math(String.raw`\alpha=2^{m_{\mathrm{old}}-m_{\mathrm{new}}}`),
        text('.'),
      ], ['flash4-exp2-log2-scale', 'flash4-skip-rescale-candidate'], 1),
      row('flash4-skip-branch-row', [
        strong('If the max barely moves, make rescale a no-op. '),
        text('When '),
        math(String.raw`m_{\mathrm{old}}-m_{\mathrm{new}}\ge -\tau`),
        text(', the code keeps the old row max and sets '),
        math(String.raw`\alpha=1`),
        text(', skipping the near-identity online rescale.'),
      ], ['flash4-skip-rescale-branch'], 1),
      row('flash4-skip-valid-row', [
        strong('Softmax still has a valid reference point. '),
        text('The probabilities are then computed relative to the possibly pinned row max; the row sum absorbs that common shift, while the O accumulator avoids multiplying by an almost-one correction.'),
      ], ['flash4-skip-rescale-apply', 'flash4-exp2-apply'], 1),
      row('flash4-skip-store-row', [
        strong('The converted probabilities keep the same downstream contract. '),
        text('After the conditional choice, '),
        math(String.raw`\operatorname{apply\_exp2\_convert}`),
        text(' still emits the FP8 probability tile consumed by PV MMA.'),
      ], ['flash4-exp2-convert-store', 'flash4-fp8-p-cast'], 1),
    ],
  },
  {
    id: 'flash4-scheduling-note',
    title: 'Scheduling',
    require: [
      text('Code-pane sketch of static or CLC tile scheduling, causal/local longest-processing-time-first (LPT) block order, and variable-length Q sequence metadata.'),
    ],
    rows: [
      row('flash4-schedule-select-row', [
        strong('The sketch starts by choosing the tile scheduler. '),
        text('Packed Q sequences use '),
        math(String.raw`\mathrm{SingleTileVarlenScheduler}`),
        text('; causal, local, and CLC modes use '),
        math(String.raw`\mathrm{SingleTileLPTScheduler}`),
        text('; the dense nonpersistent case falls back to a plain single-tile scheduler.'),
      ], ['flash4-schedule-select'], 1),
      row('flash4-schedule-args-row', [
        strong('The scheduler receives the problem geometry. '),
        text('It is parameterized by M blocks, heads, batches, splits, K/V byte geometry, tile shape, cluster shape, and the '),
        math(String.raw`\mathrm{lpt}`),
        text(' flag derived from causal or local masking.'),
      ], ['flash4-schedule-args'], 1),
      row('flash4-schedule-lpt-definition-row', [
        strong('LPT means longest-processing-time-first. '),
        text('It is a load-balancing heuristic: estimate which tiles will run longer, issue those tiles earlier, and let shorter tiles fill the tail so one heavy row is less likely to hold the whole grid open.'),
      ], ['flash4-schedule-args', 'flash4-schedule-lpt-causal'], 1),
      row('flash4-schedule-causal-lpt-row', [
        strong('For causal masking, tile cost is mostly the number of valid K blocks. '),
        text('The static scheduler first swizzles head-batch rows into L2-sized sections, then maps block '),
        math(String.raw`b`),
        text(' to '),
        math(String.raw`B-1-b`),
        text(' so the longer causal rows launch first.'),
      ], ['flash4-schedule-lpt-causal'], 1),
      row('flash4-schedule-varlen-prefix-row', [
        strong('Variable-length Q scheduling decodes a flat tile id by prefix sums. '),
        text('The scheduler reads '),
        math(String.raw`\mathrm{mCuSeqlensQ}`),
        text(' or '),
        math(String.raw`\mathrm{mSeqUsedQ}`),
        text(', computes each batch\'s M-block count, and uses warp prefix sums to find the batch that owns the current tile.'),
      ], ['flash4-schedule-varlen-input', 'flash4-schedule-varlen-prefix'], 1),
      row('flash4-schedule-varlen-lpt-row', [
        strong('For variable lengths, the longest row is sequence-local. '),
        text('After the flat tile id has been decoded into a sequence, the scheduler picks a per-sequence head section size, then reverses inside that sequence\'s own M-block count.'),
      ], ['flash4-schedule-varlen-lpt'], 1),
      row('flash4-schedule-work-loop-row', [
        strong('The work loop is per warp group. '),
        text('This page models the upstream shape: a scheduler object is passed into load, MMA, softmax, and correction warp groups, and each group runs its own '),
        math(String.raw`\mathrm{initial\_work\_tile\_info}`),
        text(' / '),
        math(String.raw`\mathrm{advance\_to\_next\_work}`),
        text(' loop around its own tile body. The real scheduler classes carry more persistent/CLC bookkeeping than this compact map.'),
      ], [
        'flash4-schedule-create',
        'flash4-schedule-work-load',
        'flash4-schedule-work-mma',
        'flash4-schedule-work-softmax',
        'flash4-schedule-work-correction',
        'flash4-schedule-clc-loop',
      ], 1),
    ],
  },
]

function withCausalRows(row: AlgorithmLineSpec): AlgorithmLineSpec {
  if (row.id === 'flash4-mma-softmax-bridge') {
    return addCodeRefs(
      {
        ...row,
        parts: [
          strong('Softmax is the tensor-memory bridge. '),
          text('Softmax warps load '),
          math(String.raw`S`),
          text(' from tensor memory, apply the causal mask and base-2 online normalization, and store FP8 probabilities back in the PV operand layout.', 'mask'),
        ],
      },
      'flash4-fwd-causal-mask'
    )
  }

  return row
}

const causalFlash4MmaRows = flash4MmaRows.map(withCausalRows)

const flash4Forward = defineAttentionContent({
  rawCode: flashAttention4CuteDslCode,
  require: flash4MmaRequire,
  rows: flash4MmaRows,
  notes: flash4IdeaNotes,
  ignoredUnusedRefs: causalCodeRefIds,
})

const causalFlash4Forward = defineAttentionContent({
  rawCode: flashAttention4CuteDslCode,
  require: causalFlash4MmaRequire,
  rows: causalFlash4MmaRows,
  notes: flash4IdeaNotes,
})

export const flashAttention4Example: AttentionExample = {
  id: 'flash4',
  urlTag: 'flashattention-4',
  label: 'FlashAttention-4',
  description:
    'FlashAttention-4 moves the FA2/FA3 tiled-attention schedule onto Blackwell SM100 with FP8 tensors, TMA-fed 2-CTA clusters, tcgen05 MMA, and tensor-memory staging.',
  algorithmTitle: 'FlashAttention-4',
  content: {
    unmasked: flash4Forward,
    masked: causalFlash4Forward,
  },
}
