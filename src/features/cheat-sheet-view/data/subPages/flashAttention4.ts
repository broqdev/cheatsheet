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

const flash4HardwarePrelude: LatexBlockSpec[] = [
  {
    id: 'flash4-hardware',
    title: 'Hardware',
    require: [
      text('Blackwell hardware features'),
    ],
    rows: [
      row('flash4-hardware-mma-meaning', [
        strong('Fully asynchronous MMA. '),
        text('Matrix multiply-accumulate is the tensor-core operation that computes '),
        math(String.raw`D=A B + C`),
        text(' for matrix fragments. FlashAttention-4 maps the QK and PV products onto fully asynchronous Blackwell '),
        math(String.raw`tcgen05`),
        text('/UMMA-style MMA atoms: QK writes score tiles into tensor memory, and PV consumes tensor-memory probabilities plus transposed V to accumulate '),
        math(String.raw`O`),
        text('.'),
      ], ['flash4-mma-meaning', 'flash4-fwd-qk-mma', 'flash4-fwd-pv-mma']),
      row('flash4-hardware-cta-define', [
        strong('Thread-block cluster units (CTA). '),
        text('A cooperative thread array is one CUDA thread block with its own warps, shared memory, named barriers, and pipeline state. FlashAttention-4 binds two CTAs into one cluster tile and assigns load, MMA, softmax, correction, and empty warp groups so the pair cooperates along the '),
        math(String.raw`M`),
        text(' dimension.'),
      ], ['flash4-cta-thread-block']),
    ],
  },
]

const flash4MmaRows: AlgorithmLineSpec[] = [
  row('flash4-mma-forward-label', [strong('MMA ops in forward')], [
    'flash4-mma-atoms',
    'flash4-mma-operands',
  ]),
  row('flash4-mma-contract', [
    strong('SM100 MMA replaces the older GEMM center. '),
    text('Compared with FA2 block matmuls and FA3 Hopper WGMMA, this sketch names Blackwell '),
    math(String.raw`tcgen05`),
    text(' MMA atoms, operand major modes, CuTe fragment ownership, warp-group ownership, and tensor-memory accumulator placement.'),
  ], ['flash4-mma-atoms', 'flash4-forward-fragments'], 1),
  row('flash4-mma-operands', [
    strong('Forward is two explicit MMA products. '),
    text('QK consumes shared-memory Q/K operands; PV consumes probabilities from tensor memory plus a transposed V operand laid out for MN-major access. CuTe/CUTLASS helpers partition those operands into per-MMA views.'),
  ], ['flash4-mma-operands', 'flash4-forward-fragments', 'flash4-fp8-only-assert', 'flash4-fp8-mma-atoms'], 1),
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
    text(' from tensor memory, apply scale/mask/online normalization, store FP8 probabilities for PV, and publish row sums/maxima through '),
    math(String.raw`\mathrm{sScale}`),
    text(' plus '),
    math(String.raw`\mathrm{mbar\_softmax\_stats}`),
    text(' for correction warps.'),
  ], ['flash4-softmax-bridge', 'flash4-forward-stats-generic'], 1),
  row('flash4-mma-pv', [
    strong('PV MMA accumulates O in tensor memory. '),
    text('The second '),
    math(String.raw`cute.gemm`),
    text(' treats '),
    math(String.raw`P`),
    text(' as the A operand from tensor memory and transposed V as the B operand from shared memory; correction/epilogue then normalizes and drains O.'),
  ], ['flash4-fwd-pv-mma', 'flash4-correction-bridge', 'flash4-forward-epilogue'], 1),

  row('flash4-mma-backward-label', [strong('MMA ops in backward')], [
    'flash4-bwd-mma-atoms',
  ]),
  row('flash4-bwd-contract', [
    strong('Backward is decomposed into MMA products. '),
    text('Generic SM100 runs preprocess, a tiled-MMA main kernel, and dQ postprocess; the main kernel rebuilds scores, forms dP, accumulates dK/dV, and accumulates dQ without materializing the full attention matrix.'),
  ], ['flash4-bwd-dispatch-generic', 'flash4-bwd-preprocess', 'flash4-bwd-mma-atoms', 'flash4-bwd-fragments', 'flash4-bwd-postprocess'], 1),
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
    text(' and the load warp issues matching Q/K and V slice handles. Generic SM100 may instead use q-stage as an M-stage pipeline, so the code marks where that source path diverges.'),
  ], ['flash4-q-stage-policy', 'flash4-q-staging-hd256', 'flash4-kv-staging-hd256', 'flash4-forward-hd256-stats'], 1),
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
    'flash4-2cta-bwd-hd256-route',
    'flash4-2cta-bwd-traffic',
    'flash4-2cta-bwd-split-kernels',
    'flash4-2cta-bwd-interface-args',
    'flash4-2cta-bwd-launch-calls',
    'flash4-2cta-bwd-dq-store',
  ]),
  row('flash4-2cta-bwd-hd256-route-row', [
    strong('HD256 uses a dedicated SM100/SM110 route. '),
    text('The interface recognizes '),
    math(String.raw`\mathrm{head\_dim}=\mathrm{head\_dim}_V=256`),
    text(' as a special shape, forces the 2-CTA path, and fixes the backward tile shapes to '),
    math(String.raw`128{\times}128`),
    text(' for dQ and '),
    math(String.raw`128{\times}64`),
    text(' for dK/dV.'),
  ], ['flash4-2cta-bwd-hd256-route', 'flash4-2cta-bwd-split-kernels'], 1),
  row('flash4-2cta-bwd-traffic-row', [
    strong('Backward is shared-memory-bound. '),
    text('The FA4 paper notes that, even after TMEM staging, most backward GEMM operands still come from shared memory; 2-CTA MMA targets that bottleneck directly.'),
  ], ['flash4-2cta-bwd-traffic', 'flash4-2cta-bwd-mma-atoms'], 1),
  row('flash4-2cta-bwd-operand-b-row', [
    strong('Operand-B traffic is cut by the CTA pair. '),
    text('With an M=256, N=K=128 MMA tile, the two CTAs behave as one larger tile: each CTA stages half of operand B and keeps its own accumulator slice.'),
  ], ['flash4-2cta-bwd-traffic', 'flash4-2cta-bwd-mma-atoms'], 1),
  row('flash4-2cta-bwd-dq-reduction-row', [
    strong('dQ differs by implementation path. '),
    text('The paper/generic 2-CTA idea repacks dS through DSMEM for the query-side reduction. Current upstream HD256 code instead launches a dedicated dQ kernel and a separate dK/dV kernel.'),
  ], ['flash4-2cta-bwd-dsmem', 'flash4-2cta-bwd-hd256-route', 'flash4-2cta-bwd-split-kernels', 'flash4-2cta-bwd-launch-calls'], 1),
  row('flash4-2cta-bwd-pipeline-row', [
    strong('The inner dQ pipeline still overlaps work. '),
    text('Inside the dedicated dQ kernel, dP and dQ MMA work are ordered to reuse tensor-memory space, but the HD256 wrapper is not one monolithic backward kernel.'),
  ], ['flash4-2cta-bwd-pipeline', 'flash4-2cta-bwd-dq-mainloop', 'flash4-2cta-bwd-split-kernels', 'flash4-2cta-bwd-launch-calls'], 1),
  row('flash4-2cta-bwd-dq-kernel-row', [
    strong('The dedicated dQ kernel owns its body. '),
    text('It builds Q/K, dO/V, and dS/K fragment views, recomputes scores and dP, forms dS, accumulates dQ in tensor memory, then stores final dQ directly.'),
  ], ['flash4-2cta-bwd-dq-kernel', 'flash4-2cta-bwd-dq-fragments', 'flash4-2cta-bwd-dq-mainloop', 'flash4-2cta-bwd-dq-store'], 1),
  row('flash4-2cta-bwd-dkdv-kernel-row', [
    strong('The dedicated dK/dV kernel is separate. '),
    text('It builds K/Q, V/dO, dS/Q, and P/dO fragment views, accumulates dK and dV, and drains them through shared-memory epilogue buffers before TMA stores.'),
  ], ['flash4-2cta-bwd-dkdv-kernel', 'flash4-2cta-bwd-dkdv-fragments', 'flash4-2cta-bwd-dkdv-mainloop', 'flash4-2cta-bwd-dkdv-store'], 1),
  row('flash4-2cta-bwd-dq-store-row', [
    strong('Dedicated HD256 writes dQ through its own epilogue. '),
    text('Host dispatch passes final dQ through the historical dQ_accum ABI slot for HD256; the wrapper rejects generic semaphores, launches dQ before dK/dV, and skips the generic dQ postprocess route.'),
  ], ['flash4-2cta-bwd-interface-args', 'flash4-2cta-bwd-launch-calls', 'flash4-2cta-bwd-dq-store'], 1),
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
      row('flash4-exp-emu-detail-label', [
        strong('Inside '),
        math(String.raw`\mathrm{ex2\_emulation\_2}`),
        strong(', two lanes are split and rebuilt together.'),
      ], ['flash4-ex2-clamp-round', 'flash4-ex2-fraction', 'flash4-ex2-poly', 'flash4-ex2-combine'], 1),
      row('flash4-exp-emu-magic-row', [
        strong('The magic number makes integer bits cheap to extract. '),
        text('The constant '),
        math(String.raw`R=2^{23}+2^{22}`),
        text(' is large enough that adding a small FP32 exponent input pushes fractional bits below the mantissa precision; the '),
        math(String.raw`2^{22}`),
        text(' offset lets signed exponent offsets survive the later bit shift.'),
      ], ['flash4-ex2-clamp-round'], 2),
      row('flash4-exp-emu-round-row', [
        strong('Packed round-down creates the integer exponent. '),
        text('After clamping each lane at '),
        math(String.raw`-127`),
        text(', '),
        math(String.raw`\mathrm{add\_packed\_f32x2}`),
        text(' adds '),
        math(String.raw`R`),
        text(' with '),
        math(String.raw`\mathrm{rnd}=\mathrm{rm}`),
        text(', so subtracting '),
        math(String.raw`R`),
        text(' back gives '),
        math(String.raw`n=\lfloor x\rfloor`),
        text(' for each lane.'),
      ], ['flash4-ex2-clamp-round', 'flash4-ex2-fraction'], 2),
      row('flash4-exp-emu-frac-row', [
        strong('The remaining fraction stays in a stable interval. '),
        text('The code forms '),
        math(String.raw`f=x-n`),
        text(', so '),
        math(String.raw`f\in[0,1)`),
        text(' and the emulation only needs a small polynomial for the mantissa-like factor '),
        math(String.raw`2^f`),
        text('.'),
      ], ['flash4-ex2-fraction'], 2),
      row('flash4-exp-emu-poly-detail-row', [
        strong('The polynomial is a Horner-form cubic. '),
        text('With coefficients '),
        math(String.raw`c_0,\ldots,c_3`),
        text(', each lane evaluates '),
        math(String.raw`((c_3f+c_2)f+c_1)f+c_0\approx2^f`),
        text('; the packed pair shares the same degree-3 approximation path.'),
      ], ['flash4-ex2-poly-coefficients', 'flash4-ex2-poly'], 2),
      row('flash4-exp-emu-recompose-detail-row', [
        strong('Bit recomposition turns the pieces back into FP32. '),
        text('The rounded value still carries '),
        math(String.raw`n`),
        text(' in low mantissa bits; shifting those bits left by 23 forms the exponent-field delta, and adding the polynomial FP32 bits keeps the '),
        math(String.raw`2^f`),
        text(' mantissa while applying the '),
        math(String.raw`2^n`),
        text(' scale.'),
      ], ['flash4-ex2-combine', 'flash4-ex2-output'], 2),
      row('flash4-exp-convert-row', [
        strong('The result is conversion, not just approximation. '),
        text('After either path, the row fragment is stored into the converted FP8 probability tile that PV MMA consumes from tensor memory.'),
      ], ['flash4-exp2-apply', 'flash4-exp2-convert-store', 'flash4-fp8-p-cast'], 1),
    ],
  },
  {
    id: 'flash4-skip-rescale-note',
    title: 'FP8 online softmax rescaling threshold',
    require: [
      text('Online softmax row state, a dtype-dependent rescale threshold, and FP8 probability conversion.'),
    ],
    rows: [
      row('flash4-skip-threshold-row', [
        strong('FP8 keeps the rescale threshold at zero. '),
        text('The SM100 FP8 path carries '),
        math(String.raw`\tau=\mathrm{rescale\_threshold}=0.0`),
        text(', so any row-max increase triggers the normal output-accumulator rescale.'),
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
        strong('Positive thresholds are the skip-rescale optimization. '),
        text('Only when '),
        math(String.raw`\tau>0`),
        text(' and '),
        math(String.raw`m_{\mathrm{old}}-m_{\mathrm{new}}\ge -\tau`),
        text(' does the code keep the old row max and set '),
        math(String.raw`\alpha=1`),
        text(', skipping the near-identity online rescale; for FP8, '),
        math(String.raw`\tau=0`),
        text(' disables this shortcut.'),
      ], ['flash4-skip-rescale-branch'], 1),
      row('flash4-skip-fp8-range-row', [
        strong('The zero threshold protects FP8 dynamic range. '),
        text('FP8 probability storage uses a positive exponent offset; if the old max were allowed to lag by '),
        math(String.raw`\tau`),
        text(', the largest stored probability could scale like '),
        math(String.raw`2^{\mathrm{max\_offset}+\tau}`),
        text('. With '),
        math(String.raw`\mathrm{max\_offset}=8`),
        text(' and the stale '),
        math(String.raw`\tau=4`),
        text(' case, that reaches '),
        math(String.raw`2^{12}`),
        text(', beyond FP8 E4M3 range, so FP8 uses '),
        math(String.raw`\tau=0`),
        text('.'),
      ], ['flash4-skip-rescale-threshold', 'flash4-fp8-max-offset', 'flash4-skip-rescale-branch', 'flash4-exp2-apply'], 1),
      row('flash4-skip-valid-row', [
        strong('Softmax stays on the ordinary online path for FP8. '),
        text('The probabilities are computed relative to the current row max plus the FP8 offset, the row sum tracks that same offset, and the O accumulator is rescaled whenever the max increases.'),
      ], ['flash4-skip-rescale-apply', 'flash4-exp2-apply'], 1),
      row('flash4-skip-store-row', [
        strong('The converted probabilities keep the same downstream contract. '),
        text('After the threshold decision, '),
        math(String.raw`\operatorname{apply\_exp2\_convert}`),
        text(' still emits the FP8 probability tile consumed by PV MMA.'),
      ], ['flash4-exp2-convert-store', 'flash4-fp8-p-cast'], 1),
    ],
  },
  {
    id: 'flash4-scheduling-note',
    title: 'Scheduling',
    require: [
      text('Code-pane sketch of the three main upstream scheduling families: plain single-tile, LPT single-tile, and varlen single-tile. CLC, Cluster Launch Control, is a scheduling mode layered onto the LPT/varlen-style work fetch, not a separate coordinate family.'),
    ],
    rows: [
      row('flash4-schedule-select-row', [
        strong('The sketch starts by choosing a scheduler family. '),
        text('Packed Q sequences use '),
        math(String.raw`\mathrm{SingleTileVarlenScheduler}`),
        text('; causal/local ordering uses '),
        math(String.raw`\mathrm{SingleTileLPTScheduler}`),
        text('; dense nonpersistent work falls back to a plain single-tile scheduler. CLC changes the work-fetch mode inside the capable schedulers.'),
      ], ['flash4-schedule-select'], 1),
      row('flash4-schedule-clc-definition-row', [
        strong('CLC means Cluster Launch Control. '),
        text('Instead of assigning each CTA a fixed static tile id, CLC lets the kernel pull raw work from a hardware queue. The scheduler still converts that raw work into the same '),
        math(String.raw`(M,H,B,\mathrm{split})`),
        text(' coordinate shape consumed by the load, MMA, softmax, and correction loops.'),
      ], ['flash4-schedule-args', 'flash4-schedule-clc-map'], 1),
      row('flash4-schedule-args-row', [
        strong('The scheduler receives the problem geometry. '),
        text('It is parameterized by M blocks, heads, batches, splits, K/V byte geometry, tile shape, cluster shape, and the '),
        math(String.raw`\mathrm{lpt}`),
        text(' flag derived from causal or local masking; CLC is carried as a scheduling mode, not another LPT predicate.'),
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
      row('flash4-schedule-clc-row', [
        strong('CLC fetches raw work, then maps it. '),
        text('Upstream CLC mode asks the hardware queue for a raw '),
        math(String.raw`(\mathrm{block}, \mathrm{head}, \mathrm{batch/split})`),
        text(' tile, then '),
        math(String.raw`\mathrm{clc\_work\_to\_coords}`),
        text(' applies cluster division, optional LPT reversal, split-KV unpacking, and cluster-lane expansion.'),
      ], ['flash4-schedule-clc-map', 'flash4-schedule-clc-loop'], 1),
      row('flash4-schedule-work-loop-row', [
        strong('The work loop is per warp group. '),
        text('This page models the upstream shape: a scheduler object is passed into load, MMA, softmax, and correction warp groups, and each group runs its own '),
        math(String.raw`\mathrm{initial\_work\_tile\_info}`),
        text(' / '),
        math(String.raw`\mathrm{advance\_to\_next\_work}`),
        text(' loop around its own tile body. The real scheduler classes carry more persistent/CLC bookkeeping than this compact map.'),
      ], [
        'flash4-schedule-create',
        'flash4-schedule-work-loop',
        'flash4-schedule-work-load',
        'flash4-schedule-work-mma',
        'flash4-schedule-work-softmax',
        'flash4-schedule-work-correction',
        'flash4-schedule-clc-map',
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
          text(' Row sums/maxima still travel through the generic stats handoff.', 'mask'),
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
  prelude: flash4HardwarePrelude,
  rows: flash4MmaRows,
  notes: flash4IdeaNotes,
  ignoredUnusedRefs: causalCodeRefIds,
})

const causalFlash4Forward = defineAttentionContent({
  rawCode: flashAttention4CuteDslCode,
  require: causalFlash4MmaRequire,
  prelude: flash4HardwarePrelude,
  rows: causalFlash4MmaRows,
  notes: flash4IdeaNotes,
})

export const flashAttention4Example: AttentionExample = {
  id: 'flash4',
  urlTag: 'flashattention-4',
  label: 'FlashAttention-4',
  description:
    'FlashAttention-4 refines exact tiled attention for Blackwell with FP8-oriented scheduling and cooperative tile execution.',
  algorithmTitle: 'FlashAttention-4',
  content: {
    unmasked: flash4Forward,
    masked: causalFlash4Forward,
  },
}
