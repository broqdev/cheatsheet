import math
from functools import partial
from typing import NamedTuple, Optional, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass import pipeline
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute import utils
from flash_attn.cute.fast_math import clz
from flash_attn.cute.mask import AttentionMask


# FlashAttention-4 / SM100 CuTe DSL FP8 content sketch.
# Mirrors Dao-AILab's Blackwell forward structure:
# - Q and K are K-major UMMA operands.
# - V is transposed to (D, S_K, H_K, B) so PV sees an MN-major B operand.
# - Q, K, and V use TMA into shared memory.
# - S = QK, P, and partial O live in Blackwell tensor memory.
# - Load, MMA, softmax, correction, and epilogue run as specialized warp groups.
#
# @ref flash4-mma-meaning
# MMA means matrix multiply-accumulate: a tensor-core tile computes D = A @ B + C.
# On Blackwell, this sketch uses tcgen05 / UMMA-style atoms, which can target
# tensor memory and participate in 2-CTA MMA.
# @end

LOG2_E = 1.4426950408889634
LN2 = 0.6931471805599453
SM100_TMEM_COLUMNS = 512
FP8_DTYPE = cutlass.Float8E4M3FN
POLY_EX2_DEGREE_3 = (
    Float32(1.0),
    Float32(0.6951461434364319),
    Float32(0.22756439447402954),
    Float32(0.07711908966302872),
)


@cute.jit
def evaluate_exp2_fraction_pair(x_frac: Float32, y_frac: Float32):
# @ref flash4-ex2-poly
    c0, c1, c2, c3 = POLY_EX2_DEGREE_3
    x_frac_ex2 = ((c3 * x_frac + c2) * x_frac + c1) * x_frac + c0
    y_frac_ex2 = ((c3 * y_frac + c2) * y_frac + c1) * y_frac + c0
    return x_frac_ex2, y_frac_ex2
# @end


@cute.jit
def combine_int_frac_ex2(x_rounded: Float32, frac_ex2: Float32):
# @ref flash4-ex2-combine
    rounded_bits = cute.arch.bitcast(Int32, x_rounded)
    frac_bits = cute.arch.bitcast(Int32, frac_ex2)
    exponent_bits = rounded_bits << 23
    return cute.arch.bitcast(Float32, exponent_bits + frac_bits)
# @end


@cute.jit
def ex2_emulation_2(x: Float32, y: Float32, poly_degree: int = 3):
# @ref flash4-ex2-clamp-round
    fp32_round_int = Float32(2**23 + 2**22)
    xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    xy_rounded = cute.arch.add_packed_f32x2(
        xy_clamped, (fp32_round_int, fp32_round_int), rnd="rm"
    )
# @end
# @ref flash4-ex2-fraction
    xy_rounded_back = (
        xy_rounded[0] - fp32_round_int,
        xy_rounded[1] - fp32_round_int,
    )
    xy_frac = (
        xy_clamped[0] - xy_rounded_back[0],
        xy_clamped[1] - xy_rounded_back[1],
    )
# @end
    assert poly_degree == 3
    xy_frac_ex2 = evaluate_exp2_fraction_pair(xy_frac[0], xy_frac[1])
# @ref flash4-ex2-output
    x_out = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0])
    y_out = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1])
    return x_out, y_out
# @end


class DescaleTensors(NamedTuple):
    q_descale: Optional[cute.Tensor] = None
    k_descale: Optional[cute.Tensor] = None
    v_descale: Optional[cute.Tensor] = None


class SketchWorkTile(NamedTuple):
    tile_idx: Tuple[Int32, Int32, Int32, Int32]
    is_valid_tile: bool


class FlashAttention4SchedulingMap:
    """Code-pane sketch of the SM100 tile scheduler.

    The real Dao-AILab kernel routes through scheduler classes in
    flash_attn.cute.tile_scheduler. This map is a compact, code-pane model of
    the major branches: static LPT's L2 swizzle, CLC raw-work mapping, varlen
    prefix-sum decoding, cluster-M expansion, and the per-warp-group loop shape.
    """

    def __init__(
        self,
        num_block: Int32,
        num_head: Int32,
        num_batch: Int32,
        num_splits: Int32,
        seqlen_k: Int32,
        total_q: Int32,
        head_dim: Int32,
        head_dim_v: Int32,
        element_size: int = 1,
        tile_shape_mn=(128, 128),
        cluster_shape_mn=(2, 1),
        qhead_per_kvhead_packgqa: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_varlen_q: bool = False,
        is_split_kv: bool = False,
        head_swizzle: bool = False,
        use_clc_scheduler: bool = False,
        use_cluster_idx: bool = True,
    ):
        self.num_block = num_block
        self.num_head = num_head
        self.num_batch = num_batch
        self.num_splits = num_splits
        self.seqlen_k = seqlen_k
        self.total_q = total_q
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.element_size = element_size
        self.tile_shape_mn = tile_shape_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.cluster_shape_m = cluster_shape_mn[0]
        self.qhead_per_kvhead_packgqa = qhead_per_kvhead_packgqa
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_varlen_q = is_varlen_q
        self.is_split_kv = is_split_kv
        self.head_swizzle = head_swizzle
        self.use_clc_scheduler = use_clc_scheduler
        self.use_cluster_idx = use_cluster_idx
        self._is_first_block = True

# @ref flash4-schedule-select
        if is_varlen_q:
            self.scheduler_name = "SingleTileVarlenScheduler"
        elif is_causal or is_local or use_clc_scheduler:
            self.scheduler_name = "SingleTileLPTScheduler"
        else:
            self.scheduler_name = "SingleTileScheduler"
# @end

# @ref flash4-schedule-args
        # TileSchedulerArguments in Dao's code carries the same geometry plus
        # FP8 element size, cluster shape, split-KV state, and optional varlen
        # metadata. CLC is Cluster Launch Control: a hardware work-queue mode,
        # not a separate coordinate family. The LPT scheduler derives its L2
        # swizzle from K/V bytes when the work is statically assigned.
        self.scheduler_args = dict(
            num_block=num_block,
            num_head=num_head,
            num_batch=num_batch,
            num_splits=num_splits,
            seqlen_k=seqlen_k,
            headdim=head_dim,
            headdim_v=head_dim_v,
            total_q=total_q,
            tile_shape_mn=tile_shape_mn,
            cluster_shape_mn=cluster_shape_mn,
            lpt=is_causal or is_local,
            scheduling_mode="CLC" if use_clc_scheduler else "STATIC",
            qhead_per_kvhead_packgqa=qhead_per_kvhead_packgqa,
            element_size=element_size,
            is_varlen_q=is_varlen_q,
            is_split_kv=is_split_kv,
            head_swizzle=head_swizzle,
            use_cluster_idx=use_cluster_idx,
        )

        size_one_head = Int64(seqlen_k) * (head_dim + head_dim_v) * element_size
        size_l2 = 50 * 1024 * 1024
        log2_floor = lambda n: 31 - clz(n)
        swizzle = 1 if size_l2 < size_one_head else 1 << log2_floor(Int32(size_l2 // size_one_head))
        num_hb = num_head * num_batch
        num_hb_remainder = num_hb % swizzle
        self.l2_minor = Int32(swizzle)
        self.num_hb_quotient = Int32(num_hb // swizzle)
        self.num_head_divmod = FastDivmodDivisor(num_head)
        self.l2_minor_divmod = FastDivmodDivisor(swizzle)
        self.l2_major_divmod = FastDivmodDivisor(swizzle * num_block)
        self.l2_minor_residual_divmod = FastDivmodDivisor(max(num_hb_remainder, 1))

        kv_block_size = (head_dim + head_dim_v) * element_size * tile_shape_mn[1]
        if head_swizzle:
            kv_block_size += head_dim * 4 * tile_shape_mn[1]
        self.max_kvblock_in_l2 = Int32(size_l2 // kv_block_size)
# @end

    @cute.jit
    def lpt_static_coord(self, tile_idx: Int32, split_idx: Int32):
# @ref flash4-schedule-lpt-causal
        # Static SingleTileLPTScheduler first groups head-batch rows into L2
        # sections. The final residual section uses its own divisor instead of
        # pretending it contains a full swizzle group.
        bidhb, l2_mod = divmod(tile_idx, self.l2_major_divmod)
        if bidhb < self.num_hb_quotient:
            block, bidhb_residual = divmod(l2_mod, self.l2_minor_divmod)
        else:
            block, bidhb_residual = divmod(l2_mod, self.l2_minor_residual_divmod)
        bidhb_actual = bidhb * self.l2_minor + bidhb_residual
        batch_idx, head_idx = divmod(bidhb_actual, self.num_head_divmod)
        if const_expr(self.is_causal or self.is_local):
            # Later M blocks have more live K tiles under a causal/local mask,
            # so LPT reverses block order after the L2 swizzle is decoded.
            block = self.num_block - 1 - block
        is_valid = tile_idx < self.num_block * self.num_head * self.num_batch
        return (Int32(block), Int32(head_idx), Int32(batch_idx), split_idx), is_valid
# @end

    @cute.jit
    def clc_work_to_coords(self, work):
# @ref flash4-schedule-clc-map
        # Upstream SingleTileLPTScheduler uses clc_work_to_coords for CLC mode:
        # hardware chooses the work order, then the scheduler converts the raw
        # (block, head, batch/split) response back into WorkTileInfo coordinates.
        block_idx, head_idx, batch_split_idx = work.tile_idx
        if const_expr(self.cluster_shape_m > 1):
            block_idx = block_idx // self.cluster_shape_m
        if const_expr(self.is_causal or self.is_local):
            num_block = (
                self.num_block // self.cluster_shape_m
                if const_expr(self.cluster_shape_m > 1 and not self.use_cluster_idx)
                else self.num_block
            )
            block_idx = num_block - 1 - block_idx
        if const_expr(self.is_split_kv):
            batch_idx, split_idx = divmod(batch_split_idx, FastDivmodDivisor(self.num_splits))
        else:
            batch_idx, split_idx = batch_split_idx, Int32(0)
        if const_expr(self.cluster_shape_m > 1 and not self.use_cluster_idx):
            block_idx = block_idx * self.cluster_shape_m + cute.arch.block_in_cluster_idx()[0]
        return SketchWorkTile(
            (Int32(block_idx), Int32(head_idx), Int32(batch_idx), Int32(split_idx)),
            work.is_valid_tile,
        )
# @end

    @cute.jit
    def varlen_num_m_blocks(self, lane: Int32, bidb_start: Int32, mCuSeqlensQ=None, mSeqUsedQ=None):
# @ref flash4-schedule-varlen-input
        batch_idx = lane + bidb_start
        if const_expr(mSeqUsedQ is not None):
            seqlen_q = Int32(0)
            if batch_idx < self.num_batch:
                seqlen_q = mSeqUsedQ[batch_idx]
        else:
            cur_cu_seqlen = Int32(0)
            if batch_idx <= self.num_batch:
                cur_cu_seqlen = mCuSeqlensQ[batch_idx]
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen_q = next_cu_seqlen - cur_cu_seqlen
        if const_expr(self.qhead_per_kvhead_packgqa > 1):
            seqlen_q *= self.qhead_per_kvhead_packgqa
        num_tiles = cute.ceil_div(seqlen_q, self.tile_shape_mn[0])
        return (
            cute.ceil_div(num_tiles, self.cluster_shape_m)
            if batch_idx < self.num_batch and lane < cute.arch.WARP_SIZE - 1
            else Int32(0)
        )
# @end

    @cute.jit
    def pick_nheads_in_l2(self, num_n_blocks: Int32):
        if num_n_blocks * 16 <= self.max_kvblock_in_l2:
            return Int32(16)
        if num_n_blocks * 8 <= self.max_kvblock_in_l2:
            return Int32(8)
        if num_n_blocks * 4 <= self.max_kvblock_in_l2:
            return Int32(4)
        if num_n_blocks * 2 <= self.max_kvblock_in_l2:
            return Int32(2)
        return Int32(1)

    @cute.jit
    def varlen_coord_map(self, tile_idx: Int32, split_idx: Int32, mCuSeqlensQ=None, mSeqUsedQ=None):
        lane_idx = cute.arch.lane_idx()
# @ref flash4-schedule-varlen-prefix
        # Varlen scheduling decodes flat tile ids in groups of 31 batches. Lane
        # 31 carries the inclusive prefix total for the group; other lanes carry
        # candidate batch sizes.
        num_m_blocks = self.varlen_num_m_blocks(lane_idx, Int32(0), mCuSeqlensQ, mSeqUsedQ)
        num_m_blocks_cumulative = utils.warp_prefix_sum(num_m_blocks, lane_idx)
        m_blocks_in_group = cute.arch.shuffle_sync(num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1)
        group_end_tile = m_blocks_in_group * self.num_head
        block, head_idx, batch_idx = Int32(0), Int32(0), Int32(0)
        next_tile_idx = tile_idx // self.cluster_shape_m
        while group_end_tile <= next_tile_idx:
            batch_idx += cute.arch.WARP_SIZE - 1
            if batch_idx >= self.num_batch:
                batch_idx = Int32(self.num_batch)
                group_end_tile = next_tile_idx + 1
            else:
                num_m_blocks = self.varlen_num_m_blocks(lane_idx, batch_idx, mCuSeqlensQ, mSeqUsedQ)
                num_m_blocks_cumulative = utils.warp_prefix_sum(num_m_blocks, lane_idx)
                m_blocks_in_group = cute.arch.shuffle_sync(
                    num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1
                )
                group_end_tile += m_blocks_in_group * self.num_head
        is_valid = False
        if batch_idx >= self.num_batch:
            return (Int32(0), Int32(0), Int32(self.num_batch), Int32(0)), is_valid

        group_start_tile = group_end_tile - m_blocks_in_group * self.num_head
        batch_idx_in_group = cute.arch.popc(
            cute.arch.vote_ballot_sync(
                group_start_tile + num_m_blocks_cumulative * self.num_head <= next_tile_idx
            )
        )
        batch_idx += batch_idx_in_group
        num_m_blocks_prev_lane = (
            Int32(0)
            if batch_idx_in_group == 0
            else cute.arch.shuffle_sync(num_m_blocks_cumulative, batch_idx_in_group - 1)
        )
        num_m_blocks = cute.arch.shuffle_sync(num_m_blocks, batch_idx_in_group)
        mh_block = next_tile_idx - group_start_tile - num_m_blocks_prev_lane * self.num_head
# @end
# @ref flash4-schedule-varlen-lpt
        if const_expr(self.is_causal or self.is_local or self.head_swizzle):
            # Same LPT idea as static scheduling, but the L2 head section size
            # depends on this batch's sequence length.
            num_n_blocks = (
                num_m_blocks
                * self.tile_shape_mn[0]
                * self.cluster_shape_m
                // self.qhead_per_kvhead_packgqa
                // self.tile_shape_mn[1]
            )
            nheads_in_l2 = cutlass.min(self.pick_nheads_in_l2(num_n_blocks), self.num_head)
            mh_in_l2 = nheads_in_l2 * num_m_blocks
            section_idx = mh_block // mh_in_l2
            l2_mod = mh_block - section_idx * mh_in_l2
            nheads_in_this_section = (
                nheads_in_l2
                if nheads_in_l2 * (section_idx + 1) <= self.num_head
                else self.num_head - section_idx * nheads_in_l2
            )
            block = l2_mod // nheads_in_this_section
            head_idx = section_idx * nheads_in_l2 + (l2_mod - block * nheads_in_this_section)
            if const_expr(self.is_causal or self.is_local):
                block = num_m_blocks - 1 - block
        else:
            head_idx = mh_block // num_m_blocks
            block = mh_block - head_idx * num_m_blocks
# @end
        is_valid = self._is_first_block and batch_idx < self.num_batch
        if const_expr(self.cluster_shape_m > 1):
            block = block * self.cluster_shape_m + cute.arch.block_in_cluster_idx()[0]
        return (Int32(block), Int32(head_idx), Int32(batch_idx), split_idx), is_valid

    @cute.jit
    def get_current_work(self, clc=None):
        if const_expr(self.use_clc_scheduler):
            work = clc.get_current_work()
            return self.clc_work_to_coords(work)
        block_idx = cute.arch.block_idx()
        tile_idx = block_idx[0]
        split_idx = block_idx[1] if const_expr(self.is_split_kv) else Int32(0)
        if const_expr(self.is_varlen_q):
            tile_coord, is_valid = self.varlen_coord_map(tile_idx, split_idx)
        else:
            tile_coord, is_valid = self.lpt_static_coord(tile_idx, split_idx)
        return SketchWorkTile(tile_coord, is_valid)

    @cute.jit
    def initial_work_tile_info(self, clc=None):
        if const_expr(self.use_clc_scheduler):
            work = clc.initial_work_tile_info()
            return self.clc_work_to_coords(work)
        return self.get_current_work()

    def prefetch_next_work(self, clc=None):
        if const_expr(self.use_clc_scheduler):
            clc.prefetch_next_work()

    @cute.jit
    def advance_to_next_work(self, clc=None):
        if const_expr(self.use_clc_scheduler):
            clc.consumer_wait()
            work_tile = self.get_current_work(clc)
            clc.consumer_release()
            return work_tile
        # Static single-tile schedulers consume one work tile per CTA.
        self._is_first_block = False
        return SketchWorkTile(
            (Int32(0), Int32(0), Int32(self.num_batch), Int32(0)),
            False,
        )

    def producer_tail(self, clc=None):
        if const_expr(self.use_clc_scheduler):
            clc.producer_tail()

    @cute.jit
    def clc_scheduler_warp(self, clc):
# @ref flash4-schedule-clc-loop
        work_tile = self.initial_work_tile_info(clc)
        while work_tile.is_valid_tile:
            self.prefetch_next_work(clc)
            work_tile = self.advance_to_next_work(clc)
        self.producer_tail(clc)
# @end


class FlashAttention4Sm100CuteDsl:
    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: Optional[int] = None,
        kv_stage: int = 2,
    ):
        head_dim_v = head_dim if head_dim_v is None else head_dim_v
        assert head_dim in (128, 256)
        assert head_dim_v == head_dim
        assert m_block_size == 128 and n_block_size == 128

        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.qk_acc_stage = 2
        self.mma_corr_stage = 2
# @ref flash4-skip-rescale-threshold
        # FP8 SM100 uses a positive threshold so nearly identical row maxima do
        # not force a full output-accumulator rescale on every K tile.
        self.rescale_threshold = 4.0
# @end
        self.cta_tiler = (m_block_size, n_block_size, head_dim)
        self.qk_mma_tiler = (2 * m_block_size, n_block_size, min(head_dim, 128))
        self.pv_mma_tiler = self.qk_mma_tiler
        self.iterations_qk = head_dim // self.qk_mma_tiler[2]
        self.iterations_pv = head_dim_v // self.pv_mma_tiler[1]
# @ref flash4-q-stage-policy
        # Generic SM100 forward can use q_stage=1 or q_stage=2 as an M-stage
        # pipeline. The dedicated HD256 forward also has two 128-wide
        # head-dimension slices, so this compact page keeps the same knob but
        # comments where the source paths diverge.
        if q_stage is None:
            q_stage = 2
        assert q_stage in (1, 2)
        if head_dim == 256:
            # In sm100_hd256_2cta_fmha_forward.py, q_stage is accepted for
            # interface parity and then fixed to iterations_qk == 2.
            assert q_stage == self.iterations_qk
        self.q_stage = q_stage
        self.kv_stage = 4 if head_dim == 256 and kv_stage == 2 else kv_stage
# @end

# @ref flash4-2cta-shape
        self.cluster_shape_mn = (2, 1)
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
# @end
# @ref flash4-cta-thread-block
        self.softmax_warp_ids = (0, 1, 2, 3)
        self.correction_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.empty_warp_ids = (10, 11)
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.softmax_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                *self.empty_warp_ids,
            )
        )
# @end

        self.tmem_s_offset = 0
        self.tmem_p_offset = self.tmem_s_offset
        self.tmem_o_offset = 256
        self.num_regs_softmax = 256 if head_dim == 256 else 176
        self.num_regs_correction = 160 if head_dim == 256 else 88
        self.num_regs_other = 32
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        descale_tensors: Optional[DescaleTensors] = None,
        stream: cuda.CUstream = None,
    ):
        q, k, v, o, lse, problem = self._make_problem_tensors(
            mQ, mK, mV, mO, mLSE, mCuSeqlensQ, mCuSeqlensK
        )
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type

# @ref flash4-fp8-only-assert
        if const_expr(
            q.element_type.width != 8
            or k.element_type.width != 8
            or v.element_type.width != 8
            or o.element_type.width != 8
        ):
            raise TypeError("FlashAttention-4 code content models the SM100 FP8 path only")
        self.fp8_dtype = FP8_DTYPE
# @end
        self.qk_acc_dtype = cutlass.Float32
        self.pv_acc_dtype = cutlass.Float32

# @ref flash4-2cta-cgroup
        cta_group = tcgen05.CtaGroup.TWO
# @end
        q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        if const_expr(q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("SM100 QK expects Q to be K-major")
        if const_expr(k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("SM100 QK expects K to be K-major")
        if const_expr(v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("SM100 PV expects transposed V to be MN-major")

# @ref flash4-mma-atoms flash4-fp8-mma-atoms
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            tcgen05.OperandMajorMode.K,
            v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )
# @end
# @ref flash4-2cta-cluster-layout
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (qk_tiled_mma.thr_id.shape,)
        )
# @end

# @ref flash4-mma-operands flash4-2cta-tma
        # The SMEM layouts are the contract between the TMA producer and the
        # UMMA consumer; their staged mode count must match the pipeline stages.
        q_smem_layout = sm100_utils.make_smem_layout_a(
            qk_tiled_mma, self.qk_mma_tiler, self.q_dtype, self.q_stage
        )
        k_smem_layout = sm100_utils.make_smem_layout_b(
            qk_tiled_mma, self.qk_mma_tiler, self.k_dtype, self.kv_stage
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma, self.pv_mma_tiler, self.q_dtype, self.qk_acc_stage
        )
        v_smem_layout = sm100_utils.make_smem_layout_b(
            pv_tiled_mma, self.pv_mma_tiler, self.v_dtype, self.kv_stage
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_atom_q, tma_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            cute.select(q_smem_layout, mode=[0, 1, 2]),
            self.qk_mma_tiler,
            qk_tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_k, tma_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            cute.select(k_smem_layout, mode=[0, 1, 2]),
            self.qk_mma_tiler,
            qk_tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_v, tma_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            cute.select(v_smem_layout, mode=[0, 1, 2]),
            self.pv_mma_tiler,
            pv_tiled_mma,
            cluster_layout_vmnk.shape,
        )
# @end

# @ref flash4-2cta-copy-bytes
        self.tma_copy_q_bytes = cute.size_in_bytes(
            self.q_dtype, cute.select(q_smem_layout, mode=[0, 1, 2])
        ) * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_kv_bytes = cute.size_in_bytes(
            self.k_dtype, cute.select(k_smem_layout, mode=[0, 1, 2])
        ) * cute.size(qk_tiled_mma.thr_id.shape)
# @end

        @cute.struct
        class SharedStorage:
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            softmax_stats_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
            sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]
            mma_corr_mbar_ptr: cute.struct.MemRange[Int64, self.mma_corr_stage * 2]
            tmem_dealloc_mbar_ptr: Int64
            tmem_holding_buf: Int32

        self.shared_storage = SharedStorage
        grid = cute.round_up(
            (cute.ceil_div(problem.seqlen_q, self.m_block_size), problem.num_heads_q, problem.batch),
            self.cluster_shape_mnk,
        )
        scale_softmax_log2 = softmax_scale * math.log2(math.exp(1.0))

        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_q,
            tma_atom_k,
            tma_k,
            tma_atom_v,
            tma_v,
            o,
            lse,
            scale_softmax_log2,
            softmax_scale,
            descale_tensors,
            cluster_layout_vmnk,
            q_smem_layout,
            k_smem_layout,
            p_tmem_layout_staged,
            v_smem_layout,
            problem,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _make_problem_tensors(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
    ):
        batch = mQ.shape[0] if const_expr(mCuSeqlensQ is None) else mCuSeqlensQ.shape[0] - 1
        seqlen_q = mQ.shape[1] if const_expr(mCuSeqlensQ is None) else mQ.shape[0]
        seqlen_k = mK.shape[1] if const_expr(mCuSeqlensK is None) else mK.shape[0]
        num_heads_q = mQ.shape[2]
        num_heads_k = mK.shape[2]
        head_ratio = num_heads_q // num_heads_k
        head_dim = cute.assume(Int64(self.head_dim), divby=128)
        stride_b_qo = Int64(seqlen_q) * Int64(num_heads_q) * head_dim
        stride_b_kv = Int64(seqlen_k) * Int64(num_heads_k) * head_dim

        q_layout = cute.make_layout(
            (seqlen_q, self.head_dim, ((head_ratio, num_heads_k), batch)),
            stride=(head_dim * Int64(num_heads_q), 1, ((head_dim, head_dim * head_ratio), stride_b_qo)),
        )
        k_layout = cute.make_layout(
            (seqlen_k, self.head_dim, ((head_ratio, num_heads_k), batch)),
            stride=(head_dim * Int64(num_heads_k), 1, ((0, head_dim), stride_b_kv)),
        )
        v_layout = cute.make_layout(
            (self.head_dim_v, seqlen_k, ((head_ratio, num_heads_k), batch)),
            stride=(1, head_dim * Int64(num_heads_k), ((0, head_dim), stride_b_kv)),
        )
        o_layout = cute.make_layout(
            (seqlen_q, self.head_dim_v, ((head_ratio, num_heads_k), batch)),
            stride=(head_dim * Int64(num_heads_q), 1, ((head_dim, head_dim * head_ratio), stride_b_qo)),
        )
        lse_layout = cute.make_layout(
            (seqlen_q, ((head_ratio, num_heads_k), batch)),
            stride=(1, ((Int64(seqlen_q), Int64(head_ratio) * Int64(seqlen_q)), Int64(num_heads_q) * Int64(seqlen_q))),
        )

        q = cute.make_tensor(mQ.iterator, q_layout)
        k = cute.make_tensor(mK.iterator, k_layout)
        v = cute.make_tensor(mV.iterator, v_layout)
        o = cute.make_tensor(mO.iterator, o_layout)
        lse = cute.make_tensor(mLSE.iterator, lse_layout) if const_expr(mLSE is not None) else None

        @cute.struct
        class ProblemShape:
            batch: Int32
            num_heads_q: Int32
            num_heads_k: Int32
            head_ratio: Int32
            seqlen_q: Int32
            seqlen_k: Int32

        problem = ProblemShape(
            Int32(batch),
            Int32(num_heads_q),
            Int32(num_heads_k),
            Int32(head_ratio),
            Int32(seqlen_q),
            Int32(seqlen_k),
        )
        return q, k, v, o, lse, problem

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        tma_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_v: cute.Tensor,
        o: cute.Tensor,
        lse: Optional[cute.Tensor],
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        descale_tensors: Optional[DescaleTensors],
        cluster_layout_vmnk: cute.Layout,
        q_smem_layout: cute.ComposedLayout,
        k_smem_layout: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        v_smem_layout: cute.ComposedLayout,
        problem,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        ThreadGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = ThreadGroup(1)
        mma_warp = ThreadGroup(1)
        softmax_threads = ThreadGroup(self.threads_per_warp * len(self.softmax_warp_ids))
        correction_threads = ThreadGroup(self.threads_per_warp * len(self.correction_warp_ids))
        softmax_threads_cluster = ThreadGroup(
            self.threads_per_warp * len(self.softmax_warp_ids) * self.cluster_shape_mn[0]
        )
        correction_threads_cluster = ThreadGroup(
            self.threads_per_warp * len(self.correction_warp_ids) * self.cluster_shape_mn[0]
        )

# @ref flash4-pipelines flash4-2cta-pipelines
        # The direction in each pipeline name is intentional:
        # TMA feeds UMMA, UMMA feeds softmax, softmax feeds PV, and MMA feeds correction.
        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            num_stages=self.q_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_q_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        load_kv_producer, load_kv_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_kv_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        mma_s_producer, mma_s_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            num_stages=self.qk_acc_stage,
            producer_group=mma_warp,
            consumer_group=softmax_threads_cluster,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        p_mma_producer, p_mma_consumer = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.p_mma_mbar_ptr.data_ptr(),
            num_stages=self.qk_acc_stage,
            producer_group=softmax_threads_cluster,
            consumer_group=mma_warp,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        softmax_stats_producer, softmax_stats_consumer = pipeline.PipelineAsync.create(
            barrier_storage=storage.softmax_stats_mbar_ptr.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
        )
        mma_corr_producer, mma_corr_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mma_corr_mbar_ptr.data_ptr(),
            num_stages=self.mma_corr_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads_cluster,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
# @end

# @ref flash4-tmem flash4-2cta-tmem
        tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_barrier,
            allocator_warp_id=self.correction_warp_ids[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )
        tmem.allocate(SM100_TMEM_COLUMNS)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
# @end

        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)
        sQ = smem.allocate_tensor(self.q_dtype, q_smem_layout.outer, q_smem_layout.inner, self.buffer_align_bytes)
        sK = smem.allocate_tensor(self.k_dtype, k_smem_layout.outer, k_smem_layout.inner, self.buffer_align_bytes)
        sV = smem.allocate_tensor(self.v_dtype, v_smem_layout.outer, v_smem_layout.inner, self.buffer_align_bytes)
# @ref flash4-forward-stats-generic
        # Generic SM100 forward keeps row sums and row maxima in shared-memory
        # sScale and synchronizes those scalars with mbar_softmax_stats. The
        # upstream file also uses a named barrier per softmax warp; this compact
        # pane exposes the pipeline state that ties softmax to correction.
        sScale = storage.sScale.get_tensor(cute.make_layout(self.q_stage * self.m_block_size * 2))
# @end

# @ref flash4-2cta-rank
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        mma_tile_coord_v = cta_rank % cute.size(qk_tiled_mma.thr_id.shape)
        qk_thr_mma = qk_tiled_mma.get_slice(mma_tile_coord_v)
        pv_thr_mma = pv_tiled_mma.get_slice(mma_tile_coord_v)
# @end
# @ref flash4-schedule-create
        effective_m_block = (
            self.m_block_size
            if const_expr(self.head_dim == 256)
            else self.m_block_size * self.q_stage
        )
        tile_scheduler = FlashAttention4SchedulingMap(
            num_block=cute.ceil_div(problem.seqlen_q, effective_m_block),
            num_head=problem.num_heads_q,
            num_batch=problem.batch,
            num_splits=Int32(1),
            seqlen_k=problem.seqlen_k,
            total_q=problem.seqlen_q * problem.batch,
            head_dim=Int32(self.head_dim),
            head_dim_v=Int32(self.head_dim_v),
            element_size=1,
            tile_shape_mn=(self.m_block_size, self.n_block_size),
            cluster_shape_mn=self.cluster_shape_mn,
            qhead_per_kvhead_packgqa=1,
            is_causal=self.is_causal,
            is_varlen_q=False,
        )
# @end
# @ref flash4-forward-fragments
        # make_fragment_A/B/C are CuTe/CUTLASS tiler internals. They return
        # per-MMA operand or accumulator views over the shared-memory / TMEM
        # tiles; the page names them because upstream relies on the library's
        # partitioning rules rather than implementing those rules locally.
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)
        tStS = qk_thr_mma.make_fragment_C(
            cute.append(qk_thr_mma.partition_shape_C(self.qk_mma_tiler[:2]), self.qk_acc_stage)
        )
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
        tOtO = pv_thr_mma.make_fragment_C(
            cute.append(pv_thr_mma.partition_shape_C(self.pv_mma_tiler[:2]), self.mma_corr_stage)
        )
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset, tOtO.layout)
# @end

        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
# @ref flash4-warp-specialization
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self.load_loop(tma_atom_q, tma_q, tma_atom_k, tma_k, tma_atom_v, tma_v, sQ, sK, sV, load_q_producer, load_kv_producer, problem, tile_scheduler)
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self.mma_loop(qk_tiled_mma, pv_tiled_mma, tSrQ, tSrK, tOrV, tStS, tOtO, p_tmem_layout_staged, load_q_consumer, load_kv_consumer, mma_s_producer, p_mma_consumer, mma_corr_producer, problem, tile_scheduler)
        elif warp_idx >= self.softmax_warp_ids[0] and warp_idx < self.correction_warp_ids[0]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)
            self.softmax_loop(tStS, sScale, scale_softmax_log2, descale_tensors, mma_s_consumer, p_mma_producer, softmax_stats_producer, problem, tile_scheduler)
        elif warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_correction)
            self.correction_loop(tOtO, sScale, o, lse, scale_softmax_log2, scale_softmax, descale_tensors, softmax_stats_consumer, mma_corr_consumer, problem, tile_scheduler)
        else:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
# @end

# @ref flash4-2cta-free
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)
# @end

    @cute.jit
    def load_loop(self, tma_atom_q, tma_q, tma_atom_k, tma_k, tma_atom_v, tma_v, sQ, sK, sV, load_q_producer, load_kv_producer, problem, tile_scheduler):
# @ref flash4-schedule-work-loop flash4-schedule-work-load
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_tile, head_q, batch, split_idx = work_tile.tile_idx
# @end
            kv_head = head_q // problem.head_ratio
# @ref flash4-q-staging-hd256
            # Producer count mirrors the consumer loop. HD256 loads two Q slices;
            # the compact HD128 path loads the single full-head tile.
            for q_iter in cutlass.range(self.q_stage, unroll=1):
                q_handle = load_q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    self.q_gmem_slice(tma_q, m_tile, q_iter, head_q, batch),
                    sQ[None, None, None, q_handle.index],
                    tma_bar_ptr=q_handle.barrier,
                )
                q_handle.commit()
# @end
            for n_tile in cutlass.range(cute.ceil_div(problem.seqlen_k, self.n_block_size), unroll=1):
# @ref flash4-kv-staging-hd256
                # K feeds the QK MMA and V feeds the later PV MMA, but both share
                # the same load/consume pipeline in this compact sketch.
                for k_iter in cutlass.range(self.iterations_qk, unroll=1):
                    k_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        self.k_gmem_slice(tma_k, n_tile, k_iter, kv_head, batch),
                        sK[None, None, None, k_handle.index],
                        tma_bar_ptr=k_handle.barrier,
                    )
                    k_handle.commit()
                for v_iter in cutlass.range(self.iterations_pv, unroll=1):
                    v_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_v,
                        self.v_gmem_slice(tma_v, n_tile, v_iter, kv_head, batch),
                        sV[None, None, None, v_handle.index],
                        tma_bar_ptr=v_handle.barrier,
                    )
                    v_handle.commit()
# @end
# @ref flash4-schedule-work-loop flash4-schedule-work-load
            work_tile = tile_scheduler.advance_to_next_work()
# @end
        load_kv_producer.tail()
        load_q_producer.tail()

    @cute.jit
    def mma_loop(self, qk_tiled_mma, pv_tiled_mma, tSrQ, tSrK, tOrV, tStS, tOtO, p_tmem_layout_staged, load_q_consumer, load_kv_consumer, mma_s_producer, p_mma_consumer, mma_corr_producer, problem, tile_scheduler):
        q_releaser = load_q_consumer.clone()
        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
# @ref flash4-schedule-work-loop flash4-schedule-work-mma
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
# @end
            for n_tile in cutlass.range(cute.ceil_div(problem.seqlen_k, self.n_block_size), unroll=1):
                s_handle = mma_s_producer.acquire_and_advance()
                tStS_slice = tStS[None, None, None, s_handle.index]
                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                # In HD256, k_iter walks the two 128-wide D slices. For HD128,
                # iterations_qk is one and this degenerates to a single QK MMA.
                for k_iter in cutlass.range(self.iterations_qk, unroll=1):
                    load_q_consumer.wait_and_advance()
                    k_handle = load_kv_consumer.wait_and_advance()
                    q_fragment = tSrQ[None, None, None, k_iter]
                    k_fragment = tSrK[None, None, None, k_handle.index]
                    for kphase in cutlass.range(cute.size(q_fragment, mode=[2]), unroll_full=True):
# @ref flash4-fwd-qk-mma
                        cute.gemm(qk_tiled_mma, tStS_slice, q_fragment[None, None, kphase], k_fragment[None, None, kphase], tStS_slice)
                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
# @end
                    k_handle.release()
                    q_releaser.release()
                    q_releaser.advance()
                s_handle.commit()

                p_handle = p_mma_consumer.wait_and_advance()
                o_handle = mma_corr_producer.acquire_and_advance()
                pv_was_accumulating = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                for v_iter in cutlass.range(self.iterations_pv, unroll=1):
                    v_handle = load_kv_consumer.wait_and_advance()
                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_was_accumulating)
                    tP = cute.make_tensor(tStS[None, None, None, p_handle.index].iterator, p_tmem_layout_staged.outer)
                    tOrP = pv_tiled_mma.get_slice(0).make_fragment_A(tP)
                    tOrP = cute.make_tensor(cute.recast_ptr(tStS[None, None, None, p_handle.index].iterator, dtype=self.q_dtype), tOrP.layout)
                    v_fragment = tOrV[None, None, None, v_handle.index]
                    for kphase in cutlass.range(cute.size(v_fragment, mode=[2]), unroll_full=True):
# @ref flash4-fwd-pv-mma
                        cute.gemm(pv_tiled_mma, tOtO[None, None, None, v_iter], tOrP[None, None, kphase], v_fragment[None, None, kphase], tOtO[None, None, None, v_iter])
                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
# @end
                    v_handle.release()
                o_handle.commit()
                p_handle.release()
# @ref flash4-schedule-work-loop flash4-schedule-work-mma
            work_tile = tile_scheduler.advance_to_next_work()
# @end

    @cute.jit
    def apply_exp2_convert(
        self,
        acc_s_row: cute.Tensor,
        acc_s_row_converted: cute.Tensor,
        ex2_emu_freq: cutlass.Constexpr[int] = 0,
        ex2_emu_res: cutlass.Constexpr[int] = 4,
        ex2_emu_start_frg: cutlass.Constexpr[int] = 0,
    ):
# @ref flash4-exp2-convert-setup
        assert cute.size(acc_s_row.shape) % 2 == 0
        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_s_row) // frg_tile
        assert cute.size(acc_s_row) % frg_tile == 0
        acc_s_row_frg = cute.logical_divide(acc_s_row, cute.make_layout(frg_tile))
        acc_s_row_converted_frg = cute.logical_divide(
            acc_s_row_converted, cute.make_layout(frg_tile)
        )
# @end
        for j in cutlass.range_constexpr(frg_cnt):
            for k_idx in cutlass.range_constexpr(0, cute.size(acc_s_row_frg, mode=[0]), 2):
                if cutlass.const_expr(ex2_emu_freq == 0):
# @ref flash4-exp2-mufu-path
                    # Normal path: emit exp2 directly and let codegen lower to
                    # the SM100 special-function path for most lanes.
                    acc_s_row_frg[k_idx, j] = cute.math.exp2(
                        acc_s_row_frg[k_idx, j], fastmath=True
                    )
                    acc_s_row_frg[k_idx + 1, j] = cute.math.exp2(
                        acc_s_row_frg[k_idx + 1, j], fastmath=True
                    )
# @end
                else:
                    if cutlass.const_expr(
                        k_idx % ex2_emu_freq < ex2_emu_freq - ex2_emu_res
                        or j >= frg_cnt - 1
                        or j < ex2_emu_start_frg
                    ):
# @ref flash4-exp2-mufu-path
                        # Non-selected residues still take the hardware exp2
                        # path, which is why this is partial emulation.
                        acc_s_row_frg[k_idx, j] = cute.math.exp2(
                            acc_s_row_frg[k_idx, j], fastmath=True
                        )
                        acc_s_row_frg[k_idx + 1, j] = cute.math.exp2(
                            acc_s_row_frg[k_idx + 1, j], fastmath=True
                        )
# @end
                    else:
# @ref flash4-exp2-emulation
                        # Selected lane pairs bypass MUFU.EX2 and rebuild exp2
                        # from integer exponent bits plus a polynomial mantissa.
                        acc_s_row_frg[k_idx, j], acc_s_row_frg[k_idx + 1, j] = (
                            ex2_emulation_2(
                                acc_s_row_frg[k_idx, j],
                                acc_s_row_frg[k_idx + 1, j],
                            )
                        )
# @end
# @ref flash4-exp2-convert-store flash4-fp8-p-cast
            acc_s_row_converted_frg[None, j].store(
                acc_s_row_frg[None, j].load().to(acc_s_row_converted.element_type)
            )
# @end

    @cute.jit
    def softmax_loop(self, tStS, sScale, scale_softmax_log2, descale_tensors, mma_s_consumer, p_mma_producer, softmax_stats_producer, problem, tile_scheduler):
# @ref flash4-schedule-work-loop flash4-schedule-work-softmax
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
# @end
            row_max = Float32(-cutlass.inf)
            row_sum = Float32(0.0)
            for n_tile in cutlass.range(cute.ceil_div(problem.seqlen_k, self.n_block_size), unroll=1):
# @ref flash4-softmax-bridge flash4-exp2-log2-scale
                s_handle = mma_s_consumer.wait_and_advance()
                scores_log2 = self.load_scores_from_tmem(tStS[None, None, None, s_handle.index])
                # From here until LSE writeback, row_max is in log2-scaled logit
                # units: raw QK * softmax_scale * log2(e) * FP8 descale.
                scores_log2 = scores_log2 * (scale_softmax_log2 * self.load_qk_descale(descale_tensors))
# @end
# @ref flash4-fwd-causal-mask
                if const_expr(self.is_causal):
                    scores_log2 = AttentionMask.apply_causal(scores_log2, n_tile, self.m_block_size, self.n_block_size)
# @end
# @ref flash4-softmax-bridge flash4-exp2-log2-scale
                row_max_candidate = cute.maximum(row_max, cute.max(scores_log2, axis=1))
# @end
# @ref flash4-skip-rescale-candidate
                row_scale_log2 = row_max - row_max_candidate
                correction = cute.math.exp2(row_scale_log2, fastmath=True)
# @end
# @ref flash4-skip-rescale-branch
                if const_expr(self.rescale_threshold > 0.0):
                    if row_scale_log2 >= -self.rescale_threshold:
                        # The new max is close enough to the old max that the
                        # output correction would be almost one, so pin the max.
                        row_max_candidate = row_max
                        correction = Float32(1.0)
# @end
# @ref flash4-softmax-bridge flash4-exp2-apply flash4-skip-rescale-apply
                probs = scores_log2 - row_max_candidate
                probs_converted = cute.make_rmem_tensor(probs.shape, self.q_dtype)
                # apply_exp2_convert mutates probs from log2 residuals into
                # probabilities, then stores the FP8 version for PV MMA.
                self.apply_exp2_convert(probs, probs_converted, ex2_emu_freq=8, ex2_emu_res=2)
                row_sum = row_sum * correction + cute.sum(probs, axis=1)
                row_max = row_max_candidate

                p_handle = p_mma_producer.acquire_and_advance()
                self.store_probs_to_tmem(tStS[None, None, None, s_handle.index], probs_converted)
                p_handle.commit()
                s_handle.release()
# @end

                stats_handle = softmax_stats_producer.acquire_and_advance()
# @ref flash4-forward-stats-generic
                # Source shape: softmax writes row_sum and row_max to sScale,
                # then signals correction through mbar_softmax_stats. The full
                # upstream code adds a per-warp named barrier; the shared memory
                # slot and stats pipeline are the visible teaching contract here.
                self.store_softmax_stats(sScale, stats_handle.index, row_max, row_sum)
# @end
                stats_handle.commit()
# @ref flash4-schedule-work-loop flash4-schedule-work-softmax
            work_tile = tile_scheduler.advance_to_next_work()
# @end

    @cute.jit
    def correction_loop(self, tOtO, sScale, o, lse, scale_softmax_log2, scale_softmax, descale_tensors, softmax_stats_consumer, mma_corr_consumer, problem, tile_scheduler):
# @ref flash4-schedule-work-loop flash4-schedule-work-correction
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
# @end
            old_row_max = Float32(-cutlass.inf)
            final_row_max = Float32(-cutlass.inf)
            final_row_sum = Float32(1.0)
            for n_tile in cutlass.range(cute.ceil_div(problem.seqlen_k, self.n_block_size), unroll=1):
                stats_handle = softmax_stats_consumer.wait_and_advance()
# @ref flash4-correction-bridge flash4-forward-stats-generic
                row_max, row_sum = self.load_softmax_stats(sScale, stats_handle.index)
                # row_max is already scaled into log2 units, so the correction is
                # exp2(old - new), with no second softmax_scale factor.
                scale = cute.math.exp2(old_row_max - row_max, fastmath=True)
                old_row_max = row_max
                final_row_max = row_max
                final_row_sum = row_sum
                o_handle = mma_corr_consumer.wait_and_advance()
                if n_tile > 0:
                    self.rescale_o_in_tmem(tOtO[None, None, None, o_handle.index], scale)
                stats_handle.release()
                o_handle.release()
# @end

            self.correction_epilogue(o, lse, tOtO, final_row_max, final_row_sum, descale_tensors)
# @ref flash4-schedule-work-loop flash4-schedule-work-correction
            work_tile = tile_scheduler.advance_to_next_work()
# @end

    @cute.jit
    def load_scores_from_tmem(self, tStS_slice):
        load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype)
        tiled_load = tcgen05.make_tmem_copy(load_atom, tStS_slice)
        thr_load = tiled_load.get_slice(cute.arch.thread_idx()[0])
        fragment = cute.make_rmem_tensor(thr_load.partition_D(tStS_slice).shape, Float32)
        cute.copy(tiled_load, thr_load.partition_S(tStS_slice), fragment)
        cute.arch.fence_view_async_tmem_load()
        return fragment

    @cute.jit
    def q_gmem_slice(self, tma_q, m_tile, q_iter, head_q, batch):
        # HD256's extra index is the D-slice; generic SM100 q_stage advances
        # the M block so two Q stages cover two consecutive M tiles.
        if const_expr(self.head_dim == 256):
            return tma_q[(m_tile, q_iter, (head_q, batch))]
        return tma_q[(self.q_stage * m_tile + q_iter, None, (head_q, batch))]

    @cute.jit
    def k_gmem_slice(self, tma_k, n_tile, k_iter, kv_head, batch):
        # K is split with Q for the QK MMA only when D=256.
        if const_expr(self.head_dim == 256):
            return tma_k[(n_tile, k_iter, (kv_head, batch))]
        return tma_k[(n_tile, None, (kv_head, batch))]

    @cute.jit
    def v_gmem_slice(self, tma_v, n_tile, v_iter, kv_head, batch):
        # V is stored transposed, so the D-slice appears before the K tile.
        if const_expr(self.head_dim_v == 256):
            return tma_v[(v_iter, n_tile, (kv_head, batch))]
        return tma_v[(None, n_tile, (kv_head, batch))]

    @cute.jit
    def store_probs_to_tmem(self, tStS_slice, probs):
        store_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.q_dtype)
        tiled_store = tcgen05.make_tmem_copy(store_atom, tStS_slice)
        thr_store = tiled_store.get_slice(cute.arch.thread_idx()[0])
        cute.copy(tiled_store, probs, thr_store.partition_D(tStS_slice))
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def store_softmax_stats(self, sScale, stage, row_max, row_sum):
# @ref flash4-forward-stats-generic
        tidx = cute.arch.thread_idx()[0] % self.m_block_size
        sScale[tidx + stage * self.m_block_size] = row_sum
        sScale[tidx + stage * self.m_block_size + self.q_stage * self.m_block_size] = row_max
# @end

    @cute.jit
    def load_softmax_stats(self, sScale, stage):
# @ref flash4-forward-stats-generic
        tidx = cute.arch.thread_idx()[0] % self.m_block_size
        row_sum = sScale[tidx + stage * self.m_block_size]
        row_max = sScale[tidx + stage * self.m_block_size + self.q_stage * self.m_block_size]
        return row_max, row_sum
# @end

    @cute.jit
    def rescale_o_in_tmem(self, tOtO_slice, scale):
        load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), self.pv_acc_dtype)
        store_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), self.pv_acc_dtype)
        tiled_load = tcgen05.make_tmem_copy(load_atom, tOtO_slice)
        tiled_store = tcgen05.make_tmem_copy(store_atom, tOtO_slice)
        thr_load = tiled_load.get_slice(cute.arch.thread_idx()[0])
        thr_store = tiled_store.get_slice(cute.arch.thread_idx()[0])
        fragment = cute.make_rmem_tensor(thr_load.partition_D(tOtO_slice).shape, Float32)
        cute.copy(tiled_load, thr_load.partition_S(tOtO_slice), fragment)
        cute.copy(tiled_store, fragment * scale, thr_store.partition_D(tOtO_slice))
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(self, o, lse, tOtO, row_max, row_sum, descale_tensors):
# @ref flash4-forward-epilogue
        # Generic SM100 correction_epilogue drains the O accumulator from TMEM,
        # applies the final V descale / row_sum normalization, and writes LSE.
        final_scale = self.load_v_descale(descale_tensors) / row_sum
        self.store_o_tile_from_tmem(o, tOtO, final_scale)
        if const_expr(lse is not None):
            self.store_lse_row(lse, row_max * LN2 + cute.math.log(row_sum))
# @end

    @cute.jit
    def store_o_tile_from_tmem(self, o, tOtO, scale):
# @ref flash4-forward-epilogue
        load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.pv_acc_dtype)
        tiled_load = tcgen05.make_tmem_copy(load_atom, tOtO)
        thr_load = tiled_load.get_slice(cute.arch.thread_idx()[0])
        fragment = cute.make_rmem_tensor(thr_load.partition_D(tOtO).shape, Float32)
        cute.copy(tiled_load, thr_load.partition_S(tOtO), fragment)
        cute.autovec_copy(fragment * scale, o)
# @end

    @cute.jit
    def store_lse_row(self, lse, value):
# @ref flash4-forward-epilogue
        lse[cute.arch.thread_idx()[0]] = value
# @end

    @cute.jit
    def hd256_forward_stats_path(self, sSum, row_sum, row_max, sum_producer, sum_consumer, tOtO, o, lse, descale_tensors):
# @ref flash4-forward-hd256-stats
        # Dedicated HD256 forward does not use the generic sScale handoff. It
        # stores row_sum / row_max through sSum with sum_producer, then the
        # correction_epilog waits through sum_consumer before final O/LSE work.
        sum_state = sum_producer.acquire_and_advance()
        self.store_hd256_sum_max(sSum, sum_state.index, row_sum, row_max)
        sum_state.commit()
        correction_state = sum_consumer.wait_and_advance()
        final_row_sum, final_row_max = self.load_hd256_sum_max(sSum, correction_state.index)
        self.correction_epilogue(o, lse, tOtO, final_row_max, final_row_sum, descale_tensors)
        correction_state.release()
# @end

    @cute.jit
    def store_hd256_sum_max(self, sSum, stage, row_sum, row_max):
        row = cute.arch.thread_idx()[0] % self.m_block_size
        sSum[row, 0, stage] = row_sum
        sSum[row, 1, stage] = row_max

    @cute.jit
    def load_hd256_sum_max(self, sSum, stage):
        row = cute.arch.thread_idx()[0] % self.m_block_size
        return sSum[row, 0, stage], sSum[row, 1, stage]

    @cute.jit
    def load_qk_descale(self, descale_tensors):
        qk_descale = Float32(1.0)
        if const_expr(descale_tensors is not None):
            if const_expr(descale_tensors.q_descale is not None):
                qk_descale = qk_descale * Float32(descale_tensors.q_descale[0, 0])
            if const_expr(descale_tensors.k_descale is not None):
                qk_descale = qk_descale * Float32(descale_tensors.k_descale[0, 0])
        return qk_descale

    @cute.jit
    def load_v_descale(self, descale_tensors):
        if const_expr(descale_tensors is not None and descale_tensors.v_descale is not None):
            return Float32(descale_tensors.v_descale[0, 0])
        return Float32(1.0)


class FlashAttention4GenericBackwardSm100Map:
    """Generic SM100 backward path before the HD256 split-kernel exception."""

    def __init__(
        self,
        head_dim: int = 128,
        head_dim_v: int = 128,
        m_block_size: int = 128,
        n_block_size: int = 128,
        use_2cta_instrs: bool = False,
        deterministic: bool = False,
        qhead_per_kvhead: int = 1,
    ):
        assert head_dim <= 192
        assert head_dim_v <= 128
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.qhead_per_kvhead = qhead_per_kvhead
        self.deterministic = deterministic
        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.acc_dtype = cutlass.Float32
        self.q_dtype = FP8_DTYPE
        self.do_dtype = FP8_DTYPE
        self.kq_mma_tiler = (n_block_size, m_block_size, min(head_dim, 128))
        self.vdo_mma_tiler = (n_block_size, m_block_size, min(head_dim_v, 128))
        self.pdo_mma_tiler = (n_block_size, head_dim_v, m_block_size)
        self.dsq_mma_tiler = (n_block_size, head_dim, m_block_size)
        self.dsk_mma_tiler = (m_block_size, head_dim, n_block_size)

    def compile_generic_backward_args(self, dQ_accum, dQ, dK, dV, qhead_per_kvhead: int):
# @ref flash4-bwd-dispatch-generic flash4-bwd-preprocess flash4-bwd-postprocess
        # Host dispatch allocates a float32 dQ_accum scratch for the generic
        # route. Preprocess fills dPsum/lse_log2 and zeros dQ_accum; the main
        # kernel accumulates into dQ_accum; postprocess converts it to final dQ.
        assert dQ_accum is not None
        dQ_semaphore = self.make_dq_semaphore(qhead_per_kvhead) if self.deterministic else None
        dKV_postprocess = qhead_per_kvhead > 1
        dK_argument = None if dKV_postprocess else dK
        dV_argument = None if dKV_postprocess else dV
        return dQ_accum, dQ, dK_argument, dV_argument, dQ_semaphore
# @end

    def make_dq_semaphore(self, qhead_per_kvhead: int):
        semaphore_layout = cute.make_layout((qhead_per_kvhead, cute.ceil_div(self.m_block_size, 32)))
        return cute.make_tensor(cute.make_gmem_ptr(Int32(0)), semaphore_layout)

    @cute.jit
    def make_backward_mma_atoms(self):
# @ref flash4-bwd-mma-atoms
        score_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.kq_mma_tiler[:2],
        )
        dp_mma = sm100_utils.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.vdo_mma_tiler[:2],
        )
        dv_mma = sm100_utils.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.pdo_mma_tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        dk_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.dsq_mma_tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        dq_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.dsk_mma_tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        return score_mma, dp_mma, dv_mma, dk_mma, dq_mma
# @end

    @cute.jit
    def make_backward_fragments(self, score_mma, dp_mma, dv_mma, dk_mma, dq_mma, sQ, sK, sV, sdO, sdS, tP):
# @ref flash4-bwd-fragments
        # As in forward, make_fragment_A/B/C are CuTe/CUTLASS partitioning
        # helpers. The generic backward kernel asks the tilers for K/Q, V/dO,
        # P/dO, dS/Q, and dS/K views plus accumulator fragments.
        tSrK = score_mma.make_fragment_A(sK)
        tSrQ = score_mma.make_fragment_B(sQ)
        tdPrV = dp_mma.make_fragment_A(sV)
        tdPrdO = dp_mma.make_fragment_B(sdO)
        tdVrdO = dv_mma.make_fragment_B(sdO)
        tdVrP = dv_mma.make_fragment_A(tP)
        tdKrdS = dk_mma.make_fragment_A(sdS)
        tdKrQ = dk_mma.make_fragment_B(sQ)
        tdQrdS = dq_mma.make_fragment_A(sdS)
        tdQrK = dq_mma.make_fragment_B(sK)
        return tSrK, tSrQ, tdPrV, tdPrdO, tdVrdO, tdVrP, tdKrdS, tdKrQ, tdQrdS, tdQrK
# @end

    @cute.jit
    def generic_backward_mma_loop(
        self,
        score_mma,
        dp_mma,
        dv_mma,
        dk_mma,
        dq_mma,
        k_frag,
        q_frag,
        v_frag,
        do_frag,
        p_tmem,
        ds_tmem,
        score_tmem,
        dp_tmem,
        dk_tmem,
        dv_tmem,
        dq_accum_tmem,
        lse_log2,
        dpsum,
        mdQaccum,
    ):
        score_mma.set(tcgen05.Field.ACCUMULATE, False)
# @ref flash4-bwd-score-mma
        cute.gemm(score_mma, score_tmem, k_frag, q_frag, score_tmem)
        score_mma.set(tcgen05.Field.ACCUMULATE, True)
# @end

        dp_mma.set(tcgen05.Field.ACCUMULATE, False)
# @ref flash4-bwd-dp-mma
        cute.gemm(dp_mma, dp_tmem, v_frag, do_frag, dp_tmem)
        dp_mma.set(tcgen05.Field.ACCUMULATE, True)
# @end

# @ref flash4-bwd-dsoftmax
        scores = self.load_tmem_fragment(score_tmem)
        probs = cute.math.exp2(scores - lse_log2, fastmath=True)
        dp = self.load_tmem_fragment(dp_tmem)
        ds = (dp - dpsum) * probs
        self.store_tmem_fragment(ds_tmem, ds.to(self.q_dtype))
# @end

        dv_mma.set(tcgen05.Field.ACCUMULATE, False)
# @ref flash4-bwd-dkv-mma
        cute.gemm(dv_mma, dv_tmem, p_tmem, do_frag, dv_tmem)
        dv_mma.set(tcgen05.Field.ACCUMULATE, True)
        cute.gemm(dk_mma, dk_tmem, ds_tmem, q_frag, dk_tmem)
        dk_mma.set(tcgen05.Field.ACCUMULATE, True)
# @end

        dq_mma.set(tcgen05.Field.ACCUMULATE, False)
# @ref flash4-bwd-dq-mma
        cute.gemm(dq_mma, dq_accum_tmem, ds_tmem, k_frag, dq_accum_tmem)
        dq_mma.set(tcgen05.Field.ACCUMULATE, True)
# @end

# @ref flash4-bwd-epilogue flash4-bwd-postprocess
        # Generic SM100 drains dK/dV from this kernel, but dQ is accumulated as
        # float32 dQaccum. A separate postprocess kernel converts dQaccum to dQ.
        self.store_dkv_from_tmem(dk_tmem, dv_tmem)
        self.accumulate_dq_to_global_scratch(dq_accum_tmem, mdQaccum)
# @end

    @cute.jit
    def load_tmem_fragment(self, tmem_tile):
        return tmem_tile

    @cute.jit
    def store_tmem_fragment(self, tmem_tile, value):
        return value

    @cute.jit
    def store_dkv_from_tmem(self, dk_tmem, dv_tmem):
        return dk_tmem, dv_tmem

    @cute.jit
    def accumulate_dq_to_global_scratch(self, dq_accum_tmem, mdQaccum):
        return mdQaccum


class BlackwellFusedMultiHeadAttentionBackwardDQKernel:
    """Dedicated HD256 dQ kernel sketch."""

    def __init__(self, acc_dtype, mma_tile, is_persistent: bool, split_head: bool):
# @ref flash4-2cta-bwd-dq-kernel
        self.acc_dtype = acc_dtype
        self.mma_tile = mma_tile
        self.is_persistent = is_persistent
        self.split_head = split_head
        self.cta_tiler = mma_tile
        self.qk_mma_tiler = (
            2 * mma_tile[0],
            mma_tile[1],
            min(self.cta_tiler[2], 128) if split_head else self.cta_tiler[2],
        )
        self.dov_mma_tiler = self.qk_mma_tiler
        self.dsk_mma_tiler = (
            2 * mma_tile[0],
            min(self.cta_tiler[2], 128) if split_head else self.cta_tiler[2],
            mma_tile[1],
        )
        self.dsk_block_tiler = (
            self.dsk_mma_tiler[0] // 2,
            self.dsk_mma_tiler[1],
            self.dsk_mma_tiler[2],
        )
        self.iterations_qk = self.cta_tiler[2] // self.qk_mma_tiler[2]
        self.iterations_dov = self.cta_tiler[2] // self.dov_mma_tiler[2]
        self.iterations_dsk = self.cta_tiler[2] // self.dsk_mma_tiler[1]
        self.cta_group = tcgen05.CtaGroup.TWO
        self.q_dtype = FP8_DTYPE
        self.do_dtype = FP8_DTYPE
        self.q_stage = self.iterations_qk
        self.k_stage = self.iterations_qk
        self.v_stage = self.iterations_dov
        self.do_stage = self.iterations_dov
        self.kt_stage = 1
        self.qk_acc_stage = 1
        self.dov_acc_stage = 1
        self.dsk_acc_stage = 1
        self.epi_stage = 1
        self.tmem_s_offset = 0
        self.tmem_dp_offset = 128
        self.tmem_dq_offset = 256
        self.epi_tile = (mma_tile[0], 64)
        self.cluster_shape_mnk = (2, 1, 1)
        self.buffer_align_bytes = 1024
# @end

    @cute.jit
    def make_mma_atoms(self):
        qk_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.qk_mma_tiler[:2],
        )
        dov_mma = sm100_utils.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.dov_mma_tiler[:2],
        )
        dsk_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.dsk_mma_tiler[:2],
        )
        return qk_mma, dov_mma, dsk_mma

    @cute.jit
    def make_tma_atoms(self, Q, K, V, dO, KT, dQ, qk_mma, dov_mma, dsk_mma):
# @ref flash4-2cta-bwd-dq-kernel
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        q_smem_layout = sm100_utils.make_smem_layout_a(qk_mma, self.qk_mma_tiler, self.q_dtype, self.q_stage)
        k_smem_layout = sm100_utils.make_smem_layout_b(qk_mma, self.qk_mma_tiler, self.q_dtype, self.k_stage)
        do_smem_layout = sm100_utils.make_smem_layout_a(dov_mma, self.dov_mma_tiler, self.do_dtype, self.do_stage)
        v_smem_layout = sm100_utils.make_smem_layout_b(dov_mma, self.dov_mma_tiler, self.q_dtype, self.v_stage)
        kt_smem_layout = sm100_utils.make_smem_layout_b(dsk_mma, self.dsk_mma_tiler, self.q_dtype, self.dsk_acc_stage)
        tma_atom_Q, tma_Q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            Q,
            cute.select(q_smem_layout, mode=[0, 1, 2]),
            self.qk_mma_tiler,
            qk_mma,
            self.cluster_shape_mnk,
        )
        tma_atom_K, tma_K = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            K,
            cute.select(k_smem_layout, mode=[0, 1, 2]),
            self.qk_mma_tiler,
            qk_mma,
            self.cluster_shape_mnk,
        )
        tma_atom_dO, tma_dO = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            dO,
            cute.select(do_smem_layout, mode=[0, 1, 2]),
            self.dov_mma_tiler,
            dov_mma,
            self.cluster_shape_mnk,
        )
        tma_atom_V, tma_V = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            V,
            cute.select(v_smem_layout, mode=[0, 1, 2]),
            self.dov_mma_tiler,
            dov_mma,
            self.cluster_shape_mnk,
        )
        tma_atom_KT, tma_KT = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            KT,
            cute.select(kt_smem_layout, mode=[0, 1, 2]),
            self.dsk_mma_tiler,
            dsk_mma,
            self.cluster_shape_mnk,
        )
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        sdQ_epi_layout = sm100_utils.make_smem_layout_epi(
            dQ.element_type,
            utils.LayoutEnum.from_tensor(dQ),
            self.epi_tile,
            1,
        )
        tma_atom_dQ, tma_dQ = cpasync.make_tiled_tma_atom(
            tma_store_op,
            dQ,
            cute.select(sdQ_epi_layout, mode=[0, 1]),
            self.epi_tile,
        )
        return (
            (tma_atom_Q, tma_Q),
            (tma_atom_K, tma_K),
            (tma_atom_dO, tma_dO),
            (tma_atom_V, tma_V),
            (tma_atom_KT, tma_KT),
            (tma_atom_dQ, tma_dQ),
            (q_smem_layout, k_smem_layout, do_smem_layout, v_smem_layout, kt_smem_layout, sdQ_epi_layout),
        )
# @end

    @cute.jit
    def allocate_shared_tiles(self, smem, layouts):
        q_smem_layout, k_smem_layout, do_smem_layout, v_smem_layout, kt_smem_layout, sdQ_epi_layout = layouts
        sQ = smem.allocate_tensor(self.q_dtype, q_smem_layout.outer, q_smem_layout.inner, self.buffer_align_bytes)
        sK = smem.allocate_tensor(self.q_dtype, k_smem_layout.outer, k_smem_layout.inner, self.buffer_align_bytes)
        sdO = smem.allocate_tensor(self.do_dtype, do_smem_layout.outer, do_smem_layout.inner, self.buffer_align_bytes)
        sV = smem.allocate_tensor(self.q_dtype, v_smem_layout.outer, v_smem_layout.inner, self.buffer_align_bytes)
        sKT = smem.allocate_tensor(self.q_dtype, kt_smem_layout.outer, kt_smem_layout.inner, self.buffer_align_bytes)
        s_epi_dQ = cute.make_tensor(
            cute.recast_ptr(sdO.iterator, sdQ_epi_layout.inner, self.q_dtype),
            sdQ_epi_layout.outer,
        )
        return sQ, sK, sdO, sV, sKT, s_epi_dQ

    @cute.jit
    def make_tmem_tiles(self, qk_mma, dov_mma, dsk_mma):
        tStS = qk_mma.make_fragment_C(cute.append(qk_mma.partition_shape_C(self.qk_mma_tiler[:2]), self.qk_acc_stage))
        tdPtdP = dov_mma.make_fragment_C(cute.append(dov_mma.partition_shape_C(self.dov_mma_tiler[:2]), self.dov_acc_stage))
        tdQtdQ = dsk_mma.make_fragment_C(dsk_mma.partition_shape_C(self.dsk_mma_tiler[:2]))
        tdQtdQ_staged = cute.make_tensor(
            tdQtdQ.iterator + self.tmem_dq_offset,
            cute.append(tdQtdQ.layout, cute.make_layout(self.iterations_dsk)),
        )
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
        tdPtdP = cute.make_tensor(tdPtdP.iterator + self.tmem_dp_offset, tdPtdP.layout)
        return tStS, tdPtdP, tdQtdQ_staged

    @cute.jit
    def make_fragments(self, qk_mma, dov_mma, dsk_mma, sQ, sK, sdO, sV, sKT):
# @ref flash4-2cta-bwd-dq-fragments
        # CuTe/CUTLASS owns the exact partitioning. The dQ kernel requests
        # Q/K score fragments, dO/V fragments for dP, K^T fragments for dQ,
        # and TMEM accumulator fragments for S, dP, and dQ.
        tSrQ = qk_mma.make_fragment_A(sQ)
        tSrK = qk_mma.make_fragment_B(sK)
        tdPrdO = dov_mma.make_fragment_A(sdO)
        tdPrV = dov_mma.make_fragment_B(sV)
        tdQrKT = dsk_mma.make_fragment_B(sKT)
        tStS = qk_mma.make_fragment_C(qk_mma.partition_shape_C(self.qk_mma_tiler[:2]))
        tdPtdP = dov_mma.make_fragment_C(dov_mma.partition_shape_C(self.dov_mma_tiler[:2]))
        tdQtdQ = dsk_mma.make_fragment_C(dsk_mma.partition_shape_C(self.dsk_mma_tiler[:2]))
        return tSrQ, tSrK, tdPrdO, tdPrV, tdQrKT, tStS, tdPtdP, tdQtdQ
# @end

    @cute.jit
    def __call__(self, Q, K, V, dQ, dO, lse_log2, dpsum, cumulative_s_q, cumulative_s_k, scale_softmax, stream):
# @ref flash4-2cta-bwd-dq-kernel
        qk_mma, dov_mma, dsk_mma = self.make_mma_atoms()
        tma_atoms = self.make_tma_atoms(Q, K, V, dO, K, dQ, qk_mma, dov_mma, dsk_mma)
        smem = cutlass.utils.SmemAllocator()
        sQ, sK, sdO, sV, sKT, s_epi_dQ = self.allocate_shared_tiles(smem, tma_atoms[-1])
        fragments = self.make_fragments(qk_mma, dov_mma, dsk_mma, sQ, sK, sdO, sV, sKT)
        tmem_tiles = self.make_tmem_tiles(qk_mma, dov_mma, dsk_mma)
        tdQtdQ_staged = self.mainloop(
            qk_mma,
            dov_mma,
            dsk_mma,
            fragments,
            tmem_tiles,
            lse_log2,
            dpsum,
            scale_softmax,
        )
        tma_atom_dQ, tma_dQ = tma_atoms[5]
        return self.epilogue(dQ, tma_atom_dQ, tma_dQ, tdQtdQ_staged, s_epi_dQ)
# @end

    @cute.jit
    def mainloop(self, qk_mma, dov_mma, dsk_mma, fragments, tmem_tiles, lse_log2, dpsum, scale_softmax):
# @ref flash4-2cta-bwd-pipeline flash4-2cta-bwd-dq-mainloop
        tSrQ, tSrK, tdPrdO, tdPrV, tdQrKT, _, _, _ = fragments
        tStS, tdPtdP, tdQtdQ_staged = tmem_tiles
        qk_mma.set(tcgen05.Field.ACCUMULATE, False)
        for q_iter in cutlass.range(self.q_stage, unroll=1):
            tStS_slice = tStS[None, None, None, q_iter]
            tSrQ_slice = tSrQ[None, None, None, q_iter]
            tSrK_slice = tSrK[None, None, None, q_iter]
            for kphase in cutlass.range(cute.size(tSrQ_slice, mode=[2]), unroll_full=True):
                cute.gemm(
                    qk_mma,
                    tStS_slice,
                    tSrQ_slice[None, None, kphase],
                    tSrK_slice[None, None, kphase],
                    tStS_slice,
                )
                qk_mma.set(tcgen05.Field.ACCUMULATE, True)

        dov_mma.set(tcgen05.Field.ACCUMULATE, False)
        tdPtdP_slice = tdPtdP[None, None, None, 0]
        for dov_iter in cutlass.range(self.do_stage, unroll=1):
            tdPrdO_slice = tdPrdO[None, None, None, dov_iter]
            tdPrV_slice = tdPrV[None, None, None, dov_iter]
            for kphase in cutlass.range(cute.size(tdPrdO_slice, mode=[2]), unroll_full=True):
                cute.gemm(
                    dov_mma,
                    tdPtdP_slice,
                    tdPrdO_slice[None, None, kphase],
                    tdPrV_slice[None, None, kphase],
                    tdPtdP_slice,
                )
                dov_mma.set(tcgen05.Field.ACCUMULATE, True)

        ds_tmem = self.compute_dsoftmax_from_tmem(tStS, tdPtdP, lse_log2, dpsum, scale_softmax)
        dsk_mma.set(tcgen05.Field.ACCUMULATE, False)
        for dsk_iter in cutlass.range(self.iterations_dsk, unroll=1):
            tdQtdQ_slice = tdQtdQ_staged[None, None, None, dsk_iter]
            ds_stage = dsk_iter % self.qk_acc_stage
            kt_stage = dsk_iter % self.dsk_acc_stage
            tdS = cute.make_tensor(ds_tmem[None, None, None, ds_stage].iterator, ds_tmem.layout)
            tdQrdS = dsk_mma.make_fragment_A(tdS)
            tdQrdS_slice = cute.make_tensor(cute.recast_ptr(tdS.iterator, dtype=self.q_dtype), tdQrdS.layout)
            tdQrKT_slice = tdQrKT[None, None, None, kt_stage]
            for kphase in cutlass.range(cute.size(tdQrKT_slice, mode=[2]), unroll_full=True):
                cute.gemm(
                    dsk_mma,
                    tdQtdQ_slice,
                    tdQrdS_slice[None, None, kphase],
                    tdQrKT_slice[None, None, kphase],
                    tdQtdQ_slice,
                )
                dsk_mma.set(tcgen05.Field.ACCUMULATE, True)
        # The source orders dP and dQ work to reuse TMEM slots; this is an
        # HD256 dedicated dQ pipeline, not the generic dQaccum postprocess path.
        return tdQtdQ_staged
# @end

    @cute.jit
    def compute_dsoftmax_from_tmem(self, tStS, tdPtdP, lse_log2, dpsum, scale_softmax):
        scores = self.load_tmem_fragment(tStS)
        dP = self.load_tmem_fragment(tdPtdP)
        probs = cute.math.exp2(scores * (scale_softmax * math.log2(math.e)) - lse_log2)
        dS = (dP - dpsum) * probs * scale_softmax
        self.store_tmem_fragment(tdPtdP, dS.to(self.q_dtype))
        return cute.make_tensor(tdPtdP.iterator, tdPtdP.layout)

    @cute.jit
    def load_tmem_fragment(self, tmem_tile):
        load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), self.acc_dtype)
        tiled_load = tcgen05.make_tmem_copy(load_atom, tmem_tile)
        thr_load = tiled_load.get_slice(cute.arch.thread_idx()[0])
        fragment = cute.make_rmem_tensor(thr_load.partition_D(tmem_tile).shape, self.acc_dtype)
        cute.copy(tiled_load, thr_load.partition_S(tmem_tile), fragment)
        cute.arch.fence_view_async_tmem_load()
        return fragment

    @cute.jit
    def store_tmem_fragment(self, tmem_tile, fragment):
        store_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype)
        tiled_store = tcgen05.make_tmem_copy(store_atom, tmem_tile)
        thr_store = tiled_store.get_slice(cute.arch.thread_idx()[0])
        cute.copy(tiled_store, fragment, thr_store.partition_D(tmem_tile))
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def epilogue(self, dQ, tma_atom_dQ, tma_dQ, tdQtdQ_staged, s_epi_dQ):
# @ref flash4-2cta-bwd-dq-store
        # Non-varlen HD256 stages TMEM dQ through epilogue SMEM and issues a TMA
        # shared-to-global store. Upstream keeps dsk_acc_stage as a one-deep
        # async pipeline; iterations_dsk is the logical dQ/K-slice loop.
        tidx = cute.arch.thread_idx()[0]
        tmem_copy_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype)
        for d_iter in cutlass.range(self.iterations_dsk, unroll=1):
            tdQtdQ = tdQtdQ_staged[(None, None), 0, 0, d_iter]
            tdQtdQ_epi = cute.zipped_divide(tdQtdQ, self.epi_tile)
            gdQ = cute.local_tile(tma_dQ, self.epi_tile, (0, d_iter))
            cdQ_local = cute.zipped_divide(cute.make_identity_tensor(self.epi_tile), self.epi_tile)
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tdQtdQ_epi)
            thr_tmem_load = tiled_tmem_load.get_slice(tidx)
            tTMEM_LOADtdQ = thr_tmem_load.partition_S(tdQtdQ_epi)
            tTMEM_LOADcdQ = thr_tmem_load.partition_D(cdQ_local)
            tTMrdQ = cute.make_rmem_tensor(tTMEM_LOADcdQ.shape, self.acc_dtype)
            cute.copy(tiled_tmem_load, tTMEM_LOADtdQ, tTMrdQ)
            tSMrdQ = cute.make_rmem_tensor(tTMrdQ.shape, dQ.element_type)
            tSMrdQ.store(tTMrdQ.load().to(dQ.element_type))
            for j in cutlass.range_constexpr(cute.size(tTMEM_LOADcdQ)):
                m_pos = tTMEM_LOADcdQ[j][0]
                n_pos = tTMEM_LOADcdQ[j][1]
                s_epi_dQ[m_pos, n_pos, 0] = tSMrdQ[j]
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=3, number_of_threads=128)
            td_sdQ, td_gdQ = cpasync.tma_partition(
                tma_atom_dQ,
                0,
                cute.make_layout(1),
                cute.group_modes(s_epi_dQ[None, None, 0], 0, 2),
                cute.group_modes(gdQ, 0, 2),
            )
            cute.copy(tma_atom_dQ, td_sdQ, td_gdQ)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        return dQ
# @end


class BlackwellFusedMultiHeadAttentionBackwardDKDVKernel:
    """Dedicated HD256 dK/dV kernel sketch."""

    def __init__(self, acc_dtype, mma_tile):
# @ref flash4-2cta-bwd-dkdv-kernel
        self.acc_dtype = acc_dtype
        self.mma_tile = mma_tile
        self.cta_tiler = mma_tile
        self.tile_shape_Q = mma_tile[0]
        self.tile_shape_K = mma_tile[1]
        self.tile_shape_dQ_K = mma_tile[2]
        self.tile_shape_dV_dO = mma_tile[2]
        self.KQ_mma_tiler = (
            2 * mma_tile[1],
            mma_tile[0],
            mma_tile[2],
        )
        self.VdO_mma_tiler = (
            2 * mma_tile[1],
            mma_tile[0],
            mma_tile[2],
        )
        self.PdO_mma_tiler = (
            2 * mma_tile[1],
            mma_tile[2],
            mma_tile[0],
        )
        self.dSQ_mma_tiler = (
            2 * mma_tile[1],
            mma_tile[2],
            mma_tile[0],
        )
        self.kq_mma_tiler = self.KQ_mma_tiler
        self.vdo_mma_tiler = self.VdO_mma_tiler
        self.pdo_mma_tiler = self.PdO_mma_tiler
        self.dsq_mma_tiler = self.dSQ_mma_tiler
        self.cta_group = tcgen05.CtaGroup.TWO
        self.q_dtype = FP8_DTYPE
        self.do_dtype = FP8_DTYPE
        self.load_mma_Q_stage = 1
        self.load_mma_K_stage = 1
        self.load_mma_V_stage = 1
        self.load_mma_QT_stage = 1
        self.load_mma_dO_stage = 1
        self.load_compute_LSE_stage = 1
        self.load_compute_sum_OdO_stage = 1
        self.mma_compute_S_stage = 1
        self.mma_compute_dP_stage = 1
        self.compute_mma_P_stage = 1
        self.compute_mma_dS_stage = 1
        self.mma_compute_dKdV_stage = 2
        self.total_epi_stages_dKV = self.mma_compute_dKdV_stage * 2
        self.tmem_dK_offset = 0
        self.tmem_dV_offset = 128
        self.tmem_S_offset = 256
        self.tmem_dP_offset = 320
        self.tmem_dk_offset = self.tmem_dK_offset
        self.tmem_dv_offset = self.tmem_dV_offset
        self.tmem_s_offset = self.tmem_S_offset
        self.tmem_dp_offset = self.tmem_dP_offset
        self.epi_tile = (mma_tile[1], 64)
        self.cluster_shape_mnk = (2, 1, 1)
        self.buffer_align_bytes = 1024
# @end

    @cute.jit
    def make_mma_atoms(self):
        kq_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.KQ_mma_tiler[:2],
        )
        vdo_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.VdO_mma_tiler[:2],
        )
        dSQ_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.dSQ_mma_tiler[:2],
        )
        pdO_mma = sm100_utils.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.PdO_mma_tiler[:2],
        )
        return kq_mma, vdo_mma, dSQ_mma, pdO_mma

    @cute.jit
    def make_tma_atoms(self, Q, K, V, dO, QT, dOT, dK, dV, kq_mma, vdo_mma, dSQ_mma, pdO_mma):
# @ref flash4-2cta-bwd-dkdv-kernel
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        k_smem_layout = sm100_utils.make_smem_layout_a(kq_mma, self.KQ_mma_tiler, self.q_dtype, self.load_mma_K_stage)
        v_smem_layout = sm100_utils.make_smem_layout_a(vdo_mma, self.VdO_mma_tiler, self.q_dtype, self.load_mma_V_stage)
        q_smem_layout = sm100_utils.make_smem_layout_b(kq_mma, self.KQ_mma_tiler, self.q_dtype, self.load_mma_Q_stage)
        qt_smem_layout = sm100_utils.make_smem_layout_b(dSQ_mma, self.dSQ_mma_tiler, self.q_dtype, self.load_mma_QT_stage)
        do_smem_layout = sm100_utils.make_smem_layout_b(vdo_mma, self.VdO_mma_tiler, self.do_dtype, self.load_mma_dO_stage)
        dot_smem_layout = sm100_utils.make_smem_layout_b(pdO_mma, self.PdO_mma_tiler, self.do_dtype, self.load_mma_dO_stage)
        tma_atom_K, tma_K = cute.nvgpu.make_tiled_tma_atom_A(tma_load_op, K, cute.select(k_smem_layout, mode=[0, 1, 2]), self.KQ_mma_tiler, kq_mma, self.cluster_shape_mnk)
        tma_atom_V, tma_V = cute.nvgpu.make_tiled_tma_atom_A(tma_load_op, V, cute.select(v_smem_layout, mode=[0, 1, 2]), self.VdO_mma_tiler, vdo_mma, self.cluster_shape_mnk)
        tma_atom_Q, tma_Q = cute.nvgpu.make_tiled_tma_atom_B(tma_load_op, Q, cute.select(q_smem_layout, mode=[0, 1, 2]), self.KQ_mma_tiler, kq_mma, self.cluster_shape_mnk)
        tma_atom_QT, tma_QT = cute.nvgpu.make_tiled_tma_atom_B(tma_load_op, QT, cute.select(qt_smem_layout, mode=[0, 1, 2]), self.dSQ_mma_tiler, dSQ_mma, self.cluster_shape_mnk)
        tma_atom_dO, tma_dO = cute.nvgpu.make_tiled_tma_atom_B(tma_load_op, dO, cute.select(do_smem_layout, mode=[0, 1, 2]), self.VdO_mma_tiler, vdo_mma, self.cluster_shape_mnk)
        tma_atom_dOT, tma_dOT = cute.nvgpu.make_tiled_tma_atom_B(tma_load_op, dOT, cute.select(dot_smem_layout, mode=[0, 1, 2]), self.PdO_mma_tiler, pdO_mma, self.cluster_shape_mnk)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        sdK_epi_layout = sm100_utils.make_smem_layout_epi(dK.element_type, utils.LayoutEnum.from_tensor(dK), self.epi_tile, self.total_epi_stages_dKV)
        sdV_epi_layout = sm100_utils.make_smem_layout_epi(dV.element_type, utils.LayoutEnum.from_tensor(dV), self.epi_tile, self.total_epi_stages_dKV)
        tma_atom_dK, tma_dK = cpasync.make_tiled_tma_atom(tma_store_op, dK, cute.select(sdK_epi_layout, mode=[0, 1]), self.epi_tile)
        tma_atom_dV, tma_dV = cpasync.make_tiled_tma_atom(tma_store_op, dV, cute.select(sdV_epi_layout, mode=[0, 1]), self.epi_tile)
        return (
            (tma_atom_K, tma_K),
            (tma_atom_V, tma_V),
            (tma_atom_Q, tma_Q),
            (tma_atom_QT, tma_QT),
            (tma_atom_dO, tma_dO),
            (tma_atom_dOT, tma_dOT),
            (tma_atom_dK, tma_dK),
            (tma_atom_dV, tma_dV),
            (k_smem_layout, q_smem_layout, v_smem_layout, do_smem_layout, qt_smem_layout, dot_smem_layout, sdK_epi_layout, sdV_epi_layout),
        )
# @end

    @cute.jit
    def allocate_shared_tiles(self, smem, layouts):
        k_smem_layout, q_smem_layout, v_smem_layout, do_smem_layout, qt_smem_layout, dot_smem_layout, sdK_epi_layout, sdV_epi_layout = layouts
        sK = smem.allocate_tensor(self.q_dtype, k_smem_layout.outer, k_smem_layout.inner, self.buffer_align_bytes)
        sQ = smem.allocate_tensor(self.q_dtype, q_smem_layout.outer, q_smem_layout.inner, self.buffer_align_bytes)
        sV = smem.allocate_tensor(self.q_dtype, v_smem_layout.outer, v_smem_layout.inner, self.buffer_align_bytes)
        sdO = smem.allocate_tensor(self.do_dtype, do_smem_layout.outer, do_smem_layout.inner, self.buffer_align_bytes)
        sdST = smem.allocate_tensor(self.q_dtype, qt_smem_layout.outer, qt_smem_layout.inner, self.buffer_align_bytes)
        sQT = smem.allocate_tensor(self.q_dtype, qt_smem_layout.outer, qt_smem_layout.inner, self.buffer_align_bytes)
        sP = smem.allocate_tensor(self.q_dtype, k_smem_layout.outer, k_smem_layout.inner, self.buffer_align_bytes)
        sdOT = smem.allocate_tensor(self.do_dtype, dot_smem_layout.outer, dot_smem_layout.inner, self.buffer_align_bytes)
        s_epi_dK = cute.make_tensor(cute.recast_ptr(sP.iterator, sdK_epi_layout.inner, self.q_dtype), sdK_epi_layout.outer)
        s_epi_dV = cute.make_tensor(cute.recast_ptr(sdOT.iterator, sdV_epi_layout.inner, self.q_dtype), sdV_epi_layout.outer)
        return sK, sQ, sV, sdO, sdST, sQT, sP, sdOT, s_epi_dK, s_epi_dV

    @cute.jit
    def make_tmem_tiles(self, kq_mma, vdo_mma, dSQ_mma, pdO_mma):
        tSTtST = kq_mma.make_fragment_C(kq_mma.partition_shape_C(self.KQ_mma_tiler[:2]))
        tdPTtdPT = vdo_mma.make_fragment_C(vdo_mma.partition_shape_C(self.VdO_mma_tiler[:2]))
        tdKtdK = dSQ_mma.make_fragment_C(dSQ_mma.partition_shape_C(self.dSQ_mma_tiler[:2]))
        tdVtdV = pdO_mma.make_fragment_C(pdO_mma.partition_shape_C(self.PdO_mma_tiler[:2]))
        tSTtST = cute.make_tensor(tSTtST.iterator + self.tmem_S_offset, tSTtST.layout)
        tdPTtdPT = cute.make_tensor(tdPTtdPT.iterator + self.tmem_dP_offset, tdPTtdPT.layout)
        tdKtdK = cute.make_tensor(tdKtdK.iterator + self.tmem_dK_offset, tdKtdK.layout)
        tdVtdV = cute.make_tensor(tdVtdV.iterator + self.tmem_dV_offset, tdVtdV.layout)
        return tSTtST, tdPTtdPT, tdKtdK, tdVtdV

    @cute.jit
    def make_fragments(self, kq_mma, vdo_mma, dSQ_mma, pdO_mma, sK, sQ, sV, sdO, sdST, sQT, sP, sdOT):
# @ref flash4-2cta-bwd-dkdv-fragments
        # The dK/dV kernel has its own fragment map: K/Q rebuild scores, V/dO
        # forms dP, dS/Q accumulates dK, and P/dO accumulates dV.
        tSTrK = kq_mma.make_fragment_A(sK)
        tSTrQ = kq_mma.make_fragment_B(sQ)
        tdPTrV = vdo_mma.make_fragment_A(sV)
        tdPTrdO = vdo_mma.make_fragment_B(sdO)
        tdKrdST = dSQ_mma.make_fragment_A(sdST)
        tdKrQT = dSQ_mma.make_fragment_B(sQT)
        tdVrP = pdO_mma.make_fragment_A(sP)
        tdVrdOT = pdO_mma.make_fragment_B(sdOT)
        tdKtdK = dSQ_mma.make_fragment_C(dSQ_mma.partition_shape_C(self.dSQ_mma_tiler[:2]))
        tdVtdV = pdO_mma.make_fragment_C(pdO_mma.partition_shape_C(self.PdO_mma_tiler[:2]))
        return tSTrK, tSTrQ, tdPTrV, tdPTrdO, tdKrdST, tdKrQT, tdVrP, tdVrdOT, tdKtdK, tdVtdV, sP, sdST
# @end

    @cute.jit
    def __call__(self, Q, K, V, dK, dV, dO, lse_log2, dpsum, cumulative_s_q, cumulative_s_k, scale_softmax, stream):
# @ref flash4-2cta-bwd-dkdv-kernel
        kq_mma, vdo_mma, dSQ_mma, pdO_mma = self.make_mma_atoms()
        tma_atoms = self.make_tma_atoms(Q, K, V, dO, Q, dO, dK, dV, kq_mma, vdo_mma, dSQ_mma, pdO_mma)
        smem = cutlass.utils.SmemAllocator()
        shared_tiles = self.allocate_shared_tiles(smem, tma_atoms[-1])
        fragments = self.make_fragments(kq_mma, vdo_mma, dSQ_mma, pdO_mma, *shared_tiles[:8])
        tmem_tiles = self.make_tmem_tiles(kq_mma, vdo_mma, dSQ_mma, pdO_mma)
        tdKtdK, tdVtdV = self.mainloop(
            kq_mma,
            vdo_mma,
            dSQ_mma,
            pdO_mma,
            fragments,
            tmem_tiles,
            lse_log2,
            dpsum,
            scale_softmax,
        )
        tma_atom_dK, tma_dK = tma_atoms[6]
        tma_atom_dV, tma_dV = tma_atoms[7]
        s_epi_dK, s_epi_dV = shared_tiles[8], shared_tiles[9]
        return self.epilogue(dK, dV, tma_atom_dK, tma_dK, tma_atom_dV, tma_dV, tdKtdK, tdVtdV, s_epi_dK, s_epi_dV, scale_softmax)
# @end

    @cute.jit
    def mainloop(self, kq_mma, vdo_mma, dSQ_mma, pdO_mma, fragments, tmem_tiles, lse_log2, dpsum, scale_softmax):
# @ref flash4-2cta-bwd-dkdv-mainloop
        tSTrK, tSTrQ, tdPTrV, tdPTrdO, tdKrdST, tdKrQT, tdVrP, tdVrdOT, _, _, sP, sdST = fragments
        tSTtST, tdPTtdPT, tdKtdK, tdVtdV = tmem_tiles
        kq_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kphase in cutlass.range(cute.size(tSTrK, mode=[2]), unroll_full=True):
            cute.gemm(kq_mma, tSTtST, tSTrK[None, None, kphase, 0], tSTrQ[None, None, kphase, 0], tSTtST)
            kq_mma.set(tcgen05.Field.ACCUMULATE, True)

        vdo_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kphase in cutlass.range(cute.size(tdPTrV, mode=[2]), unroll_full=True):
            cute.gemm(vdo_mma, tdPTtdPT, tdPTrV[None, None, kphase, 0], tdPTrdO[None, None, kphase, 0], tdPTtdPT)
            vdo_mma.set(tcgen05.Field.ACCUMULATE, True)

        self.compute_p_and_ds_from_tmem(tSTtST, tdPTtdPT, sP, sdST, lse_log2, dpsum, scale_softmax)
        dSQ_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kphase in cutlass.range(cute.size(tdKrdST, mode=[2]), unroll_full=True):
            cute.gemm(dSQ_mma, tdKtdK, tdKrdST[None, None, kphase, 0], tdKrQT[None, None, kphase, 0], tdKtdK)
            dSQ_mma.set(tcgen05.Field.ACCUMULATE, True)

        pdO_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kphase in cutlass.range(cute.size(tdVrP, mode=[2]), unroll_full=True):
            cute.gemm(pdO_mma, tdVtdV, tdVrP[None, None, kphase, 0], tdVrdOT[None, None, kphase, 0], tdVtdV)
            pdO_mma.set(tcgen05.Field.ACCUMULATE, True)
        return tdKtdK, tdVtdV
# @end

    @cute.jit
    def compute_p_and_ds_from_tmem(self, score_tmem, dp_tmem, sP, sdST, lse_log2, dpsum, scale_softmax):
        scores = self.load_tmem_fragment(score_tmem)
        dP = self.load_tmem_fragment(dp_tmem)
        probs = cute.math.exp2(scores * (scale_softmax * math.log2(math.e)) - lse_log2)
        dS = (dP - dpsum) * probs
        P = probs.to(self.q_dtype)
        dS_quantized = dS.to(self.q_dtype)
        self.reg_to_smem_mma128x128_2cta(P, sP)
        self.reg_to_smem_mma128x128_2cta(dS_quantized, sdST)

    @cute.jit
    def load_tmem_fragment(self, tmem_tile):
        load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), self.acc_dtype)
        tiled_load = tcgen05.make_tmem_copy(load_atom, tmem_tile)
        thr_load = tiled_load.get_slice(cute.arch.thread_idx()[0])
        fragment = cute.make_rmem_tensor(thr_load.partition_D(tmem_tile).shape, self.acc_dtype)
        cute.copy(tiled_load, thr_load.partition_S(tmem_tile), fragment)
        cute.arch.fence_view_async_tmem_load()
        return fragment

    @cute.jit
    def reg_to_smem_mma128x128_2cta(self, fragment, smem_tile):
        coords = cute.make_identity_tensor(smem_tile.shape)
        thread_idx = cute.arch.thread_idx()[0] % 128
        for i in cutlass.range_constexpr(cute.size(fragment)):
            m_pos = coords[i][0] + (thread_idx // 64) * 64
            n_pos = coords[i][1] % 64
            smem_tile[m_pos, n_pos, 0] = fragment[i]
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=4, number_of_threads=256)

    @cute.jit
    def epilogue(self, dK, dV, tma_atom_dK, tma_dK, tma_atom_dV, tma_dV, tdKtdK, tdVtdV, s_epi_dK, s_epi_dV, scale_softmax):
# @ref flash4-2cta-bwd-dkdv-store
        # HD256 stages dK/dV through shared epilogue buffers before TMA S2G
        # stores, with a fallback per-thread store where the source requires it.
        self.stage_and_store_tmem_accumulator(tdVtdV, s_epi_dV, tma_atom_dV, tma_dV, dV, Float32(1.0), barrier_id=5)
        self.stage_and_store_tmem_accumulator(tdKtdK, s_epi_dK, tma_atom_dK, tma_dK, dK, scale_softmax, barrier_id=6)
        return dK, dV
# @end

    @cute.jit
    def stage_and_store_tmem_accumulator(self, tmem_acc, s_epi, tma_atom, tma_tensor, output, scale, barrier_id: int):
        tidx = cute.arch.thread_idx()[0]
        tmem_copy_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype)
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tmem_acc)
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        coords = cute.make_identity_tensor(self.epi_tile)
        tTMEM_LOADsrc = thr_tmem_load.partition_S(tmem_acc)
        tTMEM_LOADcoord = thr_tmem_load.partition_D(coords)
        rmem_acc = cute.make_rmem_tensor(tTMEM_LOADcoord.shape, self.acc_dtype)
        cute.copy(tiled_tmem_load, tTMEM_LOADsrc, rmem_acc)
        rmem_out = cute.make_rmem_tensor(rmem_acc.shape, output.element_type)
        rmem_out.store((rmem_acc.load() * scale).to(output.element_type))
        for i in cutlass.range_constexpr(cute.size(tTMEM_LOADcoord)):
            m_pos = tTMEM_LOADcoord[i][0]
            n_pos = tTMEM_LOADcoord[i][1]
            stage_pos = n_pos // self.epi_tile[1]
            n_within = n_pos % self.epi_tile[1]
            s_epi[m_pos, n_within, stage_pos] = rmem_out[i]
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=256)
        for stage in cutlass.range_constexpr(cute.size(s_epi, mode=[2])):
            td_smem, td_gmem = cpasync.tma_partition(
                tma_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(s_epi[None, None, stage], 0, 2),
                cute.group_modes(tma_tensor[None, None, stage], 0, 2),
            )
            cute.copy(tma_atom, td_smem, td_gmem)
            cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)


class FlashAttention4BackwardMmaMap:
    """HD256 backward wrapper map for the code pane.

    Generic SM100 backward is shown in FlashAttention4GenericBackwardSm100Map.
    Dao-AILab's HD256 route is a wrapper that launches a dedicated dQ kernel
    first and a dedicated dK/dV kernel second.
    """

    def __init__(
        self,
        head_dim: int = 256,
        head_dim_v: int = 256,
        m_block_size: int = 128,
        n_block_size: int = 128,
        arch: int = 100,
        use_2cta_instrs: bool = False,
    ):
# @ref flash4-2cta-bwd-hd256-route
        arch_major = arch // 10
        use_dedicated_hd256_kernel = arch_major in (10, 11) and head_dim == 256 and head_dim_v == 256
        use_2cta_instrs = use_2cta_instrs or use_dedicated_hd256_kernel
        assert use_dedicated_hd256_kernel
        assert use_2cta_instrs
        dq_tile_mn = (128, 128)
        dkdv_tile_mn = (128, 64)
# @end
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
# @ref flash4-2cta-bwd-traffic
        self.cta_group = tcgen05.CtaGroup.TWO
        self.backward_2cta_mma_tile = (2 * m_block_size, n_block_size, n_block_size)
        self.dq_repacked_mma_tile = (m_block_size, 2 * n_block_size, head_dim)
        self.each_cta_stages_half_operand_b = True
        self.each_cta_keeps_accumulator_slice = True
# @end
        self.acc_dtype = cutlass.Float32
# @ref flash4-2cta-bwd-split-kernels
        # Upstream HD256 constructs two dedicated kernels at the wrapper boundary.
        self.dq_kernel = BlackwellFusedMultiHeadAttentionBackwardDQKernel(
            self.acc_dtype,
            (dq_tile_mn[0], dq_tile_mn[1], head_dim),
            is_persistent=False,
            split_head=False,
        )
        self.dkdv_kernel = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel(
            self.acc_dtype,
            (dkdv_tile_mn[0], dkdv_tile_mn[1], head_dim),
        )
# @end
        self.q_dtype = FP8_DTYPE
        self.do_dtype = FP8_DTYPE
        self.kq_mma_tiler = (2 * n_block_size, m_block_size, 128)
        self.vdo_mma_tiler = (2 * n_block_size, m_block_size, 128)
        self.pdo_mma_tiler = (2 * n_block_size, head_dim, m_block_size)
        self.dsq_mma_tiler = (2 * n_block_size, head_dim, m_block_size)
        self.dsk_mma_tiler = (2 * m_block_size, head_dim, n_block_size)

    @cute.jit
    def make_backward_mma_atoms(self):
# @ref flash4-2cta-bwd-mma-atoms
        score_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.kq_mma_tiler[:2],
        )
        dp_mma = sm100_utils.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.vdo_mma_tiler[:2],
        )
        dv_mma = sm100_utils.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.pdo_mma_tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        dk_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.dsq_mma_tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        dq_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.dsk_mma_tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
# @end
        return score_mma, dp_mma, dv_mma, dk_mma, dq_mma

    def compile_hd256_backward_args(self, dQ, dK, dV, qhead_per_kvhead: int = 1):
# @ref flash4-2cta-bwd-interface-args
        # Host dispatch keeps the historical argument ABI: the HD256 wrapper
        # receives final dQ through the dQ_accum slot, then the dedicated dQ
        # kernel stores dQ directly and skips generic dQaccum postprocess.
        use_dedicated_hd256_kernel = True
        dq_accum_scratch = None
        dkv_postprocess = qhead_per_kvhead > 1 and not use_dedicated_hd256_kernel
        dQ_accum_argument = dQ
        dK_argument = dK if not dkv_postprocess else None
        dV_argument = dV if not dkv_postprocess else None
        return dq_accum_scratch, dQ_accum_argument, dK_argument, dV_argument
# @end

    @cute.jit
    def launch_hd256_backward(
        self,
        Q,
        K,
        V,
        dO,
        dQ_accum_argument,
        dK,
        dV,
        lse_log2,
        dpsum,
        cumulative_s_q,
        cumulative_s_k,
        scale_softmax,
        stream,
        dQ_semaphore=None,
        dK_semaphore=None,
        dV_semaphore=None,
    ):
# @ref flash4-2cta-bwd-launch-calls
        # The wrapper receives dQ through the legacy dQ_accum argument slot and
        # forbids the generic semaphore reduction path.
        assert dQ_semaphore is None and dK_semaphore is None and dV_semaphore is None
        assert dQ_accum_argument is not None
        dQ = dQ_accum_argument
        self.dq_kernel(
            Q,
            K,
            V,
            dQ,
            dO,
            lse_log2,
            dpsum,
            cumulative_s_q,
            cumulative_s_k,
            scale_softmax,
            stream,
        )
        self.dkdv_kernel(
            Q,
            K,
            V,
            dK,
            dV,
            dO,
            lse_log2,
            dpsum,
            cumulative_s_q,
            cumulative_s_k,
            scale_softmax,
            stream,
        )
# @end

    @cute.jit
    def generic_2cta_dsmem_repack(self, sdS, sdS_repacked, cta_rank):
# @ref flash4-2cta-bwd-dsmem
        # dQ reduces over KV/N, while 2-CTA MMA naturally splits the output tile.
        # The paper/generic 2-CTA reduction can exchange dS through DSMEM and
        # repack it so each CTA owns M/2 output rows with the full 2N reduction.
        # Current upstream HD256 does not use placeholder helpers for that in
        # the wrapper; it launches the dedicated dQ kernel above.
        peer_rank = cta_rank ^ 1
        peer_sdS = cute.make_tensor(cute.arch.map_shared_rank(sdS.iterator, peer_rank), sdS.layout)
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        row_offset = cta_rank * (self.m_block_size // 2)
        for row in cutlass.range_constexpr(self.m_block_size // 2):
            for col in cutlass.range_constexpr(self.n_block_size):
                sdS_repacked[row_offset + row, col] = sdS[row, col]
                sdS_repacked[row_offset + row, col + self.n_block_size] = peer_sdS[row, col]
        cute.arch.cluster_sync()
        return sdS_repacked
# @end
