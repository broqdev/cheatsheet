import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma_init,
    warpgroup_mma,
    warpgroup_mma_wait,
)


# Hopper-focused Gluon sketch of FlashAttention-3.
# Uses Hopper TMA, mbarriers, WGMMA, and warp specialization.


@gluon.aggregate
class RingState:
    index: gl.tensor
    phase: gl.tensor
    num_buffers: gl.constexpr

    @gluon.jit
    def next(self):
        next_index = self.index + 1
        rollover = next_index == self.num_buffers
        index = gl.where(rollover, 0, next_index)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return RingState(index, phase, self.num_buffers)


@gluon.aggregate
class SmemChannel:
    mem: gl.shared_memory_descriptor
    ready: gl.shared_memory_descriptor
    empty: gl.shared_memory_descriptor
    num_buffers: gl.constexpr

    @gluon.jit
    def alloc(shape: gl.constexpr, dtype: gl.constexpr, layout: gl.constexpr, num_buffers: gl.constexpr):
        mem = gl.allocate_shared_memory(dtype, [num_buffers] + shape, layout)
        ready = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
        empty = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
        return SmemChannel(mem, ready, empty, num_buffers)

# @ref flash3-cta-pipeline
    @gluon.jit
    def init(self, producers: gl.constexpr = 1, consumers: gl.constexpr = 1):
        for i in gl.static_range(self.num_buffers):
            mbarrier.init(self.ready.index(i), count=producers)
            mbarrier.init(self.empty.index(i), count=consumers)
            mbarrier.arrive(self.empty.index(i), count=consumers)
# @end

    @gluon.jit
    def state(self):
        return RingState(gl.to_tensor(0), gl.to_tensor(0), self.num_buffers)

    @gluon.jit
    def producer_acquire(self, state):
        mbarrier.wait(self.empty.index(state.index), state.phase ^ 1)
        return self.mem.index(state.index), self.ready.index(state.index), state.next()

    @gluon.jit
    def consumer_acquire(self, state):
        mbarrier.wait(self.ready.index(state.index), state.phase)
        return self.mem.index(state.index), self.empty.index(state.index), state.next()


@gluon.aggregate
class Flash3Config:
    qk_scale: gl.tensor
    grad_scale: gl.tensor
    v_descale: gl.tensor
    Z: gl.tensor
    H: gl.tensor
    N_CTX: gl.tensor
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    HEAD_DIM: gl.constexpr
    NUM_SMS: gl.constexpr
    dtype: gl.constexpr
    q_layout: gl.constexpr
    k_layout: gl.constexpr
    v_layout: gl.constexpr
    o_layout: gl.constexpr
    row_layout: gl.constexpr
    mma_layout: gl.constexpr
    q_operand_layout: gl.constexpr
    p_operand_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, qk_scale, grad_scale, v_descale, Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM, NUM_SMS, dtype, num_warps):
        self.qk_scale = qk_scale
        self.grad_scale = grad_scale
        self.v_descale = v_descale
        self.Z = Z
        self.H = H
        self.N_CTX = N_CTX
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.NUM_SMS = gl.constexpr(NUM_SMS)
        self.dtype = gl.constexpr(dtype)
        self.q_layout = gl.constexpr(gl.NVMMASharedLayout.get_default_for([BLOCK_M, HEAD_DIM], dtype))
        self.k_layout = gl.constexpr(gl.NVMMASharedLayout.get_default_for([BLOCK_N, HEAD_DIM], dtype))
        self.v_layout = gl.constexpr(gl.NVMMASharedLayout.get_default_for([HEAD_DIM, BLOCK_N], dtype))
        self.o_layout = gl.constexpr(gl.NVMMASharedLayout.get_default_for([BLOCK_M, HEAD_DIM], dtype))
        self.row_layout = gl.constexpr(gl.NVMMASharedLayout.get_default_for([BLOCK_M, 1], gl.float32))
        self.mma_layout = gl.constexpr(
            gl.NVMMADistributedLayout(
                version=[3, 0],
                warps_per_cta=[num_warps, 1],
                instr_shape=[16, 64, 256 // dtype.primitive_bitwidth],
            )
        )
        self.q_operand_layout = gl.constexpr(
            gl.DotOperandLayout(operand_index=0, parent=self.mma_layout, k_width=32 // dtype.primitive_bitwidth)
        )
        self.p_operand_layout = gl.constexpr(
            gl.DotOperandLayout(operand_index=0, parent=self.mma_layout, k_width=32 // dtype.primitive_bitwidth)
        )


@gluon.aggregate
class Flash3Program:
    start_m: gl.tensor
    off_hz: gl.tensor
    base_y: gl.tensor
    qo_y: gl.tensor

    @gluon.jit
    def loop_bounds(self, config):
        return 0, config.N_CTX


@gluon.jit
def get_program(config, tile_id):
    pid_m = tile_id // (config.Z * config.H)
    off_hz = tile_id - pid_m * (config.Z * config.H)
    off_z = off_hz // config.H
    off_h = off_hz - off_z * config.H
    base_y = off_z * (config.N_CTX * config.H) + off_h * config.N_CTX
    return Flash3Program(pid_m, off_hz, base_y, base_y + pid_m * config.BLOCK_M)


@gluon.aggregate
class Flash3BwdProgram:
    start_n: gl.tensor
    off_hz: gl.tensor
    base_token: gl.tensor
    base_vec: gl.tensor


@gluon.jit
def get_bwd_program(config, tile_id):
    num_k_tiles = gl.cdiv(config.N_CTX, config.BLOCK_N)
    off_hz = tile_id // num_k_tiles
    pid_n = tile_id - off_hz * num_k_tiles
    base_token = off_hz * config.N_CTX
    base_vec = base_token * config.HEAD_DIM
    return Flash3BwdProgram(pid_n * config.BLOCK_N, off_hz, base_token, base_vec)


@gluon.jit
def tma_load(desc, offset_y, channel, state):
    smem, ready_bar, state = channel.producer_acquire(state)
    mbarrier.expect(ready_bar, desc.block_type.nbytes)
    tma.async_load(desc, [offset_y, 0], ready_bar, smem)
    return state


@gluon.jit
def tma_load_v_transposed(desc, offset_y, channel, state):
    smem, ready_bar, state = channel.producer_acquire(state)
    mbarrier.expect(ready_bar, desc.block_type.nbytes)
    tma.async_load(desc, [0, offset_y], ready_bar, smem)
    return state


@gluon.jit
def wgmma_qk(config, q_smem, k_smem, acc):
    q = q_smem.load(config.q_operand_layout)
    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
    return warpgroup_mma(q, k_smem.permute((1, 0)), acc, is_async=True, use_acc=False)


@gluon.jit
def wgmma_pv(config, p_smem, v_smem, acc, use_acc: gl.constexpr):
    p = p_smem.load(config.p_operand_layout)
    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
    return warpgroup_mma(p, v_smem.permute((1, 0)), acc, is_async=True, use_acc=use_acc)


# @ref flash3-cta-forward-label
@gluon.jit
def _flash3_fwd_load(config, channels, descs):
    q_chnl, k_chnl, v_chnl, p_chnl, o_chnl = channels
    desc_q, desc_k, desc_v, desc_o = descs
    q_state = q_chnl.state()
    k_state = k_chnl.state()
    v_state = v_chnl.state()
# @end

# @ref flash3-cta-producer-loop flash3-cta-wait-stage
    num_tiles = gl.cdiv(config.N_CTX, config.BLOCK_M) * config.Z * config.H
    for tile_id in range(gl.program_id(0), num_tiles, config.NUM_SMS):
        prog = get_program(config, tile_id)
        lo, hi = prog.loop_bounds(config)
# @end

# @ref flash3-cta-load-q flash3-cta-commit-q flash3-cta-wait-q flash3-consumer-wait-qk0
        q_state = tma_load(desc_q, prog.qo_y, q_chnl, q_state)
# @end

# @ref flash3-cta-load-kv flash3-cta-commit-kv flash3-cta-wait-k flash3-consumer-wait-kj
        for start_n in range(lo, hi, config.BLOCK_N):
            offset_y = prog.base_y + start_n
            k_state = tma_load(desc_k, offset_y, k_chnl, k_state)
# @end

# @ref flash3-cta-load-kv flash3-cta-commit-kv flash3-cta-wait-v flash3-consumer-wait-vprev flash3-consumer-wait-vlast flash3-fp8-v-load
            v_state = tma_load_v_transposed(desc_v, offset_y, v_chnl, v_state)
# @end


# @ref flash3-consumer-forward-label
@gluon.jit
def _flash3_fwd_compute(config, channels, descs, M):
    q_chnl, k_chnl, v_chnl, p_chnl, o_chnl = channels
    q_state = q_chnl.state()
    k_state = k_chnl.state()
    v_state = v_chnl.state()
    p_state = p_chnl.state()
    o_state = o_chnl.state()
# @end

# @ref flash3-cta-consumer-loop flash3-consumer-loop
    num_tiles = gl.cdiv(config.N_CTX, config.BLOCK_M) * config.Z * config.H
    for tile_id in range(gl.program_id(0), num_tiles, config.NUM_SMS):
        prog = get_program(config, tile_id)
        lo, hi = prog.loop_bounds(config)
# @end

# @ref flash3-cta-init-state flash3-consumer-init
        # Persistent CTAs visit multiple output tiles; each tile needs a fresh
        # online-softmax state or rows leak state from the previous tile.
        m_i = gl.full([config.BLOCK_M], -float("inf"), gl.float32)
        l_i = gl.full([config.BLOCK_M], 0.0, gl.float32)
        o_acc = warpgroup_mma_init(gl.zeros((config.BLOCK_M, config.HEAD_DIM), dtype=gl.float32, layout=config.mma_layout))
# @end

# @ref flash3-cta-wait-q flash3-consumer-wait-qk0
        q_smem, q_empty, q_state = q_chnl.consumer_acquire(q_state)
# @end

# @ref flash3-cta-wait-k flash3-consumer-wait-kj
        k_smem, k_empty, k_state = k_chnl.consumer_acquire(k_state)
# @end

# @ref flash3-cta-score flash3-consumer-score-cur flash3-fp8-qk-descale
        qk_acc = warpgroup_mma_init(
            gl.zeros((config.BLOCK_M, config.BLOCK_N), dtype=gl.float32, layout=config.mma_layout)
        )
        qk_acc = wgmma_qk(config, q_smem, k_smem, qk_acc)
        qk_acc = warpgroup_mma_wait(num_outstanding=0, deps=(qk_acc,))
# @end

# @ref flash3-consumer-release-k0 flash3-consumer-release-buffer
        fence_async_shared()
        mbarrier.arrive(k_empty)
# @end

# @ref flash3-cta-score flash3-consumer-score-cur
        s_cur = qk_acc * config.qk_scale
# @end

# @ref flash3-fwd-causal-mask
        q_rows = prog.start_m * config.BLOCK_M + gl.arange(0, config.BLOCK_M)[:, None]
        k_cols = lo + gl.arange(0, config.BLOCK_N)[None, :]
        s_cur = gl.where(q_rows >= k_cols, s_cur, -float("inf"))
# @end

# @ref flash3-cta-rowmax
        m_next = gl.maximum(m_i, gl.max(s_cur, axis=1))
# @end

# @ref flash3-cta-prob-l flash3-consumer-online-cur
        alpha = gl.exp2(m_i - m_next)
        p_cur = gl.exp2(s_cur - m_next[:, None])
        l_i = l_i * alpha + gl.sum(p_cur, axis=1)
        m_i = m_next
# @end

# @ref flash3-consumer-wait-vprev
        v_smem, v_empty, v_state = v_chnl.consumer_acquire(v_state)
# @end

# @ref flash3-cta-output flash3-consumer-output-prev flash3-fp8-p-cast
        p_smem, p_ready, p_state = p_chnl.producer_acquire(p_state)
        p_smem.store(p_cur.to(config.dtype))
        fence_async_shared()
        mbarrier.arrive(p_ready)
        o_acc = wgmma_pv(config, p_smem, v_smem, o_acc, use_acc=True)
# @end

# @ref flash3-consumer-wait-kj
        for start_n in range(lo + config.BLOCK_N, hi, config.BLOCK_N):
            k_smem, k_empty, k_state = k_chnl.consumer_acquire(k_state)
# @end

# @ref flash3-consumer-score-next flash3-fp8-qk-descale
            qk_next = warpgroup_mma_init(
                gl.zeros((config.BLOCK_M, config.BLOCK_N), dtype=gl.float32, layout=config.mma_layout)
            )
            qk_next = wgmma_qk(config, q_smem, k_smem, qk_next)
# @end

# @ref flash3-consumer-wait-score-next
            qk_next = warpgroup_mma_wait(num_outstanding=0, deps=(qk_next,))
# @end

# @ref flash3-consumer-release-buffer
            fence_async_shared()
            mbarrier.arrive(k_empty)
# @end

# @ref flash3-consumer-score-next flash3-consumer-wait-score-next
            s_next = qk_next * config.qk_scale
# @end

# @ref flash3-fwd-causal-mask
            k_cols = start_n + gl.arange(0, config.BLOCK_N)[None, :]
            s_next = gl.where(q_rows >= k_cols, s_next, -float("inf"))
# @end

# @ref flash3-consumer-online-next
            m_next = gl.maximum(m_i, gl.max(s_next, axis=1))
            p_next = gl.exp2(s_next - m_next[:, None])
# @end

# @ref flash3-consumer-rescale-output
            # The previous PV WGMMA must finish before its accumulator is
            # rescaled into the new online-softmax frame.
            o_acc = warpgroup_mma_wait(num_outstanding=0, deps=(o_acc,))
            mbarrier.arrive(v_empty)
            o_acc = o_acc * gl.exp2(m_i - m_next)[:, None]
            l_i = l_i * gl.exp2(m_i - m_next) + gl.sum(p_next, axis=1)
            m_i = m_next
# @end

# @ref flash3-consumer-copy-next
            p_cur = p_next
# @end

# @ref flash3-consumer-wait-vlast flash3-consumer-output-last flash3-fp8-p-cast
            # Every K tile has a matching transposed V tile. Keep streaming
            # P_j V_j here so middle key blocks contribute to O_i.
            v_smem, v_empty, v_state = v_chnl.consumer_acquire(v_state)
            p_smem, p_ready, p_state = p_chnl.producer_acquire(p_state)
            p_smem.store(p_cur.to(config.dtype))
            fence_async_shared()
            mbarrier.arrive(p_ready)
            o_acc = wgmma_pv(config, p_smem, v_smem, o_acc, use_acc=True)
# @end

# @ref flash3-cta-consumer-end
        o_acc = warpgroup_mma_wait(num_outstanding=0, deps=(o_acc,))
        mbarrier.arrive(q_empty)
        mbarrier.arrive(v_empty)
# @end

# @ref flash3-cta-normalize flash3-consumer-epilogue flash3-fp8-v-descale
        out = (o_acc / l_i[:, None]) * config.v_descale
        lse = (m_i + gl.log2(l_i)) * 0.6931471805599453
        offs_m = prog.start_m * config.BLOCK_M + gl.arange(0, config.BLOCK_M)
# @end

# @ref flash3-cta-write flash3-consumer-epilogue flash3-fp8-output-store
        # FA backward reads LSE, so the forward sketch must publish it beside O.
        gl.store(M + prog.off_hz * config.N_CTX + offs_m, lse)
        o_smem, o_ready, o_state = o_chnl.producer_acquire(o_state)
        o_smem.store(out.to(config.dtype))
        fence_async_shared()
        mbarrier.arrive(o_ready)
# @end


@gluon.jit
def _flash3_fwd_store(config, channels, descs):
    q_chnl, k_chnl, v_chnl, p_chnl, o_chnl = channels
    desc_q, desc_k, desc_v, desc_o = descs
    o_state = o_chnl.state()
    num_tiles = gl.cdiv(config.N_CTX, config.BLOCK_M) * config.Z * config.H
    for tile_id in range(gl.program_id(0), num_tiles, config.NUM_SMS):
        prog = get_program(config, tile_id)
        o_smem, o_empty, o_state = o_chnl.consumer_acquire(o_state)
        tma.async_copy_shared_to_global(desc_o, [prog.qo_y, 0], o_smem)
        tma.store_wait(pendings=0)
        mbarrier.arrive(o_empty)


# @ref flash3-backward-label flash3-bwd-preprocess
@gluon.jit
def flash3_bwd_preprocess(O, dO, D, N_CTX: gl.constexpr, BLOCK_M: gl.constexpr, HEAD_DIM: gl.constexpr):
    pid_m = gl.program_id(0)
    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M)
# @end

# @ref flash3-bwd-preprocess
    o = gl.load(O + offs_m[:, None] * HEAD_DIM + gl.arange(0, HEAD_DIM)[None, :])
    do = gl.load(dO + offs_m[:, None] * HEAD_DIM + gl.arange(0, HEAD_DIM)[None, :])
    gl.store(D + offs_m, gl.sum(o * do, axis=1))
# @end


@gluon.jit
def _flash3_bwd_load(config, channels, Q, K, V, dO, L, D):
    k_chnl, v_chnl, q_chnl, do_chnl, l_chnl, d_chnl, dq_chnl = channels
    k_state = k_chnl.state()
    v_state = v_chnl.state()
    q_state = q_chnl.state()
    do_state = do_chnl.state()
    l_state = l_chnl.state()
    d_state = d_chnl.state()
    num_tiles = gl.cdiv(config.N_CTX, config.BLOCK_N) * config.Z * config.H

# @ref flash3-bwd-producer-loop
    for tile_id in range(gl.program_id(0), num_tiles, config.NUM_SMS):
        prog = get_bwd_program(config, tile_id)
        offs_n = prog.start_n + gl.arange(0, config.BLOCK_N)
        offs_d = gl.arange(0, config.HEAD_DIM)
# @end

# @ref flash3-bwd-load-kv flash3-bwd-commit-kv flash3-fp8-v-load
        # Producer stages K and the FP8 transposed V tile once per K block.
        k_smem, k_ready, k_state = k_chnl.producer_acquire(k_state)
        v_smem, v_ready, v_state = v_chnl.producer_acquire(v_state)
        k_smem.store(gl.load(K + prog.base_vec + offs_n[:, None] * config.HEAD_DIM + offs_d[None, :]))
        v_smem.store(gl.load(V + offs_d[:, None] * config.N_CTX + prog.base_token + offs_n[None, :]))
        fence_async_shared()
        mbarrier.arrive(k_ready)
        mbarrier.arrive(v_ready)
# @end

# @ref flash3-bwd-load-q-do flash3-bwd-commit-q-do flash3-bwd-partition-qkv
        for start_m in range(0, config.N_CTX, config.BLOCK_M):
            offs_m = start_m + gl.arange(0, config.BLOCK_M)
            q_smem, q_ready, q_state = q_chnl.producer_acquire(q_state)
            do_smem, do_ready, do_state = do_chnl.producer_acquire(do_state)
            l_smem, l_ready, l_state = l_chnl.producer_acquire(l_state)
            d_smem, d_ready, d_state = d_chnl.producer_acquire(d_state)
            q_smem.store(gl.load(Q + prog.base_vec + offs_m[:, None] * config.HEAD_DIM + offs_d[None, :]))
            do_smem.store(gl.load(dO + prog.base_vec + offs_m[:, None] * config.HEAD_DIM + offs_d[None, :]))
            l_smem.store(gl.load(L + prog.base_token + offs_m)[:, None])
            d_smem.store(gl.load(D + prog.base_token + offs_m)[:, None])
            fence_async_shared()
            mbarrier.arrive(q_ready)
            mbarrier.arrive(do_ready)
            mbarrier.arrive(l_ready)
            mbarrier.arrive(d_ready)
# @end


@gluon.jit
def _flash3_bwd_compute(config, channels, dK, dV):
    k_chnl, v_chnl, q_chnl, do_chnl, l_chnl, d_chnl, dq_chnl = channels
    k_state = k_chnl.state()
    v_state = v_chnl.state()
    q_state = q_chnl.state()
    do_state = do_chnl.state()
    l_state = l_chnl.state()
    d_state = d_chnl.state()
    dq_state = dq_chnl.state()
    num_tiles = gl.cdiv(config.N_CTX, config.BLOCK_N) * config.Z * config.H

    for tile_id in range(gl.program_id(0), num_tiles, config.NUM_SMS):
        prog = get_bwd_program(config, tile_id)
        offs_n = prog.start_n + gl.arange(0, config.BLOCK_N)
        offs_d = gl.arange(0, config.HEAD_DIM)

# @ref flash3-bwd-wait-kv
        k_smem, k_empty, k_state = k_chnl.consumer_acquire(k_state)
        v_smem, v_empty, v_state = v_chnl.consumer_acquire(v_state)
        k = k_smem.load()
        v_t = v_smem.load()
# @end

# @ref flash3-bwd-init-dk-dv
        dk = gl.zeros([config.BLOCK_N, config.HEAD_DIM], dtype=gl.float32)
        dv = gl.zeros([config.BLOCK_N, config.HEAD_DIM], dtype=gl.float32)
# @end

# @ref flash3-bwd-consumer-loop
        for start_m in range(0, config.N_CTX, config.BLOCK_M):
# @end
            offs_m = start_m + gl.arange(0, config.BLOCK_M)

# @ref flash3-bwd-wait-qi flash3-bwd-wait-do flash3-bwd-load-li-di
            q_smem, q_empty, q_state = q_chnl.consumer_acquire(q_state)
            do_smem, do_empty, do_state = do_chnl.consumer_acquire(do_state)
            l_smem, l_empty, l_state = l_chnl.consumer_acquire(l_state)
            d_smem, d_empty, d_state = d_chnl.consumer_acquire(d_state)
            q = q_smem.load()
            do = do_smem.load()
            lse = l_smem.load()[:, 0]
            d_i = d_smem.load()[:, 0]
# @end

# @ref flash3-bwd-score flash3-bwd-prob flash3-fp8-qk-descale
            s = gl.dot(q, k.trans()) * config.qk_scale
# @end

# @ref flash3-bwd-causal-prob
            q_rows = offs_m[:, None]
            k_cols = offs_n[None, :]
            s = gl.where(q_rows >= k_cols, s, -float("inf"))
# @end

# @ref flash3-bwd-score flash3-bwd-prob
            p = gl.exp2(s - lse[:, None] * 1.44269504)
# @end

# @ref flash3-bwd-dv
            # Forward multiplies the PV result by v_descale in the epilogue, so
            # gradients flowing into V and P carry the same factor.
            dv += gl.dot(p.trans(), do) * config.v_descale
# @end

# @ref flash3-bwd-dp
            # FP8 forward keeps V transposed for WGMMA, so dP uses V^T directly.
            dp = gl.dot(do, v_t) * config.v_descale
# @end

# @ref flash3-bwd-ds
            ds = p * (dp - d_i[:, None])
# @end

# @ref flash3-bwd-dq-causal-mask
            ds = gl.where(q_rows >= k_cols, ds, 0.0)
# @end

# @ref flash3-bwd-dk
            dk += gl.dot(ds.trans(), q) * config.grad_scale
# @end

# @ref flash3-bwd-dq-local
            dq_local = gl.dot(ds, k) * config.grad_scale
            dq_smem, dq_ready, dq_state = dq_chnl.producer_acquire(dq_state)
            dq_smem.store(dq_local)
            fence_async_shared()
            mbarrier.arrive(dq_ready)
# @end

            mbarrier.arrive(q_empty)
            mbarrier.arrive(do_empty)
            mbarrier.arrive(l_empty)
            mbarrier.arrive(d_empty)

        gl.store(dK + prog.base_vec + offs_n[:, None] * config.HEAD_DIM + offs_d[None, :], dk)
        gl.store(dV + prog.base_vec + offs_n[:, None] * config.HEAD_DIM + offs_d[None, :], dv)
        mbarrier.arrive(k_empty)
        mbarrier.arrive(v_empty)


@gluon.jit
def _flash3_bwd_dq_store(config, channels, dQ):
    k_chnl, v_chnl, q_chnl, do_chnl, l_chnl, d_chnl, dq_chnl = channels
    dq_state = dq_chnl.state()
    num_tiles = gl.cdiv(config.N_CTX, config.BLOCK_N) * config.Z * config.H

    for tile_id in range(gl.program_id(0), num_tiles, config.NUM_SMS):
        prog = get_bwd_program(config, tile_id)

# @ref flash3-bwd-dq-writer-loop
        for start_m in range(0, config.N_CTX, config.BLOCK_M):
# @end
            offs_m = start_m + gl.arange(0, config.BLOCK_M)
            offs_d = gl.arange(0, config.HEAD_DIM)

# @ref flash3-bwd-dq-ready
            dq_smem, dq_empty, dq_state = dq_chnl.consumer_acquire(dq_state)
            dq_local = dq_smem.load()
# @end

# @ref flash3-bwd-dq-atomic flash3-bwd-dq-writer-end
            if prog.start_n == 0:
                gl.store(dQ + prog.base_vec + offs_m[:, None] * config.HEAD_DIM + offs_d[None, :], dq_local)
            else:
                gl.atomic_add(dQ + prog.base_vec + offs_m[:, None] * config.HEAD_DIM + offs_d[None, :], dq_local, sem="relaxed")
            mbarrier.arrive(dq_empty)
# @end


# @ref flash3-backward-label
@gluon.jit
def flash3_bwd_full(
    Q,
    K,
    V,
    dO,
    O,
    L,
    D,
    dQ,
    dK,
    dV,
    alpha: gl.constexpr,
    Q_DESCALE,
    K_DESCALE,
    V_DESCALE,
    Z,
    H,
    N_CTX,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    NUM_SMS: gl.constexpr,
    dtype: gl.constexpr,
    num_warps: gl.constexpr,
):
    q_descale = gl.load(Q_DESCALE)
    k_descale = gl.load(K_DESCALE)
    v_descale = gl.load(V_DESCALE)
    qk_grad_scale = alpha * q_descale * k_descale
    config = Flash3Config(qk_grad_scale * 1.44269504, qk_grad_scale, v_descale, Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM, NUM_SMS, dtype, num_warps)
    k_chnl = SmemChannel.alloc([BLOCK_N, HEAD_DIM], dtype, config.k_layout, num_buffers=2)
    v_chnl = SmemChannel.alloc([HEAD_DIM, BLOCK_N], dtype, config.v_layout, num_buffers=2)
    q_chnl = SmemChannel.alloc([BLOCK_M, HEAD_DIM], dtype, config.q_layout, num_buffers=2)
    do_chnl = SmemChannel.alloc([BLOCK_M, HEAD_DIM], dtype, config.q_layout, num_buffers=2)
    l_chnl = SmemChannel.alloc([BLOCK_M, 1], gl.float32, config.row_layout, num_buffers=2)
    d_chnl = SmemChannel.alloc([BLOCK_M, 1], gl.float32, config.row_layout, num_buffers=2)
    dq_chnl = SmemChannel.alloc([BLOCK_M, HEAD_DIM], gl.float32, config.o_layout, num_buffers=2)
    channels = (k_chnl, v_chnl, q_chnl, do_chnl, l_chnl, d_chnl, dq_chnl)

    k_chnl.init()
    v_chnl.init()
    q_chnl.init()
    do_chnl.init()
    l_chnl.init()
    d_chnl.init()
    dq_chnl.init()
# @end

# @ref flash3-bwd-partition-qkv flash3-bwd-partition-do-l flash3-bwd-dq-writer-else-if
    gl.warp_specialize(
        [
            (_flash3_bwd_load, (config, channels, Q, K, V, dO, L, D)),
            (_flash3_bwd_compute, (config, channels, dK, dV)),
            (_flash3_bwd_dq_store, (config, channels, dQ)),
        ],
        [1, 1, 1],
        [24, 160, 24],
    )
# @end


# @ref flash3-cta-forward-label
@gluon.jit(do_not_specialize=["Z", "H", "N_CTX"])
def flash3_fwd_full(
    sm_scale,
    Q_DESCALE,
    K_DESCALE,
    V_DESCALE,
    M,
    Z,
    H,
    N_CTX,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    NUM_SMS: gl.constexpr,
    dtype: gl.constexpr,
    num_warps: gl.constexpr,
):
# @end
# @ref flash3-fp8-descale-load
    q_descale = gl.load(Q_DESCALE)
    k_descale = gl.load(K_DESCALE)
    v_descale = gl.load(V_DESCALE)
# @end

# @ref flash3-fp8-qk-descale
    qk_grad_scale = sm_scale * q_descale * k_descale
    config = Flash3Config(qk_grad_scale * 1.44269504, qk_grad_scale, v_descale, Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM, NUM_SMS, dtype, num_warps)
# @end
    q_chnl = SmemChannel.alloc([BLOCK_M, HEAD_DIM], dtype, config.q_layout, num_buffers=2)
    k_chnl = SmemChannel.alloc([BLOCK_N, HEAD_DIM], dtype, config.k_layout, num_buffers=3)
    v_chnl = SmemChannel.alloc([HEAD_DIM, BLOCK_N], dtype, config.v_layout, num_buffers=3)
    p_chnl = SmemChannel.alloc([BLOCK_M, BLOCK_N], dtype, config.k_layout, num_buffers=2)
    o_chnl = SmemChannel.alloc([BLOCK_M, HEAD_DIM], dtype, config.o_layout, num_buffers=2)
    channels = (q_chnl, k_chnl, v_chnl, p_chnl, o_chnl)
    descs = (desc_q, desc_k, desc_v, desc_o)

    q_chnl.init()
    k_chnl.init()
    v_chnl.init()
    p_chnl.init()
    o_chnl.init()

# @ref flash3-cta-pipeline flash3-cta-consumer-loop flash3-cta-producer-registers flash3-cta-consumer-registers flash3-consumer-registers
    gl.warp_specialize(
        [
            (_flash3_fwd_compute, (config, channels, descs, M)),
            (_flash3_fwd_load, (config, channels, descs)),
            (_flash3_fwd_store, (config, channels, descs)),
        ],
        [1, 1],
        [24, 24],
    )
# @end


def attention_forward(q, k, v, q_descale, k_descale, v_descale, sm_scale, o=None, M=None, maxnreg=128):
    B, H, N_CTX, HEAD_DIM = q.shape
    if o is None:
        o = torch.empty_like(q)
    if M is None:
        M = torch.empty((B, H, N_CTX), device=q.device, dtype=torch.float32)

# @ref flash3-fp8-dtype
    dtype = gl.float8e5
# @end
    q_layout = gl.NVMMASharedLayout.get_default_for([128, HEAD_DIM], dtype)
    kv_layout = gl.NVMMASharedLayout.get_default_for([128, HEAD_DIM], dtype)
    v_layout = gl.NVMMASharedLayout.get_default_for([HEAD_DIM, 128], dtype)
    y_dim = B * H * N_CTX
    desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[128, HEAD_DIM], layout=q_layout)
    desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[128, HEAD_DIM], layout=kv_layout)
# @ref flash3-fp8-v-desc
    desc_v = TensorDescriptor(v, shape=[HEAD_DIM, y_dim], strides=[N_CTX, 1], block_shape=[HEAD_DIM, 128], layout=v_layout)
# @end
    desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[128, HEAD_DIM], layout=q_layout)
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

    flash3_fwd_full[(num_sms,)](
        sm_scale,
        q_descale,
        k_descale,
        v_descale,
        M,
        B,
        H,
        N_CTX,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        BLOCK_M=128,
        BLOCK_N=128,
        HEAD_DIM=HEAD_DIM,
        NUM_SMS=num_sms,
        dtype=dtype,
        num_warps=4,
# @ref flash3-cta-producer-registers flash3-cta-consumer-registers flash3-consumer-registers
        maxnreg=maxnreg,
# @end
    )
    return o, M
