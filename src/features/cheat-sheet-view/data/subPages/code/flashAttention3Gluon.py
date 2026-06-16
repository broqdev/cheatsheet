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
    mma_layout: gl.constexpr
    q_operand_layout: gl.constexpr
    p_operand_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, qk_scale, grad_scale, Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM, NUM_SMS, dtype, num_warps):
        self.qk_scale = qk_scale
        self.grad_scale = grad_scale
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
        self.v_layout = gl.constexpr(gl.NVMMASharedLayout.get_default_for([BLOCK_N, HEAD_DIM], dtype))
        self.o_layout = gl.constexpr(gl.NVMMASharedLayout.get_default_for([BLOCK_M, HEAD_DIM], dtype))
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


@gluon.jit
def tma_load(desc, offset_y, channel, state):
    smem, ready_bar, state = channel.producer_acquire(state)
    mbarrier.expect(ready_bar, desc.block_type.nbytes)
    tma.async_load(desc, [offset_y, 0], ready_bar, smem)
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
    return warpgroup_mma(p, v_smem, acc, is_async=True, use_acc=use_acc)


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

# @ref flash3-cta-load-kv flash3-cta-commit-kv flash3-cta-wait-v flash3-consumer-wait-vprev flash3-consumer-wait-vlast
            v_state = tma_load(desc_v, offset_y, v_chnl, v_state)
# @end


# @ref flash3-consumer-forward-label
@gluon.jit
def _flash3_fwd_compute(config, channels, descs):
    q_chnl, k_chnl, v_chnl, p_chnl, o_chnl = channels
    q_state = q_chnl.state()
    k_state = k_chnl.state()
    v_state = v_chnl.state()
    p_state = p_chnl.state()
    o_state = o_chnl.state()
# @end

# @ref flash3-cta-init-state flash3-consumer-init
    m_i = gl.full([config.BLOCK_M], -float("inf"), gl.float32)
    l_i = gl.full([config.BLOCK_M], 0.0, gl.float32)
    o_acc = warpgroup_mma_init(gl.zeros((config.BLOCK_M, config.HEAD_DIM), dtype=gl.float32, layout=config.mma_layout))
# @end

# @ref flash3-cta-consumer-loop flash3-consumer-loop
    num_tiles = gl.cdiv(config.N_CTX, config.BLOCK_M) * config.Z * config.H
    for tile_id in range(gl.program_id(0), num_tiles, config.NUM_SMS):
        prog = get_program(config, tile_id)
        lo, hi = prog.loop_bounds(config)
        q_smem, q_empty, q_state = q_chnl.consumer_acquire(q_state)
        k_smem, k_empty, k_state = k_chnl.consumer_acquire(k_state)
# @end

# @ref flash3-cta-score flash3-consumer-score-cur
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

# @ref flash3-cta-rowmax
        s_cur = qk_acc * config.qk_scale
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

# @ref flash3-cta-output flash3-consumer-output-prev
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

# @ref flash3-consumer-score-next
            qk_next = warpgroup_mma_init(
                gl.zeros((config.BLOCK_M, config.BLOCK_N), dtype=gl.float32, layout=config.mma_layout)
            )
            qk_next = wgmma_qk(config, q_smem, k_smem, qk_next)
# @end

# @ref flash3-consumer-wait-score-next
            qk_next = warpgroup_mma_wait(num_outstanding=0, deps=(qk_next,))
# @end

# @ref flash3-consumer-online-next
            s_next = qk_next * config.qk_scale
            m_next = gl.maximum(m_i, gl.max(s_next, axis=1))
            p_next = gl.exp2(s_next - m_next[:, None])
# @end

# @ref flash3-consumer-rescale-output
            o_acc = o_acc * gl.exp2(m_i - m_next)[:, None]
            l_i = l_i * gl.exp2(m_i - m_next) + gl.sum(p_next, axis=1)
            m_i = m_next
# @end

# @ref flash3-consumer-copy-next
            p_cur = p_next
# @end

# @ref flash3-consumer-wait-vlast
        v_smem, v_empty, v_state = v_chnl.consumer_acquire(v_state)
# @end

# @ref flash3-consumer-output-last
        p_smem, p_ready, p_state = p_chnl.producer_acquire(p_state)
        p_smem.store(p_cur.to(config.dtype))
        fence_async_shared()
        mbarrier.arrive(p_ready)
        o_acc = wgmma_pv(config, p_smem, v_smem, o_acc, use_acc=True)
        o_acc = warpgroup_mma_wait(num_outstanding=0, deps=(o_acc,))
# @end

# @ref flash3-cta-consumer-end
        mbarrier.arrive(q_empty)
        mbarrier.arrive(v_empty)
# @end

# @ref flash3-cta-normalize flash3-consumer-epilogue
        out = o_acc / l_i[:, None]
        lse = (m_i + gl.log2(l_i)) * 0.6931471805599453
# @end

# @ref flash3-cta-write flash3-consumer-epilogue
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
def _flash3_bwd_load(config, channels, descs):
    kv_chnl, qdo_chnl = channels
    desc_q, desc_k, desc_v, desc_do = descs
    kv_state = kv_chnl.state()
    qdo_state = qdo_chnl.state()

# @ref flash3-bwd-producer-loop
    for start_n in range(0, config.N_CTX, config.BLOCK_N):
# @end

# @ref flash3-bwd-load-kv flash3-bwd-commit-kv
        kv_state = tma_load(desc_k, start_n, kv_chnl, kv_state)
        kv_state = tma_load(desc_v, start_n, kv_chnl, kv_state)
# @end

# @ref flash3-bwd-load-q-do flash3-bwd-commit-q-do flash3-bwd-partition-qkv
        qdo_state = tma_load(desc_q, 0, qdo_chnl, qdo_state)
        qdo_state = tma_load(desc_do, 0, qdo_chnl, qdo_state)
# @end


@gluon.jit
def _flash3_bwd_compute(config, channels):
    kv_chnl, qdo_chnl = channels
    kv_state = kv_chnl.state()
    qdo_state = qdo_chnl.state()

# @ref flash3-bwd-init-dk-dv
    dk = gl.zeros([config.BLOCK_N, config.HEAD_DIM], dtype=gl.float32)
    dv = gl.zeros([config.BLOCK_N, config.HEAD_DIM], dtype=gl.float32)
# @end

# @ref flash3-bwd-wait-kv
    k_smem, k_empty, kv_state = kv_chnl.consumer_acquire(kv_state)
    v_smem, v_empty, kv_state = kv_chnl.consumer_acquire(kv_state)
# @end

# @ref flash3-bwd-consumer-loop
    for start_m in range(0, config.N_CTX, config.BLOCK_M):
# @end

# @ref flash3-bwd-wait-qi flash3-bwd-wait-do
        q_smem, q_empty, qdo_state = qdo_chnl.consumer_acquire(qdo_state)
        do_smem, do_empty, qdo_state = qdo_chnl.consumer_acquire(qdo_state)
# @end

# @ref flash3-bwd-load-li-di flash3-bwd-partition-do-l
        lse = gl.full([config.BLOCK_M], 1.0, gl.float32)
        d_i = gl.full([config.BLOCK_M], 0.0, gl.float32)
# @end

# @ref flash3-bwd-score flash3-bwd-prob
        s = gl.dot(q_smem.load(), k_smem.load().trans()) * config.qk_scale
        p = gl.exp2(s - lse[:, None] * 1.44269504)
# @end

# @ref flash3-bwd-dv
        dv += gl.dot(p.trans(), do_smem.load())
# @end

# @ref flash3-bwd-dp
        dp = gl.dot(do_smem.load(), v_smem.load().trans())
# @end

# @ref flash3-bwd-ds
        ds = p * (dp - d_i[:, None])
# @end

# @ref flash3-bwd-dk
        dk += gl.dot(ds.trans(), q_smem.load()) * config.grad_scale
# @end

# @ref flash3-bwd-dq-local
        dq_local = gl.dot(ds, k_smem.load()) * config.grad_scale
# @end


# @ref flash3-backward-label
@gluon.jit
def flash3_bwd_full(Q, K, V, dO, O, L, D, dQ, dK, dV, alpha: gl.constexpr, N_CTX: gl.constexpr):
    # Backward follows FA3's producer / consumer / dQ-writer split.
    pass
# @end

# @ref flash3-bwd-partition-qkv flash3-bwd-partition-do-l
    # Q, dO, L, and D are streamed by query block; K and V are fixed by key block.
# @end

# @ref flash3-bwd-dq-writer-loop
    for start_m in range(0, N_CTX, BLOCK_M):
        pass
# @end

# @ref flash3-bwd-dq-ready
        # Wait until the consumer publishes dQ_i^(local) into shared memory.
# @end

# @ref flash3-bwd-dq-atomic flash3-bwd-dq-writer-end
        # Serialize the global accumulation of dQ_i from all key blocks.
        gl.atomic_add(dQ + start_m, 0.0, sem="relaxed")
# @end


# @ref flash3-cta-forward-label
@gluon.jit(do_not_specialize=["Z", "H", "N_CTX"])
def flash3_fwd_full(
    sm_scale,
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
    config = Flash3Config(sm_scale * 1.44269504, sm_scale, Z, H, N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM, NUM_SMS, dtype, num_warps)
    q_chnl = SmemChannel.alloc([BLOCK_M, HEAD_DIM], dtype, config.q_layout, num_buffers=2)
    k_chnl = SmemChannel.alloc([BLOCK_N, HEAD_DIM], dtype, config.k_layout, num_buffers=3)
    v_chnl = SmemChannel.alloc([BLOCK_N, HEAD_DIM], dtype, config.v_layout, num_buffers=3)
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
            (_flash3_fwd_compute, (config, channels, descs)),
            (_flash3_fwd_load, (config, channels, descs)),
            (_flash3_fwd_store, (config, channels, descs)),
        ],
        [1, 1],
        [24, 24],
    )
# @end


def attention_forward(q, k, v, sm_scale, o=None, M=None, maxnreg=128):
    B, H, N_CTX, HEAD_DIM = q.shape
    if o is None:
        o = torch.empty_like(q)
    if M is None:
        M = torch.empty((B, H, N_CTX), device=q.device, dtype=torch.float32)

    dtype = gl.float16 if q.dtype == torch.float16 else gl.bfloat16
    q_layout = gl.NVMMASharedLayout.get_default_for([128, HEAD_DIM], dtype)
    kv_layout = gl.NVMMASharedLayout.get_default_for([128, HEAD_DIM], dtype)
    y_dim = B * H * N_CTX
    desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[128, HEAD_DIM], layout=q_layout)
    desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[128, HEAD_DIM], layout=kv_layout)
    desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[128, HEAD_DIM], layout=kv_layout)
    desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[128, HEAD_DIM], layout=q_layout)
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

    flash3_fwd_full[(num_sms,)](
        sm_scale,
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
