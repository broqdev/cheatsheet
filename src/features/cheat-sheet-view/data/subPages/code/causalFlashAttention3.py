# @ref flash3-consumer-forward-label
@triton.jit
def _flash3_fwd_stream_kv(
    acc,
    l_i,
    m_i,
    q,
    desc_k,
    desc_v,
    qk_scale: tl.constexpr,
    N_CTX: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    warp_specialize: tl.constexpr,
    offs_m,
    offs_n,
    CAUSAL: tl.constexpr,
):
# @end
# @ref flash3-cta-pipeline flash3-cta-producer-loop flash3-cta-wait-stage flash3-cta-consumer-loop flash3-consumer-loop
    # Stream every K/V tile while the current Q tile stays resident on chip.
    for start_n in tl.range(
        0,
        N_CTX,
        BLOCK_N,
        num_stages=NUM_STAGES,
        warp_specialize=warp_specialize,
    ):
        start_n = tl.multiple_of(start_n, BLOCK_N)
# @end

# @ref flash3-cta-load-kv flash3-cta-commit-kv flash3-cta-wait-k flash3-consumer-wait-qk0 flash3-consumer-wait-kj
        # Tensor descriptors issue tile loads that can lower to TMA on Hopper+.
        k = desc_k.load([start_n, 0]).T
# @end

# @ref flash3-cta-score flash3-consumer-score-cur flash3-consumer-score-next flash3-consumer-wait-score-next
        # Scores for S_i^(j) = alpha * Q_i K_j^T.
        qk = tl.dot(q, k) * qk_scale
# @end
# @ref flash3-fwd-causal-mask
        if CAUSAL:
            keep = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(keep, 0.0, -1.0e6)
# @end

# @ref flash3-cta-rowmax
        # Online softmax state for this row block.
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
# @end
# @ref flash3-cta-prob-l flash3-consumer-online-cur flash3-consumer-online-next
        p = tl.exp(qk - m_ij[:, None])
        row_scale = tl.exp(m_i - m_ij)
        l_i = l_i * row_scale + tl.sum(p, axis=1)
# @end

# @ref flash3-consumer-rescale-output
        # Rescale the previous accumulator, then add P_i^(j) V_j.
        acc = acc * row_scale[:, None]
# @end
# @ref flash3-cta-load-kv flash3-cta-commit-kv flash3-cta-wait-v flash3-consumer-wait-vprev flash3-consumer-wait-vlast
        v = desc_v.load([start_n, 0])
# @end
# @ref flash3-cta-output flash3-consumer-output-prev flash3-consumer-output-last
        acc = tl.dot(p.to(tl.float16), v, acc)
# @end

# @ref flash3-cta-consumer-end flash3-consumer-release-k0 flash3-consumer-release-buffer flash3-consumer-copy-next
        m_i = m_ij
# @end

    return acc, l_i, m_i


# @ref flash3-cta-forward-label
@triton.jit
def flash3_fwd_full(
    Q,
    K,
    V,
    O,
    L,
    alpha: tl.constexpr,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    warp_specialize: tl.constexpr,
    CAUSAL: tl.constexpr,
):
# @end
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    pid_m = tl.program_id(0)
    row_start = pid_m * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # All STRIDE_* values are element strides, not byte strides.
    # For contiguous [N_CTX, HEAD_DIM]:
    # row/token stride = HEAD_DIM, head-dim stride = 1.
    # Tensor descriptors replace FlashAttention-2 block pointers for TMA-style
    # tiled movement in the forward path.
    desc_q = tl.make_tensor_descriptor(
        Q,
        shape=[N_CTX, HEAD_DIM],
        strides=[STRIDE_QM, STRIDE_QD],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_k = tl.make_tensor_descriptor(
        K,
        shape=[N_CTX, HEAD_DIM],
        strides=[STRIDE_KM, STRIDE_KD],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_v = tl.make_tensor_descriptor(
        V,
        shape=[N_CTX, HEAD_DIM],
        strides=[STRIDE_VM, STRIDE_VD],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = tl.make_tensor_descriptor(
        O,
        shape=[N_CTX, HEAD_DIM],
        strides=[STRIDE_OM, STRIDE_OD],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

# @ref flash3-cta-load-q flash3-cta-commit-q flash3-cta-wait-q flash3-consumer-wait-qk0
    q = desc_q.load([row_start, 0])
# @end

# @ref flash3-cta-init-state flash3-consumer-init
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    qk_scale = alpha
# @end

    acc, l_i, m_i = _flash3_fwd_stream_kv(
        acc,
        l_i,
        m_i,
        q,
        desc_k,
        desc_v,
        qk_scale,
        N_CTX,
        BLOCK_N,
        NUM_STAGES,
        warp_specialize,
        offs_m,
        offs_n,
        CAUSAL,
    )

# @ref flash3-cta-normalize flash3-consumer-epilogue
    lse = m_i + tl.log(l_i)
    out = acc / l_i[:, None]
# @end

# @ref flash3-cta-write flash3-consumer-epilogue
    tl.store(L + offs_m, lse)
    desc_o.store([row_start, 0], out.to(O.type.element_ty))
# @end


# @ref flash3-backward-label flash3-bwd-preprocess
@triton.jit
def flash3_bwd_preprocess(
    O,
    dO,
    D,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
# @end
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

# @ref flash3-bwd-preprocess
    # STRIDE_OM/DOM step token rows.
    # STRIDE_OD/DOD step head-dim columns.
    o = tl.load(O + offs_m[:, None] * STRIDE_OM + offs_d[None, :] * STRIDE_OD)
    do = tl.load(dO + offs_m[:, None] * STRIDE_DOM + offs_d[None, :] * STRIDE_DOD).to(tl.float32)

    # D_i = rowsum(O_i * dO_i), reused by dK/dV and dQ.
    delta = tl.sum(o * do, axis=1)
    tl.store(D + offs_m, delta)
# @end


# @ref flash3-backward-label
@triton.jit
def _flash3_bwd_accumulate_dkdv(
    dk,
    dv,
    Q,
    k_scaled,
    v,
    dO,
    L,
    D,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_n,
    CAUSAL: tl.constexpr,
):
# @end
# @ref flash3-bwd-producer-loop flash3-bwd-consumer-loop
    # Hold K_j and V_j on chip while streaming all query blocks.
    offs_n = start_n + tl.arange(0, BLOCK_N)
    for start_m in range(0, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
# @end

# @ref flash3-bwd-load-q-do flash3-bwd-commit-q-do flash3-bwd-wait-qi flash3-bwd-wait-do flash3-bwd-partition-qkv
        # Same [N_CTX, HEAD_DIM] element-stride convention:
        # token-row stride first, head-dim stride second.
        q_ptr = tl.make_block_ptr(
            Q, (N_CTX, HEAD_DIM), (STRIDE_QM, STRIDE_QD),
            (start_m, 0), (BLOCK_M, HEAD_DIM), (1, 0)
        )
        do_ptr = tl.make_block_ptr(
            dO, (N_CTX, HEAD_DIM), (STRIDE_DOM, STRIDE_DOD),
            (start_m, 0), (BLOCK_M, HEAD_DIM), (1, 0)
        )

        q = tl.load(q_ptr)
        do = tl.load(do_ptr).to(tl.float32)
# @end
# @ref flash3-bwd-load-li-di flash3-bwd-partition-do-l
        lse = tl.load(L + offs_m)
        delta = tl.load(D + offs_m)
# @end

# @ref flash3-bwd-score flash3-bwd-prob
        # Recompute P_i^(j) instead of reading a stored N x N matrix.
        p = tl.exp(tl.dot(q, tl.trans(k_scaled)) - lse[:, None])
# @end
# @ref flash3-bwd-causal-prob
        if CAUSAL:
            keep = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(keep, p, 0.0)
# @end
# @ref flash3-bwd-dv
        dv += tl.dot(tl.trans(p.to(tl.float16)), do)
# @end

# @ref flash3-bwd-dp
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
# @end
# @ref flash3-bwd-ds
        ds = (p * (dp - delta[:, None])).to(tl.float16)
# @end
# @ref flash3-bwd-dk
        dk += tl.dot(tl.trans(ds), q)
# @end

    return dk, dv


# @ref flash3-backward-label
@triton.jit
def _flash3_bwd_accumulate_dq(
    dq,
    q,
    K,
    V,
    do,
    lse,
    delta,
    qk_scale: tl.constexpr,
    N_CTX: tl.constexpr,
    start_m: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
# @end
    # Stream K/V blocks and accumulate dQ_i = sum_j dS_i^(j) (alpha K_j).
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    for start_n in range(0, N_CTX, BLOCK_N):
        # K is loaded as [HEAD_DIM, BLOCK_N].
        # V is loaded as [BLOCK_N, HEAD_DIM].
        k_ptr = tl.make_block_ptr(
            K, (HEAD_DIM, N_CTX), (STRIDE_KD, STRIDE_KN),
            (0, start_n), (HEAD_DIM, BLOCK_N), (0, 1)
        )
        v_ptr = tl.make_block_ptr(
            V, (N_CTX, HEAD_DIM), (STRIDE_VM, STRIDE_VD),
            (start_n, 0), (BLOCK_N, HEAD_DIM), (1, 0)
        )

        k_scaled = tl.load(k_ptr) * qk_scale
        v = tl.load(v_ptr)

        p = tl.exp(tl.dot(q, k_scaled) - lse[:, None])
# @ref flash3-bwd-dq-causal-mask
        if CAUSAL:
            keep = offs_m[:, None] >= (start_n + offs_n[None, :])
            p = tl.where(keep, p, 0.0)
# @end
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = (p * (dp - delta[:, None])).to(tl.float16)
# @ref flash3-bwd-dq-local
        dq += tl.dot(ds, tl.trans(k_scaled))
# @end

    return dq


# @ref flash3-backward-label
@triton.jit
def flash3_bwd_full(
    Q,
    K,
    V,
    dO,
    L,
    D,
    dQ,
    dK,
    dV,
    alpha: tl.constexpr,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
# @end
# @ref flash3-bwd-partition-qkv
    # Program axis 0 owns one K/V block for dK_j and dV_j.
    pid_n = tl.program_id(0)
    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
# @end

# @ref flash3-bwd-load-kv flash3-bwd-commit-kv
    # Input K/V are [N_CTX, HEAD_DIM].
    # STRIDE_KM/VM step token rows.
    # STRIDE_KD/VD step head-dim columns.
    k_ptr = tl.make_block_ptr(
        K, (N_CTX, HEAD_DIM), (STRIDE_KM, STRIDE_KD),
        (start_n, 0), (BLOCK_N, HEAD_DIM), (1, 0)
    )
    v_ptr = tl.make_block_ptr(
        V, (N_CTX, HEAD_DIM), (STRIDE_VM, STRIDE_VD),
        (start_n, 0), (BLOCK_N, HEAD_DIM), (1, 0)
    )
# @end

# @ref flash3-bwd-wait-kv flash3-bwd-load-kv flash3-bwd-commit-kv
    k = tl.load(k_ptr)
    v = tl.load(v_ptr)
# @end
# @ref flash3-bwd-init-dk-dv
    dk = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    dv = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    qk_scale = alpha
# @end

    dk, dv = _flash3_bwd_accumulate_dkdv(
        dk, dv, Q, k * qk_scale, v, dO, L, D,
        N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM, start_n, CAUSAL
    )

    # dK/dV output pointers use the same element-stride rule.
    # STRIDE_DKM/DVM step token rows.
    # STRIDE_DKD/DVD step head-dim columns.
    dk = alpha * dk
    dk_ptr = dK + offs_n[:, None] * STRIDE_DKM + offs_d[None, :] * STRIDE_DKD
    dv_ptr = dV + offs_n[:, None] * STRIDE_DVM + offs_d[None, :] * STRIDE_DVD
    tl.store(dk_ptr, dk)
    tl.store(dv_ptr, dv)

# @ref flash3-bwd-dq-writer-loop
    # Program axis 1 owns one Q block for dQ_i.
    pid_m = tl.program_id(1)
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
# @end

# @ref flash3-bwd-partition-qkv flash3-bwd-partition-do-l
    # Q/dO/dQ are [N_CTX, HEAD_DIM].
    # STRIDE_*M steps token rows; STRIDE_*D steps head dims.
    q_ptr = tl.make_block_ptr(
        Q, (N_CTX, HEAD_DIM), (STRIDE_QM, STRIDE_QD),
        (start_m, 0), (BLOCK_M, HEAD_DIM), (1, 0)
    )
    do_ptr = tl.make_block_ptr(
        dO, (N_CTX, HEAD_DIM), (STRIDE_DOM, STRIDE_DOD),
        (start_m, 0), (BLOCK_M, HEAD_DIM), (1, 0)
    )
# @end

# @ref flash3-bwd-load-q-do flash3-bwd-commit-q-do
    q = tl.load(q_ptr)
    do = tl.load(do_ptr).to(tl.float32)
# @end
# @ref flash3-bwd-load-li-di flash3-bwd-partition-do-l
    lse = tl.load(L + offs_m)
    delta = tl.load(D + offs_m)
# @end
    dq = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

# @ref flash3-bwd-dq-ready
    dq = _flash3_bwd_accumulate_dq(
        dq, q, K, V, do, lse, delta,
        qk_scale, N_CTX, start_m, BLOCK_M, BLOCK_N, HEAD_DIM, CAUSAL
    )
# @end

# @ref flash3-bwd-dq-atomic flash3-bwd-dq-writer-end
    dq_ptr = dQ + offs_m[:, None] * STRIDE_DQM + offs_d[None, :] * STRIDE_DQD
    tl.store(dq_ptr, dq)
# @end
