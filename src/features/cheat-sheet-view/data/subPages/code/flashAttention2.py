@triton.jit
def _flash2_fwd_stream_kv(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    qk_scale: tl.constexpr,
    N_CTX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
# @ref flash2-inner-loop
    # Stream every K/V tile while the current Q tile stays resident on chip.
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
# @end

# @ref flash2-compute-score
        # Scores for S_i^(j) = alpha * Q_i K_j^T.
# @end
# @ref flash2-load-kv flash2-compute-score
        k = tl.load(K_block_ptr)
# @end
# @ref flash2-compute-score
        qk = tl.dot(q, k) * qk_scale
# @end

# @ref flash2-compute-online-softmax
        # Online softmax state for this row block.
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        row_scale = tl.exp(m_i - m_ij)
        l_i = l_i * row_scale + tl.sum(p, axis=1)
# @end

# @ref flash2-compute-output-accumulator
        # Rescale the previous accumulator, then add P_i^(j) V_j.
        acc = acc * row_scale[:, None]
# @end
# @ref flash2-load-kv flash2-compute-output-accumulator
        v = tl.load(V_block_ptr)
# @end
# @ref flash2-compute-output-accumulator
        acc = tl.dot(p.to(tl.float16), v, acc)
# @end

# @ref flash2-end-inner
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
# @end

# @ref flash2-end-inner
    return acc, l_i, m_i
# @end


# @ref flash2-outer-loop
@triton.jit
def flash2_fwd_full(
    Q,
    K,
    V,
    O,
    L,
    alpha: tl.constexpr,
    N_CTX: tl.constexpr,
# @end
# @ref flash2-forward-label flash2-outer-loop
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
# @end
# @ref flash2-outer-loop
    HEAD_DIM: tl.constexpr,
):
# @end
# @ref flash2-partition flash2-outer-loop
    pid_m = tl.program_id(0)
    row_start = pid_m * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)
# @end
# @ref flash2-partition

    # All STRIDE_* values are element strides, not byte strides.
    # For contiguous [N_CTX, HEAD_DIM]:
    # row/token stride = HEAD_DIM, head-dim stride = 1.
    # Q/V/O use token-row then head-dim strides.
    q_ptr = tl.make_block_ptr(
        Q, (N_CTX, HEAD_DIM), (STRIDE_QM, STRIDE_QD),
        (row_start, 0), (BLOCK_M, HEAD_DIM), (1, 0)
    )
    # K is addressed as [HEAD_DIM, N_CTX] for tl.dot(q, k).
    # STRIDE_KD steps head dims; STRIDE_KN steps key tokens.
    k_ptr = tl.make_block_ptr(
        K, (HEAD_DIM, N_CTX), (STRIDE_KD, STRIDE_KN),
        (0, 0), (HEAD_DIM, BLOCK_N), (0, 1)
    )
    v_ptr = tl.make_block_ptr(
        V, (N_CTX, HEAD_DIM), (STRIDE_VM, STRIDE_VD),
        (0, 0), (BLOCK_N, HEAD_DIM), (1, 0)
    )
# @end
# @ref flash2-partition flash2-divide-output
    o_ptr = tl.make_block_ptr(
        O, (N_CTX, HEAD_DIM), (STRIDE_OM, STRIDE_OD),
        (row_start, 0), (BLOCK_M, HEAD_DIM), (1, 0)
    )
# @end

# @ref flash2-load-query
    q = tl.load(q_ptr)
# @end
# @ref flash2-initialize-state
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
# @end
# @ref flash2-set-scale
    qk_scale = alpha
# @end

# @ref flash2-inner-loop
    acc, l_i, m_i = _flash2_fwd_stream_kv(
        acc, l_i, m_i, q, k_ptr, v_ptr,
        qk_scale, N_CTX, BLOCK_N
    )
# @end

# @ref flash2-divide-output flash2-compute-lse flash2-end-outer
    lse = m_i + tl.log(l_i)
# @end
# @ref flash2-divide-output flash2-compute-output flash2-end-outer
    out = acc / l_i[:, None]
# @end
# @ref flash2-divide-output flash2-write-lse flash2-end-outer flash2-return-forward
    tl.store(L + offs_m, lse)
# @end
# @ref flash2-divide-output flash2-write-output flash2-end-outer flash2-return-forward
    tl.store(o_ptr, out.to(O.type.element_ty))
# @end


# @ref flash2-bwd-label
@triton.jit
# @end
# @ref flash2-bwd-label flash2-bwd-divide-state
def flash2_bwd_preprocess(
# @end
# @ref flash2-bwd-divide-state
    O,
    dO,
    D,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # STRIDE_OM/DOM step token rows.
    # STRIDE_OD/DOD step head-dim columns.
# @end
# @ref flash2-bwd-divide-state flash2-bwd-compute-d
    o = tl.load(O + offs_m[:, None] * STRIDE_OM + offs_d[None, :] * STRIDE_OD)
    do = tl.load(dO + offs_m[:, None] * STRIDE_DOM + offs_d[None, :] * STRIDE_DOD).to(tl.float32)

    # D_i = rowsum(O_i * dO_i), reused by dK/dV and dQ.
    delta = tl.sum(o * do, axis=1)
    tl.store(D + offs_m, delta)
# @end


# @ref flash2-bwd-label
@triton.jit
def _flash2_bwd_accumulate_dkdv(
# @end
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
):
# @ref flash2-bwd-query-loop
    # Hold K_j and V_j on chip while streaming all query blocks.
# @end
# @ref flash2-bwd-preprocess flash2-bwd-query-loop
    for start_m in range(0, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
# @end
# @ref flash2-bwd-preprocess

# @end
# @ref flash2-bwd-preprocess flash2-bwd-load-query
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
# @end
# @ref flash2-bwd-load-query

        q = tl.load(q_ptr)
        do = tl.load(do_ptr).to(tl.float32)
# @end
# @ref flash2-bwd-divide-state flash2-bwd-load-query
        lse = tl.load(L + offs_m)
        delta = tl.load(D + offs_m)
# @end

# @ref flash2-bwd-prob
        # Recompute P_i^(j) instead of reading a stored N x N matrix.
        p = tl.exp(tl.dot(q, tl.trans(k_scaled)) - lse[:, None])
# @end
# @ref flash2-bwd-dv-dp
        dv += tl.dot(tl.trans(p.to(tl.float16)), do)
# @end

# @ref flash2-bwd-dp
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
# @end
# @ref flash2-bwd-ds
        ds = (p * (dp - delta[:, None])).to(tl.float16)
# @end
# @ref flash2-bwd-dk
        dk += tl.dot(tl.trans(ds), q)
# @end

# @ref flash2-bwd-end-query
    return dk, dv
# @end


# @ref flash2-bwd-label
@triton.jit
def _flash2_bwd_accumulate_dq(
# @end
    dq,
    q,
    K,
    V,
    do,
    lse,
    delta,
    qk_scale: tl.constexpr,
    N_CTX: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
# @ref flash2-bwd-dq-compute
    # Stream K/V blocks and accumulate dQ_i = sum_j dS_i^(j) (alpha K_j).
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
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = (p * (dp - delta[:, None])).to(tl.float16)
        dq += tl.dot(ds, tl.trans(k_scaled))
# @end

    return dq


# @ref flash2-bwd-label flash2-bwd-kv-loop
@triton.jit
def flash2_bwd_full(
# @end
# @ref flash2-bwd-kv-loop
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
):
    # Program axis 0 owns one K/V block for dK_j and dV_j.
    pid_n = tl.program_id(0)
    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
# @end

# @ref flash2-bwd-load-kv
    # Input K/V are [N_CTX, HEAD_DIM].
    # STRIDE_KM/VM step token rows.
    # STRIDE_KD/VD step head-dim columns.
# @end
# @ref flash2-bwd-preprocess flash2-bwd-load-kv
    k_ptr = tl.make_block_ptr(
        K, (N_CTX, HEAD_DIM), (STRIDE_KM, STRIDE_KD),
        (start_n, 0), (BLOCK_N, HEAD_DIM), (1, 0)
    )
    v_ptr = tl.make_block_ptr(
        V, (N_CTX, HEAD_DIM), (STRIDE_VM, STRIDE_VD),
        (start_n, 0), (BLOCK_N, HEAD_DIM), (1, 0)
    )
# @end
# @ref flash2-bwd-load-kv

    k = tl.load(k_ptr)
    v = tl.load(v_ptr)
# @end
# @ref flash2-bwd-initialize-gradients flash2-bwd-init-dkdv
    dk = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    dv = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
# @end
# @ref flash2-bwd-init-dkdv
    qk_scale = alpha
# @end

    dk, dv = _flash2_bwd_accumulate_dkdv(
        dk, dv, Q, k * qk_scale, v, dO, L, D,
        N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM
    )

# @ref flash2-bwd-write-dkdv flash2-bwd-return
    # dK/dV output pointers use the same element-stride rule.
    # STRIDE_DKM/DVM step token rows.
    # STRIDE_DKD/DVD step head-dim columns.
    dk = alpha * dk
# @end
# @ref flash2-bwd-initialize-gradients flash2-bwd-write-dkdv flash2-bwd-return
    dk_ptr = dK + offs_n[:, None] * STRIDE_DKM + offs_d[None, :] * STRIDE_DKD
    dv_ptr = dV + offs_n[:, None] * STRIDE_DVM + offs_d[None, :] * STRIDE_DVD
    tl.store(dk_ptr, dk)
    tl.store(dv_ptr, dv)
# @end

# @ref flash2-bwd-dq-loop flash2-bwd-end-dq-loop
    # Program axis 1 owns one Q block for dQ_i.
    pid_m = tl.program_id(1)
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
# @end
# @ref flash2-bwd-end-dq-loop

# @end
# @ref flash2-bwd-dq-load flash2-bwd-end-dq-loop
    # Q/dO/dQ are [N_CTX, HEAD_DIM].
    # STRIDE_*M steps token rows; STRIDE_*D steps head dims.
# @end
# @ref flash2-bwd-preprocess flash2-bwd-dq-load flash2-bwd-end-dq-loop
    q_ptr = tl.make_block_ptr(
        Q, (N_CTX, HEAD_DIM), (STRIDE_QM, STRIDE_QD),
        (start_m, 0), (BLOCK_M, HEAD_DIM), (1, 0)
    )
    do_ptr = tl.make_block_ptr(
        dO, (N_CTX, HEAD_DIM), (STRIDE_DOM, STRIDE_DOD),
        (start_m, 0), (BLOCK_M, HEAD_DIM), (1, 0)
    )
# @end
# @ref flash2-bwd-dq-load flash2-bwd-end-dq-loop

    q = tl.load(q_ptr)
    do = tl.load(do_ptr).to(tl.float32)
# @end
# @ref flash2-bwd-divide-state flash2-bwd-dq-load flash2-bwd-end-dq-loop
    lse = tl.load(L + offs_m)
    delta = tl.load(D + offs_m)
# @end
# @ref flash2-bwd-initialize-gradients flash2-bwd-dq-load flash2-bwd-end-dq-loop
    dq = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
# @end
# @ref flash2-bwd-end-dq-loop

# @end
# @ref flash2-bwd-dq-compute flash2-bwd-end-dq-loop
    dq = _flash2_bwd_accumulate_dq(
        dq, q, K, V, do, lse, delta,
        qk_scale, N_CTX, BLOCK_N, HEAD_DIM
    )
# @end
# @ref flash2-bwd-end-dq-loop

# @end
# @ref flash2-bwd-initialize-gradients flash2-bwd-dq-write flash2-bwd-end-dq-loop flash2-bwd-return
    dq_ptr = dQ + offs_m[:, None] * STRIDE_DQM + offs_d[None, :] * STRIDE_DQD
    tl.store(dq_ptr, dq)
# @end
