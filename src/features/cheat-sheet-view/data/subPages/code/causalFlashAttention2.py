@triton.jit
def _flash2_fwd_stream_kv_stage(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    qk_scale: tl.constexpr,
    lo: tl.constexpr,
    hi: tl.constexpr,
    offs_m,
    offs_n,
    BLOCK_N: tl.constexpr,
    APPLY_CAUSAL: tl.constexpr,
):
# @ref flash2-causal-prefix-loop
    # Advance to the first K/V tile handled by this stage.
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

# @end
# @ref flash2-causal-prefix-loop flash2-causal-prefix-load-kv flash2-causal-prefix-score flash2-causal-diagonal-load
        k = tl.load(K_block_ptr)
# @end
# @ref flash2-causal-prefix-loop flash2-causal-prefix-score flash2-causal-diagonal-score
        qk = tl.dot(q, k) * qk_scale
# @end

# @ref flash2-causal-diagonal-score
        if APPLY_CAUSAL:
            # Only diagonal tiles need the r >= c predicate.
            keep = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(keep, 0.0, -1.0e6)
# @end

# @ref flash2-causal-prefix-state flash2-causal-diagonal-state-output
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        row_scale = tl.exp(m_i - m_ij)
        l_i = l_i * row_scale + tl.sum(p, axis=1)
# @end
# @ref flash2-causal-diagonal-state-output

# @end
# @ref flash2-causal-prefix-output flash2-causal-diagonal-state-output
        acc = acc * row_scale[:, None]
# @end
# @ref flash2-causal-prefix-load-kv flash2-causal-prefix-output flash2-causal-diagonal-load flash2-causal-diagonal-state-output
        v = tl.load(V_block_ptr)
# @end
# @ref flash2-forward-label flash2-causal-prefix-output flash2-causal-diagonal-state-output
        acc = tl.dot(p.to(tl.float16), v, acc)
# @end
# @ref flash2-forward-label

# @end
# @ref flash2-causal-end-prefix flash2-causal-end-diagonal
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
# @end

# @ref flash2-causal-end-prefix flash2-causal-end-diagonal
    return acc, l_i, m_i
# @end


# @ref flash2-causal-outer-loop
@triton.jit
def flash2_fwd_causal(
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
):
# @end
# @ref flash2-block-pointers flash2-causal-outer-loop
    pid_m = tl.program_id(0)
    row_start = pid_m * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
# @end
# @ref flash2-block-pointers

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
# @ref flash2-block-pointers flash2-causal-divide-output
    o_ptr = tl.make_block_ptr(
        O, (N_CTX, HEAD_DIM), (STRIDE_OM, STRIDE_OD),
        (row_start, 0), (BLOCK_M, HEAD_DIM), (1, 0)
    )
# @end

# @ref flash2-causal-load-query
    q = tl.load(q_ptr)
# @end
# @ref flash2-causal-init
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
# @end
# @ref flash2-causal-set-scale
    qk_scale = alpha
# @end

# @ref flash2-causal-prefix-loop
    # Stage 1 in the Triton tutorial: prefix K/V blocks are fully visible.
    acc, l_i, m_i = _flash2_fwd_stream_kv_stage(
        acc, l_i, m_i, q, k_ptr, v_ptr,
        qk_scale, 0, row_start, offs_m, offs_n, BLOCK_N, False
# @end
# @ref flash2-causal-prefix-loop flash2-causal-end-prefix
    )
# @end

# @ref flash2-causal-diagonal-loop
    # Stage 2 in the Triton tutorial: the diagonal tile applies the mask.
    acc, l_i, m_i = _flash2_fwd_stream_kv_stage(
        acc, l_i, m_i, q, k_ptr, v_ptr,
        qk_scale, row_start, row_start + BLOCK_M,
        offs_m, offs_n, BLOCK_N, True
# @end
# @ref flash2-causal-diagonal-loop flash2-causal-end-diagonal
    )
# @end

# @ref flash2-causal-divide-output flash2-causal-compute-lse flash2-causal-end-outer
    lse = m_i + tl.log(l_i)
# @end
# @ref flash2-causal-divide-output flash2-causal-compute-output flash2-causal-end-outer
    out = acc / l_i[:, None]
# @end
# @ref flash2-causal-divide-output flash2-causal-write-lse flash2-causal-end-outer flash2-causal-return-forward
    tl.store(L + offs_m, lse)
# @end
# @ref flash2-causal-divide-output flash2-causal-write-output flash2-causal-end-outer flash2-causal-return-forward
    tl.store(o_ptr, out.to(O.type.element_ty))
# @end


# @ref flash2-bwd-label
@triton.jit
# @end
# @ref flash2-bwd-label flash2-causal-bwd-divide-state
def flash2_bwd_preprocess(
# @end
# @ref flash2-causal-bwd-divide-state
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
# @ref flash2-causal-bwd-divide-state flash2-causal-bwd-compute-d
    o = tl.load(O + offs_m[:, None] * STRIDE_OM + offs_d[None, :] * STRIDE_OD)
    do = tl.load(dO + offs_m[:, None] * STRIDE_DOM + offs_d[None, :] * STRIDE_DOD).to(tl.float32)

    # D_i = rowsum(O_i * dO_i), reused by dK/dV and dQ.
    delta = tl.sum(o * do, axis=1)
    tl.store(D + offs_m, delta)
# @end


# @ref flash2-bwd-label
@triton.jit
def _flash2_bwd_accumulate_dkdv_visible(
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
    start_n: tl.constexpr,
    start_m: tl.constexpr,
    end_m: tl.constexpr,
    APPLY_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    offs_n = start_n + tl.arange(0, BLOCK_N)

    # Stream only query blocks that can see this K/V block.
# @ref flash2-bwd-preprocess
    for row_start in range(start_m, end_m, BLOCK_M):
        offs_m = row_start + tl.arange(0, BLOCK_M)

# @end
# @ref flash2-bwd-preprocess flash2-causal-bwd-load-query
        # Same [N_CTX, HEAD_DIM] element-stride convention:
        # token-row stride first, head-dim stride second.
        q_ptr = tl.make_block_ptr(
            Q, (N_CTX, HEAD_DIM), (STRIDE_QM, STRIDE_QD),
            (row_start, 0), (BLOCK_M, HEAD_DIM), (1, 0)
        )
        do_ptr = tl.make_block_ptr(
            dO, (N_CTX, HEAD_DIM), (STRIDE_DOM, STRIDE_DOD),
            (row_start, 0), (BLOCK_M, HEAD_DIM), (1, 0)
        )

        q = tl.load(q_ptr)
        do = tl.load(do_ptr).to(tl.float32)
# @end
# @ref flash2-bwd-preprocess flash2-causal-bwd-divide-state flash2-causal-bwd-load-query
        lse = tl.load(L + offs_m)
        delta = tl.load(D + offs_m)
# @end

# @ref flash2-causal-bwd-prob
        p = tl.exp(tl.dot(q, tl.trans(k_scaled)) - lse[:, None])
        if APPLY_CAUSAL:
            keep = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(keep, p, 0.0)
# @end

# @ref flash2-causal-bwd-dv
        dv += tl.dot(tl.trans(p.to(tl.float16)), do)
# @end
# @ref flash2-causal-bwd-dp
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
# @end
# @ref flash2-causal-bwd-ds
        ds = (p * (dp - delta[:, None])).to(tl.float16)
# @end
# @ref flash2-causal-bwd-dk
        dk += tl.dot(tl.trans(ds), q)
# @end

# @ref flash2-causal-bwd-end-query
    return dk, dv
# @end


# @ref flash2-bwd-label
@triton.jit
def _flash2_bwd_accumulate_dq_visible(
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
    start_m: tl.constexpr,
    start_n: tl.constexpr,
    end_n: tl.constexpr,
    APPLY_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

# @ref flash2-causal-bwd-dq-prefix
    # Stream only key blocks visible to this query block.
    for col_start in range(start_n, end_n, BLOCK_N):
        # K is loaded as [HEAD_DIM, BLOCK_N].
        # V is loaded as [BLOCK_N, HEAD_DIM].
# @end
# @ref flash2-bwd-preprocess flash2-causal-bwd-dq-prefix
        k_ptr = tl.make_block_ptr(
            K, (HEAD_DIM, N_CTX), (STRIDE_KD, STRIDE_KN),
            (0, col_start), (HEAD_DIM, BLOCK_N), (0, 1)
        )
        v_ptr = tl.make_block_ptr(
            V, (N_CTX, HEAD_DIM), (STRIDE_VM, STRIDE_VD),
            (col_start, 0), (BLOCK_N, HEAD_DIM), (1, 0)
        )
# @end
# @ref flash2-causal-bwd-dq-prefix

        k_scaled = tl.load(k_ptr) * qk_scale
        v = tl.load(v_ptr)
        p = tl.exp(tl.dot(q, k_scaled) - lse[:, None])
# @end

# @ref flash2-causal-bwd-dq-diagonal
        if APPLY_CAUSAL:
            keep = offs_m[:, None] >= (col_start + offs_n[None, :])
            p = tl.where(keep, p, 0.0)

# @end
# @ref flash2-causal-bwd-dq-prefix flash2-causal-bwd-dq-diagonal
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = (p * (dp - delta[:, None])).to(tl.float16)
        dq += tl.dot(ds, tl.trans(k_scaled))
# @end

    return dq


# @ref flash2-bwd-label
@triton.jit
def flash2_bwd_causal(
# @end
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
    tl.static_assert(BLOCK_M == BLOCK_N)

# @ref flash2-causal-bwd-kv-loop
    # One program id owns K/V block j for dK_j/dV_j and Q block i for dQ_i.
    pid = tl.program_id(0)
    start_n = pid * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
# @end

# @ref flash2-causal-bwd-load-kv
    # Input K/V are [N_CTX, HEAD_DIM].
    # STRIDE_KM/VM step token rows.
    # STRIDE_KD/VD step head-dim columns.
# @end
# @ref flash2-bwd-preprocess flash2-causal-bwd-load-kv
    k_ptr = tl.make_block_ptr(
        K, (N_CTX, HEAD_DIM), (STRIDE_KM, STRIDE_KD),
        (start_n, 0), (BLOCK_N, HEAD_DIM), (1, 0)
    )
    v_ptr = tl.make_block_ptr(
        V, (N_CTX, HEAD_DIM), (STRIDE_VM, STRIDE_VD),
        (start_n, 0), (BLOCK_N, HEAD_DIM), (1, 0)
    )
# @end
# @ref flash2-causal-bwd-load-kv

    k = tl.load(k_ptr)
    v = tl.load(v_ptr)
# @end
# @ref flash2-causal-bwd-initialize-gradients flash2-causal-bwd-load-kv flash2-causal-bwd-init-dkdv
    dk = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    dv = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
# @end
# @ref flash2-causal-bwd-load-kv flash2-causal-bwd-init-dkdv
    qk_scale = alpha
    k_scaled = k * qk_scale
# @end

# @ref flash2-causal-bwd-query-loop
    # Diagonal query block: visible entries need the causal predicate.
    dk, dv = _flash2_bwd_accumulate_dkdv_visible(
        dk, dv, Q, k_scaled, v, dO, L, D,
        N_CTX, start_n, start_n, start_n + BLOCK_N, True,
        BLOCK_M, BLOCK_N, HEAD_DIM
    )

    # Later query blocks can see this whole K/V block.
    dk, dv = _flash2_bwd_accumulate_dkdv_visible(
        dk, dv, Q, k_scaled, v, dO, L, D,
        N_CTX, start_n, start_n + BLOCK_N, N_CTX, False,
        BLOCK_M, BLOCK_N, HEAD_DIM
    )
# @end

# @ref flash2-causal-bwd-write-dkdv flash2-causal-bwd-return
    # dK/dV output pointers use the same element-stride rule.
    # STRIDE_DKM/DVM step token rows.
    # STRIDE_DKD/DVD step head-dim columns.
    dk = alpha * dk
# @end
# @ref flash2-causal-bwd-initialize-gradients flash2-causal-bwd-write-dkdv flash2-causal-bwd-return
    dk_ptr = dK + offs_n[:, None] * STRIDE_DKM + offs_d[None, :] * STRIDE_DKD
    dv_ptr = dV + offs_n[:, None] * STRIDE_DVM + offs_d[None, :] * STRIDE_DVD
    tl.store(dk_ptr, dk)
    tl.store(dv_ptr, dv)
# @end

# @ref flash2-causal-bwd-dq-loop flash2-causal-bwd-end-dq-loop
    # Reuse the same program id for the matching Q block.
    start_m = pid * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
# @end
# @ref flash2-causal-bwd-end-dq-loop

# @end
# @ref flash2-causal-bwd-dq-load flash2-causal-bwd-end-dq-loop
    # Q/dO/dQ are [N_CTX, HEAD_DIM].
    # STRIDE_*M steps token rows; STRIDE_*D steps head dims.
# @end
# @ref flash2-bwd-preprocess flash2-causal-bwd-dq-load flash2-causal-bwd-end-dq-loop
    q_ptr = tl.make_block_ptr(
        Q, (N_CTX, HEAD_DIM), (STRIDE_QM, STRIDE_QD),
        (start_m, 0), (BLOCK_M, HEAD_DIM), (1, 0)
    )
    do_ptr = tl.make_block_ptr(
        dO, (N_CTX, HEAD_DIM), (STRIDE_DOM, STRIDE_DOD),
        (start_m, 0), (BLOCK_M, HEAD_DIM), (1, 0)
    )
# @end
# @ref flash2-causal-bwd-dq-load flash2-causal-bwd-end-dq-loop

    q = tl.load(q_ptr)
    do = tl.load(do_ptr).to(tl.float32)
# @end
# @ref flash2-causal-bwd-divide-state flash2-causal-bwd-dq-load flash2-causal-bwd-end-dq-loop
    lse = tl.load(L + offs_m)
    delta = tl.load(D + offs_m)
# @end
# @ref flash2-causal-bwd-initialize-gradients flash2-causal-bwd-dq-load flash2-causal-bwd-end-dq-loop
    dq = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
# @end
# @ref flash2-causal-bwd-end-dq-loop

# @end
# @ref flash2-causal-bwd-dq-prefix flash2-causal-bwd-end-dq-loop
    # Prefix key blocks are fully visible to this query block.
    dq = _flash2_bwd_accumulate_dq_visible(
        dq, q, K, V, do, lse, delta,
        qk_scale, N_CTX, start_m, 0, start_m, False,
        BLOCK_M, BLOCK_N, HEAD_DIM
    )
# @end
# @ref flash2-causal-bwd-end-dq-loop

# @end
# @ref flash2-causal-bwd-dq-diagonal flash2-causal-bwd-end-dq-loop
    # Diagonal key block needs the causal predicate.
    dq = _flash2_bwd_accumulate_dq_visible(
        dq, q, K, V, do, lse, delta,
        qk_scale, N_CTX, start_m, start_m, start_m + BLOCK_M, True,
        BLOCK_M, BLOCK_N, HEAD_DIM
    )
# @end
# @ref flash2-causal-bwd-end-dq-loop

# @end
# @ref flash2-causal-bwd-initialize-gradients flash2-causal-bwd-dq-write flash2-causal-bwd-end-dq-loop flash2-causal-bwd-return
    dq_ptr = dQ + offs_m[:, None] * STRIDE_DQM + offs_d[None, :] * STRIDE_DQD
    tl.store(dq_ptr, dq)
# @end
