@triton.jit
def flash1_fwd(Q, K, V, M, O, ell, m, R, alpha: tl.constexpr, N_CTX: tl.constexpr):
# @ref fwd-rng
    R = init_rng_state()
    save_rng_state(R)
# @end
# @ref fwd-blocks
    Bc = BLOCK_N
    Br = BLOCK_M
# @end

# @ref fwd-init
    offs_m = tl.arange(0, Br)
    offs_n = tl.arange(0, Bc)
    offs_d = tl.arange(0, HEAD_DIM)
# @end

# @ref fwd-loop-kv
    for j in range(0, N_CTX, Bc):
# @end
# @ref fwd-load-kv
        k = tl.load(K + (j + offs_n)[:, None] * STRIDE_KN + offs_d[None, :])
        v = tl.load(V + (j + offs_n)[:, None] * STRIDE_VN + offs_d[None, :])
# @end

# @ref fwd-loop-q
        for i in range(0, N_CTX, Br):
# @end
# @ref fwd-load-q-state
            q = tl.load(Q + (i + offs_m)[:, None] * STRIDE_QM + offs_d[None, :])
            o = tl.load(O + (i + offs_m)[:, None] * STRIDE_OM + offs_d[None, :])
            ell_i = tl.load(ell + i + offs_m)
            m_i = tl.load(m + i + offs_m)
# @end

# @ref fwd-score
            s = alpha * tl.dot(q, tl.trans(k))
# @end
# @ref fwd-mask
            mask = tl.load(M + (i + offs_m)[:, None] * STRIDE_MM + (j + offs_n)[None, :])
            s_masked = s + mask
# @end
# @ref fwd-rowmax
            m_tile = tl.max(s_masked, axis=1)
            p_tilde = tl.exp(s_masked - m_tile[:, None])
            ell_tilde = tl.sum(p_tilde, axis=1)
# @end

# @ref fwd-update
            m_new = tl.maximum(m_i, m_tile)
            old_scale = tl.exp(m_i - m_new)
            tile_scale = tl.exp(m_tile - m_new)
            ell_new = old_scale * ell_i + tile_scale * ell_tilde
# @end

# @ref fwd-dropout
            z = dropout_mask(R, i + offs_m[:, None], j + offs_n[None, :], P_DROP)
            p_drop = p_tilde * z
# @end
# @ref fwd-output-numer
            numer = old_scale[:, None] * ell_i[:, None] * o
            numer += tile_scale[:, None] * tl.dot(p_drop, v)
            o_new = numer / ell_new[:, None]
# @end

# @ref fwd-store-output
            tl.store(O + (i + offs_m)[:, None] * STRIDE_OM + offs_d[None, :], o_new)
# @end
# @ref fwd-write-state
            tl.store(ell + i + offs_m, ell_new)
            tl.store(m + i + offs_m, m_new)
# @end
# @ref fwd-return
    return O, ell, m, R
# @end

# @ref bwd-signature
@triton.jit
def flash1_bwd(Q, K, V, M, O, dO, ell, m, R, dQ, dK, dV, alpha: tl.constexpr, N_CTX: tl.constexpr):
# @end
# @ref bwd-rng
    restore_rng_state(R)
# @end
# @ref bwd-blocks
    Bc = BLOCK_N
    Br = BLOCK_M
# @end
# @ref bwd-init-grads
    offs_m = tl.arange(0, Br)
    offs_n = tl.arange(0, Bc)
    offs_d = tl.arange(0, HEAD_DIM)
# @end

# @ref bwd-loop-kv
    for j in range(0, N_CTX, Bc):
# @end
# @ref bwd-load-kv
        k = tl.load(K + (j + offs_n)[:, None] * STRIDE_KN + offs_d[None, :])
        v = tl.load(V + (j + offs_n)[:, None] * STRIDE_VN + offs_d[None, :])
# @end
# @ref bwd-init-kv-grads
        dk = tl.zeros_like(k)
        dv = tl.zeros_like(v)
# @end

# @ref bwd-loop-q
        for i in range(0, N_CTX, Br):
# @end
# @ref bwd-load-q-state bwd-load-divide-state
            q = tl.load(Q + (i + offs_m)[:, None] * STRIDE_QM + offs_d[None, :])
            o = tl.load(O + (i + offs_m)[:, None] * STRIDE_OM + offs_d[None, :])
            do = tl.load(dO + (i + offs_m)[:, None] * STRIDE_DOM + offs_d[None, :])
# @end
# @ref bwd-load-q-state
            dq = tl.load(dQ + (i + offs_m)[:, None] * STRIDE_DQM + offs_d[None, :])
# @end
# @ref bwd-load-q-state bwd-load-divide-state
            ell_i = tl.load(ell + i + offs_m)
            m_i = tl.load(m + i + offs_m)
# @end

# @ref bwd-score
            s = alpha * tl.dot(q, tl.trans(k))
# @end
# @ref bwd-mask
            mask = tl.load(M + (i + offs_m)[:, None] * STRIDE_MM + (j + offs_n)[None, :])
            s_masked = s + mask
# @end
# @ref bwd-prob
            p = tl.exp(s_masked - m_i[:, None]) / ell_i[:, None]
# @end
# @ref bwd-dropout-mask
            z = dropout_mask(R, i + offs_m[:, None], j + offs_n[None, :], P_DROP)
# @end
# @ref bwd-dropped-prob
            p_drop = p * z
# @end

# @ref bwd-dv
            dv += tl.dot(tl.trans(p_drop), do)
# @end
# @ref bwd-dp
            dp_drop = tl.dot(do, tl.trans(v))
# @end
# @ref bwd-dp-after-dropout
            dp = dp_drop * z
# @end
# @ref bwd-d
            D = tl.sum(do * o, axis=1)
# @end
# @ref bwd-ds
            ds = p * (dp - D[:, None])
# @end

# @ref bwd-dq
            tl.store(dQ + (i + offs_m)[:, None] * STRIDE_DQM + offs_d[None, :], dq + alpha * tl.dot(ds, k))
# @end
# @ref bwd-dk
            dk += alpha * tl.dot(tl.trans(ds), q)
# @end

# @ref bwd-write-kv
        tl.store(dK + (j + offs_n)[:, None] * STRIDE_DKN + offs_d[None, :], dk)
        tl.store(dV + (j + offs_n)[:, None] * STRIDE_DVN + offs_d[None, :], dv)
# @end
# @ref bwd-return
    return dQ, dK, dV
# @end
