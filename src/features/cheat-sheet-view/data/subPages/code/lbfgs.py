import torch


# @ref flatten-helpers
def _gather_flat_grad(params):
    return torch.cat(
        [
            torch.zeros_like(param).reshape(-1)
            if param.grad is None
            else param.grad.detach().reshape(-1)
            for param in params
        ]
    )


def _add_flat_update(params, update):
    offset = 0
    with torch.no_grad():
        for param in params:
            count = param.numel()
            param.add_(update[offset : offset + count].view_as(param))
            offset += count
# @end


# @ref loss-gradient-helper
def _loss_and_grad(params, loss_gen):
    # loss_gen() returns scalar L(theta); this helper refreshes parameter gradients.
    for param in params:
        param.grad = None
    loss = loss_gen()
    loss.backward()
    return loss.item(), _gather_flat_grad(params)
# @end


# @ref bfgs-signature
def bfgs_step(
    params,
    loss_gen,
    state,
    lr=1.0,
    epsilon_c=1e-10,
):
# @end
# @ref bfgs-state-init
    params = list(params)
    n = sum(param.numel() for param in params)
    M = state.get("inverse_hessian")

    if M is None:
        first = params[0]
        M = torch.eye(n, device=first.device, dtype=first.dtype)
# @end
# @ref bfgs-initial-evaluation
    loss, g = _loss_and_grad(params, loss_gen)
# @end
# @ref bfgs-direction
    p = -(M @ g)
# @end
# @ref bfgs-parameter-update
    s = lr * p
    _add_flat_update(params, s)
# @end
# @ref bfgs-reevaluate
    loss, g_next = _loss_and_grad(params, loss_gen)
    y = g_next - g
# @end
# @ref bfgs-curvature
    c = torch.dot(y, s)
# @end
# @ref bfgs-hessian-update
    # Positive curvature preserves a positive-definite inverse-Hessian estimate.
    if c > epsilon_c:
        I = torch.eye(n, device=M.device, dtype=M.dtype)
        V = I - torch.outer(s, y) / c
        M = V @ M @ V.T + torch.outer(s, s) / c
# @end
# @ref bfgs-state-store
    state["inverse_hessian"] = M
    return params, loss, state
# @end


# @ref two-loop-signature
def _two_loop_recursion(g, s_history, y_history, c_history):
# @end
# @ref two-loop-start
    q = g.clone()
    a_values = []
# @end
# @ref two-loop-backward
    # Apply V_j.T from the newest correction to the oldest.
    for s, y, c in zip(
        reversed(s_history), reversed(y_history), reversed(c_history), strict=True
    ):
        a = torch.dot(s, q) / c
        a_values.append(a)
        q.add_(y, alpha=-a)
# @end
# @ref initial-hessian-scale
    if c_history:
        y_last = y_history[-1]
        m0_scale = c_history[-1] / torch.dot(y_last, y_last)
    else:
        m0_scale = 1.0
    r = m0_scale * q
# @end
# @ref two-loop-forward
    # Apply V_j and its rank-one correction from oldest to newest.
    for s, y, c, a in zip(
        s_history, y_history, c_history, reversed(a_values), strict=True
    ):
        b = torch.dot(y, r) / c
        r.add_(s, alpha=a - b)
    return r
# @end


# @ref history-helper
def _append_correction(s_history, y_history, c_history, s, y, history_size):
    c = torch.dot(y, s)
    if not (c > 1e-10):
        return
    if len(s_history) == history_size:
        for history in (s_history, y_history, c_history):
            history.pop(0)
    s_history.append(s)
    y_history.append(y)
    c_history.append(c)
# @end


# @ref step-signature
def lbfgs_step(
    params,
    loss_gen,
    state,
    lr=1.0,
    max_iter=20,
    history_size=100,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
):
# @end
# @ref state-init
    params = list(params)
    s_history = state.setdefault("s_history", [])
    y_history = state.setdefault("y_history", [])
    c_history = state.setdefault("c_history", [])
# @end
# @ref initial-evaluation
    loss, g = _loss_and_grad(params, loss_gen)
    initial_loss = loss

    if g.abs().max() <= tolerance_grad:
        return initial_loss, state
    if max_iter <= 0:
        return initial_loss, state
# @end
# @ref pending-history
    pending_step = state.pop("pending_step", None)
    if pending_step is not None:
        s_prev, g_prev = pending_step
        _append_correction(
            s_history, y_history, c_history, s_prev, g - g_prev, history_size
        )
# @end
# @ref inner-loop
    for k in range(max_iter):
# @end
# @ref direction
        r = _two_loop_recursion(g, s_history, y_history, c_history)
        p = -r
# @end
# @ref step-size
        if k == 0 and not s_history:
            alpha = min(1.0, 1.0 / g.abs().sum()) * lr
        else:
            alpha = lr

        if torch.dot(g, p) > -tolerance_change:
            break
# @end
# @ref save-iterate
        s = alpha * p
# @end
# @ref parameter-update
        _add_flat_update(params, s)
# @end
# @ref defer-final-evaluation
        if k == max_iter - 1:
            state["pending_step"] = (s, g)
            break
# @end
        g_prev = g
        loss_prev = loss
# @ref reevaluate
        loss, g_next = _loss_and_grad(params, loss_gen)
# @end
# @ref history-update
        _append_correction(
            s_history, y_history, c_history, s, g_next - g_prev, history_size
        )
        g = g_next
# @end
# @ref convergence
        if (
            g.abs().max() <= tolerance_grad
            or s.abs().max() <= tolerance_change
            or abs(loss - loss_prev) < tolerance_change
        ):
            break
# @end
# @ref state-store
    state["s_history"] = s_history
    state["y_history"] = y_history
    state["c_history"] = c_history
    return initial_loss, state
# @end
