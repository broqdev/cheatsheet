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


# @ref two-loop-signature
def _two_loop_direction(grad, s_history, y_history, rho_history):
    """Apply the limited-memory inverse-Hessian approximation."""
# @end
# @ref two-loop-start
    q = grad.clone()
    alphas = []
# @end
# @ref two-loop-backward
    for s, y, rho in zip(
        reversed(s_history), reversed(y_history), reversed(rho_history), strict=True
    ):
        alpha = rho * torch.dot(s, q)
        alphas.append(alpha)
        q.add_(y, alpha=-alpha)
# @end
# @ref initial-hessian-scale
    if s_history:
        s_last = s_history[-1]
        y_last = y_history[-1]
        h0 = torch.dot(s_last, y_last) / torch.dot(y_last, y_last)
    else:
        h0 = 1.0
    r = h0 * q
# @end
# @ref two-loop-forward
    for s, y, rho, alpha in zip(
        s_history, y_history, rho_history, reversed(alphas), strict=True
    ):
        beta = rho * torch.dot(y, r)
        r.add_(s, alpha=alpha - beta)
    return -r
# @end


# @ref step-signature
def lbfgs_step(
    params,
    closure,
    state,
    lr=1.0,
    max_iter=20,
    history_size=100,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
):
    """One fixed-step L-BFGS call.

    The closure must clear gradients, recompute the loss, and backpropagate.
    """
# @end
# @ref state-init
    params = list(params)
    s_history = state.setdefault("s_history", [])
    y_history = state.setdefault("y_history", [])
    rho_history = state.setdefault("rho_history", [])
# @end
# @ref initial-evaluation
    loss = float(closure())
    grad = _gather_flat_grad(params)
    initial_loss = loss

    if grad.abs().max() <= tolerance_grad:
        return initial_loss, state
# @end
# @ref inner-loop
    for iteration in range(max_iter):
# @end
# @ref direction
        direction = _two_loop_direction(
            grad, s_history, y_history, rho_history
        )
# @end
# @ref step-size
        if iteration == 0 and not s_history:
            step_size = min(1.0, 1.0 / grad.abs().sum()) * lr
        else:
            step_size = lr

        if torch.dot(grad, direction) > -tolerance_change:
            break
# @end
# @ref save-iterate
        s = step_size * direction
        previous_grad = grad.clone()
        previous_loss = loss
# @end
# @ref parameter-update
        _add_flat_update(params, s)
# @end
# @ref reevaluate
        loss = float(closure())
        grad = _gather_flat_grad(params)
# @end
# @ref history-update
        y = grad - previous_grad
        curvature = torch.dot(y, s)

        if curvature > 1e-10:
            if len(s_history) == history_size:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)
            s_history.append(s)
            y_history.append(y)
            rho_history.append(1.0 / curvature)
# @end
# @ref convergence
        if grad.abs().max() <= tolerance_grad:
            break
        if s.abs().max() <= tolerance_change:
            break
        if abs(loss - previous_loss) < tolerance_change:
            break
# @end
# @ref state-store
    state["s_history"] = s_history
    state["y_history"] = y_history
    state["rho_history"] = rho_history
    return initial_loss, state
# @end
