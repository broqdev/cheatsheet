import torch


# @ref ns-signature
def zeropower_via_newtonschulz5(update, steps=5, eps=1e-7):
    """Approximate the orthogonal polar factor with quintic Newton-Schulz."""
# @end
# @ref ns-assert
    assert update.ndim == 2
# @end
# @ref ns-coefficients
    a, b, c = (3.4445, -4.7750, 2.0315)
# @end
# @ref ns-cast
    X = update.bfloat16()
# @end
# @ref ns-transpose
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
# @end
# @ref ns-normalize
    X = X / X.norm().clamp(min=eps)
# @end
# @ref ns-iteration
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
# @end
# @ref ns-restore
    if transposed:
        X = X.T
    return X.to(update.dtype)
# @end


# @ref step-signature
def muon_step(params, grads, state, lr, momentum=0.95, ns_steps=5):
    """One Muon step with Moonshot-style LR adjustment."""
# @end
# @ref no-grad
    with torch.no_grad():
# @end
# @ref parameter-loop
        for index, (param, grad) in enumerate(zip(params, grads, strict=True)):
            if grad is None:
                continue
            if param.ndim != 2:
                raise ValueError("Muon only supports 2D parameters")
# @end
# @ref state-init
            param_state = state.setdefault(index, {})
            buffer = param_state.get("momentum_buffer")

            if buffer is None:
                buffer = torch.zeros_like(grad)
# @end
# @ref momentum
            buffer.mul_(momentum).add_(grad)
# @end
# @ref direction
            update = buffer
# @end
# @ref orthogonalize
            update = zeropower_via_newtonschulz5(update, steps=ns_steps)
# @end
# @ref adjust-lr
            adjusted_lr = 0.2 * lr * max(param.size(0), param.size(1)) ** 0.5
# @end
# @ref update
            param.add_(update, alpha=-adjusted_lr)
# @end
# @ref state-store
            param_state["momentum_buffer"] = buffer
# @end
# @ref return-state
    return params, state
# @end
