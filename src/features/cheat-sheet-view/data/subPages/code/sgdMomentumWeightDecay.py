import torch


# @ref step-signature
def sgd_step(params, grads, state, lr, momentum=0.9, weight_decay=0.0, dampening=0.0):
    """One PyTorch-style SGD step with momentum and coupled L2 decay."""
# @end
# @ref no-grad
    with torch.no_grad():
# @end
# @ref parameter-loop
        for index, (param, grad) in enumerate(zip(params, grads, strict=True)):
            if grad is None:
                continue
# @end
# @ref gradient-direction
            d_p = grad
# @end
# @ref weight-decay
            if weight_decay != 0.0:
                d_p = d_p.add(param, alpha=weight_decay)
# @end
# @ref state-init
            param_state = state.setdefault(index, {})
            buffer = param_state.get("momentum_buffer")
# @end
# @ref momentum
            if buffer is None:
                buffer = torch.clone(d_p).detach()
            else:
                buffer.mul_(momentum).add_(d_p, alpha=1.0 - dampening)
            d_p = buffer
# @end
# @ref update
            param.add_(d_p, alpha=-lr)
# @end
# @ref state-store
            param_state["momentum_buffer"] = buffer
# @end
# @ref return-state
    return params, state
# @end
