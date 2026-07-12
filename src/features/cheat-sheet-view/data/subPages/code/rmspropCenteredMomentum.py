import torch


# @ref step-signature
def rmsprop_step(
    params,
    grads,
    state,
    lr,
    alpha=0.99,
    eps=1e-8,
    momentum=0.9,
):
    """One centered RMSProp step with momentum."""
# @end
# @ref no-grad
    with torch.no_grad():
# @end
# @ref parameter-loop
        for index, (param, grad) in enumerate(zip(params, grads, strict=True)):
            if grad is None:
                continue
# @end
# @ref state-init
            param_state = state.setdefault(index, {})
            if (square_avg := param_state.get("square_avg")) is None:
                square_avg = param_state["square_avg"] = torch.zeros_like(param)
# @end
# @ref centered-state-init
            if (grad_avg := param_state.get("grad_avg")) is None:
                grad_avg = param_state["grad_avg"] = torch.zeros_like(param)
# @end
# @ref momentum-state-init
            if (buffer := param_state.get("momentum_buffer")) is None:
                buffer = param_state["momentum_buffer"] = torch.zeros_like(param)
# @end
# @ref gradient-direction
            d_p = grad
# @end
# @ref square-average
            square_avg.mul_(alpha).addcmul_(d_p, d_p, value=1.0 - alpha)
# @end
# @ref centered-average
            grad_avg.mul_(alpha).add_(d_p, alpha=1.0 - alpha)
            avg = square_avg.addcmul(grad_avg, grad_avg, value=-1.0)
# @end
# @ref denominator
            avg = avg.sqrt().add_(eps)
# @end
# @ref momentum
            buffer.mul_(momentum).addcdiv_(d_p, avg)
# @end
# @ref update
            param.add_(buffer, alpha=-lr)
# @end
# @ref return-state
    return params, state
# @end
