import torch


# @ref step-signature
def rmsprop_step(params, grads, state, lr, alpha=0.99, eps=1e-8, centered=False):
    """One RMSprop step. state maps parameter index to running averages."""
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
# @ref gradient-direction
            d_p = grad
# @end
# @ref square-average
            square_avg.mul_(alpha).addcmul_(d_p, d_p, value=1.0 - alpha)
# @end
# @ref centered-average
            avg = square_avg
            if centered:
# @end
# @ref centered-state-init centered-average
                if (grad_avg := param_state.get("grad_avg")) is None:
                    grad_avg = param_state["grad_avg"] = torch.zeros_like(param)
# @end
# @ref centered-average
                grad_avg.mul_(alpha).add_(d_p, alpha=1.0 - alpha)
                avg = avg.addcmul(grad_avg, grad_avg, value=-1.0)
# @end
# @ref denominator update
            param.addcdiv_(d_p, avg.sqrt().add_(eps), value=-lr)
# @end
# @ref return-state
    return params, state
# @end
