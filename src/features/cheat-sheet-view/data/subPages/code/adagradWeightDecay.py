import torch


# @ref step-signature
def adagrad_step(params, grads, state, lr, lr_decay=0.0, initial_accumulator_value=0.0, eps=1e-10, weight_decay=0.0):
    """One dense AdaGrad step with coupled L2 weight decay."""
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
            step = param_state.get("step", 0) + 1

            if (square_sum := param_state.get("sum")) is None:
                square_sum = param_state["sum"] = torch.full_like(param, initial_accumulator_value)
# @end
# @ref gradient-direction
            d_p = grad
# @end
# @ref weight-decay
            # Coupled decay: the penalty enters the squared-gradient accumulator.
            if weight_decay != 0.0:
                d_p = d_p.add(param, alpha=weight_decay)
# @end
# @ref learning-rate-decay
            step_size = lr / (1.0 + (step - 1.0) * lr_decay)
# @end
# @ref accumulator
            square_sum.addcmul_(d_p, d_p, value=1.0)
# @end
# @ref denominator
            denom = square_sum.sqrt().add_(eps)
# @end
# @ref update
            param.addcdiv_(d_p, denom, value=-step_size)
# @end
# @ref state-store
            param_state["step"] = step
# @end
# @ref return-state
    return params, state
# @end
