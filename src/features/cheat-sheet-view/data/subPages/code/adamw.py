import torch


# @ref step-signature
def adamw_step(params, grads, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """AdamW with weight_decay=0.0, matching Adam's moment update."""
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
            exp_avg = param_state.get("exp_avg")
            exp_avg_sq = param_state.get("exp_avg_sq")

            if exp_avg is None:
                exp_avg = torch.zeros_like(param)
                exp_avg_sq = torch.zeros_like(param)
# @end
# @ref gradient-direction
            d_p = grad
# @end
# @ref first-moment
            exp_avg.mul_(beta1).add_(d_p, alpha=1.0 - beta1)
# @end
# @ref second-moment
            exp_avg_sq.mul_(beta2).addcmul_(d_p, d_p, value=1.0 - beta2)
# @end
# @ref bias-correction
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            step_size = lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(eps)
# @end
# @ref update
            param.addcdiv_(exp_avg, denom, value=-step_size)
# @end
# @ref state-store
            param_state["step"] = step
            param_state["exp_avg"] = exp_avg
            param_state["exp_avg_sq"] = exp_avg_sq
# @end
# @ref return-state
    return params, state
# @end
