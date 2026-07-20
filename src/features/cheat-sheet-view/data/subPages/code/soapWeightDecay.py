import torch


# @ref eigenvectors-signature
def eigenvectors_descending(matrix):
    """Return an orthonormal eigenbasis ordered by decreasing eigenvalue."""
# @end
# @ref eigendecomposition
    _, eigenvectors = torch.linalg.eigh(matrix)
    return eigenvectors.flip(dims=(1,))
# @end


# @ref basis-refresh-signature
def refresh_eigenbasis(matrix, basis):
    """Advance an eigenbasis with one power iteration followed by QR."""
# @end
# @ref power-iteration
    next_basis, _ = torch.linalg.qr(matrix @ basis)
    return next_basis
# @end


# @ref step-signature
def soap_step(
    params,
    grads,
    state,
    lr,
    betas=(0.95, 0.95),
    eps=1e-8,
    precondition_frequency=10,
    weight_decay=0.0,
):
    """One SOAP step with decoupled weight decay for 2D matrix parameters."""
    if precondition_frequency < 1:
        raise ValueError("precondition_frequency must be positive")
# @end
# @ref no-grad
    with torch.no_grad():
# @end
# @ref parameter-loop
        for index, (param, grad) in enumerate(zip(params, grads, strict=True)):
            if grad is None:
                continue
            if param.ndim != 2:
                raise ValueError("This SOAP sketch only supports 2D parameters")
            matrix_grad = grad if grad.dtype == torch.float64 else grad.float()
# @end
# @ref state-init
            param_state = state.setdefault(index, {})
            step = param_state.get("step", 0) + 1
            beta1, beta2 = betas
            first_moment = param_state.get("exp_avg")

            if first_moment is None:
                first_moment = torch.zeros_like(matrix_grad)
                second_moment = torch.zeros_like(matrix_grad)
                left = (1.0 - beta2) * (matrix_grad @ matrix_grad.T)
                right = (1.0 - beta2) * (matrix_grad.T @ matrix_grad)
                q_left = eigenvectors_descending(left)
                q_right = eigenvectors_descending(right)
                initialized_preconditioner = True
            else:
                second_moment = param_state["exp_avg_sq"]
                left = param_state["left"]
                right = param_state["right"]
                q_left = param_state["q_left"]
                q_right = param_state["q_right"]
                initialized_preconditioner = False
# @end
# @ref project-gradient
            projected_grad = q_left.T @ matrix_grad @ q_right
# @end
# @ref first-moment
            first_moment.lerp_(matrix_grad, 1.0 - beta1)
            projected_first_moment = q_left.T @ first_moment @ q_right
# @end
# @ref second-moment
            second_moment.lerp_(projected_grad.square(), 1.0 - beta2)
# @end
# @ref bias-correction
            corrected_first = projected_first_moment / (1.0 - beta1**step)
            corrected_second = second_moment / (1.0 - beta2**step)
# @end
# @ref normalize
            projected_update = corrected_first / (corrected_second.sqrt() + eps)
# @end
# @ref project-back
            update = q_left @ projected_update @ q_right.T
# @end
# @ref weight-decay
            if weight_decay != 0.0:
                param.mul_(1.0 - lr * weight_decay)
# @end
# @ref update
            param.add_(update.to(param.dtype), alpha=-lr)
# @end
# @ref preconditioner-update
            if not initialized_preconditioner:
                left.lerp_(matrix_grad @ matrix_grad.T, 1.0 - beta2)
                right.lerp_(matrix_grad.T @ matrix_grad, 1.0 - beta2)
# @end
# @ref basis-refresh
            if not initialized_preconditioner and step % precondition_frequency == 0:
                q_left = refresh_eigenbasis(left, q_left)
                q_right = refresh_eigenbasis(right, q_right)
# @end
# @ref state-store
            param_state.update(
                step=step,
                exp_avg=first_moment,
                exp_avg_sq=second_moment,
                left=left,
                right=right,
                q_left=q_left,
                q_right=q_right,
            )
# @end
# @ref return-state
    return params, state
# @end
