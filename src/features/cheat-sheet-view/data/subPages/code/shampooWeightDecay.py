import torch


# @ref matrix-power-signature
def symmetric_matrix_power(matrix, exponent, eps):
    """Raise a symmetric positive-semidefinite matrix to a real power."""
# @end
# @ref matrix-power-eigh
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    powered = eigenvalues.clamp_min(eps).pow(exponent)
# @end
# @ref matrix-power-reconstruct
    return (eigenvectors * powered.unsqueeze(0)) @ eigenvectors.T
# @end


# @ref step-signature
def shampoo_step(params, grads, state, lr, weight_decay=0.0, eps=1e-8):
    """One original Shampoo step with coupled L2 weight decay."""
# @end
# @ref no-grad
    with torch.no_grad():
# @end
# @ref parameter-loop
        for index, (param, grad) in enumerate(zip(params, grads, strict=True)):
            if grad is None:
                continue
            if param.ndim != 2:
                raise ValueError("This Shampoo sketch only supports 2D parameters")
            matrix_grad = grad if grad.dtype == torch.float64 else grad.float()
# @end
# @ref state-init
            param_state = state.setdefault(index, {})
            rows, columns = param.shape
            left = param_state.get("left")
            right = param_state.get("right")

            if left is None:
                left = eps * torch.eye(rows, dtype=matrix_grad.dtype, device=grad.device)
                right = eps * torch.eye(columns, dtype=matrix_grad.dtype, device=grad.device)
# @end
# @ref weight-decay
            if weight_decay != 0.0:
                matrix_grad = matrix_grad.add(param, alpha=weight_decay)
# @end
# @ref preconditioner-update
            left.addmm_(matrix_grad, matrix_grad.T)
            right.addmm_(matrix_grad.T, matrix_grad)
# @end
# @ref inverse-roots
            left_inverse_root = symmetric_matrix_power(left, -0.25, eps)
            right_inverse_root = symmetric_matrix_power(right, -0.25, eps)
# @end
# @ref precondition-gradient
            update = left_inverse_root @ matrix_grad @ right_inverse_root
# @end
# @ref update
            param.add_(update.to(param.dtype), alpha=-lr)
# @end
# @ref state-store
            param_state["left"] = left
            param_state["right"] = right
# @end
# @ref return-state
    return params, state
# @end
