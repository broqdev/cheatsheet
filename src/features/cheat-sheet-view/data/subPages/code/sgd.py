import torch


# @ref step-signature
def sgd_step(params, grads, lr):
    """One plain SGD step over matching parameter and gradient lists."""
# @end
# @ref no-grad
    with torch.no_grad():
# @end
# @ref parameter-loop
        for param, grad in zip(params, grads, strict=True):
            if grad is None:
                continue
# @end
# @ref gradient-direction
            d_p = grad
# @end
# @ref update
            param.add_(d_p, alpha=-lr)
# @end
# @ref return-params
    return params
# @end
