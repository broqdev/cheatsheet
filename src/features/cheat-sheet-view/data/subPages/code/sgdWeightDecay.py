import torch


# @ref step-signature
def sgd_step(params, grads, lr, weight_decay=0.0):
    """One SGD step with coupled L2 weight decay."""
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
# @ref weight-decay
            if weight_decay != 0.0:
                d_p = d_p.add(param, alpha=weight_decay)
# @end
# @ref update
            param.add_(d_p, alpha=-lr)
# @end
# @ref return-params
    return params
# @end
