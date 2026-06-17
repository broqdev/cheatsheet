import torch

# @ref forward-signature
def attention_forward(Q, K, V, mask):
# @end
# @ref score
    # Additive mask is 0 for visible keys and -inf for blocked keys.
    alpha = 1.0 / torch.sqrt(torch.tensor(Q.shape[-1], device=Q.device))
    S = alpha * (Q @ K.transpose(-2, -1))
    S = S + mask
# @end
# @ref probability
    P = torch.softmax(S, dim=-1)
# @end
# @ref output
    O = P @ V
    return O, P
# @end

# @ref backward-signature
def attention_backward(Q, K, V, P, O, dO):
# @end
# @ref dv-dp
    # Masked positions already have P=0, so dS is zero there.
    dV = P.transpose(-2, -1) @ dO
    dP = dO @ V.transpose(-2, -1)
# @end

# @ref softmax-call
    # Rowwise softmax backward: dS = P * (dP - rowsum(P * dP)).
# @end
# @ref softmax-call-result softmax-call
    dS = softmax_backward(P, dP)
# @end

# @ref dq-dk
    alpha = 1.0 / torch.sqrt(torch.tensor(Q.shape[-1], device=Q.device))
    dQ = alpha * (dS @ K)
    dK = alpha * (dS.transpose(-2, -1) @ Q)
    return dQ, dK, dV
# @end

# @ref softmax-def softmax-body
def softmax_backward(P, dP):
# @end
# @ref softmax-comment softmax-body
    # Subtract one correction scalar per row before scaling by P.
# @end
# @ref softmax-correction softmax-body
    D = (P * dP).sum(dim=-1, keepdim=True)
# @end
# @ref softmax-return softmax-body
    return P * (dP - D)
# @end
