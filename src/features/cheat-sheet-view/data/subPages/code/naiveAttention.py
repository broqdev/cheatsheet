import torch

# @ref forward-signature
def attention_forward(Q, K, V):
# @end
# @ref score
    # alpha scales QK scores; 1/sqrt(d) is the usual transformer choice.
    alpha = 1.0 / torch.sqrt(torch.tensor(Q.shape[-1], device=Q.device))
    S = alpha * (Q @ K.transpose(-2, -1))
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
    # Value gradient is the easy matrix multiply.
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
