"""Multi-head self-attention and cross-attention with numpy."""
import numpy as np
from picochem.ops import softmax_forward, softmax_backward, linear_forward, linear_backward


def scaled_dot_product_attention_forward(Q, K, V, mask=None):
    '''
    Args:
        Q: (B, H, T, Dh) queries
        K: (B, H, S, Dh) keys
        V: (B, H, S, Dh) values
        mask: (B, 1, T, S) or (1, 1, T, S) additive mask. 0 for allowed, -inf for blocked.

    Returns:
        out: (B, H, T, Dh)
        cache: values needed for backward
    '''
    Dh = Q.shape[-1]
    scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(Dh)
    if mask is not None:
        scores = scores + mask

    weights, softmax_cache = softmax_forward(scores)
    out = np.matmul(weights, V)
    cache = (Q, K, V, weights, softmax_cache, mask)
    return out, cache


def scaled_dot_product_attention_backward(grad_out, cache):
    Q, K, V, weights, softmax_cache, mask = cache
    Dh = Q.shape[-1]

    grad_weights = np.matmul(grad_out, V.swapaxes(-2, -1))
    grad_V = np.matmul(weights.swapaxes(-2, -1), grad_out)

    grad_scores, = softmax_backward(grad_weights, softmax_cache)
    grad_scores = grad_scores / np.sqrt(Dh)

    grad_Q = np.matmul(grad_scores, K)
    grad_K = np.matmul(grad_scores.swapaxes(-2, -1), Q)

    return grad_Q, grad_K, grad_V
