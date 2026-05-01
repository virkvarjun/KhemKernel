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

def multihead_self_attention_forward(x, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o,
                                     n_heads, mask=None):
    """Multi-head self-attention forward pass.

    Args:
        x: (B, S, D) input sequence
        W_q, W_k, W_v: (D, D) projection matrices for Q, K, V
        W_o: (D, D) output projection
        b_q, b_k, b_v, b_o: (D,) bias vectors
        n_heads: number of attention heads
        mask: optional additive mask, shape broadcastable to (B, n_heads, S, S)

    Returns:
        out: (B, S, D)
        cache: values needed for backward
    """
    B, S, D = x.shape
    H = n_heads
    Dh = D // H

    # Project to Q, K, V: (B*S, D)
    Q, q_cache = linear_forward(x.reshape(B * S, D), W_q, b_q)
    K, k_cache = linear_forward(x.reshape(B * S, D), W_k, b_k)
    V, v_cache = linear_forward(x.reshape(B * S, D), W_v, b_v)

    # Split heads: (B*S, D) → (B, S, H, Dh) → (B, H, S, Dh)
    Q = Q.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    attn_out, attn_cache = scaled_dot_product_attention_forward(Q, K, V, mask=mask)

    # Concatenate heads: (B, H, S, Dh) → (B, S, H, Dh) → (B, S, D)
    concat = attn_out.transpose(0, 2, 1, 3).reshape(B, S, D)

    # Output projection
    out, o_cache = linear_forward(concat.reshape(B * S, D), W_o, b_o)
    out = out.reshape(B, S, D)

    cache = (B, S, D, H, Dh, x.shape, q_cache, k_cache, v_cache, attn_cache, o_cache)
    return out, cache


def multihead_self_attention_backward(grad_out, cache):
    B, S, D, H, Dh, x_shape, q_cache, k_cache, v_cache, attn_cache, o_cache = cache

    # Backward through output projection
    grad_concat_flat, grad_W_o, grad_b_o = linear_backward(grad_out.reshape(B * S, D), o_cache)

    # Reverse concatenate-heads: (B*S, D) → (B, S, H, Dh) → (B, H, S, Dh)
    grad_attn_out = grad_concat_flat.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)

    # Backward through scaled dot-product attention
    grad_Q_heads, grad_K_heads, grad_V_heads = scaled_dot_product_attention_backward(grad_attn_out, attn_cache)

    # Reverse split-heads: (B, H, S, Dh) → (B, S, H, Dh) → (B*S, D)
    grad_Q_flat = grad_Q_heads.transpose(0, 2, 1, 3).reshape(B * S, D)
    grad_K_flat = grad_K_heads.transpose(0, 2, 1, 3).reshape(B * S, D)
    grad_V_flat = grad_V_heads.transpose(0, 2, 1, 3).reshape(B * S, D)

    # Backward through QKV projections
    grad_x_q, grad_W_q, grad_b_q = linear_backward(grad_Q_flat, q_cache)
    grad_x_k, grad_W_k, grad_b_k = linear_backward(grad_K_flat, k_cache)
    grad_x_v, grad_W_v, grad_b_v = linear_backward(grad_V_flat, v_cache)

    # Sum contributions to grad_x (Q, K, V all projected from the same x)
    grad_x = (grad_x_q + grad_x_k + grad_x_v).reshape(x_shape)

    return grad_x, grad_W_q, grad_W_k, grad_W_v, grad_W_o, grad_b_q, grad_b_k, grad_b_v, grad_b_o