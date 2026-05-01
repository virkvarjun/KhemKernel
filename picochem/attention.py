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
    
    # Project to Q, K, V
    Q, q_cache = linear_forward(x.reshape(B * S, D), W_q, b_q)  # (B*S, D)
    K, k_cache = linear_forward(x.reshape(B * S, D), W_k, b_k)
    V, v_cache = linear_forward(x.reshape(B * S, D), W_v, b_v)
    
    # Reshape and Split heads: (B*S, D) → (B, S, H, Dh) → (B, H, S, Dh)
    Q = Q.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
    
    # Attention 
    attn_out, attn_cache = scaled_dot_product_attention_forward(Q, K, V, mask=mask)

    # Cancatenate heads: (B, H, S, Dh) → (B, S, H, Dh) → (B, S, D)
    concat = attn_out.transpose(0, 2, 1, 3).reshape(B, S, D) 

    # Output Projection
    out, o_cache = linear_forward(concat.reshape(B*S, D), W_o, b_o) 
    return out, cache 

def multihead_self_attention_backwards(grad_out, cache): 
    

