"""Tests for attention mechanisms."""

import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from tests.test_ops import check_gradient
from picochem.attention import (
    scaled_dot_product_attention_forward,
    scaled_dot_product_attention_backward,
)


def test_scaled_dot_product_attention_no_mask():
    rng = np.random.default_rng(0)
    B, H, T, S, Dh = 2, 2, 3, 4, 5
    Q = rng.standard_normal((B, H, T, Dh)).astype(np.float64)
    K = rng.standard_normal((B, H, S, Dh)).astype(np.float64)
    V = rng.standard_normal((B, H, S, Dh)).astype(np.float64)
    
    check_gradient(
        scaled_dot_product_attention_forward,
        scaled_dot_product_attention_backward,
        [Q, K, V],
    )


def test_scaled_dot_product_attention_causal_mask():
    rng = np.random.default_rng(1)
    B, H, T, Dh = 2, 2, 4, 5
    Q = rng.standard_normal((B, H, T, Dh)).astype(np.float64)
    K = rng.standard_normal((B, H, T, Dh)).astype(np.float64)
    V = rng.standard_normal((B, H, T, Dh)).astype(np.float64)
    
    # Causal mask: -inf above diagonal
    causal = np.triu(np.full((T, T), -1e9), k=1)
    causal = causal[None, None, :, :]  # broadcast over (B, H)
    
    def fwd(q, k, v):
        return scaled_dot_product_attention_forward(q, k, v, mask=causal)
    
    check_gradient(fwd, scaled_dot_product_attention_backward, [Q, K, V])

from picochem.attention import (
    multihead_self_attention_forward,
    multihead_self_attention_backward,
)


def test_multihead_self_attention():
    rng = np.random.default_rng(2)
    B, S, D, H = 2, 4, 8, 2  # D must be divisible by H
    
    x = rng.standard_normal((B, S, D)).astype(np.float64)
    W_q = rng.standard_normal((D, D)).astype(np.float64) * 0.1
    W_k = rng.standard_normal((D, D)).astype(np.float64) * 0.1
    W_v = rng.standard_normal((D, D)).astype(np.float64) * 0.1
    W_o = rng.standard_normal((D, D)).astype(np.float64) * 0.1
    b_q = rng.standard_normal((D,)).astype(np.float64) * 0.01
    b_k = rng.standard_normal((D,)).astype(np.float64) * 0.01
    b_v = rng.standard_normal((D,)).astype(np.float64) * 0.01
    b_o = rng.standard_normal((D,)).astype(np.float64) * 0.01
    
    def fwd(x, Wq, Wk, Wv, Wo, bq, bk, bv, bo):
        return multihead_self_attention_forward(
            x, Wq, Wk, Wv, Wo, bq, bk, bv, bo, n_heads=H,
        )
    
    check_gradient(
        fwd, multihead_self_attention_backward,
        [x, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o],
    )

def multihead_cross_attention_forward(x_dec, x_enc, W_q, W_k, W_v, W_o,
                                       b_q, b_k, b_v, b_o, n_heads, mask=None):
    """Cross-attention: Q from decoder, K/V from encoder.
    
    Args:
        x_dec: (B, T, D) decoder hidden state — provides Q
        x_enc: (B, S, D) encoder output — provides K, V
        W_q, W_k, W_v, W_o, b_*: projection matrices and biases
        n_heads: number of heads
        mask: optional, shape broadcastable to (B, H, T, S). Typically a padding mask
              for the encoder side; no causal mask in cross-attention.
    
    Returns:
        out: (B, T, D)
        cache: values needed for backward
    """
    B, T, D = x_dec.shape
    _, S, _ = x_enc.shape
    H = n_heads 
    Dh = D // H 

    # Q from decoder 
    Q, q_cache = linear_forward(x_dec.reshape(B*T, D), W_q, b_q) 
    # K, V from decoder 
    K, k_cache = linear_forward(x_enc.reshape(B*S, D), W_k, b_k) 
    V, v_cache = linear_forward(x_enc.reshape(B*S, D), W_v, b_v) 
    
    # Reshape and split heads 
    Q = Q.reshape(B, T, H, Dh).transpose(0, 2, 1, 3) 
    K = K.reshape(B, S, H, Dh).transpose(0, 2, 1, 3) 
    V = V.reshape(B, S, H, Dh).transpose(0, 2, 1, 3) 

    attn_out, attn_cache = scaled_dot_product_attention_forward(Q, K, V, mask=mask): 
    concat = attn_out.transpose(0, 2, 1, 3).reshape(B, T, D)
    out, o_cache = linear_forward(concat.reshape(B * T, D), W_o, b_o)
    out = out.reshape(B, T, D)
    
    cache = (B, T, S, D, H, Dh, x_dec.shape, x_enc.shape,
             q_cache, k_cache, v_cache, attn_cache, o_cache, concat.shape)
    return out, cache

def multihead_cross_attention_backward(grad_out, cache):
    (B, T, S, D, H, Dh, x_dec_shape, x_enc_shape,
     q_cache, k_cache, v_cache, attn_cache, o_cache, concat_shape) = cache
    
    # Backward through output projection 
    grad_concat_flat, grad_W_o, grad_b_o = linear_backward(
        grad_out.reshape(B * T, D), o_cache
    )
    grad_concat = grad_concat_flat.reshape(concat_shape)
    grad_attn_out = grad_concat.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
    
    # Backward through attention
    grad_Q_heads, grad_K_heads, grad_V_heads = scaled_dot_product_attention_backward(
        grad_attn_out, attn_cache
    )
    
    grad_Q_flat = grad_Q_heads.transpose(0, 2, 1, 3).reshape(B * T, D)
    grad_K_flat = grad_K_heads.transpose(0, 2, 1, 3).reshape(B * S, D)
    grad_V_flat = grad_V_heads.transpose(0, 2, 1, 3).reshape(B * S, D)
    
    # Backward through QKV projections
    grad_x_dec_q, grad_W_q, grad_b_q = linear_backward(grad_Q_flat, q_cache)
    grad_x_enc_k, grad_W_k, grad_b_k = linear_backward(grad_K_flat, k_cache)
    grad_x_enc_v, grad_W_v, grad_b_v = linear_backward(grad_V_flat, v_cache)
    
    # x_dec contributes only through Q
    grad_x_dec = grad_x_dec_q.reshape(x_dec_shape)
    # x_enc contributes through both K and V — sum the contributions
    grad_x_enc = (grad_x_enc_k + grad_x_enc_v).reshape(x_enc_shape)
    
    return (grad_x_dec, grad_x_enc,
            grad_W_q, grad_W_k, grad_W_v, grad_W_o,
            grad_b_q, grad_b_k, grad_b_v, grad_b_o)