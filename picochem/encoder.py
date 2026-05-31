"""Transformer encoder block and stacked encoder."""

import numpy as np

from picochem.ops import layer_norm_forward, layer_norm_backward
from picochem.attention import (
    multihead_self_attention_forward,
    multihead_self_attention_backward,
)
from picochem.ffn import ffn_forward, ffn_backward


def encoder_block_forward(x, params, n_heads, padding_mask=None):
      # x: (B, S, D)
    # params: dict with 'ln1_gamma', 'ln1_beta', 'attn_W*', 'attn_b*',
    #                   'ln2_gamma', 'ln2_beta', 'ffn_W1', 'ffn_b1', 'ffn_W2', 'ffn_b2'
    # returns: (B, S, D), cache (also a dict)

    # Sub-layer 1: Self-Attnetion
    x_norm1, ln1_cache = layer_norm_forward(x, params['ln1_gamma'], params['ln1_beta'])
    attn_out, attn_cache = multihead_self_attention_forward(
        x_norm1,
        params['attn_Wq'], params['attn_Wk'], params['attn_Wv'], params['attn_Wo'],
        params['attn_bq'], params['attn_bk'], params['attn_bv'], params['attn_bo'],
        n_heads=n_heads, mask=padding_mask,
    )
    x1 = x + attn_out

    # Sub-layer 2: FFN
    x_norm2, ln2_cache = layer_norm_forward(x1, params['ln2_gamma'], params['ln2_beta'])
    ffn_out, ffn_cache = ffn_forward (
        x_norm2, params['ffn_W1'], params['ffn_b1'],
        params['ffn_W2'], params['ffn_b2'],
    )
    out = x1 + ffn_out

    cache = {'ln1': ln1_cache, 'attn': attn_cache,
             'ln2': ln2_cache, 'ffn': ffn_cache}
    return out, cache

def encoder_block_backward(grad_out, cache):
    #   Sub layer 2: FFN
    grad_ffn_out = grad_out
    grad_x1_residual = grad_out
    grad_x_norm2, grad_W1, grad_b1, grad_W2, grad_b2 = ffn_backward(grad_ffn_out, cache['ffn'])
    grad_x1_ffn, grad_ln2_gamma, grad_ln2_beta = layer_norm_backward(grad_x_norm2, cache['ln2'])
    grad_x1 = grad_x1_residual + grad_x1_ffn

    # Sub layer 1: Self-Attention
    grad_attn_out = grad_x1
    grad_x_residual = grad_x1

    (grad_x_norm1, grad_attn_Wq, grad_attn_Wk, grad_attn_Wv, grad_attn_Wo,
     grad_attn_bq, grad_attn_bk, grad_attn_bv, grad_attn_bo) = (
        multihead_self_attention_backward(grad_attn_out, cache['attn'])
    )
    grad_x_attn, grad_ln1_gamma, grad_ln1_beta = layer_norm_backward(grad_x_norm1, cache['ln1'])
    grad_x = grad_x_residual + grad_x_attn

    grads = {
        'ln1_gamma': grad_ln1_gamma, 'ln1_beta': grad_ln1_beta,
        'attn_Wq': grad_attn_Wq, 'attn_Wk': grad_attn_Wk,
        'attn_Wv': grad_attn_Wv, 'attn_Wo': grad_attn_Wo,
        'attn_bq': grad_attn_bq, 'attn_bk': grad_attn_bk,
        'attn_bv': grad_attn_bv, 'attn_bo': grad_attn_bo,
        'ln2_gamma': grad_ln2_gamma, 'ln2_beta': grad_ln2_beta,
        'ffn_W1': grad_W1, 'ffn_b1': grad_b1,
        'ffn_W2': grad_W2, 'ffn_b2': grad_b2,
    }
    return grad_x, grads