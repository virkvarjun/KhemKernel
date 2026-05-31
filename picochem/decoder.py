"""Transformer decoder block and stacked decoder (masked self-attn + cross-attn)."""
import numpy as np
from picochem.ops import layer_norm_forward, layer_norm_backward
from picochem.attention import (
    multihead_self_attention_forward,
    multihead_self_attention_backward,
    multihead_cross_attention_forward,
    multihead_cross_attention_backward,
)
from picochem.ffn import ffn_forward, ffn_backward
def decoder_block_forward(x, encoder_output, params, n_heads, causal_mask=None, encoder_padding_mask=None):
    # x: (B, T, D)
    # encoder_output: (B, S, D)
    # params: dict with three sets of layer norms, self-attn weights,
    #         cross-attn weights, FFN weights
    # returns: (B, T, D), cache

    # Sub-layer 1: causal self-attention
    x_norm1, ln1_cache = layer_norm_forward(x, params['ln1_gamma'], params['ln1_beta'])
    self_attn_out, self_attn_cache = multihead_self_attention_forward(
         x_norm1,
        params['self_Wq'], params['self_Wk'], params['self_Wv'], params['self_Wo'],
        params['self_bq'], params['self_bk'], params['self_bv'], params['self_bo'],
        n_heads=n_heads, mask=causal_mask,
    )
    x1 = x + self_attn_out

    # Sub Layer 2: Cross-Attention
    x_norm2, ln2_cache = layer_norm_forward(x1, params['ln2_gamma'], params['ln2_beta'])
    cross_attn_out, cross_attn_cache = multihead_cross_attention_forward(
         x_norm2, encoder_output,
        params['cross_Wq'], params['cross_Wk'], params['cross_Wv'], params['cross_Wo'],
        params['cross_bq'], params['cross_bk'], params['cross_bv'], params['cross_bo'],
        n_heads=n_heads, mask=encoder_padding_mask,
    )
    x2 = x1 + cross_attn_out

    # Sub-Layer 3: FFN
    x_norm3, ln3_cache = layer_norm_forward(x2, params['ln3_gamma'], params['ln3_beta'])
    ffn_out, ffn_cache = ffn_forward(
        x_norm3, params['ffn_W1'], params['ffn_b1'],
        params['ffn_W2'], params['ffn_b2'],
    )
    out = x2 + ffn_out

    cache = {
        'ln1': ln1_cache, 'self_attn': self_attn_cache,
        'ln2': ln2_cache, 'cross_attn': cross_attn_cache,
        'ln3': ln3_cache, 'ffn': ffn_cache,
    }
    return out, cache

def decoder_block_backward(grad_out, cache):
    # Sub-layer 3: FFN
    grad_ffn_out = grad_out
    grad_x2_residual = grad_out
    grad_x_norm3, grad_W1, grad_b1, grad_W2, grad_b2 = ffn_backward(grad_ffn_out, cache['ffn'])
    grad_x2_ffn, grad_ln3_gamma, grad_ln3_beta = layer_norm_backward(grad_x_norm3, cache['ln3'])
    grad_x2 = grad_x2_residual + grad_x2_ffn

    # Sub-layer 2: cross-attention
    grad_cross_out = grad_x2
    grad_x1_residual = grad_x2
    (grad_x_norm2, grad_enc,
     grad_cross_Wq, grad_cross_Wk, grad_cross_Wv, grad_cross_Wo,
     grad_cross_bq, grad_cross_bk, grad_cross_bv, grad_cross_bo) = (
        multihead_cross_attention_backward(grad_cross_out, cache['cross_attn'])
    )
    grad_x1_cross, grad_ln2_gamma, grad_ln2_beta = layer_norm_backward(grad_x_norm2, cache['ln2'])
    grad_x1 = grad_x1_residual + grad_x1_cross

    # Sub-layer 1: causal self-attention
    grad_self_out = grad_x1
    grad_x_residual = grad_x1
    (grad_x_norm1,
     grad_self_Wq, grad_self_Wk, grad_self_Wv, grad_self_Wo,
     grad_self_bq, grad_self_bk, grad_self_bv, grad_self_bo) = (
        multihead_self_attention_backward(grad_self_out, cache['self_attn'])
    )
    grad_x_self, grad_ln1_gamma, grad_ln1_beta = layer_norm_backward(grad_x_norm1, cache['ln1'])
    grad_x = grad_x_residual + grad_x_self

    grads = {
        'ln1_gamma': grad_ln1_gamma, 'ln1_beta': grad_ln1_beta,
        'self_Wq': grad_self_Wq, 'self_Wk': grad_self_Wk,
        'self_Wv': grad_self_Wv, 'self_Wo': grad_self_Wo,
        'self_bq': grad_self_bq, 'self_bk': grad_self_bk,
        'self_bv': grad_self_bv, 'self_bo': grad_self_bo,
        'ln2_gamma': grad_ln2_gamma, 'ln2_beta': grad_ln2_beta,
        'cross_Wq': grad_cross_Wq, 'cross_Wk': grad_cross_Wk,
        'cross_Wv': grad_cross_Wv, 'cross_Wo': grad_cross_Wo,
        'cross_bq': grad_cross_bq, 'cross_bk': grad_cross_bk,
        'cross_bv': grad_cross_bv, 'cross_bo': grad_cross_bo,
        'ln3_gamma': grad_ln3_gamma, 'ln3_beta': grad_ln3_beta,
        'ffn_W1': grad_W1, 'ffn_b1': grad_b1,
        'ffn_W2': grad_W2, 'ffn_b2': grad_b2,
    }
    return grad_x, grad_enc, grads