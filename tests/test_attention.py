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

from picochem.attention import (
    multihead_cross_attention_forward,
    multihead_cross_attention_backward,
)


def test_multihead_cross_attention():
    rng = np.random.default_rng(3)
    B, T, S, D, H = 2, 3, 5, 8, 2

    x_dec = rng.standard_normal((B, T, D)).astype(np.float64)
    x_enc = rng.standard_normal((B, S, D)).astype(np.float64)
    W_q = rng.standard_normal((D, D)).astype(np.float64) * 0.1
    W_k = rng.standard_normal((D, D)).astype(np.float64) * 0.1
    W_v = rng.standard_normal((D, D)).astype(np.float64) * 0.1
    W_o = rng.standard_normal((D, D)).astype(np.float64) * 0.1
    b_q = rng.standard_normal((D,)).astype(np.float64) * 0.01
    b_k = rng.standard_normal((D,)).astype(np.float64) * 0.01
    b_v = rng.standard_normal((D,)).astype(np.float64) * 0.01
    b_o = rng.standard_normal((D,)).astype(np.float64) * 0.01

    def fwd(x_dec, x_enc, Wq, Wk, Wv, Wo, bq, bk, bv, bo):
        return multihead_cross_attention_forward(
            x_dec, x_enc, Wq, Wk, Wv, Wo, bq, bk, bv, bo, n_heads=H,
        )

    check_gradient(
        fwd, multihead_cross_attention_backward,
        [x_dec, x_enc, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o],
    )

