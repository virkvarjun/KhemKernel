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