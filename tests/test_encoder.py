"""Test FFN with gradient checking."""

import sys
sys.path.insert(0, ".")

import numpy as np

from tests.test_ops import check_gradient
from picochem.ffn import ffn_forward, ffn_backward


def test_ffn_gradient():
    rng = np.random.default_rng(0)
    B, S, D, DF = 2, 3, 4, 8

    x = rng.standard_normal((B, S, D)).astype(np.float64)
    W1 = rng.standard_normal((D, DF)).astype(np.float64) * 0.1
    b1 = rng.standard_normal((DF,)).astype(np.float64) * 0.01
    W2 = rng.standard_normal((DF, D)).astype(np.float64) * 0.1
    b2 = rng.standard_normal((D,)).astype(np.float64) * 0.01

    check_gradient(ffn_forward, ffn_backward, [x, W1, b1, W2, b2])