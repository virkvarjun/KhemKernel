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

"""Test encoder block with gradient checking."""

import sys
sys.path.insert(0, ".")

import numpy as np

from picochem.encoder import encoder_block_forward, encoder_block_backward


def init_encoder_params(D, DF, rng):
    return {
        'ln1_gamma': rng.standard_normal((D,)).astype(np.float64),
        'ln1_beta': rng.standard_normal((D,)).astype(np.float64),
        'attn_Wq': rng.standard_normal((D, D)).astype(np.float64) * 0.1,
        'attn_Wk': rng.standard_normal((D, D)).astype(np.float64) * 0.1,
        'attn_Wv': rng.standard_normal((D, D)).astype(np.float64) * 0.1,
        'attn_Wo': rng.standard_normal((D, D)).astype(np.float64) * 0.1,
        'attn_bq': rng.standard_normal((D,)).astype(np.float64) * 0.01,
        'attn_bk': rng.standard_normal((D,)).astype(np.float64) * 0.01,
        'attn_bv': rng.standard_normal((D,)).astype(np.float64) * 0.01,
        'attn_bo': rng.standard_normal((D,)).astype(np.float64) * 0.01,
        'ln2_gamma': rng.standard_normal((D,)).astype(np.float64),
        'ln2_beta': rng.standard_normal((D,)).astype(np.float64),
        'ffn_W1': rng.standard_normal((D, DF)).astype(np.float64) * 0.1,
        'ffn_b1': rng.standard_normal((DF,)).astype(np.float64) * 0.01,
        'ffn_W2': rng.standard_normal((DF, D)).astype(np.float64) * 0.1,
        'ffn_b2': rng.standard_normal((D,)).astype(np.float64) * 0.01,
    }


def numerical_grad_param(forward_fn, params, key, eps=1e-5):
    """Compute numerical gradient for one parameter in the params dict."""
    grad = np.zeros_like(params[key])
    it = np.nditer(params[key], flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original = params[key][idx]

        params[key][idx] = original + eps
        f_plus, _ = forward_fn(params)
        plus_loss = f_plus.sum()

        params[key][idx] = original - eps
        f_minus, _ = forward_fn(params)
        minus_loss = f_minus.sum()

        params[key][idx] = original
        grad[idx] = (plus_loss - minus_loss) / (2 * eps)
        it.iternext()

    return grad


def test_encoder_block_grad_x():
    """Test gradient w.r.t. input x via finite differences."""
    rng = np.random.default_rng(0)
    B, S, D, DF, H = 2, 3, 8, 16, 2

    x = rng.standard_normal((B, S, D)).astype(np.float64)
    params = init_encoder_params(D, DF, rng)

    out, cache = encoder_block_forward(x, params, n_heads=H)
    grad_out = np.ones_like(out)
    grad_x, grads = encoder_block_backward(grad_out, cache)

    # Numerical gradient w.r.t. x
    eps = 1e-5
    num_grad_x = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        out_plus, _ = encoder_block_forward(x, params, n_heads=H)
        x[idx] = original - eps
        out_minus, _ = encoder_block_forward(x, params, n_heads=H)
        x[idx] = original
        num_grad_x[idx] = (out_plus.sum() - out_minus.sum()) / (2 * eps)
        it.iternext()

    diff = np.abs(grad_x - num_grad_x).max()
    assert diff < 1e-4, f"grad_x mismatch: {diff:.2e}"


def test_encoder_block_grad_params():
    """Test gradient w.r.t. a few key parameters."""
    rng = np.random.default_rng(1)
    B, S, D, DF, H = 2, 3, 8, 16, 2

    x = rng.standard_normal((B, S, D)).astype(np.float64)
    params = init_encoder_params(D, DF, rng)

    out, cache = encoder_block_forward(x, params, n_heads=H)
    grad_out = np.ones_like(out)
    _, analytical_grads = encoder_block_backward(grad_out, cache)

    def fwd_with_params(p):
        return encoder_block_forward(x, p, n_heads=H)

    # Check a few key parameters
    for key in ['attn_Wq', 'ffn_W1', 'ln1_gamma']:
        num_grad = numerical_grad_param(fwd_with_params, params, key)
        diff = np.abs(analytical_grads[key] - num_grad).max()
        assert diff < 1e-4, f"{key}: diff = {diff:.2e}"