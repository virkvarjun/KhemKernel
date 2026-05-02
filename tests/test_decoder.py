"""Test decoder block with gradient checking."""

import sys
sys.path.insert(0, ".")

import numpy as np

from picochem.decoder import decoder_block_forward, decoder_block_backward


def init_decoder_params(D, DF, rng):
    p = {}
    for prefix in ['self', 'cross']:
        for w in ['Wq', 'Wk', 'Wv', 'Wo']:
            p[f'{prefix}_{w}'] = rng.standard_normal((D, D)).astype(np.float64) * 0.1
        for b in ['bq', 'bk', 'bv', 'bo']:
            p[f'{prefix}_{b}'] = rng.standard_normal((D,)).astype(np.float64) * 0.01
    for i in [1, 2, 3]:
        p[f'ln{i}_gamma'] = rng.standard_normal((D,)).astype(np.float64)
        p[f'ln{i}_beta'] = rng.standard_normal((D,)).astype(np.float64)
    p['ffn_W1'] = rng.standard_normal((D, DF)).astype(np.float64) * 0.1
    p['ffn_b1'] = rng.standard_normal((DF,)).astype(np.float64) * 0.01
    p['ffn_W2'] = rng.standard_normal((DF, D)).astype(np.float64) * 0.1
    p['ffn_b2'] = rng.standard_normal((D,)).astype(np.float64) * 0.01
    return p


def test_decoder_block_grad_x_and_enc():
    rng = np.random.default_rng(0)
    B, T, S, D, DF, H = 2, 3, 4, 8, 16, 2

    x = rng.standard_normal((B, T, D)).astype(np.float64)
    enc = rng.standard_normal((B, S, D)).astype(np.float64)
    params = init_decoder_params(D, DF, rng)

    out, cache = decoder_block_forward(x, enc, params, n_heads=H)
    grad_out = np.ones_like(out)
    grad_x, grad_enc, _ = decoder_block_backward(grad_out, cache)

    eps = 1e-5

    # Check grad_x
    num_grad_x = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        out_p, _ = decoder_block_forward(x, enc, params, n_heads=H)
        x[idx] = original - eps
        out_m, _ = decoder_block_forward(x, enc, params, n_heads=H)
        x[idx] = original
        num_grad_x[idx] = (out_p.sum() - out_m.sum()) / (2 * eps)
        it.iternext()

    assert np.abs(grad_x - num_grad_x).max() < 1e-4

    # Check grad_enc
    num_grad_enc = np.zeros_like(enc)
    it = np.nditer(enc, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original = enc[idx]
        enc[idx] = original + eps
        out_p, _ = decoder_block_forward(x, enc, params, n_heads=H)
        enc[idx] = original - eps
        out_m, _ = decoder_block_forward(x, enc, params, n_heads=H)
        enc[idx] = original
        num_grad_enc[idx] = (out_p.sum() - out_m.sum()) / (2 * eps)
        it.iternext()

    assert np.abs(grad_enc - num_grad_enc).max() < 1e-4