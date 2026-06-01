"""Parity of the device-resident layers (picochem.device_layers) vs the NumPy
model. Skipped unless picochem_cuda is built and importable.

Run on the GPU pod:  PYTHONPATH=picochem/kernels pytest tests/test_device_layers.py -v
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "picochem/kernels")

import numpy as np
import pytest

try:
    import picochem_cuda
    import picochem.device_layers as dl
    HAS_CUDA = True
except Exception:
    HAS_CUDA = False

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="picochem_cuda not built")

rng = np.random.default_rng(0)


def _to_dt(params):
    """numpy param dict -> DeviceTensor dict (float32)."""
    return {k: picochem_cuda.DeviceTensor(np.asarray(v, dtype=np.float32)) for k, v in params.items()}


def _close(dev, ref, rtol, atol=1e-4):
    # Mixed rtol/atol: the key-projection bias gradient is structurally zero
    # (softmax is invariant to a per-row constant from a uniform key shift), so a
    # pure relative metric divides fp32 noise by fp64 noise. atol handles those.
    np.testing.assert_allclose(np.asarray(dev), np.asarray(ref), rtol=rtol, atol=atol)


def test_encoder_block_parity():
    from picochem.model import init_params, make_padding_mask
    from picochem.encoder import encoder_block_forward, encoder_block_backward

    B, S, D, H = 2, 7, 32, 4
    cfg = dict(src_vocab=10, tgt_vocab=10, d_model=D, n_heads=H, d_ff=64,
               n_enc_layers=1, n_dec_layers=1, max_src_len=S, max_tgt_len=S)
    params = init_params(cfg, np.random.default_rng(1))
    bp = params['encoder_blocks'][0]               # one block's params (float64)

    x = rng.standard_normal((B, S, D)).astype(np.float64)
    # padding mask: last 2 positions of sequence 0 are padding
    token_mask = np.ones((B, S), dtype=np.float64)
    token_mask[0, -2:] = 0.0
    pad = make_padding_mask(token_mask)            # (B,1,1,S) additive

    # numpy reference
    out_ref, cache_ref = encoder_block_forward(x, bp, H, padding_mask=pad)
    grad_out = rng.standard_normal((B, S, D)).astype(np.float64)
    gx_ref, grads_ref = encoder_block_backward(grad_out, cache_ref)

    # device: materialize the additive mask to (B*H, S, S)
    mask_full = np.broadcast_to(pad, (B, H, S, S)).reshape(B * H, S, S).astype(np.float32)
    p_dt = _to_dt(bp)
    x_dt = picochem_cuda.DeviceTensor(x.astype(np.float32))
    mask_dt = picochem_cuda.DeviceTensor(mask_full)

    out, cache = dl.encoder_block_forward(x_dt, p_dt, H, mask_dt=mask_dt)
    gx, grads = dl.encoder_block_backward(picochem_cuda.DeviceTensor(grad_out.astype(np.float32)), cache)

    _close(out.numpy(), out_ref, rtol=2e-3)
    _close(gx.numpy(), gx_ref, rtol=5e-3)
    for k, v in grads_ref.items():
        _close(grads[k].numpy(), v, rtol=1e-2)


def test_decoder_block_parity():
    from picochem.model import init_params, make_padding_mask, make_causal_mask
    from picochem.decoder import decoder_block_forward, decoder_block_backward
    import picochem.device_layers as dl

    B, T, S, D, H = 2, 6, 7, 32, 4
    cfg = dict(src_vocab=10, tgt_vocab=10, d_model=D, n_heads=H, d_ff=64,
               n_enc_layers=1, n_dec_layers=1, max_src_len=S, max_tgt_len=T)
    params = init_params(cfg, np.random.default_rng(2))
    bp = params['decoder_blocks'][0]

    x = rng.standard_normal((B, T, D)).astype(np.float64)        # decoder input
    enc = rng.standard_normal((B, S, D)).astype(np.float64)      # encoder output
    tgt_tok = np.ones((B, T)); tgt_tok[0, -1:] = 0.0
    src_tok = np.ones((B, S)); src_tok[1, -2:] = 0.0
    self_mask = make_causal_mask(T) + make_padding_mask(tgt_tok)  # (B,1,T,T)
    enc_mask = make_padding_mask(src_tok)                         # (B,1,1,S)

    out_ref, c = decoder_block_forward(x, enc, bp, H, causal_mask=self_mask, encoder_padding_mask=enc_mask)
    grad_out = rng.standard_normal((B, T, D)).astype(np.float64)
    gx_ref, genc_ref, grads_ref = decoder_block_backward(grad_out, c)

    selfm = np.broadcast_to(self_mask, (B, H, T, T)).reshape(B * H, T, T).astype(np.float32)
    encm = np.broadcast_to(enc_mask, (B, H, T, S)).reshape(B * H, T, S).astype(np.float32)
    pdt = _to_dt(bp)
    out, cache = dl.decoder_block_forward(
        picochem_cuda.DeviceTensor(x.astype(np.float32)),
        picochem_cuda.DeviceTensor(enc.astype(np.float32)), pdt, H,
        causal_mask_dt=picochem_cuda.DeviceTensor(selfm),
        enc_mask_dt=picochem_cuda.DeviceTensor(encm))
    gx, genc, grads = dl.decoder_block_backward(
        picochem_cuda.DeviceTensor(grad_out.astype(np.float32)), cache)

    _close(out.numpy(), out_ref, rtol=2e-3)
    _close(gx.numpy(), gx_ref, rtol=5e-3)
    _close(genc.numpy(), genc_ref, rtol=5e-3)
    for k, v in grads_ref.items():
        _close(grads[k].numpy(), v, rtol=1e-2)
