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


def test_full_model_parity():
    """End-to-end: device model + dt_cross_entropy vs the numpy model, incl.
    embedding-table gradients reconstructed via host scatter (as model.py does)."""
    from picochem.model import (init_params, model_forward, model_backward,
                                compute_loss, loss_backward, make_padding_mask, make_causal_mask)
    import picochem.device_layers as dl

    B, S, T, D, H = 2, 6, 5, 32, 4
    cfg = dict(src_vocab=12, tgt_vocab=15, d_model=D, n_heads=H, d_ff=64,
               n_enc_layers=2, n_dec_layers=2, max_src_len=S, max_tgt_len=T)
    p = init_params(cfg, np.random.default_rng(3))
    src_ids = rng.integers(3, cfg['src_vocab'], size=(B, S)).astype(np.int32)
    tgt_ids = rng.integers(3, cfg['tgt_vocab'], size=(B, T)).astype(np.int32)
    tgt_out = rng.integers(3, cfg['tgt_vocab'], size=(B, T)).astype(np.int32)
    tgt_out[0, -1] = -1  # exercise ignore_index
    src_mask = np.ones((B, S)); tgt_mask = np.ones((B, T))

    # ---- numpy reference ----
    logits_ref, fwd_cache = model_forward(src_ids, tgt_ids, src_mask, tgt_mask, p, cfg)
    loss_ref, lcache = compute_loss(logits_ref, tgt_out, ignore_index=-1)
    grads_ref = model_backward(loss_backward(1.0, lcache), fwd_cache, p, cfg)

    # ---- device ----
    DTt = picochem_cuda.DeviceTensor
    src_emb = (p['src_token_embed'][src_ids] + p['src_pos_embed'][:S]).astype(np.float32)
    tgt_emb = (p['tgt_token_embed'][tgt_ids] + p['tgt_pos_embed'][:T]).astype(np.float32)
    self_mask = make_causal_mask(T) + make_padding_mask(tgt_mask)
    selfm = np.broadcast_to(self_mask, (B, H, T, T)).reshape(B * H, T, T).astype(np.float32)
    enc_p = [_to_dt(bp) for bp in p['encoder_blocks']]
    dec_p = [_to_dt(bp) for bp in p['decoder_blocks']]
    tgt_embed_dt = DTt(p['tgt_token_embed'].astype(np.float32))

    logits, cache = dl.model_forward(
        DTt(src_emb), DTt(tgt_emb), enc_p, dec_p,
        DTt(p['final_ln_gamma'].astype(np.float32)), DTt(p['final_ln_beta'].astype(np.float32)),
        tgt_embed_dt, H, enc_mask_dt=None, self_mask_dt=DTt(selfm), cross_mask_dt=None)
    loss, n_valid = picochem_cuda.dt_cross_entropy_forward(logits, tgt_out.reshape(-1), -1)
    grad_logits = picochem_cuda.dt_cross_entropy_backward(logits, tgt_out.reshape(-1), -1, n_valid, 1.0)
    g = dl.model_backward(grad_logits, cache)

    # forward correctness
    _close(logits.numpy().reshape(B, T, cfg['tgt_vocab']), logits_ref, rtol=2e-2)
    assert abs(loss - float(loss_ref)) < 1e-3

    # stack grads
    for i in range(cfg['n_enc_layers']):
        for k, v in grads_ref['encoder_blocks'][i].items():
            _close(g['encoder_blocks'][i][k].numpy(), v, rtol=3e-2)
    for i in range(cfg['n_dec_layers']):
        for k, v in grads_ref['decoder_blocks'][i].items():
            _close(g['decoder_blocks'][i][k].numpy(), v, rtol=3e-2)
    _close(g['final_ln_gamma'].numpy(), grads_ref['final_ln_gamma'], rtol=2e-2)
    _close(g['final_ln_beta'].numpy(), grads_ref['final_ln_beta'], rtol=2e-2)

    # embedding-table grads, reconstructed via host scatter (as model.py does)
    gse = g['grad_src_emb'].numpy(); gte = g['grad_tgt_emb'].numpy()
    g_src_tok = np.zeros_like(p['src_token_embed']); np.add.at(g_src_tok, src_ids, gse)
    g_tgt_tok = np.zeros_like(p['tgt_token_embed']); np.add.at(g_tgt_tok, tgt_ids, gte)
    g_tgt_tok += g['grad_tgt_embed_proj'].numpy()
    g_src_pos = np.zeros_like(p['src_pos_embed']); g_src_pos[:S] = gse.sum(0)
    g_tgt_pos = np.zeros_like(p['tgt_pos_embed']); g_tgt_pos[:T] = gte.sum(0)
    _close(g_src_tok, grads_ref['src_token_embed'], rtol=3e-2)
    _close(g_tgt_tok, grads_ref['tgt_token_embed'], rtol=3e-2)
    _close(g_src_pos, grads_ref['src_pos_embed'], rtol=3e-2)
    _close(g_tgt_pos, grads_ref['tgt_pos_embed'], rtol=3e-2)
