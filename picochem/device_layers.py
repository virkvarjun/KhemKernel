"""Device-resident transformer layers built on the picochem_cuda DeviceTensor ops.

This is the GPU re-implementation of the NumPy model in picochem/{ops,attention,
ffn,encoder,decoder,model}.py. Every op keeps its tensors on the GPU
(DeviceTensor) across the whole forward/backward, so a training step incurs no
per-op host<->device copies — only the batch goes in and the loss/grads come out.

The math mirrors the NumPy reference exactly (same residual/pre-norm structure,
same Linear x@W+b convention), so each piece is gradient-checked against it.

Requires the built extension: `bash scripts/build_cuda.sh` then ensure
picochem/kernels is importable (PYTHONPATH=picochem/kernels).
"""
import picochem_cuda as pc

DT = pc.DeviceTensor


# ── Linear (operates on 2-D (M, K)) ──────────────────────────────────────────

def linear_forward(x2, W, b):
    """y = x2 @ W + b. x2:(M,K) W:(K,N) b:(N,) — all DeviceTensors."""
    y = pc.dt_add_bias(pc.dt_matmul(x2, W), b)
    return y, (x2, W)


def linear_backward(grad_y, cache):
    x2, W = cache
    grad_x = pc.dt_matmul_dA(grad_y, W)   # grad_y @ W.T
    grad_W = pc.dt_matmul_dB(x2, grad_y)  # x2.T @ grad_y
    grad_b = pc.dt_colsum(grad_y)
    return grad_x, grad_W, grad_b


# ── Feed-forward network (Linear -> GeLU -> Linear) ──────────────────────────

def ffn_forward(x3, W1, b1, W2, b2):
    """x3:(B,S,D) -> (B,S,D). Pre-norm caller handles the residual."""
    B, S, D = x3.shape
    M = B * S
    x2 = pc.dt_reshape(x3, [M, D])
    h, l1 = linear_forward(x2, W1, b1)
    a = pc.dt_gelu_forward(h)
    out2, l2 = linear_forward(a, W2, b2)
    out = pc.dt_reshape(out2, [B, S, D])
    return out, (B, S, D, l1, h, a, l2)


def ffn_backward(grad_out3, cache):
    B, S, D, l1, h, a, l2 = cache
    M = B * S
    grad_out2 = pc.dt_reshape(grad_out3, [M, D])
    grad_a, grad_W2, grad_b2 = linear_backward(grad_out2, l2)
    grad_h = pc.dt_gelu_backward(grad_a, h)
    grad_x2, grad_W1, grad_b1 = linear_backward(grad_h, l1)
    grad_x = pc.dt_reshape(grad_x2, [B, S, D])
    return grad_x, grad_W1, grad_b1, grad_W2, grad_b2


# ── Multi-head self-attention ────────────────────────────────────────────────
# mask_dt, when given, is an additive mask already materialized to (B*H, T, S).

def mha_self_forward(x3, Wq, Wk, Wv, Wo, bq, bk, bv, bo, n_heads, mask_dt=None):
    import math
    B, S, D = x3.shape
    H = n_heads
    Dh = D // H
    M = B * S
    scale = 1.0 / math.sqrt(Dh)

    x2 = pc.dt_reshape(x3, [M, D])
    Q, qc = linear_forward(x2, Wq, bq)
    K, kc = linear_forward(x2, Wk, bk)
    V, vc = linear_forward(x2, Wv, bv)
    Qh = pc.dt_split_heads(pc.dt_reshape(Q, [B, S, D]), H)   # (B*H, S, Dh)
    Kh = pc.dt_split_heads(pc.dt_reshape(K, [B, S, D]), H)
    Vh = pc.dt_split_heads(pc.dt_reshape(V, [B, S, D]), H)

    scores = pc.dt_scale(pc.dt_bmm(Qh, Kh, transB=True), scale)  # (B*H, S, S)
    if mask_dt is not None:
        scores = pc.dt_add(scores, mask_dt)
    W = pc.dt_softmax(scores)
    ctx = pc.dt_bmm(W, Vh)                                       # (B*H, S, Dh)
    concat2 = pc.dt_reshape(pc.dt_merge_heads(ctx, H), [M, D])
    out2, oc = linear_forward(concat2, Wo, bo)
    out = pc.dt_reshape(out2, [B, S, D])
    cache = (B, S, D, H, Dh, scale, qc, kc, vc, oc, Qh, Kh, Vh, W, x2)
    return out, cache


def mha_self_backward(grad_out3, cache):
    B, S, D, H, Dh, scale, qc, kc, vc, oc, Qh, Kh, Vh, W, x2 = cache
    M = B * S
    grad_out2 = pc.dt_reshape(grad_out3, [M, D])
    grad_concat2, grad_Wo, grad_bo = linear_backward(grad_out2, oc)
    grad_ctx = pc.dt_split_heads(pc.dt_reshape(grad_concat2, [B, S, D]), H)  # (B*H,S,Dh)

    grad_W = pc.dt_bmm(grad_ctx, Vh, transB=True)   # (B*H,S,S)
    grad_Vh = pc.dt_bmm(W, grad_ctx, transA=True)   # (B*H,S,Dh)
    grad_scores = pc.dt_scale(pc.dt_softmax_backward(grad_W, W), scale)
    grad_Qh = pc.dt_bmm(grad_scores, Kh)            # (B*H,S,Dh)
    grad_Kh = pc.dt_bmm(grad_scores, Qh, transA=True)

    grad_Q2 = pc.dt_reshape(pc.dt_merge_heads(grad_Qh, H), [M, D])
    grad_K2 = pc.dt_reshape(pc.dt_merge_heads(grad_Kh, H), [M, D])
    grad_V2 = pc.dt_reshape(pc.dt_merge_heads(grad_Vh, H), [M, D])

    grad_xq, grad_Wq, grad_bq = linear_backward(grad_Q2, qc)
    grad_xk, grad_Wk, grad_bk = linear_backward(grad_K2, kc)
    grad_xv, grad_Wv, grad_bv = linear_backward(grad_V2, vc)
    grad_x2 = pc.dt_add(pc.dt_add(grad_xq, grad_xk), grad_xv)
    grad_x = pc.dt_reshape(grad_x2, [B, S, D])
    grads = {
        'attn_Wq': grad_Wq, 'attn_Wk': grad_Wk, 'attn_Wv': grad_Wv, 'attn_Wo': grad_Wo,
        'attn_bq': grad_bq, 'attn_bk': grad_bk, 'attn_bv': grad_bv, 'attn_bo': grad_bo,
    }
    return grad_x, grads


# ── LayerNorm helpers (thin wrappers; cache mirrors the kernel's outputs) ─────

def layer_norm_forward(x3, gamma, beta):
    B, S, D = x3.shape
    x2 = pc.dt_reshape(x3, [B * S, D])
    y2, x_hat, inv_std = pc.dt_layer_norm(x2, gamma, beta)
    y = pc.dt_reshape(y2, [B, S, D])
    return y, (B, S, D, x_hat, gamma, inv_std)


def layer_norm_backward(grad_y3, cache):
    B, S, D, x_hat, gamma, inv_std = cache
    grad_y2 = pc.dt_reshape(grad_y3, [B * S, D])
    grad_x2, grad_gamma, grad_beta = pc.dt_layer_norm_backward(grad_y2, x_hat, gamma, inv_std)
    grad_x = pc.dt_reshape(grad_x2, [B, S, D])
    return grad_x, grad_gamma, grad_beta


# ── Encoder block (pre-norm: x + sublayer(LN(x))) ────────────────────────────

def encoder_block_forward(x3, p, n_heads, mask_dt=None):
    xn1, ln1c = layer_norm_forward(x3, p['ln1_gamma'], p['ln1_beta'])
    attn, attnc = mha_self_forward(
        xn1, p['attn_Wq'], p['attn_Wk'], p['attn_Wv'], p['attn_Wo'],
        p['attn_bq'], p['attn_bk'], p['attn_bv'], p['attn_bo'], n_heads, mask_dt)
    x1 = pc.dt_add(x3, attn)
    xn2, ln2c = layer_norm_forward(x1, p['ln2_gamma'], p['ln2_beta'])
    ffn, ffnc = ffn_forward(xn2, p['ffn_W1'], p['ffn_b1'], p['ffn_W2'], p['ffn_b2'])
    out = pc.dt_add(x1, ffn)
    return out, (ln1c, attnc, ln2c, ffnc)


def encoder_block_backward(grad_out3, cache):
    ln1c, attnc, ln2c, ffnc = cache
    # FFN sublayer: out = x1 + ffn(LN2(x1))
    grad_x1 = grad_out3                      # residual path
    g_ffn, gW1, gb1, gW2, gb2 = ffn_backward(grad_out3, ffnc)
    g_xn2, g_ln2g, g_ln2b = layer_norm_backward(g_ffn, ln2c)
    grad_x1 = pc.dt_add(grad_x1, g_xn2)
    # Attention sublayer: x1 = x + attn(LN1(x))
    grad_x = grad_x1                          # residual path
    g_attn, attn_grads = mha_self_backward(grad_x1, attnc)
    g_xn1, g_ln1g, g_ln1b = layer_norm_backward(g_attn, ln1c)
    grad_x = pc.dt_add(grad_x, g_xn1)
    grads = {
        'ln1_gamma': g_ln1g, 'ln1_beta': g_ln1b,
        'ln2_gamma': g_ln2g, 'ln2_beta': g_ln2b,
        'ffn_W1': gW1, 'ffn_b1': gb1, 'ffn_W2': gW2, 'ffn_b2': gb2,
    }
    grads.update(attn_grads)
    return grad_x, grads


# ── Multi-head cross-attention (Q from decoder, K/V from encoder) ─────────────

def mha_cross_forward(x_dec3, x_enc3, Wq, Wk, Wv, Wo, bq, bk, bv, bo, n_heads, mask_dt=None):
    import math
    B, T, D = x_dec3.shape
    _, S, _ = x_enc3.shape
    H = n_heads
    Dh = D // H
    scale = 1.0 / math.sqrt(Dh)

    xdec2 = pc.dt_reshape(x_dec3, [B * T, D])
    xenc2 = pc.dt_reshape(x_enc3, [B * S, D])
    Q, qc = linear_forward(xdec2, Wq, bq)
    K, kc = linear_forward(xenc2, Wk, bk)
    V, vc = linear_forward(xenc2, Wv, bv)
    Qh = pc.dt_split_heads(pc.dt_reshape(Q, [B, T, D]), H)   # (B*H,T,Dh)
    Kh = pc.dt_split_heads(pc.dt_reshape(K, [B, S, D]), H)   # (B*H,S,Dh)
    Vh = pc.dt_split_heads(pc.dt_reshape(V, [B, S, D]), H)

    scores = pc.dt_scale(pc.dt_bmm(Qh, Kh, transB=True), scale)  # (B*H,T,S)
    if mask_dt is not None:
        scores = pc.dt_add(scores, mask_dt)
    W = pc.dt_softmax(scores)
    ctx = pc.dt_bmm(W, Vh)                                       # (B*H,T,Dh)
    concat2 = pc.dt_reshape(pc.dt_merge_heads(ctx, H), [B * T, D])
    out2, oc = linear_forward(concat2, Wo, bo)
    out = pc.dt_reshape(out2, [B, T, D])
    cache = (B, T, S, D, H, Dh, scale, qc, kc, vc, oc, Qh, Kh, Vh, W)
    return out, cache


def mha_cross_backward(grad_out3, cache):
    B, T, S, D, H, Dh, scale, qc, kc, vc, oc, Qh, Kh, Vh, W = cache
    grad_out2 = pc.dt_reshape(grad_out3, [B * T, D])
    grad_concat2, grad_Wo, grad_bo = linear_backward(grad_out2, oc)
    grad_ctx = pc.dt_split_heads(pc.dt_reshape(grad_concat2, [B, T, D]), H)  # (B*H,T,Dh)

    grad_W = pc.dt_bmm(grad_ctx, Vh, transB=True)   # (B*H,T,S)
    grad_Vh = pc.dt_bmm(W, grad_ctx, transA=True)   # (B*H,S,Dh)
    grad_scores = pc.dt_scale(pc.dt_softmax_backward(grad_W, W), scale)
    grad_Qh = pc.dt_bmm(grad_scores, Kh)            # (B*H,T,Dh)
    grad_Kh = pc.dt_bmm(grad_scores, Qh, transA=True)  # (B*H,S,Dh)

    grad_Q2 = pc.dt_reshape(pc.dt_merge_heads(grad_Qh, H), [B * T, D])
    grad_K2 = pc.dt_reshape(pc.dt_merge_heads(grad_Kh, H), [B * S, D])
    grad_V2 = pc.dt_reshape(pc.dt_merge_heads(grad_Vh, H), [B * S, D])

    grad_xdec2, grad_Wq, grad_bq = linear_backward(grad_Q2, qc)
    grad_xenc_k2, grad_Wk, grad_bk = linear_backward(grad_K2, kc)
    grad_xenc_v2, grad_Wv, grad_bv = linear_backward(grad_V2, vc)
    grad_xdec = pc.dt_reshape(grad_xdec2, [B, T, D])
    grad_xenc = pc.dt_reshape(pc.dt_add(grad_xenc_k2, grad_xenc_v2), [B, S, D])
    grads = {
        'attn_Wq': grad_Wq, 'attn_Wk': grad_Wk, 'attn_Wv': grad_Wv, 'attn_Wo': grad_Wo,
        'attn_bq': grad_bq, 'attn_bk': grad_bk, 'attn_bv': grad_bv, 'attn_bo': grad_bo,
    }
    return grad_xdec, grad_xenc, grads


# ── Decoder block (causal self-attn -> cross-attn -> FFN, all pre-norm) ───────

def _reprefix(grads, new):
    """'attn_Wq' -> '<new>_Wq', etc."""
    return {f"{new}_{k[len('attn_'):]}": v for k, v in grads.items()}


def decoder_block_forward(x3, enc_out3, p, n_heads, causal_mask_dt=None, enc_mask_dt=None):
    xn1, ln1c = layer_norm_forward(x3, p['ln1_gamma'], p['ln1_beta'])
    sattn, sc = mha_self_forward(
        xn1, p['self_Wq'], p['self_Wk'], p['self_Wv'], p['self_Wo'],
        p['self_bq'], p['self_bk'], p['self_bv'], p['self_bo'], n_heads, causal_mask_dt)
    x1 = pc.dt_add(x3, sattn)
    xn2, ln2c = layer_norm_forward(x1, p['ln2_gamma'], p['ln2_beta'])
    cattn, cc = mha_cross_forward(
        xn2, enc_out3, p['cross_Wq'], p['cross_Wk'], p['cross_Wv'], p['cross_Wo'],
        p['cross_bq'], p['cross_bk'], p['cross_bv'], p['cross_bo'], n_heads, enc_mask_dt)
    x2 = pc.dt_add(x1, cattn)
    xn3, ln3c = layer_norm_forward(x2, p['ln3_gamma'], p['ln3_beta'])
    ffn, ffnc = ffn_forward(xn3, p['ffn_W1'], p['ffn_b1'], p['ffn_W2'], p['ffn_b2'])
    out = pc.dt_add(x2, ffn)
    return out, (ln1c, sc, ln2c, cc, ln3c, ffnc)


def decoder_block_backward(grad_out3, cache):
    ln1c, sc, ln2c, cc, ln3c, ffnc = cache
    # FFN sublayer
    grad_x2 = grad_out3
    g_ffn, gW1, gb1, gW2, gb2 = ffn_backward(grad_out3, ffnc)
    g_xn3, g_ln3g, g_ln3b = layer_norm_backward(g_ffn, ln3c)
    grad_x2 = pc.dt_add(grad_x2, g_xn3)
    # Cross-attention sublayer (also produces the encoder-output gradient)
    grad_x1 = grad_x2
    g_cross_dec, grad_enc, cross_grads = mha_cross_backward(grad_x2, cc)
    g_xn2, g_ln2g, g_ln2b = layer_norm_backward(g_cross_dec, ln2c)
    grad_x1 = pc.dt_add(grad_x1, g_xn2)
    # Causal self-attention sublayer
    grad_x = grad_x1
    g_self, self_grads = mha_self_backward(grad_x1, sc)
    g_xn1, g_ln1g, g_ln1b = layer_norm_backward(g_self, ln1c)
    grad_x = pc.dt_add(grad_x, g_xn1)
    grads = {
        'ln1_gamma': g_ln1g, 'ln1_beta': g_ln1b,
        'ln2_gamma': g_ln2g, 'ln2_beta': g_ln2b,
        'ln3_gamma': g_ln3g, 'ln3_beta': g_ln3b,
        'ffn_W1': gW1, 'ffn_b1': gb1, 'ffn_W2': gW2, 'ffn_b2': gb2,
    }
    grads.update(_reprefix(self_grads, 'self'))
    grads.update(_reprefix(cross_grads, 'cross'))
    return grad_x, grad_enc, grads


# ── Full model (embedded inputs in, logits out) ──────────────────────────────
# Embeddings (token gather + positional add) and their table gradients are
# handled by the caller at the host boundary; the whole transformer stack —
# encoder, decoder, final LayerNorm, and the weight-tied output projection —
# runs on resident DeviceTensors. tgt_embed (V, D) is the tied output weight.

def model_forward(src_emb, tgt_emb, enc_params, dec_params,
                  final_ln_gamma, final_ln_beta, tgt_embed, n_heads,
                  enc_mask_dt, self_mask_dt, cross_mask_dt):
    enc_x = src_emb
    enc_caches = []
    for p in enc_params:
        enc_x, c = encoder_block_forward(enc_x, p, n_heads, mask_dt=enc_mask_dt)
        enc_caches.append(c)
    enc_out = enc_x

    dec_x = tgt_emb
    dec_caches = []
    for p in dec_params:
        dec_x, c = decoder_block_forward(dec_x, enc_out, p, n_heads,
                                         causal_mask_dt=self_mask_dt, enc_mask_dt=cross_mask_dt)
        dec_caches.append(c)

    dec_normed, lnc = layer_norm_forward(dec_x, final_ln_gamma, final_ln_beta)
    B, T, D = dec_normed.shape
    M = B * T
    V = tgt_embed.shape[0]
    dec_flat = pc.dt_reshape(dec_normed, [M, D])
    # logits = dec_flat @ tgt_embed.T  (batch-1 bmm with transposed B)
    logits = pc.dt_bmm(pc.dt_reshape(dec_flat, [1, M, D]),
                       pc.dt_reshape(tgt_embed, [1, V, D]), transB=True)  # (1,M,V)
    logits2 = pc.dt_reshape(logits, [M, V])
    cache = (enc_caches, dec_caches, lnc, dec_flat, tgt_embed, B, T, D, M, V)
    return logits2, cache


def model_backward(grad_logits2, cache):
    enc_caches, dec_caches, lnc, dec_flat, tgt_embed, B, T, D, M, V = cache
    grad_logits = pc.dt_reshape(grad_logits2, [1, M, V])
    # grad_dec_flat = grad_logits @ tgt_embed ; grad_tgt_embed(proj) = grad_logits.T @ dec_flat
    grad_dec_flat = pc.dt_reshape(
        pc.dt_bmm(grad_logits, pc.dt_reshape(tgt_embed, [1, V, D])), [M, D])
    grad_tgt_embed_proj = pc.dt_reshape(
        pc.dt_bmm(grad_logits, pc.dt_reshape(dec_flat, [1, M, D]), transA=True), [V, D])

    grad_dec_x = pc.dt_reshape(grad_dec_flat, [B, T, D])
    grad_dec_x, g_fg, g_fb = layer_norm_backward(grad_dec_x, lnc)

    dec_grads = [None] * len(dec_caches)
    grad_enc = None
    for i in reversed(range(len(dec_caches))):
        grad_dec_x, g_enc_c, dgrads = decoder_block_backward(grad_dec_x, dec_caches[i])
        grad_enc = g_enc_c if grad_enc is None else pc.dt_add(grad_enc, g_enc_c)
        dec_grads[i] = dgrads

    grad_enc_x = grad_enc
    enc_grads = [None] * len(enc_caches)
    for i in reversed(range(len(enc_caches))):
        grad_enc_x, egrads = encoder_block_backward(grad_enc_x, enc_caches[i])
        enc_grads[i] = egrads

    return {
        'encoder_blocks': enc_grads,
        'decoder_blocks': dec_grads,
        'final_ln_gamma': g_fg, 'final_ln_beta': g_fb,
        'grad_src_emb': grad_enc_x,           # grad w.r.t. encoder embedded input (B,S,D)
        'grad_tgt_emb': grad_dec_x,           # grad w.r.t. decoder embedded input (B,T,D)
        'grad_tgt_embed_proj': grad_tgt_embed_proj,  # (V,D) tied-weight grad from projection
    }
