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
