"""Tests for the pybind11 CUDA bindings (picochem_cuda module).

Skipped automatically when picochem_cuda is not built.
Build with: bash scripts/build_cuda.sh
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

try:
    import picochem_cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="picochem_cuda not built")

rng = np.random.default_rng(42)


def test_vector_add():
    N = 1024
    a = rng.standard_normal(N).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    expected = a + b
    out = picochem_cuda.vector_add(a, b)
    assert out.shape == (N,)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_matmul_naive():
    M, K, N = 32, 64, 48
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    expected = A @ B
    out = picochem_cuda.matmul_naive(A, B)
    assert out.shape == (M, N)
    np.testing.assert_allclose(out, expected, atol=1e-3)


def test_matmul_tiled():
    M, K, N = 64, 64, 64
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    expected = A @ B
    out = picochem_cuda.matmul_tiled(A, B)
    assert out.shape == (M, N)
    np.testing.assert_allclose(out, expected, atol=1e-3)


def test_device_tensor_roundtrip():
    """Upload to GPU and back must be a no-op (float32)."""
    x = rng.standard_normal((7, 13)).astype(np.float32)
    dt = picochem_cuda.DeviceTensor(x)
    assert list(dt.shape) == [7, 13]
    np.testing.assert_array_equal(dt.numpy(), x)


def test_dt_matmul_parity():
    """Device-resident matmul matches numpy; inputs stay on the GPU."""
    M, K, N = 64, 48, 80
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    dA = picochem_cuda.DeviceTensor(A)
    dB = picochem_cuda.DeviceTensor(B)
    dC = picochem_cuda.dt_matmul(dA, dB)
    assert list(dC.shape) == [M, N]
    np.testing.assert_allclose(dC.numpy(), A @ B, atol=1e-3)
    # Result is itself a DeviceTensor and can feed another op without leaving the GPU.
    dD = picochem_cuda.dt_matmul(dC, picochem_cuda.DeviceTensor(rng.standard_normal((N, 16)).astype(np.float32)))
    assert list(dD.shape) == [M, 16]


def test_dt_matmul_backward_parity():
    """Device-resident Linear backward matches numpy."""
    M, K, N = 40, 24, 56
    W = rng.standard_normal((K, N)).astype(np.float32)
    x = rng.standard_normal((M, K)).astype(np.float32)
    grad_y = rng.standard_normal((M, N)).astype(np.float32)
    DT = picochem_cuda.DeviceTensor
    gx = picochem_cuda.dt_matmul_dA(DT(grad_y), DT(W)).numpy()   # grad_y @ W.T
    gW = picochem_cuda.dt_matmul_dB(DT(x), DT(grad_y)).numpy()   # x.T @ grad_y
    np.testing.assert_allclose(gx, grad_y @ W.T, atol=1e-3)
    np.testing.assert_allclose(gW, x.T @ grad_y, atol=1e-3)


def test_dt_add_parity():
    a = rng.standard_normal((16, 32)).astype(np.float32)
    b = rng.standard_normal((16, 32)).astype(np.float32)
    DT = picochem_cuda.DeviceTensor
    np.testing.assert_allclose(picochem_cuda.dt_add(DT(a), DT(b)).numpy(), a + b, atol=1e-5)


def test_dt_gelu_parity():
    from picochem.ops import gelu_forward, gelu_backward
    x = (rng.standard_normal((16, 64)).astype(np.float32) * 3.0)
    grad_y = rng.standard_normal((16, 64)).astype(np.float32)
    DT = picochem_cuda.DeviceTensor
    fwd_ref, cache = gelu_forward(x.astype(np.float64))
    (bwd_ref,) = gelu_backward(grad_y.astype(np.float64), cache)
    np.testing.assert_allclose(picochem_cuda.dt_gelu_forward(DT(x)).numpy(), fwd_ref, atol=1e-3)
    np.testing.assert_allclose(picochem_cuda.dt_gelu_backward(DT(grad_y), DT(x)).numpy(), bwd_ref, atol=1e-3)


def test_matmul_dA():
    """dA = dC @ Bᵀ for C = A @ B."""
    M, K, N = 40, 24, 56
    B = rng.standard_normal((K, N)).astype(np.float32)
    dC = rng.standard_normal((M, N)).astype(np.float32)
    expected = dC @ B.T
    out = picochem_cuda.matmul_dA(dC, B)
    assert out.shape == (M, K)
    np.testing.assert_allclose(out, expected, atol=1e-3)


def test_matmul_dB():
    """dB = Aᵀ @ dC for C = A @ B."""
    M, K, N = 40, 24, 56
    A = rng.standard_normal((M, K)).astype(np.float32)
    dC = rng.standard_normal((M, N)).astype(np.float32)
    expected = A.T @ dC
    out = picochem_cuda.matmul_dB(A, dC)
    assert out.shape == (K, N)
    np.testing.assert_allclose(out, expected, atol=1e-3)


def test_matmul_backward_matches_linear_backward():
    """End-to-end: the two kernels reproduce ops.linear_backward's grad_x/grad_W."""
    from picochem.ops import linear_forward, linear_backward
    M, K, N = 32, 48, 64
    x = rng.standard_normal((M, K)).astype(np.float32)
    W = rng.standard_normal((K, N)).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    y, cache = linear_forward(x, W, b)
    grad_y = rng.standard_normal((M, N)).astype(np.float32)
    grad_x_ref, grad_W_ref, _ = linear_backward(grad_y, cache)
    # grad_x = grad_y @ Wᵀ  (dA form);  grad_W = xᵀ @ grad_y  (dB form)
    grad_x = picochem_cuda.matmul_dA(grad_y, W)
    grad_W = picochem_cuda.matmul_dB(x, grad_y)
    np.testing.assert_allclose(grad_x, grad_x_ref, atol=1e-2)
    np.testing.assert_allclose(grad_W, grad_W_ref, atol=1e-2)


def test_softmax_2d():
    M, N = 16, 100
    x = rng.standard_normal((M, N)).astype(np.float32)
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    expected = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
    out = picochem_cuda.softmax(x)
    assert out.shape == (M, N)
    np.testing.assert_allclose(out, expected, atol=1e-5)
    # Each row should sum to 1.
    np.testing.assert_allclose(out.sum(axis=-1), np.ones(M), atol=1e-5)


def test_softmax_4d():
    """Binding must handle arbitrary leading dims by flattening."""
    B, H, T, S = 2, 4, 8, 16
    x = rng.standard_normal((B, H, T, S)).astype(np.float32)
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    expected = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
    out = picochem_cuda.softmax(x)
    assert out.shape == (B, H, T, S)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_layer_norm():
    M, N = 16, 64
    x = rng.standard_normal((M, N)).astype(np.float32)
    gamma = rng.standard_normal(N).astype(np.float32)
    beta = rng.standard_normal(N).astype(np.float32)

    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + 1e-5)
    x_hat = (x - mean) * inv_std
    expected = (gamma * x_hat + beta).astype(np.float32)

    out = picochem_cuda.layer_norm(x, gamma, beta)
    assert out.shape == (M, N)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_layer_norm_3d():
    """Binding must handle arbitrary leading dims."""
    B, S, D = 2, 10, 32
    x = rng.standard_normal((B, S, D)).astype(np.float32)
    gamma = np.ones(D, dtype=np.float32)
    beta = np.zeros(D, dtype=np.float32)

    x_flat = x.reshape(-1, D)
    mean = x_flat.mean(axis=-1, keepdims=True)
    var = x_flat.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + 1e-5)
    expected = (gamma * (x_flat - mean) * inv_std + beta).reshape(B, S, D)

    out = picochem_cuda.layer_norm(x, gamma, beta)
    assert out.shape == (B, S, D)
    np.testing.assert_allclose(out, expected, atol=1e-5)


# ── backward-pass kernels (parity vs picochem.ops) ──────────────────────────

def test_gelu_forward_parity():
    from picochem.ops import gelu_forward
    x = (rng.standard_normal((16, 64)).astype(np.float32) * 3.0)
    expected, _ = gelu_forward(x.astype(np.float64))
    out = picochem_cuda.gelu_forward(x)
    assert out.shape == x.shape
    np.testing.assert_allclose(out, expected, atol=1e-3)


def test_gelu_backward_parity():
    from picochem.ops import gelu_forward, gelu_backward
    x = (rng.standard_normal((16, 64)).astype(np.float32) * 3.0)
    grad_y = rng.standard_normal((16, 64)).astype(np.float32)
    _, cache = gelu_forward(x.astype(np.float64))
    (expected,) = gelu_backward(grad_y.astype(np.float64), cache)
    out = picochem_cuda.gelu_backward(grad_y, x)
    assert out.shape == x.shape
    np.testing.assert_allclose(out, expected, atol=1e-3)


def test_softmax_backward_parity():
    from picochem.ops import softmax_backward_pure
    B, H, T, S = 2, 4, 8, 16
    logits = rng.standard_normal((B, H, T, S)).astype(np.float32)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
    grad_out = rng.standard_normal((B, H, T, S)).astype(np.float32)
    (expected,) = softmax_backward_pure(grad_out.astype(np.float64), (probs.astype(np.float64),))
    out = picochem_cuda.softmax_backward(grad_out, probs)
    assert out.shape == probs.shape
    np.testing.assert_allclose(out, expected, atol=1e-4)


def test_layer_norm_backward_parity():
    from picochem.ops import layer_norm_forward, layer_norm_backward
    M, N = 24, 64
    x = rng.standard_normal((M, N)).astype(np.float64)
    gamma = (rng.standard_normal(N) + 1.0).astype(np.float64)
    beta = rng.standard_normal(N).astype(np.float64)
    grad_y = rng.standard_normal((M, N)).astype(np.float64)

    _, cache = layer_norm_forward(x, gamma, beta)
    x_hat, _gamma, inv_std = cache  # inv_std is (M, 1)
    gx_ref, gg_ref, gb_ref = layer_norm_backward(grad_y, cache)

    gx, gg, gb = picochem_cuda.layer_norm_backward(
        grad_y.astype(np.float32),
        x_hat.astype(np.float32),
        gamma.astype(np.float32),
        inv_std.reshape(-1).astype(np.float32),
    )
    np.testing.assert_allclose(gx, gx_ref, atol=1e-3)
    np.testing.assert_allclose(gg, gg_ref, atol=1e-2)
    np.testing.assert_allclose(gb, gb_ref, atol=1e-2)


def test_cross_entropy_parity():
    from picochem.ops import (
        softmax_cross_entropy_forward, softmax_cross_entropy_backward,
    )
    M, V, ignore = 32, 200, -1
    logits = rng.standard_normal((M, V)).astype(np.float32)
    targets = rng.integers(0, V, size=M).astype(np.int32)
    targets[::5] = ignore  # mask some rows

    loss_ref, cache = softmax_cross_entropy_forward(
        logits.astype(np.float64), targets, ignore_index=ignore
    )
    grad_ref, _ = softmax_cross_entropy_backward(1.0, cache)

    loss, n_valid = picochem_cuda.cross_entropy_forward(logits, targets, ignore)
    grad = picochem_cuda.cross_entropy_backward(logits, targets, ignore, n_valid, 1.0)

    assert abs(loss - loss_ref) < 1e-3
    assert n_valid == (targets != ignore).sum()
    np.testing.assert_allclose(grad, grad_ref, atol=1e-4)


def test_embedding_backward_parity():
    from picochem.embeddings import token_embedding_forward, token_embedding_backward
    V, D = 20, 32
    table = rng.standard_normal((V, D)).astype(np.float64)
    token_ids = rng.integers(0, V, size=(4, 8)).astype(np.int32)  # (B, S), with repeats
    _, cache = token_embedding_forward(token_ids, table)
    grad_out = rng.standard_normal((4, 8, D)).astype(np.float32)
    (grad_table_ref,) = token_embedding_backward(grad_out.astype(np.float64), cache)

    grad_table = picochem_cuda.embedding_backward(
        grad_out.reshape(-1, D), token_ids.reshape(-1), V
    )
    assert grad_table.shape == (V, D)
    np.testing.assert_allclose(grad_table, grad_table_ref, atol=1e-3)


def test_adam_update_parity():
    from picochem.optimizer import adam_step
    n = 512
    param = rng.standard_normal(n).astype(np.float64)
    grad = rng.standard_normal(n).astype(np.float64)
    m0 = rng.standard_normal(n).astype(np.float64) * 0.1
    v0 = (rng.standard_normal(n).astype(np.float64) * 0.1) ** 2
    step, lr, b1, b2, eps = 7, 1e-3, 0.9, 0.999, 1e-8

    # adam_step rebinds state["m"]/["v"] to new arrays, so read the updated
    # values back out of the dict rather than from the originals.
    p_ref = param.copy()
    state_ref = {"m": m0.copy(), "v": v0.copy()}
    adam_step(p_ref, grad, state_ref, step, lr=lr, beta1=b1, beta2=b2, eps=eps)

    p_out, m_out, v_out = picochem_cuda.adam_update(
        param.astype(np.float32), grad.astype(np.float32),
        m0.astype(np.float32), v0.astype(np.float32),
        step, lr, b1, b2, eps,
    )
    np.testing.assert_allclose(p_out, p_ref, atol=1e-5)
    np.testing.assert_allclose(m_out, state_ref["m"], atol=1e-5)
    np.testing.assert_allclose(v_out, state_ref["v"], atol=1e-6)
