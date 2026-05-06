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
