"""Parity tests: numpy backend vs CUDA backend.

For each modified op, run a small forward pass with numpy, then with CUDA,
and compare outputs within the expected tolerance (1e-3 for all, because of
float32/float64 conversion on the CUDA path).

CUDA tests are skipped automatically when picochem_cuda is not built.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

import picochem.backend as backend
from picochem.ops import linear_forward, layer_norm_forward

try:
    import picochem_cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

rng = np.random.default_rng(0)
ATOL = 1e-3  # loose because float32↔float64 conversion on the CUDA path


# ── helpers ───────────────────────────────────────────────────────────────────

def with_numpy(fn):
    backend.set_backend('numpy')
    try:
        return fn()
    finally:
        backend.set_backend('numpy')


def with_cuda(fn):
    if not HAS_CUDA:
        pytest.skip("picochem_cuda not built")
    backend.set_backend('cuda')
    try:
        return fn()
    finally:
        backend.set_backend('numpy')


# ── matmul ────────────────────────────────────────────────────────────────────

def test_matmul_numpy_matches_reference():
    A = rng.standard_normal((32, 64))
    B = rng.standard_normal((64, 48))
    np_out = with_numpy(lambda: backend.matmul(A, B))
    np.testing.assert_allclose(np_out, A @ B, atol=1e-10)


def test_matmul_cuda_parity():
    A = rng.standard_normal((32, 64))
    B = rng.standard_normal((64, 48))
    np_out  = with_numpy(lambda: backend.matmul(A, B))
    cu_out  = with_cuda(lambda: backend.matmul(A, B))
    np.testing.assert_allclose(cu_out, np_out, atol=ATOL)


# ── softmax ───────────────────────────────────────────────────────────────────

def test_softmax_numpy_sums_to_one():
    x = rng.standard_normal((4, 2, 8, 16))
    probs, _ = with_numpy(lambda: backend.softmax(x))
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones((4, 2, 8)), atol=1e-10)


def test_softmax_cuda_parity():
    x = rng.standard_normal((4, 2, 8, 16))
    np_probs, _ = with_numpy(lambda: backend.softmax(x))
    cu_probs, _ = with_cuda(lambda: backend.softmax(x))
    np.testing.assert_allclose(cu_probs, np_probs, atol=ATOL)


# ── layer_norm ────────────────────────────────────────────────────────────────

def test_layer_norm_numpy_matches_reference():
    x     = rng.standard_normal((2, 10, 16))
    gamma = np.ones(16)
    beta  = np.zeros(16)
    y, _  = with_numpy(lambda: layer_norm_forward(x, gamma, beta))
    # With gamma=1, beta=0: y should have zero mean and unit variance per row.
    x_flat = x.reshape(-1, 16)
    y_flat = y.reshape(-1, 16)
    np.testing.assert_allclose(y_flat.mean(axis=-1), np.zeros(x_flat.shape[0]), atol=1e-10)
    # var is var/(var+eps) due to the eps offset — 1e-4 tolerance is appropriate.
    np.testing.assert_allclose(y_flat.var(axis=-1),  np.ones(x_flat.shape[0]),  atol=1e-4)


def test_layer_norm_cuda_parity():
    x     = rng.standard_normal((2, 10, 16))
    gamma = rng.standard_normal(16) + 1.0
    beta  = rng.standard_normal(16) * 0.1
    np_y, _ = with_numpy(lambda: layer_norm_forward(x, gamma, beta))
    cu_y, _ = with_cuda(lambda: layer_norm_forward(x, gamma, beta))
    np.testing.assert_allclose(cu_y, np_y, atol=ATOL)


# ── linear_forward (uses backend.matmul) ──────────────────────────────────────

def test_linear_forward_cuda_parity():
    x = rng.standard_normal((16, 32))
    W = rng.standard_normal((32, 64))
    b = rng.standard_normal(64)
    np_y, _ = with_numpy(lambda: linear_forward(x, W, b))
    cu_y, _ = with_cuda(lambda: linear_forward(x, W, b))
    np.testing.assert_allclose(cu_y, np_y, atol=ATOL)
