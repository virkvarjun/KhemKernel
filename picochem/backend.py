"""Backend dispatcher for picochem forward-pass operations.

Two backends are available:
  'numpy' (default) — pure NumPy, float64 throughout.
  'cuda'            — CUDA kernels via picochem_cuda (float32 internally).

Only forward-pass operations are dispatched here.  Backward passes always
run in NumPy because CUDA backward kernels are not yet implemented.  The
float32/float64 conversion on the CUDA path is intentional; accuracy
differences of up to 1e-3 relative to the NumPy path are expected.

Usage:
    import picochem.backend as backend
    backend.set_backend('cuda')   # once at startup
    y = backend.matmul(A, B)
"""

import numpy as np

_backend: str = 'numpy'


def set_backend(name: str) -> None:
    global _backend
    if name not in ('numpy', 'cuda'):
        raise ValueError(f"backend must be 'numpy' or 'cuda', got {name!r}")
    _backend = name


def get_backend() -> str:
    return _backend


def _cuda_module():
    try:
        import picochem_cuda
        return picochem_cuda
    except ImportError:
        raise RuntimeError(
            "CUDA backend selected but picochem_cuda module not found. "
            "Run: bash scripts/build_cuda.sh"
        )


# ── matmul ────────────────────────────────────────────────────────────────────

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """2-D matrix multiply.  Caller ensures inputs are 2-D."""
    if _backend == 'numpy':
        return np.matmul(A, B)
    cuda = _cuda_module()
    dtype = A.dtype
    out = cuda.matmul_tiled(
        np.ascontiguousarray(A, dtype=np.float32),
        np.ascontiguousarray(B, dtype=np.float32),
    )
    return out.astype(dtype)


# ── matmul backward ─────────────────────────────────────────────────────────

def matmul_dA(grad_y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Backward w.r.t. the left operand: grad_x = grad_y @ Wᵀ. Inputs 2-D."""
    if _backend == 'numpy':
        return grad_y @ W.T
    cuda = _cuda_module()
    dtype = grad_y.dtype
    out = cuda.matmul_dA(
        np.ascontiguousarray(grad_y, dtype=np.float32),
        np.ascontiguousarray(W, dtype=np.float32),
    )
    return out.astype(dtype)


def matmul_dB(x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
    """Backward w.r.t. the right operand: grad_W = xᵀ @ grad_y. Inputs 2-D."""
    if _backend == 'numpy':
        return x.T @ grad_y
    cuda = _cuda_module()
    dtype = grad_y.dtype
    out = cuda.matmul_dB(
        np.ascontiguousarray(x, dtype=np.float32),
        np.ascontiguousarray(grad_y, dtype=np.float32),
    )
    return out.astype(dtype)


# ── softmax ───────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray):
    """Numerically stable softmax along the last axis.

    Returns (probs, cache) where cache = (probs,).
    The cache is consumed by picochem.ops.softmax_backward_pure.
    """
    if _backend == 'numpy':
        shifted = x - x.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        probs = e / e.sum(axis=-1, keepdims=True)
        return probs, (probs,)
    cuda = _cuda_module()
    dtype = x.dtype
    orig_shape = x.shape
    x_f32 = np.ascontiguousarray(x.reshape(-1, x.shape[-1]), dtype=np.float32)
    probs = cuda.softmax(x_f32).astype(dtype).reshape(orig_shape)
    return probs, (probs,)


# ── layer_norm ────────────────────────────────────────────────────────────────

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    """Layer norm along the last axis.

    Returns y only.  The caller (layer_norm_forward in ops.py) separately
    computes x_hat and inv_std in NumPy so that the backward pass can run
    without a CUDA kernel.
    """
    if _backend == 'numpy':
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + eps)
        x_hat = (x - mean) * inv_std
        return gamma * x_hat + beta
    cuda = _cuda_module()
    dtype = x.dtype
    orig_shape = x.shape
    x_f32 = np.ascontiguousarray(x.reshape(-1, x.shape[-1]), dtype=np.float32)
    g_f32 = np.ascontiguousarray(gamma.reshape(-1), dtype=np.float32)
    b_f32 = np.ascontiguousarray(beta.reshape(-1), dtype=np.float32)
    y = cuda.layer_norm(x_f32, g_f32, b_f32).astype(dtype).reshape(orig_shape)
    return y
