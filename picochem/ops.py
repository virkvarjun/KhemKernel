"""Forward and backward primitives: linear, layernorm, embeddings, softmax, etc."""
import numpy as np
import picochem.backend as backend

# Linear (forward)
def linear_forward(x, W, b):
    # x: (batch_size, input_dim)
    # W: (input_dim, output_dim)
    # b: (output_dim,)
    y = backend.matmul(x, W) + b
    cache = (x, W)
    return y, cache

def linear_backward(grad_y, cache):
    x, W = cache
    grad_x = grad_y @ W.T
    grad_W = x.T @ grad_y
    grad_b = grad_y.sum(axis=0)
    return grad_x, grad_W, grad_b


# Here, let's work on the GeLU
GELU_CONST = np.sqrt(2.0 / np.pi)
def gelu_forward(x):
    inner = GELU_CONST * (x + 0.044715 * x ** 3)
    tanh_inner = np.tanh(inner)
    y = 0.5 * x * (1 + tanh_inner)
    cache = (x, tanh_inner)
    return y, cache

def gelu_backward(grad_y, cache):
    x, tanh_inner = cache
    # Derivative of the GeLU with respect to x
    sech_sq = 1.0 - tanh_inner ** 2
    d_inner = GELU_CONST * (1.0 + 3.0 * 0.044715 * x ** 2)
    d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_sq * d_inner
    grad_x = grad_y * d_gelu
    return (grad_x,)


# Softmax + Cross-Entropy (used by the training loss, not by attention)
def softmax_cross_entropy_forward(logits, targets, ignore_index=-1):
    # Stable Softmax with cross-entropy loss
    logits_max = logits.max(axis=-1, keepdims=True)
    log_sum_exp = np.log(np.exp(logits - logits_max).sum(axis=-1, keepdims=True)) + logits_max
    log_probs = logits - log_sum_exp

    # Mask out ignored targets
    mask = (targets != ignore_index).astype(np.float64)
    n_valid = mask.sum()
    if n_valid == 0:
        n_valid = 1.0

    # NLL of the correct class for each example
    batch_idx = np.arange(len(targets))
    safe_targets = np.where(targets == ignore_index, 0, targets)
    nll = -log_probs[batch_idx, safe_targets]

    loss = (nll * mask).sum() / n_valid

    cache = (log_probs, targets, mask, n_valid, ignore_index)
    return loss, cache

def softmax_cross_entropy_backward(grad_loss, cache):
    log_probs, targets, mask, n_valid, ignore_index = cache
    probs = np.exp(log_probs)
    grad_logits = probs.copy()
    batch_idx = np.arange(len(targets))
    safe_targets = np.where(targets == ignore_index, 0, targets)
    grad_logits[batch_idx, safe_targets] -= 1.0

    # apply mask and normalize
    grad_logits = grad_logits * mask[:, None] / n_valid
    grad_logits = grad_logits * grad_loss  # chain-rule: multiply by upstream gradient

    return (grad_logits, None)  # no gradient wrt integer targets


# Pure softmax backward (used by attention — forward dispatches via backend.softmax)
def softmax_backward_pure(grad_out, cache):
    """Jacobian-vector product for a pure softmax (no cross-entropy).

    cache = (probs,) as returned by backend.softmax.
    """
    (probs,) = cache
    dot = (grad_out * probs).sum(axis=-1, keepdims=True)
    return (probs * (grad_out - dot),)


# Layer Normalization
def layer_norm_forward(x, gamma, beta, eps=1e-5):
    # Compute cache values in NumPy — needed for the backward pass.
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * inv_std
    # Dispatch the output computation to the active backend.
    y = backend.layer_norm(x, gamma, beta, eps)
    cache = (x_hat, gamma, inv_std)
    return y, cache

def layer_norm_backward(grad_y, cache):
    x_hat, gamma, inv_std = cache
    N = x_hat.shape[-1]

    # Gradient w.r.t. learned parameters (sum over all batch dims)
    reduce_axes = tuple(range(grad_y.ndim - 1))
    grad_gamma = (grad_y * x_hat).sum(axis=reduce_axes)
    grad_beta = grad_y.sum(axis=reduce_axes)

    # Gradient w.r.t. input
    dxhat = grad_y * gamma
    grad_x = (1.0 / N) * inv_std * (
        N * dxhat
        - dxhat.sum(axis=-1, keepdims=True)
        - x_hat * (dxhat * x_hat).sum(axis=-1, keepdims=True)
    )

    return grad_x, grad_gamma, grad_beta
