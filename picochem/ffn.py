# Feedforward Neural Network

from picochem.ops import linear_forward, linear_backward, gelu_forward, gelu_backward


def ffn_forward(x, W1, b1, W2, b2):
    """Two-layer FFN with GELU activation."""
    # FNN gets called with 3D, not 2D
    B, S, D = x.shape
    x_flat = x.reshape (B*S, D)

    h, l1_cache = linear_forward(x_flat, W1, b1)
    a, gelu_cache = gelu_forward(h)
    out_flat, l2_cache = linear_forward(a, W2, b2)

    out = out_flat.reshape(B, S, D)
    cache = (B, S, D, l1_cache, gelu_cache, l2_cache)
    return out, cache

def ffn_backward(grad_out, cache):
    B, S, D, l1_cache, gelu_cache, l2_cache = cache
    grad_out_flat = grad_out.reshape(B*S, D)
    grad_a, grad_W2, grad_b2 = linear_backward(grad_out_flat, l2_cache)
    grad_h, = gelu_backward(grad_a, gelu_cache)
    grad_x_flat, grad_W1, grad_b1 = linear_backward(grad_h, l1_cache)

    grad_x = grad_x_flat.reshape(B, S, D)
    return grad_x, grad_W1, grad_b1, grad_W2, grad_b2


