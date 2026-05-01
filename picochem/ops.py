"""Forward and backward primitives: linear, layernorm, embeddings, softmax, etc."""
import numpy as np 

# Linear (forward) 
def linear_forward(x, W, b): 
    # x: (batch_size, input_dim)
    # W: (input_dim, output_dim)
    # b: (output_dim,)
    y = x @ W + b 
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
    inner = GELU_CONST * (x*0.044715 * x ** 3) 
    tanh_inner = np.tanh(inner) 
    y = 0.5 * x * (1 + tanh_inner) 
    cache = (x, tanh_inner) 
    return y, cache 
def gelu_backward(grad_y, cache): 
    x, tanh_inner = cache 
    # Derivative of the GeLU with respect to x 
    sech_sq = 1.0 - tanh_inner ** 2
    d_inner = GELU_CONST * (1.0+3.0 * 0.04475 * x ** 2) 
    d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_sq * d_inner 
    grad_x = grad_y * d_gelu 
    return (grad_x,) 
