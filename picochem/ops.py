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

