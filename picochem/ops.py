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

# Softmax and Cross-Entropy 
def softmax_forward(logits, targets, ignore_index=1): 
    # Stable Softmax with cross-entropy loss
    logits_max = logits.max(axis=-1, keepdim=True) 
    log_sum_exp = np.log(np.exp(logits - logits_max).sum(axis=-1, keepdims=True)) + logits_max
    log_probs = logits - log_sum_exp 
     
    # Mask out ignored targets 
    mask = (targets != ignore_index).astype(np.float64) 
    n_valid = mask.sum() 
    if n_valid == 0: 
        n_valid = 1.0 # Avoid division by zero
    
    # NLL of the correct class for each example
    batch_idx = np.arange(len(targets))
    safe_targets = np.where(targets == ignore_index, 0, targets)
    nll = -log_probs[batch_idx, safe_targets]
    
    loss = (nll * mask).sum() / n_valid
    
    cache = (log_probs, targets, mask, n_valid, ignore_index)
    return loss, cache
