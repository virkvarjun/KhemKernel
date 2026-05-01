import numpy as np 
import pytest 

def numerical_gradient(f, x, eps=1e-5): 
    # We compute the numerical gradient of scalar function f(x) using central differences principle
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished: 
        idx = it.multi_index 
        original = x[idx] 
        x[idx] = original + eps 
        f_plus = f(x) 
        x[idx] = original - eps 
        f_minus = f(x) 
        x[idx] = original 
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad

def check_gradient(forward_fn, backward_fn, inputs, output_grad=None, tol=1e-5): 
    # Check if the analytical gradient from backward_fn matches the numerical gradient from forward_fn
    # Run forward to get output and cache
    output, cache = forward_fn(*inputs)
    
    if output_grad is None:
        output_grad = np.ones_like(output)
    
    # Get analytical gradients
    analytical = backward_fn(output_grad, cache)
    if not isinstance(analytical, tuple): # Ensure analytical is a tuple for consistent indexing
        analytical = (analytical,) 
    
    # For each input, compute numerical gradient by perturbing it
    for i, x in enumerate(inputs):
        if not isinstance(x, np.ndarray) or x.dtype.kind != "f":
            continue  # skip non-float inputs (e.g., integer indices)
        
        # We need a scalar function to differentiate. Define it as
        # sum(output * output_grad), which has the right gradient structure.
        def f(perturbed):
            modified_inputs = list(inputs)
            modified_inputs[i] = perturbed
            out, _ = forward_fn(*modified_inputs)
            return np.sum(out * output_grad)
        
        numerical = numerical_gradient(f, x.copy())
        
        diff = np.abs(analytical[i] - numerical).max()
        rel_diff = diff / (np.abs(numerical).max() + 1e-10)
        
        assert diff < tol or rel_diff < tol, (
            f"Gradient check failed for input {i}: "
            f"max abs diff = {diff:.2e}, max rel diff = {rel_diff:.2e}"
        )
    
    return True

import sys
sys.path.insert(0, ".")  # so we can import picochem

from picochem.ops import linear_forward, linear_backward


def test_linear_gradient():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 5)).astype(np.float64)
    W = rng.standard_normal((5, 3)).astype(np.float64)
    b = rng.standard_normal((3,)).astype(np.float64)
    
    check_gradient(linear_forward, linear_backward, [x, W, b])