"""Adam optimizer with weight decay and optional learning-rate schedule."""
import numpy as np 

def init_adam(params): 
    """Create m and v state mirroring the params structure."""
    if isinstance(params, np.ndarray):
        return {'m': np.zeros_like(params), 'v': np.zeros_like(params)}
    if isinstance(params, dict):
        return {k: init_adam_state(v) for k, v in params.items()}
    if isinstance(params, list):
        return [init_adam_state(item) for item in params]
    raise TypeError(f"Unsupported params type: {type(params)}")

