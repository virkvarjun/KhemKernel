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

def adam_step(params, grads, state, step, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    if isinstance(params, np.ndarray): 
        g = grads 
        s = state 
        s['m'] = beta1 * s['m'] + (1-beta1)*g 
        s['v'] = beta2 * s['v'] + (1 - beta2) * (g ** 2)
        m_hat = s['m'] / (1 - beta1 ** step)
        v_hat = s['v'] / (1 - beta2 ** step)
        params -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return params 
    if isinstance(params, dict): 
        for key in params: 
            params[key] = adam_step(params[key], grads[key], state[key], step, lr, beta1, beta2, eps)
            return params 
    if isinstance(params, list):
        for i in range(len(params)):
            params[i] = adam_step(params[i], grads[i], state[i], step,
                                   lr, beta1, beta2, eps)
        return params

    raise TypeError(f"Unsupported type: {type(params)}")