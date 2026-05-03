"""Adam optimizer with weight decay and optional learning-rate schedule."""
import numpy as np 

def init_adam_state(params):
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

# Gradient clipping 
def clip_grad_norm(grads, max_norm): 
    total_sq = 0.0 
    def accumulate(g):
        nonlocal total_sq
        if isinstance(g, np.ndarray):
            total_sq += (g ** 2).sum()
        elif isinstance(g, dict):
            for v in g.values():
                accumulate(v)
        elif isinstance(g, list):
            for v in g:
                accumulate(v)

    accumulate(grads)
    total_norm = np.sqrt(total_sq)

    # Scale down if needed
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)

        def rescale(g):
            if isinstance(g, np.ndarray):
                g *= scale
            elif isinstance(g, dict):
                for v in g.values():
                    rescale(v)
            elif isinstance(g, list):
                for v in g:
                    rescale(v)

        rescale(grads)

    return total_norm