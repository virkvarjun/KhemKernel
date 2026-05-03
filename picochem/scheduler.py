"""Learning-rate schedules: linear warmup + cosine or linear decay."""
import numpy as np


def linear_warmup_cosine_decay(step, warmup_steps, total_steps, peak_lr, min_lr=0.0):
    """LR schedule: linear ramp to peak_lr, then cosine decay to min_lr.

    Parameters
    ----------
    step : int
        Current training step (1-indexed; step 0 returns 0.0).
    warmup_steps : int
        Number of steps to ramp from 0 to peak_lr.
    total_steps : int
        Total training steps; LR equals min_lr at this step and beyond.
    peak_lr : float
    min_lr : float
    """
    if step <= 0:
        return 0.0
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    if step >= total_steps:
        return float(min_lr)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(min_lr + (peak_lr - min_lr) * cosine_factor)


def linear_warmup_linear_decay(step, warmup_steps, total_steps, peak_lr, min_lr=0.0):
    """LR schedule: linear ramp to peak_lr, then linear decay to min_lr.

    Parameters match :func:`linear_warmup_cosine_decay`.
    """
    if step <= 0:
        return 0.0
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    if step >= total_steps:
        return float(min_lr)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return float(peak_lr - (peak_lr - min_lr) * progress)
