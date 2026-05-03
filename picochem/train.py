"""Training loop for picochem."""

import numpy as np

from picochem.model import init_params, model_forward, model_backward, compute_loss, loss_backward
from picochem.optimizer import init_adam_state, adam_step, clip_grad_norm


def train_step(src_ids, tgt_in, tgt_out, src_mask, tgt_mask,
               params, state, step, config,
               lr=1e-3, max_grad_norm=1.0, ignore_index=-1):
    """One training step: forward, loss, backward, optimizer.

    Args:
        src_ids: (B, S) source token IDs
        tgt_in: (B, T) decoder input (target shifted right)
        tgt_out: (B, T) decoder targets (what to predict at each position)
        src_mask, tgt_mask: (B, S), (B, T) attention masks
        params: model parameters (modified in place)
        state: optimizer state (modified in place)
        step: integer step number, starting at 1
        config: model config
        lr, max_grad_norm, ignore_index: hyperparameters

    Returns:
        loss: scalar
        grad_norm: scalar (pre-clipping)
    """
    # Forward
    logits, fwd_cache = model_forward(
        src_ids, tgt_in, src_mask, tgt_mask, params, config,
    )

    # Loss (only on tgt_out positions, ignoring padding)
    loss, loss_cache = compute_loss(logits, tgt_out, ignore_index=ignore_index)

    # Backward
    grad_logits = loss_backward(1.0, loss_cache)
    grads = model_backward(grad_logits, fwd_cache, params, config)

    # Clip and step
    grad_norm = clip_grad_norm(grads, max_grad_norm)
    adam_step(params, grads, state, step, lr=lr)

    return loss, grad_norm