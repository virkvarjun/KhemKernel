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


def compute_val_loss(val_pairs, params, config, batch_size, src_pad, tgt_pad, n_batches=20):
    """Estimate validation loss without updating params or optimizer state.

    Parameters
    ----------
    val_pairs : list of (src_ids, tgt_ids)
    params : dict
    config : dict
    batch_size : int
    src_pad : int
    tgt_pad : int
    n_batches : int
        Number of random batches to average over.

    Returns
    -------
    float : mean cross-entropy loss across the sampled batches
    """
    from picochem.data_loader import make_batch
    rng = np.random.default_rng(0)   # fixed seed → deterministic estimate
    n = min(n_batches, max(1, len(val_pairs) // max(1, batch_size)))
    total = 0.0
    for _ in range(n):
        src_ids, tgt_in, tgt_out, src_mask, tgt_mask = make_batch(
            val_pairs, batch_size, src_pad, tgt_pad, rng
        )
        logits, _ = model_forward(src_ids, tgt_in, src_mask, tgt_mask, params, config)
        loss, _ = compute_loss(logits, tgt_out, ignore_index=-1)
        total += float(loss)
    return total / n