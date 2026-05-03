"""Overfit test: prove the training stack works by memorizing 32 examples."""

import numpy as np

from picochem.model import init_params
from picochem.optimizer import init_adam_state
from picochem.train import train_step


def make_synthetic_batch(rng, batch_size, src_len, tgt_len, src_vocab, tgt_vocab):
    """Create a small synthetic dataset for overfit testing."""
    src_ids = rng.integers(2, src_vocab, size=(batch_size, src_len))  # avoid pad/special
    tgt_full = rng.integers(2, tgt_vocab, size=(batch_size, tgt_len + 1))

    # Shifted teacher forcing
    tgt_in = tgt_full[:, :-1]    # (B, T)
    tgt_out = tgt_full[:, 1:]    # (B, T)

    src_mask = np.ones((batch_size, src_len), dtype=np.float64)
    tgt_mask = np.ones((batch_size, tgt_len), dtype=np.float64)

    return src_ids, tgt_in, tgt_out, src_mask, tgt_mask


def main():
    config = {
        'src_vocab': 50, 'tgt_vocab': 100,
        'd_model': 64, 'n_heads': 4, 'd_ff': 256,
        'n_enc_layers': 2, 'n_dec_layers': 2,
        'max_src_len': 20, 'max_tgt_len': 30,
    }

    rng = np.random.default_rng(42)
    params = init_params(config, rng)
    state = init_adam_state(params)

    # Generate one fixed batch and train on it repeatedly
    batch = make_synthetic_batch(rng, batch_size=32, src_len=12, tgt_len=16,
                                  src_vocab=config['src_vocab'], tgt_vocab=config['tgt_vocab'])

    print(f"Overfit test: training on {32} fixed examples.")
    print(f"Initial loss should be ~log({config['tgt_vocab']}) = {np.log(config['tgt_vocab']):.2f}")
    print(f"Target loss: < 0.1\n")

    n_steps = 500
    for step in range(1, n_steps + 1):
        loss, grad_norm = train_step(
            *batch, params=params, state=state, step=step, config=config,
            lr=3e-4, max_grad_norm=1.0,
        )

        if step == 1 or step % 25 == 0:
            print(f"step {step:4d}  loss {loss:.4f}  |grad| {grad_norm:.3f}")

        if loss < 0.05:
            print(f"\nOverfit succeeded at step {step}! Loss = {loss:.4f}")
            return

    print(f"\nFinal loss: {loss:.4f}")
    if loss < 0.5:
        print("Loss dropped substantially. Likely working but didn't fully converge.")
    else:
        print("Loss did not drop enough. Something is likely broken.")


if __name__ == "__main__":
    main()