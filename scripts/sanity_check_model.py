"""Quick sanity check: build a tiny model, run one forward pass, print loss.

Usage:
    python scripts/sanity_check_model.py
    python scripts/sanity_check_model.py --backend cuda
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")

import picochem.backend as _backend_mod
from picochem.model import init_params, model_forward, compute_loss, loss_backward
from picochem.optimizer import init_adam_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["numpy", "cuda"], default="numpy")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    _backend_mod.set_backend(args.backend)
    print(f"Backend : {args.backend}")

    config = {
        "src_vocab": 50, "tgt_vocab": 50,
        "d_model": 32,   "n_heads": 2,
        "d_ff": 64,
        "n_enc_layers": 1, "n_dec_layers": 1,
        "max_src_len": 20, "max_tgt_len": 20,
    }

    rng = np.random.default_rng(0)
    params = init_params(config, rng)

    B, S, T = args.batch_size, 10, 8
    src_ids = rng.integers(2, 50, size=(B, S))
    tgt_ids = rng.integers(2, 50, size=(B, T))
    tgt_out = rng.integers(2, 50, size=(B, T))

    # src_mask: 1 = keep, 0 = pad (all-ones = no padding)
    src_mask = np.ones((B, S))
    tgt_mask = np.ones((B, T))

    t0 = time.perf_counter()
    logits, fwd_cache = model_forward(src_ids, tgt_ids, src_mask, tgt_mask, params, config)
    loss, loss_cache = compute_loss(logits, tgt_out)
    elapsed = time.perf_counter() - t0

    print(f"Logits shape: {logits.shape}")
    print(f"Loss    : {float(loss):.4f}")
    print(f"Time    : {elapsed*1000:.1f} ms")
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
