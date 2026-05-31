"""Training script for picochem.

Usage (fresh run)
-----------------
    python scripts/train.py --data data/raw_pairs.parquet \\
        --d_model 256 --n_heads 4 --d_ff 1024 \\
        --n_enc_layers 3 --n_dec_layers 3 \\
        --total_steps 100000 --batch_size 32 --peak_lr 3e-4

Usage (resume)
--------------
    python scripts/train.py --resume_from runs/run_<ts>/ckpt_latest.npz ...
"""
import argparse
import datetime
import json
import os
import sys
import traceback

import numpy as np
from tqdm import tqdm

sys.path.insert(0, ".")

from picochem.checkpointing import save_checkpoint, load_checkpoint
from picochem.data import load_vocab
from picochem.data_loader import load_dataset, make_batch, split_dataset
from picochem.logging_utils import TrainLogger
from picochem.model import init_params
from picochem.optimizer import init_adam_state
from picochem.scheduler import (
    linear_warmup_cosine_decay,
    linear_warmup_linear_decay,
)
from picochem.train import train_step, compute_val_loss


def _get_lr(step, args):
    if args.schedule == "flat":
        return args.peak_lr
    if args.schedule == "cosine":
        return linear_warmup_cosine_decay(
            step, args.warmup_steps, args.total_steps, args.peak_lr, args.min_lr
        )
    return linear_warmup_linear_decay(
        step, args.warmup_steps, args.total_steps, args.peak_lr, args.min_lr
    )


def _save(path, params, state, step, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_checkpoint(path, params, state, step, config)


def main():
    parser = argparse.ArgumentParser(description="Train picochem.")

    # Data
    parser.add_argument("--data",         default="data/raw_pairs.parquet")
    parser.add_argument("--smiles_vocab", default="data/smiles_vocab.json")
    parser.add_argument("--iupac_vocab",  default="data/iupac_vocab.json")

    # Model
    parser.add_argument("--d_model",      type=int,   default=256)
    parser.add_argument("--n_heads",      type=int,   default=4)
    parser.add_argument("--d_ff",         type=int,   default=1024)
    parser.add_argument("--n_enc_layers", type=int,   default=3)
    parser.add_argument("--n_dec_layers", type=int,   default=3)
    parser.add_argument("--max_src_len",  type=int,   default=150)
    parser.add_argument("--max_tgt_len",  type=int,   default=200)

    # Optimisation
    parser.add_argument("--total_steps",   type=int,   default=100_000)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--peak_lr",       type=float, default=3e-4)
    parser.add_argument("--min_lr",        type=float, default=1e-5)
    parser.add_argument("--warmup_steps",  type=int,   default=1000)
    parser.add_argument("--schedule",      choices=["flat", "cosine", "linear"],
                        default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Checkpointing
    parser.add_argument("--run_dir",               default=None,
                        help="Directory for this run. Defaults to runs/run_<timestamp>.")
    parser.add_argument("--resume_from",           default=None,
                        help="Path to ckpt_latest.npz to resume from.")
    parser.add_argument("--checkpoint_every",      type=int, default=500,
                        help="Save ckpt_latest.npz every N steps.")
    parser.add_argument("--keep_checkpoint_every", type=int, default=5000,
                        help="Keep a numbered ckpt_NNNN.npz every N steps.")

    # Validation
    parser.add_argument("--val_every",    type=int,   default=500)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--val_batches",  type=int,   default=20)

    # Evaluation (OPSIN)
    parser.add_argument("--eval_every",    type=int, default=5000,
                        help="Run OPSIN structure-match eval every N steps.")
    parser.add_argument("--eval_n_samples", type=int, default=100)

    # Logging
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--plot_every", type=int, default=500)

    # Backend
    parser.add_argument("--backend", choices=["numpy", "cuda"], default="numpy",
                        help="Compute backend for forward-pass ops (default: numpy).")

    args = parser.parse_args()

    # ── Backend ───────────────────────────────────────────────────────────
    from picochem import backend as _backend_mod
    _backend_mod.set_backend(args.backend)
    if args.backend != "numpy":
        print(f"Backend: {args.backend}")

    # ── Run directory ──────────────────────────────────────────────────────
    if args.run_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = os.path.join("runs", f"run_{ts}")
    os.makedirs(args.run_dir, exist_ok=True)

    # ── Vocabs ────────────────────────────────────────────────────────────
    smiles_vocab, smiles_itos = load_vocab(args.smiles_vocab)
    iupac_vocab,  iupac_itos  = load_vocab(args.iupac_vocab)

    src_pad = smiles_vocab["<pad>"]
    tgt_pad = iupac_vocab["<pad>"]

    # ── Dataset ───────────────────────────────────────────────────────────
    print(f"Loading data from {args.data} ...")
    all_pairs = load_dataset(
        args.data, smiles_vocab, iupac_vocab,
        args.max_src_len, args.max_tgt_len,
    )
    train_pairs, val_pairs = split_dataset(all_pairs, val_fraction=args.val_fraction)
    print(f"  train: {len(train_pairs):,}  val: {len(val_pairs):,}")

    # ── Model config ──────────────────────────────────────────────────────
    config = {
        "src_vocab":    len(smiles_vocab),
        "tgt_vocab":    len(iupac_vocab),
        "d_model":      args.d_model,
        "n_heads":      args.n_heads,
        "d_ff":         args.d_ff,
        "n_enc_layers": args.n_enc_layers,
        "n_dec_layers": args.n_dec_layers,
        "max_src_len":  args.max_src_len,
        "max_tgt_len":  args.max_tgt_len,
    }

    # ── Init or resume ────────────────────────────────────────────────────
    start_step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from {args.resume_from}")
        params, state, start_step, config = load_checkpoint(args.resume_from)
        print(f"  Resumed at step {start_step}")
    else:
        rng = np.random.default_rng(0)
        params = init_params(config, rng)
        state  = init_adam_state(params)

    # Save args for resume_training.py
    with open(os.path.join(args.run_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Logger ────────────────────────────────────────────────────────────
    logger = TrainLogger(args.run_dir)

    # ── Eval integration ──────────────────────────────────────────────────
    try:
        from picochem.evaluate import evaluate_model as _eval_model
        _EVAL_AVAILABLE = True
    except Exception:
        _EVAL_AVAILABLE = False

    # ── Training loop ─────────────────────────────────────────────────────
    rng = np.random.default_rng(start_step)
    best_val_loss = float("inf")

    ckpt_best   = os.path.join(args.run_dir, "ckpt_best.npz")
    ckpt_latest = os.path.join(args.run_dir, "ckpt_latest.npz")

    print(f"Training for {args.total_steps} steps  (run_dir: {args.run_dir})\n")

    def _do_save_latest(step):
        _save(ckpt_latest, params, state, step, config)

    def _do_checkpoint(step):
        _do_save_latest(step)
        if step % args.keep_checkpoint_every == 0:
            numbered = os.path.join(args.run_dir, f"ckpt_{step:07d}.npz")
            _save(numbered, params, state, step, config)

    try:
        pbar = tqdm(range(start_step + 1, args.total_steps + 1), initial=start_step,
                    total=args.total_steps, dynamic_ncols=True)
        for step in pbar:
            lr = _get_lr(step, args)

            batch = make_batch(train_pairs, args.batch_size, src_pad, tgt_pad, rng)
            loss, grad_norm = train_step(
                *batch,
                params=params, state=state, step=step, config=config,
                lr=lr, max_grad_norm=args.max_grad_norm,
            )

            if step % args.log_every == 0:
                logger.log_step(step, loss, grad_norm, lr)
                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}")

            # ── Val loss ──────────────────────────────────────────────────
            if step % args.val_every == 0 and len(val_pairs) > 0:
                val_loss = compute_val_loss(
                    val_pairs, params, config,
                    args.batch_size, src_pad, tgt_pad,
                    n_batches=args.val_batches,
                )
                logger.log_val(step, val_loss)
                tqdm.write(
                    f"step {step:6d}  train_loss {loss:.4f}  val_loss {val_loss:.4f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save(ckpt_best, params, state, step, config)
                    tqdm.write(f"  → new best val loss {best_val_loss:.4f}")

            # ── OPSIN eval ────────────────────────────────────────────────
            if _EVAL_AVAILABLE and step % args.eval_every == 0 and len(val_pairs) > 0:
                try:
                    res = _eval_model(
                        params, config, val_pairs,
                        smiles_itos, iupac_itos,
                        n_samples=args.eval_n_samples,
                        max_length=config["max_tgt_len"],
                    )
                    logger.log_eval(
                        step,
                        res["trace_validity_rate"],
                        res["opsin_parse_rate"],
                        res["structure_match_rate"],
                    )
                    tqdm.write(
                        f"  OPSIN eval: validity={res['trace_validity_rate']:.3f} "
                        f"parse={res['opsin_parse_rate']:.3f} "
                        f"match={res['structure_match_rate']:.3f}"
                    )
                except Exception as e:
                    tqdm.write(f"  eval error: {e}")

            # ── Checkpoint ────────────────────────────────────────────────
            if step % args.checkpoint_every == 0:
                _do_checkpoint(step)

            # ── Plot ──────────────────────────────────────────────────────
            if step % args.plot_every == 0:
                logger.plot()

    except KeyboardInterrupt:
        tqdm.write("\nInterrupted — saving ckpt_latest.npz ...")
        _do_save_latest(step)
        tqdm.write(f"Saved to {ckpt_latest}")
        sys.exit(0)

    except Exception:
        tqdm.write("\nUnexpected error — saving ckpt_latest.npz ...")
        _do_save_latest(step)
        raise

    # Save final checkpoint
    _do_save_latest(args.total_steps)
    logger.plot()
    print(f"\nDone. Final checkpoint at {ckpt_latest}")


if __name__ == "__main__":
    main()
