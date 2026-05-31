"""Evaluate a checkpoint against OPSIN on a held-out test set.

Usage
-----
    python scripts/evaluate.py --checkpoint runs/run_<ts>/ckpt_best.npz \\
                                --n_samples 500 --seed 42
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from picochem.checkpointing import load_checkpoint
from picochem.data import load_vocab
from picochem.data_loader import load_dataset, split_dataset
from picochem.evaluate import evaluate_model, OPSIN_AVAILABLE, _OPSIN_BACKEND


def main():
    parser = argparse.ArgumentParser(description="Evaluate a picochem checkpoint.")
    parser.add_argument("--checkpoint",   required=True,
                        help="Path to a .npz checkpoint file.")
    parser.add_argument("--data",         default="data/traces.parquet")
    parser.add_argument("--smiles_vocab", default="data/smiles_vocab.json")
    parser.add_argument("--iupac_vocab",  default="data/iupac_vocab.json")
    parser.add_argument("--n_samples",    type=int, default=500)
    parser.add_argument("--seed",         type=int, default=42,
                        help="Seed for reproducible test-set selection.")
    parser.add_argument("--output",       default=None,
                        help="Path to save results JSON.")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    params, _, step, config = load_checkpoint(args.checkpoint)
    print(f"  Step: {step}")

    smiles_vocab, smiles_itos = load_vocab(args.smiles_vocab)
    iupac_vocab,  iupac_itos  = load_vocab(args.iupac_vocab)

    print(f"Loading dataset: {args.data}")
    all_pairs = load_dataset(
        args.data, smiles_vocab, iupac_vocab,
        config["max_src_len"], config["max_tgt_len"],
    )
    # Use the val split (same seed → same split as training)
    _, val_pairs = split_dataset(all_pairs, val_fraction=0.05, seed=0)

    rng = np.random.default_rng(args.seed)
    n   = min(args.n_samples, len(val_pairs))
    idx = rng.choice(len(val_pairs), size=n, replace=False)
    test_pairs = [val_pairs[i] for i in idx]

    print(f"Evaluating {n} examples  (OPSIN backend: {_OPSIN_BACKEND})\n")

    results = evaluate_model(
        params, config, test_pairs,
        smiles_itos, iupac_itos,
        n_samples=n,
        max_length=config["max_tgt_len"],
    )

    N = results["n_evaluated"]
    n_v = int(results["trace_validity_rate"]  * N)
    n_o = int(results["opsin_parse_rate"]     * N)
    n_m = int(results["structure_match_rate"] * N)

    print(f"{'─'*45}")
    print(f"  Step:              {step}")
    print(f"  N evaluated:       {N}")
    print(f"  Trace validity:    {results['trace_validity_rate']*100:5.1f}%  ({n_v} / {N})")
    if OPSIN_AVAILABLE:
        print(f"  OPSIN parse:       {results['opsin_parse_rate']*100:5.1f}%  ({n_o} / {N})")
        print(f"  Structure match:   {results['structure_match_rate']*100:5.1f}%  ({n_m} / {N})")
    else:
        print("  OPSIN not available — install py2opsin + Java for parse/match metrics")
    print(f"{'─'*45}\n")

    out_path = args.output
    if out_path is None:
        os.makedirs("results", exist_ok=True)
        out_path = os.path.join("results", f"eval_{step}.json")

    with open(out_path, "w") as f:
        summary = {k: v for k, v in results.items() if k != "samples"}
        summary["step"] = step
        summary["opsin_backend"] = _OPSIN_BACKEND
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
