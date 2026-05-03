"""Evaluate a checkpoint against OPSIN on a held-out test set.

Usage
-----
    python scripts/evaluate.py --checkpoint checkpoints/step_10000.npz \\
                                --n_eval 500 --seed 42
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, ".")

from picochem.checkpointing import load_checkpoint
from picochem.data import load_vocab, encode_smiles, decode_iupac
from picochem.data_loader import load_dataset, make_batch
from picochem.model import model_forward

# ---------------------------------------------------------------------------
# OPSIN integration
# ---------------------------------------------------------------------------

_OPSIN_AVAILABLE = False
_name_to_smiles  = None

try:
    import py2opsin as _py2opsin
    def _name_to_smiles_py2opsin(name):
        result = _py2opsin.name2smiles(name)
        return result if result else None
    _name_to_smiles = _name_to_smiles_py2opsin
    _OPSIN_AVAILABLE = True
    _OPSIN_BACKEND = "py2opsin"
except ImportError:
    pass

if not _OPSIN_AVAILABLE:
    import subprocess as _subprocess
    _OPSIN_JAR = os.environ.get("OPSIN_JAR", "opsin.jar")
    if os.path.exists(_OPSIN_JAR):
        def _name_to_smiles_jar(name):
            try:
                result = _subprocess.run(
                    ["java", "-jar", _OPSIN_JAR, "-osmi", name],
                    capture_output=True, text=True, timeout=10,
                )
                smi = result.stdout.strip()
                return smi if smi else None
            except Exception:
                return None
        _name_to_smiles = _name_to_smiles_jar
        _OPSIN_AVAILABLE = True
        _OPSIN_BACKEND = "opsin-jar"

if not _OPSIN_AVAILABLE:
    _OPSIN_BACKEND = "none"


# ---------------------------------------------------------------------------
# RDKit canonicalization
# ---------------------------------------------------------------------------

try:
    from rdkit import Chem as _Chem
    from rdkit import RDLogger as _RDLogger
    _RDLogger.DisableLog("rdApp.*")

    def _canonicalize(smi):
        if not smi:
            return None
        mol = _Chem.MolFromSmiles(smi)
        return _Chem.MolToSmiles(mol) if mol else None
    _RDKIT_AVAILABLE = True
except ImportError:
    def _canonicalize(smi):
        return smi
    _RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Greedy decoder
# ---------------------------------------------------------------------------

def greedy_decode(src_ids, src_mask, params, config, max_len, start_id, end_id):
    """Greedy left-to-right decoding for a single example (B=1)."""
    tgt = np.array([[start_id]], dtype=np.int32)
    for _ in range(max_len):
        T = tgt.shape[1]
        tgt_mask = np.ones((1, T), dtype=np.float64)
        logits, _ = model_forward(src_ids, tgt, src_mask, tgt_mask, params, config)
        next_tok = int(logits[0, -1, :].argmax())
        tgt = np.concatenate([tgt, np.array([[next_tok]], dtype=np.int32)], axis=1)
        if next_tok == end_id:
            break
    return tgt[0]   # 1-D array


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"<name>(.*?)</name>", re.DOTALL)


def parse_name(trace_str):
    """Extract the IUPAC name from a decoded trace string, or None."""
    m = _NAME_RE.search(trace_str)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a picochem checkpoint.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a .npz checkpoint file.")
    parser.add_argument("--data", default="data/traces.parquet",
                        help="Path to traces.parquet.")
    parser.add_argument("--smiles_vocab", default="data/smiles_vocab.json")
    parser.add_argument("--iupac_vocab",  default="data/iupac_vocab.json")
    parser.add_argument("--n_eval", type=int, default=200,
                        help="Number of held-out examples to evaluate.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Fixed RNG seed for reproducible test-set selection.")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    params, _, step, config = load_checkpoint(args.checkpoint)

    smiles_vocab, _ = load_vocab(args.smiles_vocab)
    iupac_vocab,  iupac_itos = load_vocab(args.iupac_vocab)

    start_id = iupac_vocab["<start>"]
    end_id   = iupac_vocab["<end>"]
    pad_id   = iupac_vocab["<pad>"]

    print(f"Loading dataset: {args.data}")
    pairs = load_dataset(
        args.data, smiles_vocab, iupac_vocab,
        config["max_src_len"], config["max_tgt_len"],
    )

    rng = np.random.default_rng(args.seed)
    n = min(args.n_eval, len(pairs))
    test_idx = rng.choice(len(pairs), size=n, replace=False)
    test_pairs = [pairs[i] for i in test_idx]

    print(f"Evaluating {n} examples  (OPSIN backend: {_OPSIN_BACKEND})")
    print()

    n_valid_trace  = 0   # has parseable <name>...</name>
    n_opsin_ok     = 0   # OPSIN converted name → SMILES
    n_struct_match = 0   # canonicalized SMILES match

    for src_ids, tgt_ids in tqdm(test_pairs, desc="eval"):
        src = src_ids[np.newaxis, :]                            # (1, S)
        src_mask = np.ones((1, len(src_ids)), dtype=np.float64)

        gen_ids = greedy_decode(
            src, src_mask, params, config,
            max_len=config["max_tgt_len"],
            start_id=start_id, end_id=end_id,
        )
        gen_str = decode_iupac(gen_ids, iupac_itos)
        ref_str = decode_iupac(tgt_ids, iupac_itos)

        pred_name = parse_name(gen_str)
        ref_name  = parse_name(ref_str)

        if pred_name is not None:
            n_valid_trace += 1

        if _OPSIN_AVAILABLE and pred_name:
            pred_smi = _name_to_smiles(pred_name)
            if pred_smi:
                n_opsin_ok += 1
                if _RDKIT_AVAILABLE and ref_name:
                    ref_smi = _name_to_smiles(ref_name)
                    if ref_smi:
                        if _canonicalize(pred_smi) == _canonicalize(ref_smi):
                            n_struct_match += 1

    trace_validity  = n_valid_trace  / n
    opsin_parse     = n_opsin_ok     / n
    struct_match    = n_struct_match / n

    print(f"\n{'='*45}")
    print(f"  Step:                {step}")
    print(f"  N evaluated:         {n}")
    print(f"  Trace validity rate: {trace_validity:.3f}")
    if _OPSIN_AVAILABLE:
        print(f"  OPSIN parse rate:    {opsin_parse:.3f}")
        print(f"  Structure match:     {struct_match:.3f}")
    else:
        print("  OPSIN not available — skipping parse/structure metrics")
    print(f"{'='*45}\n")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"eval_{step}.json")
    result = {
        "step": step,
        "n_eval": n,
        "seed": args.seed,
        "opsin_backend": _OPSIN_BACKEND,
        "trace_validity_rate": trace_validity,
        "opsin_parse_rate": opsin_parse if _OPSIN_AVAILABLE else None,
        "structure_match_rate": struct_match if _OPSIN_AVAILABLE else None,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
