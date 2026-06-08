"""Separate stereochemistry errors from constitution errors.

For each val molecule we beam-decode and check, across the beam:
  * exact_any  — some candidate's OPSIN SMILES == target (stereo-sensitive)
  * const_any  — some candidate matches the target with stereochemistry removed
                 (right skeleton + substituents, possibly wrong stereo)

The gap (const_any - exact_any) is the share the model gets constitutionally
right but stereochemically wrong — the "if we ignored stereo" headroom.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from picochem.checkpointing import load_checkpoint
from picochem.bpe import BPETokenizer
from picochem.data import load_vocab, decode_smiles, decode_iupac
from picochem.data_loader import load_dataset, split_dataset
from picochem.model import beam_decode
from picochem.evaluate import parse_trace, name_to_smiles, _canonicalize


def strip_stereo(smi):
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if not m:
        return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/traces.parquet")
    ap.add_argument("--smiles_vocab", default="data/smiles_vocab.json")
    ap.add_argument("--iupac_bpe", required=True)
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--beam_width", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    params, _, step, cfg = load_checkpoint(args.checkpoint)
    sv, sv_itos = load_vocab(args.smiles_vocab)
    tok = BPETokenizer.load(args.iupac_bpe)
    start, end, pad = tok.vocab["<start>"], tok.vocab["<end>"], tok.vocab["<pad>"]

    pairs = load_dataset(args.data, sv, tok, cfg["max_src_len"], cfg["max_tgt_len"])
    _, val = split_dataset(pairs, val_fraction=0.05, seed=0)
    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, len(val))
    idx = rng.choice(len(val), size=n, replace=False)

    exact_n = const_n = stereo_gap = has_stereo = 0
    for j, i in enumerate(idx):
        src_ids, _ = val[i]
        src = src_ids[np.newaxis, :]
        smask = np.ones((1, len(src_ids)), dtype=np.float64)
        tgt = _canonicalize(decode_smiles(src_ids, sv_itos))
        tgt_flat = strip_stereo(tgt)
        if tgt != tgt_flat:
            has_stereo += 1

        beams = beam_decode(src, smask, params, cfg, start, end, pad,
                            max_length=cfg["max_tgt_len"], beam_width=args.beam_width)
        cand_smis = []
        for toks, _ in beams:
            nm = parse_trace(decode_iupac(np.array(toks), tok.itos))
            cand_smis.append(_canonicalize(name_to_smiles(nm)) if nm else None)

        exact = any(s and s == tgt for s in cand_smis)
        const = exact or any(s and strip_stereo(s) == tgt_flat for s in cand_smis)
        exact_n += exact
        const_n += const
        if const and not exact:
            stereo_gap += 1
        if (j + 1) % 200 == 0:
            print(f"  ...{j+1}/{n}", flush=True)

    print(f"\n{'='*56}\nStereo breakdown — step {step}, {n} samples (beam {args.beam_width})\n{'='*56}")
    print(f"  exact match (stereo-sensitive):  {exact_n}/{n}  ({exact_n/n*100:.1f}%)")
    print(f"  constitutional match (stereo-blind): {const_n}/{n}  ({const_n/n*100:.1f}%)")
    print(f"  ── stereo-only near-misses:      {stereo_gap}/{n}  ({stereo_gap/n*100:.1f} pts of the gap)")
    print(f"  targets that HAVE stereochemistry: {has_stereo}/{n}  ({has_stereo/n*100:.1f}%)")
    if has_stereo:
        print(f"  of stereo-bearing targets, stereo-only errors ≈ {stereo_gap/has_stereo*100:.1f}%")


if __name__ == "__main__":
    main()
