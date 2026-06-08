"""How much does the model 'know' beyond its top-1? Beam-width ceiling sweep.

Decodes a wide beam (default 20) and reports, at top-k for k in {1,5,10,20}:
  * exact@k  — some candidate in the top-k exactly matches the target
  * const@k  — some candidate matches the target stereo-blind

Reading it:
  * exact@1  ≈ greedy accuracy
  * exact@5  ≈ current beam+rerank
  * if exact@20 >> exact@5  → the model already knows answers it ranks low
        → RL / rejection-sampling can harvest them (cheap win)
  * if the curve plateaus    → the model genuinely doesn't know those cases
        → need capability (graph/atom features, bigger model, more data)
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
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--beam_width", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ks = [k for k in (1, 5, 10, 20) if k <= args.beam_width]

    params, _, step, cfg = load_checkpoint(args.checkpoint)
    sv, sv_itos = load_vocab(args.smiles_vocab)
    tok = BPETokenizer.load(args.iupac_bpe)
    start, end, pad = tok.vocab["<start>"], tok.vocab["<end>"], tok.vocab["<pad>"]

    pairs = load_dataset(args.data, sv, tok, cfg["max_src_len"], cfg["max_tgt_len"])
    _, val = split_dataset(pairs, val_fraction=0.05, seed=0)
    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, len(val))
    idx = rng.choice(len(val), size=n, replace=False)

    exact_at = {k: 0 for k in ks}
    const_at = {k: 0 for k in ks}

    for j, i in enumerate(idx):
        src_ids, _ = val[i]
        src = src_ids[np.newaxis, :]
        smask = np.ones((1, len(src_ids)), dtype=np.float64)
        tgt = _canonicalize(decode_smiles(src_ids, sv_itos))
        tgt_flat = strip_stereo(tgt)

        beams = beam_decode(src, smask, params, cfg, start, end, pad,
                            max_length=cfg["max_tgt_len"],
                            beam_width=args.beam_width, n_return=args.beam_width)
        smis = []
        for toks, _ in beams:
            nm = parse_trace(decode_iupac(np.array(toks), tok.itos))
            smis.append(_canonicalize(name_to_smiles(nm)) if nm else None)

        for k in ks:
            top = smis[:k]
            if any(s and s == tgt for s in top):
                exact_at[k] += 1
                const_at[k] += 1
            elif any(s and strip_stereo(s) == tgt_flat for s in top):
                const_at[k] += 1
        if (j + 1) % 100 == 0:
            print(f"  ...{j+1}/{n}", flush=True)

    print(f"\n{'='*52}\nBeam-width ceiling — step {step}, {n} samples\n{'='*52}")
    print(f"  {'top-k':>6} | {'exact':>8} | {'stereo-blind':>12}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*12}")
    for k in ks:
        print(f"  {k:>6} | {exact_at[k]/n*100:7.1f}% | {const_at[k]/n*100:11.1f}%")
    if len(ks) >= 2:
        lo, hi = ks[0], ks[-1]
        print(f"\n  exact headroom top-{lo}→top-{hi}: "
              f"+{(exact_at[hi]-exact_at[lo])/n*100:.1f} pts "
              f"(answers the model knows but ranks below {lo})")


if __name__ == "__main__":
    main()
