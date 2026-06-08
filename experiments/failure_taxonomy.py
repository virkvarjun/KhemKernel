"""Categorize model failures by chemistry-relevant types.

Runs beam + OPSIN-rerank over a val sample, collects the cases the model still
gets wrong, and buckets them so we know what to fix next:

  * unparseable     — no beam candidate produced an OPSIN-valid IUPAC name
  * wrong_structure — produced a valid name, but for the wrong molecule
      - same_formula (isomer confusion: right atoms, wrong connectivity)
      - stereo_involved (target has stereochemistry)
      - by heavy-atom-count bucket (are big molecules harder?)

Usage:
    PATH="/opt/homebrew/opt/openjdk/bin:$PATH" .venv/bin/python \
        experiments/failure_taxonomy.py \
        --checkpoint runs/device_bpe_d512_v2/ckpt_latest.npz \
        --iupac_bpe data/iupac_bpe_v2.json --n_samples 1200
"""
import argparse
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, ".")

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

from picochem.checkpointing import load_checkpoint
from picochem.bpe import BPETokenizer
from picochem.data import load_vocab, decode_smiles, decode_iupac
from picochem.data_loader import load_dataset, split_dataset
from picochem.model import beam_decode
from picochem.evaluate import parse_trace, name_to_smiles, _canonicalize


def _heavy_atoms(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return m.GetNumHeavyAtoms() if m else None


def _formula(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return rdMolDescriptors.CalcMolFormula(m) if m else None


def _has_stereo(smi):
    if not smi:
        return False
    if "@" in smi or "/" in smi or "\\" in smi:
        return True
    m = Chem.MolFromSmiles(smi)
    if not m:
        return False
    return len(Chem.FindMolChiralCenters(m, useLegacyImplementation=False)) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/traces.parquet")
    ap.add_argument("--smiles_vocab", default="data/smiles_vocab.json")
    ap.add_argument("--iupac_bpe", required=True)
    ap.add_argument("--n_samples", type=int, default=1200)
    ap.add_argument("--beam_width", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()

    params, _, step, cfg = load_checkpoint(args.checkpoint)
    sv, sv_itos = load_vocab(args.smiles_vocab)
    tok = BPETokenizer.load(args.iupac_bpe)
    start, end, pad = tok.vocab["<start>"], tok.vocab["<end>"], tok.vocab["<pad>"]

    all_pairs = load_dataset(args.data, sv, tok, cfg["max_src_len"], cfg["max_tgt_len"])
    _, val = split_dataset(all_pairs, val_fraction=0.05, seed=0)
    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, len(val))
    idx = rng.choice(len(val), size=n, replace=False)

    cats = Counter()
    wrong = {"same_formula": 0, "stereo_involved": 0}
    size_total = Counter()
    size_fail = Counter()
    examples = []

    def bucket(h):
        if h is None:
            return "?"
        return "1-10" if h <= 10 else "11-20" if h <= 20 else "21-30" if h <= 30 else "31+"

    for j, i in enumerate(idx):
        src_ids, _ = val[i]
        src = src_ids[np.newaxis, :]
        smask = np.ones((1, len(src_ids)), dtype=np.float64)
        target = _canonicalize(decode_smiles(src_ids, sv_itos))
        size_total[bucket(_heavy_atoms(target))] += 1

        beams = beam_decode(src, smask, params, cfg, start, end, pad,
                            max_length=cfg["max_tgt_len"], beam_width=args.beam_width)
        cands = []
        for toks, _ in beams:
            nm = parse_trace(decode_iupac(np.array(toks), tok.itos))
            sm = _canonicalize(name_to_smiles(nm)) if nm else None
            cands.append((nm, sm))
        verified = next(((nm, sm) for nm, sm in cands if sm and sm == target), None)
        if verified:
            cats["correct"] += 1
            continue

        size_fail[bucket(_heavy_atoms(target))] += 1
        parseable = next(((nm, sm) for nm, sm in cands if sm), None)
        if not parseable:
            cats["unparseable"] += 1
            ex_pred = (cands[0][0], None)
        else:
            cats["wrong_structure"] += 1
            pname, psmi = parseable
            ex_pred = (pname, psmi)
            if _formula(psmi) == _formula(target):
                wrong["same_formula"] += 1
            if _has_stereo(target):
                wrong["stereo_involved"] += 1
        if len(examples) < args.examples:
            examples.append((target, ex_pred[0], ex_pred[1]))
        if (j + 1) % 200 == 0:
            print(f"  ...{j+1}/{n}", flush=True)

    total = sum(cats.values())
    nfail = cats["unparseable"] + cats["wrong_structure"]
    print(f"\n{'='*60}\nFailure taxonomy — step {step}, {total} samples\n{'='*60}")
    print(f"  correct:          {cats['correct']:4d}  ({cats['correct']/total*100:.1f}%)")
    print(f"  FAILURES:         {nfail:4d}  ({nfail/total*100:.1f}%)")
    print(f"    unparseable:    {cats['unparseable']:4d}  (no valid IUPAC name from any beam)")
    print(f"    wrong_structure:{cats['wrong_structure']:4d}  (valid name, wrong molecule)")
    if cats["wrong_structure"]:
        ws = cats["wrong_structure"]
        print(f"      ├ same formula (isomer confusion): {wrong['same_formula']}/{ws} "
              f"({wrong['same_formula']/ws*100:.0f}%)")
        print(f"      └ target has stereochemistry:      {wrong['stereo_involved']}/{ws} "
              f"({wrong['stereo_involved']/ws*100:.0f}%)")
    print(f"\n  Failure rate by molecule size (heavy atoms):")
    for b in ("1-10", "11-20", "21-30", "31+"):
        if size_total[b]:
            print(f"    {b:>6}: {size_fail[b]:3d}/{size_total[b]:3d}  "
                  f"({size_fail[b]/size_total[b]*100:.0f}% fail)")
    print(f"\n  Example failures (target_smiles | predicted_name | predicted_smiles):")
    for tsmi, pname, psmi in examples:
        print(f"    {tsmi}\n       -> {pname!r}  => {psmi}")


if __name__ == "__main__":
    main()
