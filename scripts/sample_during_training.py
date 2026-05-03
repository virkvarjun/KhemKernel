"""Print greedy-decoded traces for 5 fixed molecules from a checkpoint.

Usage
-----
    python scripts/sample_during_training.py --checkpoint checkpoints/step_5000.npz
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from picochem.checkpointing import load_checkpoint
from picochem.data import load_vocab, encode_smiles, decode_iupac
from picochem.model import model_forward

# Five fixed test molecules (covers different functional groups and sizes)
TEST_SMILES = [
    "c1ccccc1",                        # benzene
    "CC(=O)O",                         # acetic acid
    "c1ccc(O)cc1",                     # phenol
    "CC(N)C(=O)O",                     # alanine
    "c1ccc2ccccc2c1",                  # naphthalene
]


def greedy_decode(src_ids, src_mask, params, config, max_len, start_id, end_id):
    """Greedy autoregressive decode for a single example."""
    tgt = np.array([[start_id]], dtype=np.int32)
    for _ in range(max_len):
        T = tgt.shape[1]
        tgt_mask = np.ones((1, T), dtype=np.float64)
        logits, _ = model_forward(src_ids, tgt, src_mask, tgt_mask, params, config)
        next_tok = int(logits[0, -1, :].argmax())
        tgt = np.concatenate([tgt, np.array([[next_tok]], dtype=np.int32)], axis=1)
        if next_tok == end_id:
            break
    return tgt[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .npz checkpoint.")
    parser.add_argument("--smiles_vocab", default="data/smiles_vocab.json")
    parser.add_argument("--iupac_vocab",  default="data/iupac_vocab.json")
    args = parser.parse_args()

    params, _, step, config = load_checkpoint(args.checkpoint)
    smiles_vocab, _ = load_vocab(args.smiles_vocab)
    _, iupac_itos    = load_vocab(args.iupac_vocab)
    iupac_stoi, _   = load_vocab(args.iupac_vocab)

    start_id = iupac_stoi["<start>"]
    end_id   = iupac_stoi["<end>"]

    print(f"=== Samples at step {step} ===\n")
    for smi in TEST_SMILES:
        src_ids = encode_smiles(smi, smiles_vocab)
        if len(src_ids) == 0 or len(src_ids) > config["max_src_len"]:
            print(f"SMILES: {smi}  [skipped — length {len(src_ids)}]\n")
            continue

        src     = src_ids[np.newaxis, :]
        src_mask = np.ones((1, len(src_ids)), dtype=np.float64)

        gen_ids = greedy_decode(
            src, src_mask, params, config,
            max_len=config["max_tgt_len"],
            start_id=start_id, end_id=end_id,
        )
        gen_str = decode_iupac(gen_ids, iupac_itos)

        print(f"SMILES:    {smi}")
        print(f"Generated: {gen_str}")
        print()


if __name__ == "__main__":
    main()
