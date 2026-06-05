"""Train the BPE tokenizer for the IUPAC/trace side and report length stats.

    python scripts/build_bpe.py --vocab_size 4000

Reads the ``trace`` column of data/traces.parquet, learns merges, writes
data/iupac_bpe.json, and prints the token-length distribution so you can pick
``--max_tgt_len`` for training.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from picochem.bpe import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/traces.parquet")
    ap.add_argument("--out", default="data/iupac_bpe.json")
    ap.add_argument("--vocab_size", type=int, default=4000)
    ap.add_argument("--train_lines", type=int, default=300_000,
                    help="How many traces to learn merges from (merges are stable "
                         "well before the full corpus; encoding still applies to all).")
    args = ap.parse_args()

    print(f"Loading {args.data}")
    df = pd.read_parquet(args.data)
    col = "trace" if "trace" in df.columns else "IUPAC"
    corpus = df[col].astype(str).tolist()
    train_corpus = corpus[: args.train_lines]
    print(f"Training BPE (vocab_size {args.vocab_size}) on {len(train_corpus):,} of {len(corpus):,} traces")

    tok = BPETokenizer.train(train_corpus, vocab_size=args.vocab_size, verbose=True)
    tok.save(args.out)
    print(f"\nSaved tokenizer -> {args.out}  (vocab {len(tok.vocab)}, merges {len(tok.merges)})")

    # Length distribution over the WHOLE corpus (includes <start>/<end>).
    lens = np.fromiter((len(tok.encode(t)) for t in corpus), dtype=np.int32, count=len(corpus))
    unk = tok.vocab["<unk>"]
    n_unk = sum(1 for t in corpus[:20000] if unk in tok.encode(t))
    print(f"\nToken-length over {len(corpus):,} traces:")
    print(f"  mean {lens.mean():.1f}  p50 {int(np.percentile(lens,50))}  "
          f"p95 {int(np.percentile(lens,95))}  p99 {int(np.percentile(lens,99))}  max {int(lens.max())}")
    print(f"  <unk> traces in first 20k: {n_unk}")
    print(f"\nSuggested --max_tgt_len: {int(np.percentile(lens,99)) + 8}  (covers p99 + margin)")


if __name__ == "__main__":
    main()
