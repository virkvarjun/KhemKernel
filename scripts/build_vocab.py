"""Scan training data, build character/token vocabularies, and write vocab files."""
import json
import os
from collections import Counter

import pandas as pd
from tqdm import tqdm

from picochem.data import tokenize_smiles, tokenize_iupac

INPUT_PATH = "data/raw_pairs.parquet"
SMILES_VOCAB_PATH = "data/smiles_vocab.json"
IUPAC_VOCAB_PATH = "data/iupac_vocab.json"
IUPAC_MIN_FREQ = 5  # IUPAC tokens appearing fewer times become <unk>

SPECIAL_TOKENS = ["<pad>", "<start>", "<end>", "<unk>"]  # Special strings for the model

# Trace structure tokens — always included in the IUPAC vocab regardless of
# frequency so that encoded traces never map these to <unk>.
TRACE_TOKENS = [
    "<parent>", "</parent>",
    "<groups>", "</groups>",
    "<atoms>",  "</atoms>",
    "<rings>",  "</rings>",
    "<name>",   "</name>",
    ";",
]
# <pad> - fills out short sequences in a a batch so they have the equal length
# <start> - beginning of the iupac sequence we want decoder to generate
# <end> - Tells the decoder to stop generating stuff
# <unk> - replaces tokens not in the vocabulary (i.e. rare tokens that we want to replace with a common token to avoid overfitting)

def build_vocab(token_lists, min_freq=1, extra_tokens=None) -> dict[str, int]:
    """Build a token-to-index vocabulary from lists of tokens.

    ``extra_tokens`` are inserted right after SPECIAL_TOKENS and are included
    unconditionally (useful for trace structure tokens).
    """
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    reserved = set(SPECIAL_TOKENS) | set(extra_tokens or [])
    vocab = list(SPECIAL_TOKENS)
    if extra_tokens:
        for tok in extra_tokens:
            if tok not in SPECIAL_TOKENS:
                vocab.append(tok)
    for token, count in sorted(counter.items()):
        if count >= min_freq and token not in reserved:
            vocab.append(token)
    return {token: idx for idx, token in enumerate(vocab)}


def main():
    print(f"Loading {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df):,} pairs")
    print("\nTokenizing SMILES")
    smiles_tokens = [tokenize_smiles(s) for s in tqdm(df["SMILES"])]
    print("Tokenizing IUPAC names...")
    iupac_tokens = [tokenize_iupac(s) for s in tqdm(df["IUPAC"])]

    smiles_vocab = build_vocab(smiles_tokens, min_freq=1)
    iupac_vocab  = build_vocab(iupac_tokens, min_freq=IUPAC_MIN_FREQ,
                               extra_tokens=TRACE_TOKENS)

    print(f"\nSMILES vocab size: {len(smiles_vocab)}")
    print(f"IUPAC vocab size: {len(iupac_vocab)}")

    os.makedirs(os.path.dirname(SMILES_VOCAB_PATH), exist_ok=True)
    with open(SMILES_VOCAB_PATH, "w") as f:
        json.dump(smiles_vocab, f, indent=2)
    with open(IUPAC_VOCAB_PATH, "w") as f:
        json.dump(iupac_vocab, f, indent=2)

    print(f"\nVocabularies saved to {SMILES_VOCAB_PATH} and {IUPAC_VOCAB_PATH}")

if __name__ == "__main__":
    main()
