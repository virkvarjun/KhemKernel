"""Load and batch traces.parquet for training."""
import numpy as np
import pandas as pd

from picochem.data import encode_smiles, encode_iupac


def load_dataset(parquet_path, smiles_vocab, iupac_vocab, max_src_len, max_tgt_len):
    """Load traces.parquet and encode all pairs, discarding those that exceed length limits.

    Parameters
    ----------
    parquet_path : str
        Path to a parquet file with columns ``smiles`` and ``trace``.
    smiles_vocab : dict[str, int]
        Token-to-index mapping for SMILES.
    iupac_vocab : dict[str, int]
        Token-to-index mapping for the target (traces / IUPAC).
    max_src_len : int
    max_tgt_len : int

    Returns
    -------
    list of (src_ids, tgt_ids) where each is a 1-D int32 numpy array.
    The tgt_ids include the ``<start>`` and ``<end>`` boundary tokens.
    """
    df = pd.read_parquet(parquet_path)

    # Support both raw_pairs.parquet (SMILES/IUPAC columns) and
    # traces.parquet (smiles/trace columns).
    if "smiles" in df.columns and "trace" in df.columns:
        src_col, tgt_col = "smiles", "trace"
    elif "SMILES" in df.columns and "IUPAC" in df.columns:
        src_col, tgt_col = "SMILES", "IUPAC"
    else:
        raise ValueError(
            f"Expected columns (smiles, trace) or (SMILES, IUPAC), "
            f"got {list(df.columns)}"
        )

    pairs = []
    for _, row in df.iterrows():
        src_ids = encode_smiles(str(row[src_col]), smiles_vocab)
        tgt_ids = encode_iupac(str(row[tgt_col]), iupac_vocab)
        if len(src_ids) > max_src_len or len(tgt_ids) > max_tgt_len:
            continue
        pairs.append((src_ids, tgt_ids))

    return pairs


def make_batch(pairs, batch_size, src_pad_id, tgt_pad_id, rng):
    """Sample and pad a batch of (src, tgt) pairs for teacher-forced training.

    Parameters
    ----------
    pairs : list of (src_ids, tgt_ids)
        Full encoded sequences; tgt_ids begin with ``<start>`` and end with ``<end>``.
    batch_size : int
    src_pad_id : int
    tgt_pad_id : int
    rng : np.random.Generator

    Returns
    -------
    src_ids  : (B, S) int32
    tgt_in   : (B, T) int32  — tgt[:-1], decoder input
    tgt_out  : (B, T) int32  — tgt[1:], what the model predicts;
                              pad positions are set to -1 (ignore_index)
    src_mask : (B, S) float64
    tgt_mask : (B, T) float64
    """
    batch_size = min(batch_size, len(pairs))
    idx = rng.choice(len(pairs), size=batch_size, replace=False)
    batch = [pairs[i] for i in idx]

    S = max(len(s) for s, _ in batch)
    T = max(len(t) for _, t in batch) - 1  # shifted sequences are one shorter

    src_ids = np.full((batch_size, S), src_pad_id, dtype=np.int32)
    tgt_in  = np.full((batch_size, T), tgt_pad_id, dtype=np.int32)
    tgt_out = np.full((batch_size, T), -1,          dtype=np.int32)
    src_mask = np.zeros((batch_size, S), dtype=np.float64)
    tgt_mask = np.zeros((batch_size, T), dtype=np.float64)

    for i, (src, tgt) in enumerate(batch):
        src_ids[i, :len(src)] = src
        src_mask[i, :len(src)] = 1.0

        t_in  = tgt[:-1]   # everything except <end>
        t_out = tgt[1:]    # everything except <start>
        tgt_in[i,  :len(t_in)]  = t_in
        tgt_mask[i, :len(t_in)] = 1.0
        tgt_out[i, :len(t_out)] = t_out  # pad positions remain -1

    return src_ids, tgt_in, tgt_out, src_mask, tgt_mask
