"""Run picochem.traces over the dataset to produce reasoning-augmented training data.

Output: data/traces.parquet
"""
"""Generate chemistry reasoning traces for the full dataset."""

import os

import pandas as pd
from tqdm import tqdm

from picochem.traces import build_trace


INPUT_PATH = "data/raw_pairs.parquet"
OUTPUT_PATH = "data/traces.parquet"


def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df):,} pairs")
    
    traces = []
    skipped = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building traces"):
        trace = build_trace(row["smiles"], row["iupac"])
        if trace is None:
            skipped += 1
            continue
        traces.append({
            "smiles": row["smiles"],
            "iupac": row["iupac"],
            "trace": trace,
        })
    
    out_df = pd.DataFrame(traces)
    out_df.to_parquet(OUTPUT_PATH, index=False)
    
    print(f"\nGenerated traces: {len(out_df):,}")
    print(f"Skipped (invalid SMILES): {skipped:,}")
    print(f"Saved to {OUTPUT_PATH}")
    print(f"Disk size: {os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()