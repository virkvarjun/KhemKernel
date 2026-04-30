# picochem

Encoder-decoder transformer from scratch in NumPy that translates between SMILES molecular structure strings and IUPAC chemical names.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`rdkit` installed via `pip install rdkit` (tested with rdkit==2026.3.1 on macOS arm64).

## Data

Download and filter 1M SMILES↔IUPAC pairs from PubChem (streams from HuggingFace, no full download):

```bash
python scripts/download_data.py
```

```
README.md: 3.28kB [00:00, 1.27MB/s]
Resolving data files: 100%|█| 124/124 [00:00<
Streamed 100,000 | Kept: 67,792
Streamed 200,000 | Kept: 135,266
Streamed 400,000 | Kept: 269,279
Streamed 500,000 | Kept: 336,906
Streamed 600,000 | Kept: 403,002
Streamed 800,000 | Kept: 538,171
Streamed 1,100,000 | Kept: 738,354
Streamed 1,200,000 | Kept: 805,799
Streamed 1,400,000 | Kept: 939,862

Final: 1,000,000 pairs saved to data/raw_pairs.parquet
Disk size: 54.2 MB
```

~1.4M rows streamed to collect 1M after filtering (no mixtures, SMILES ≤ 100 chars, IUPAC ≤ 100 chars). Output saved to `data/raw_pairs.parquet` (gitignored).
