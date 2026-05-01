# picochem

A chemistry reasoning model with explanation traces, built from scratch in NumPy — then accelerated with hand-written GPU kernels. Translates between SMILES molecular structure strings and IUPAC chemical names, producing structured reasoning traces that explain each step. Includes interpretability experiments (linear probing, attention faithfulness) and a GPU acceleration phase progressing from Numba through Triton to raw CUDA.

## Three pillars

1. **Reasoning traces** — structured chemistry explanations generated from (SMILES, IUPAC) pairs via RDKit, used as the model's training target
2. **Interpretability** — linear probing on encoder activations, attention-faithfulness experiments, and failure taxonomy
3. **GPU kernels** — NumPy CPU baseline → Numba CUDA → Triton → raw CUDA C++, benchmarked throughout

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`rdkit` installed via `pip install rdkit` (tested with rdkit==2026.3.1 on macOS arm64). GPU dependencies are listed but commented out in `requirements.txt` — install when entering the kernel phase.

## Data

Download and filter 1M SMILES↔IUPAC pairs from PubChem (streams from HuggingFace, no full download):

```bash
python scripts/download_data.py
```

```
Streamed 1,400,000 | Kept: 939,862
Final: 1,000,000 pairs saved to data/raw_pairs.parquet
Disk size: 54.2 MB
```

~1.4M rows streamed to collect 1M after filtering (no mixtures, SMILES ≤ 100 chars, IUPAC ≤ 100 chars).

## Tokenizers

```bash
python scripts/build_vocab.py
```

```
SMILES vocab size: 341
IUPAC vocab size: 11891
Vocabularies saved to data/smiles_vocab.json and data/iupac_vocab.json
```

SMILES uses the Schwaller et al. regex tokenizer (handles multi-char atoms like `[C@@H]`, `Cl`, `Br`). IUPAC is split on word boundaries, digits, and punctuation. Rare IUPAC tokens (< 5 occurrences) map to `<unk>`.
