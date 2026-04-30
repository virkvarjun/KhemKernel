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
