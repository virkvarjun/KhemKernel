# ChemKernel demo

A small local web app to try the trained model in both directions:

- **SMILES to IUPAC name** runs the trained transformer. When OPSIN is available it beam decodes and keeps the candidate whose name parses back to your input molecule, so the answer is verified. Without OPSIN it falls back to a single greedy pass.
- **IUPAC name to SMILES** runs OPSIN, the same parser the eval uses for scoring, so you can check either direction.

No web framework. It is Python's standard library `http.server` plus the model's own dependencies.

```
demo/
  server.py           loads the checkpoint once, serves the JSON endpoints and the pages
  static/index.html   the two-box UI
  static/writeup.html the technical writeup, linked from the main page
```

## Run it

From the repo root, pointing at the trained checkpoint and its tokenizer:

```bash
PATH="/opt/homebrew/opt/openjdk/bin:$PATH" \
PICOCHEM_CKPT="$(pwd)/runs/device_bpe_d512_v2/ckpt_latest.npz" \
PICOCHEM_IUPAC_BPE="$(pwd)/data/iupac_bpe_v2.json" \
.venv/bin/python demo/server.py
# open http://localhost:8000   (writeup at http://localhost:8000/writeup)
```

The SMILES to IUPAC direction only needs NumPy and the model. The reverse direction and the verified reranking need OPSIN, which needs Java.

## Prerequisites

The server needs a checkpoint, the matching IUPAC tokenizer, and the SMILES vocab. These are gitignored because they are large, and they regenerate deterministically. The tokenizer has to be the one the checkpoint was trained with, since the token ids index the trained embeddings.

| Need | Path | How to get it |
|---|---|---|
| Checkpoint | `runs/device_bpe_d512_v2/ckpt_latest.npz` | produced by `scripts/run_retrain.sh` |
| IUPAC tokenizer | `data/iupac_bpe_v2.json` | the byte pair tokenizer built during the same run (`scripts/build_bpe.py`) |
| SMILES vocab | `data/smiles_vocab.json` | `python scripts/download_data.py && python scripts/build_vocab.py` (comes out to 341 tokens) |
| Java (OPSIN) | system | `brew install openjdk` on macOS, plus `uv pip install py2opsin` |

`run_retrain.sh` produces a checkpoint and a tokenizer that match each other, so the simplest path on a fresh machine is to run that and point the server at its outputs. The server auto-adds a Homebrew JDK (`/opt/homebrew/opt/openjdk/bin`) to `PATH`, so on a Mac you usually do not need to configure Java yourself.

## Config

Environment variables, all optional:

- `PORT` listen port, default `8000`
- `PICOCHEM_CKPT` checkpoint path
- `PICOCHEM_IUPAC_BPE` byte pair tokenizer path; when unset the server uses the legacy word vocab at `data/iupac_vocab.json`
- `PICOCHEM_BEAM` beam width for the verified SMILES to IUPAC direction, default `20`. Lower it (for example `10` or `5`) for faster responses at slightly lower accuracy.

## API

```bash
curl -X POST localhost:8000/api/smiles2iupac -d '{"smiles":"c1ccc(O)cc1"}'
# {"ok": true, "name": "phenol", "verified": true, "opsin_smiles": "Oc1ccccc1", "decode": "beam20+rerank", "trace": "..."}

curl -X POST localhost:8000/api/iupac2smiles -d '{"name":"phenol"}'
# {"ok": true, "smiles": "C1(=CC=CC=C1)O"}

curl localhost:8000/api/health
# {"ok": true, "step": 120000, "opsin": true}
```
