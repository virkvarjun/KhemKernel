# ChemKernel demo + interactive guide

The frontend is now an interactive technical guide (Vite + React + TypeScript,
Computer Modern throughout) that explains the whole project from the chemistry
down to the CUDA kernels. It embeds the original live demo: a two-box panel that
runs the trained model in both directions.

- **SMILES to IUPAC name** runs the trained transformer. When OPSIN is available it beam decodes and keeps the candidate whose name parses back to your input molecule, so the answer is verified. Without OPSIN it falls back to a single greedy pass.
- **IUPAC name to SMILES** runs OPSIN, the same parser the eval uses for scoring, so you can check either direction.

The server has no web framework. It is Python's standard library `http.server`
plus the model's own dependencies, and it serves the built guide as static files.

```
demo/
  server.py        loads the checkpoint once, serves the JSON API + the built guide
  web/             the interactive guide (Vite + React); see web/README.md
  web/dist/        the built static site server.py serves (produced by `npm run build`)
```

## Run it

First build the guide once (Node 18+), then start the server from the repo root:

```bash
cd demo/web && npm install && npm run build && cd ../..

PATH="/opt/homebrew/opt/openjdk/bin:$PATH" \
PICOCHEM_CKPT="$(pwd)/runs/device_bpe_d512_v2/ckpt_latest.npz" \
PICOCHEM_IUPAC_BPE="$(pwd)/data/iupac_bpe_v2.json" \
.venv/bin/python demo/server.py
# open http://localhost:8000
```

The server now defaults `PICOCHEM_CKPT` to the BPE d512 checkpoint and
`PICOCHEM_IUPAC_BPE` to `data/iupac_bpe_v2.json` when those files exist, so on a
normal checkout the env vars above are optional. The SMILES to IUPAC direction
only needs NumPy and the model; the reverse direction and the verified reranking
need OPSIN, which needs Java.

For guide development with hot reload (and live inference proxied to the Python
backend), see `demo/web/README.md`.

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
