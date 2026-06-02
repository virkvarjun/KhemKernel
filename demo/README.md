# ChemKernel demo

A tiny local web app to play with the trained model in both directions:

- **SMILES → IUPAC name** — runs *your trained transformer* (greedy decode, on CPU).
- **IUPAC name → SMILES** — runs OPSIN (`py2opsin`), the same reference parser the eval uses for scoring.

No web framework — just Python's stdlib `http.server` plus the model's own deps.

```
demo/
  server.py          # loads the checkpoint once, serves 2 JSON endpoints + the page
  static/index.html  # single-page UI
```

## Run it

From the repo root, with the project venv:

```bash
.venv/bin/python demo/server.py
# then open http://localhost:8000
```

That's it for the **SMILES → IUPAC** direction (only needs `numpy` + the model).

## Prerequisites

The server needs three things on disk. Two of them are **gitignored** (too large
for the repo), so on a fresh clone you regenerate them — it's fully deterministic:

| Need | Path | How to get it |
|------|------|---------------|
| Trained checkpoint | `runs/device_full/ckpt_latest.npz` | produced by training (`scripts/train_device.py`) |
| Vocab files | `data/smiles_vocab.json`, `data/iupac_vocab.json` | `python scripts/download_data.py && python scripts/build_vocab.py` |
| Java (for IUPAC→SMILES only) | system | `brew install openjdk` (macOS) + `uv pip install py2opsin` |

The vocab build streams the same fixed PubChem dataset and sorts tokens, so it
reproduces the exact vocab the model trained on — you can confirm the sizes come
out to **341** (SMILES) and **11902** (IUPAC), matching the checkpoint's config.

The IUPAC→SMILES card is optional: if Java/`py2opsin` aren't present, the
SMILES→IUPAC direction still works and the page shows OPSIN as unavailable.
The server auto-adds a Homebrew JDK (`/opt/homebrew/opt/openjdk/bin`) to `PATH`,
so you usually don't need to configure Java yourself.

## Config

Environment variables (all optional):

- `PORT` — listen port (default `8000`)
- `PICOCHEM_CKPT` — checkpoint path (default `runs/device_full/ckpt_latest.npz`)

## API

```bash
curl -X POST localhost:8000/api/smiles2iupac -d '{"smiles":"c1ccc(O)cc1"}'
# {"ok": true, "name": "phenol", "trace": "<parent>benzene</parent>...<name>phenol</name>"}

curl -X POST localhost:8000/api/iupac2smiles -d '{"name":"phenol"}'
# {"ok": true, "smiles": "C1(=CC=CC=C1)O"}

curl localhost:8000/api/health
# {"ok": true, "step": 100000, "opsin": true}
```
