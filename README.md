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

## How to run

### Fresh training run

```bash
python scripts/train.py \
    --data data/raw_pairs.parquet \
    --d_model 256 --n_heads 4 --d_ff 1024 \
    --n_enc_layers 3 --n_dec_layers 3 \
    --total_steps 100000 --batch_size 32 \
    --peak_lr 3e-4 --warmup_steps 2000 --schedule cosine \
    --checkpoint_every 500 --val_every 500 --eval_every 5000
```

Each run creates a timestamped directory under `runs/`, containing:
- `ckpt_latest.npz` — overwritten every `--checkpoint_every` steps; use this to resume after a crash
- `ckpt_best.npz` — saved whenever validation loss improves
- `ckpt_NNNNNNN.npz` — milestone snapshots every `--keep_checkpoint_every` steps (default 5000)
- `log.jsonl` — one JSON line per logged step (train loss, val loss, LR, OPSIN eval)
- `loss_curve.png` / `training_progress.png` — 2-panel training plot (loss + structure match rate)

> **Expected timeline:** On CPU (NumPy), expect ~2–5 seconds per step with batch size 32 and `d_model=256`. A full 100k-step run takes roughly 60–120 hours. Checkpoint 11 onwards moves to GPU, which will reduce this by ~100×.

### Resume after a crash

```bash
python scripts/resume_training.py
```

Automatically finds the most recently modified run in `runs/`, reads its `run_args.json`, and re-launches `scripts/train.py` with `--resume_from` pointing at `ckpt_latest.npz`. Additional arguments appended on the command line override saved args:

```bash
python scripts/resume_training.py --total_steps 200000
```

### Evaluate a checkpoint

```bash
python scripts/evaluate.py \
    --checkpoint runs/run_<timestamp>/ckpt_best.npz \
    --n_samples 500
```

Prints:

```
─────────────────────────────────────────────
  Step:              12500
  N evaluated:       500
  Trace validity:     83.4%  (417 / 500)
  OPSIN parse:        61.2%  (306 / 500)
  Structure match:    38.8%  (194 / 500)
─────────────────────────────────────────────
```

Requires Java 8+ for OPSIN. Install with `pip install py2opsin` then ensure `java` is on `$PATH`. Without Java, only trace validity rate is reported.

### Sample generated traces

```bash
python scripts/sample_during_training.py \
    --checkpoint runs/run_<timestamp>/ckpt_best.npz
```

Decodes 5 fixed SMILES strings and prints the generated reasoning traces to stdout.

---

## Ops (`picochem/ops.py`)

All forward and backward passes are implemented from scratch in NumPy. Each op returns a cache for the backward pass; gradients are verified against finite differences via pytest.

| Op | Forward | Backward |
|---|---|---|
| Linear | `y = xW + b` | gradients w.r.t. `x`, `W`, `b` |
| GeLU | tanh approximation (Hendrycks & Gimpel) | analytic derivative via chain rule |
| Softmax + cross-entropy | numerically stable log-softmax; masks `<pad>` tokens via `ignore_index` | gradient w.r.t. logits only; integer targets have no gradient |
| Layer norm | normalizes over last axis; learnable `γ`, `β` | full Bessel-corrected backward; reduces `grad_γ`, `grad_β` over batch dims |

## Attention (`picochem/attention.py`)

Three attention primitives, all gradient-checked against finite differences.

| Function | Description |
|---|---|
| `scaled_dot_product_attention_forward/backward` | Core `QKᵀ/√Dh` attention; supports additive mask (causal or padding) |
| `multihead_self_attention_forward/backward` | Full MHA: Q/K/V projections → split heads → SDPA → concat → output projection; Q, K, V all come from the same input `x` |
| `multihead_cross_attention_forward/backward` | Cross-attention: Q from decoder `x_dec`, K/V from encoder `x_enc`; gradients flow back to both sequences independently |

All three ops use the same shape convention: `(B, H, S, Dh)` inside the attention kernel, with head-split and head-merge transposes handled in the multihead wrappers. Caches store only what is needed for the backward pass (no redundant copies).
