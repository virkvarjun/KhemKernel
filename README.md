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
