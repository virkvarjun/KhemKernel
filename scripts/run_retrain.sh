#!/usr/bin/env bash
# Full retrain pipeline for a fresh RunPod GPU pod: build CUDA, fetch data,
# build the BPE tokenizer, and train the scaled (d_model 512) model with a
# warmup->cosine LR schedule on the BPE target.
#
# Usage (on the pod, from the repo root):
#   bash scripts/run_retrain.sh
#
# Tunables via env, e.g.:  STEPS=150000 DMODEL=512 bash scripts/run_retrain.sh
set -uo pipefail
cd "$(dirname "$0")/.."

# Put CUDA on PATH so build_cuda's arch auto-detect works (and so a Blackwell
# sm_120 GPU on an older toolkit falls back to compute_90 PTX automatically).
export PATH="$(ls -d /usr/local/cuda*/bin 2>/dev/null | head -1):$PATH"

DMODEL=${DMODEL:-512}
HEADS=${HEADS:-8}
DFF=${DFF:-2048}
ENC=${ENC:-3}
DEC=${DEC:-3}
STEPS=${STEPS:-120000}
BATCH=${BATCH:-32}
LR=${LR:-3e-4}
WARMUP=${WARMUP:-1500}
MAXTGT=${MAXTGT:-64}
MAXSRC=${MAXSRC:-100}
VOCAB=${VOCAB:-4000}
TRAINLINES=${TRAINLINES:-80000}
RUNDIR=${RUNDIR:-runs/device_bpe_d${DMODEL}}

echo "=== [deps] datasets + rdkit ==="; date
pip install -q datasets rdkit py2opsin 2>&1 | tail -2

echo "=== [build] CUDA extension ==="; date
bash scripts/build_cuda.sh

echo "=== [1/4] download pairs (skip if present) ==="; date
[ -f data/raw_pairs.parquet ] || python3 scripts/download_data.py

echo "=== [2/4] generate traces (skip if present) ==="; date
[ -f data/traces.parquet ] || python3 scripts/generate_traces.py

echo "=== [3/4] build vocab: SMILES (build_vocab) + IUPAC BPE (skip if present) ==="; date
[ -f data/smiles_vocab.json ] || python3 scripts/build_vocab.py
[ -f data/iupac_bpe.json ] || python3 scripts/build_bpe.py --vocab_size "$VOCAB" --train_lines "$TRAINLINES"

echo "=== [smoke] tiny synthetic step to confirm kernels handle d_model=$DMODEL ==="; date
PYTHONPATH=picochem/kernels python3 scripts/train_device.py --synthetic \
  --d_model "$DMODEL" --n_heads "$HEADS" --d_ff "$DFF" \
  --n_enc_layers "$ENC" --n_dec_layers "$DEC" --total_steps 5 --batch_size 8 \
  --checkpoint_every 0 || { echo "SMOKE FAILED — aborting before full run"; exit 1; }

echo "=== [4/4] TRAIN (device-resident, BPE target, cosine LR) ==="; date
PYTHONPATH=picochem/kernels python3 scripts/train_device.py \
  --data data/traces.parquet --iupac_bpe data/iupac_bpe.json \
  --d_model "$DMODEL" --n_heads "$HEADS" --d_ff "$DFF" \
  --n_enc_layers "$ENC" --n_dec_layers "$DEC" \
  --max_src_len "$MAXSRC" --max_tgt_len "$MAXTGT" \
  --batch_size "$BATCH" --lr "$LR" --schedule cosine --warmup_steps "$WARMUP" \
  --total_steps "$STEPS" --log_every 50 --checkpoint_every 1000 \
  --run_dir "$RUNDIR"
echo "=== DONE ==="; date
echo "checkpoint: $RUNDIR/ckpt_latest.npz   tokenizer: data/iupac_bpe.json"
