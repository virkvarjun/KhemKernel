#!/usr/bin/env bash
# One-shot environment setup for picochem (fresh box / RunPod pod).
#
# Creates a .venv, installs pinned dependencies, and installs picochem in
# editable mode. Idempotent: re-running reuses the existing venv.
#
# Usage:
#   bash scripts/setup_env.sh
#
# Requires Python >= 3.10 on PATH (the code uses `X | None` type unions).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

# Verify interpreter is >= 3.10.
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"Python >= 3.10 required, found {sys.version.split()[0]}. "
             "Set PYTHON=/path/to/python3.11 and re-run.")
print(f"Using Python {sys.version.split()[0]}")
PY

if [ ! -d .venv ]; then
    echo "Creating virtualenv at .venv"
    "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo
echo "=== Environment ready ==="
echo "Activate with:  source .venv/bin/activate"
echo
echo "Next steps:"
echo "  python scripts/download_data.py     # SMILES/IUPAC pairs -> data/raw_pairs.parquet"
echo "  python scripts/build_vocab.py        # tokenizers       -> data/*_vocab.json"
echo "  python scripts/generate_traces.py    # reasoning traces -> data/traces.parquet"
echo "  python scripts/train.py --data data/traces.parquet ..."
