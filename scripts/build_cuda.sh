#!/usr/bin/env bash
# Build the picochem_cuda Python extension module.
# Places picochem_cuda.so in picochem/kernels/ (importable from the repo root).
#
# Usage:
#   bash scripts/build_cuda.sh
#
# Environment variables:
#   CUDA_ARCH   GPU SM version (e.g. sm_120 for RTX 5090, sm_89 for RTX 4090).
#               Auto-detected from nvidia-smi if not set.
#   CUDA_HOME   CUDA install prefix (default /usr/local/cuda).
#   PYTHON      Python interpreter to use (default: python3 in PATH).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
CUDA_SETUP="$ROOT/picochem/kernels/cuda/setup.py"

PYTHON="${PYTHON:-python3}"

echo "=== Building picochem_cuda ==="
"$PYTHON" "$CUDA_SETUP"
echo "=== Build complete ==="
