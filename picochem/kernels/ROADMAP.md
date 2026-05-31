# CUDA training-path roadmap

Goal: run the full picochem training step (forward **and** backward + optimizer)
on the GPU, fast enough to make GPU training worthwhile.

The hand-written kernels in `cuda/` currently cover only three *forward* ops
(`matmul_tiled`, `softmax`, `layer_norm`), each doing a full host→device→host
copy per call, with all backward passes running in NumPy. That is a benchmarking
artifact, not a training accelerator. This roadmap closes the gap.

## Verification workflow (must run on a CUDA GPU)

There is no GPU on the dev laptop, so kernels are verified on the pod:

```bash
# 1. Standalone per-kernel correctness self-tests (CPU reference, prints max error)
cd picochem/kernels && make            # builds cuda/<kernel> executables
./cuda/matmul_backward                 # → "max error = ..."

# 2. Build the Python extension
bash scripts/build_cuda.sh

# 3. Parity tests against the NumPy reference (skip automatically without a GPU)
pytest tests/test_cuda_bindings.py tests/test_backend_parity.py -v
```

Every new `.cu` carries a `#ifdef BUILD_STANDALONE main()` with a CPU reference,
and a matching pytest parity test. A kernel is "done" only after both pass on the
pod.

## Phase 1 — Backward kernels (correctness, copy-based)

Each op backprop needs a kernel. Math (forward → backward):

| Kernel | Backward computed | Status |
|---|---|---|
| `matmul_backward` | `dA = dC·Bᵀ`, `dB = Aᵀ·dC` | written, unverified |
| `gelu` (fwd+bwd)  | tanh-approx gelu and its derivative | written, unverified |
| `layer_norm_backward` | `dx`, `dγ`, `dβ` (Bessel-corrected) | written, unverified |
| `softmax_backward` | `dz = p ⊙ (dy − Σ(dy⊙p))` (pure softmax, for attention) | written, unverified |
| `cross_entropy` (fwd+bwd) | NLL loss; `(softmax − onehot)/n_valid`, pad-masked | written, unverified |
| `embedding_backward` | scatter-add grad rows by token id (`atomicAdd`) | written, unverified |
| `adam_update` | in-place Adam step over a flat buffer | written, unverified |

All Phase-1 kernels are written with standalone self-tests and pytest parity
tests, but **none are verified on hardware yet** — build and run them on the pod.

Exit criterion: `--backend cuda` runs full fwd+bwd and matches the NumPy path on
a tiny model within ~1e-3 (fp32).

## Phase 2 — Device-resident tensors (speed)

- Introduce a `DeviceTensor` handle (GPU pointer + shape + dtype) exposed to Python.
- Replace per-op `cudaMalloc`/`memcpy`/`free` with persistent device buffers;
  copy only batch inputs in and loss/metrics out.
- Route attention's `np.matmul` (Q·Kᵀ, weights·V) and the tied output projection
  through the backend so they stop falling back to CPU.

## Phase 3 — Wire & validate training

- `train_step` runs end-to-end on device.
- Finite-difference gradient-check the device path vs NumPy on a tiny config.
- Benchmark steps/sec vs the NumPy baseline; populate `scripts/benchmark_kernels.py`.
