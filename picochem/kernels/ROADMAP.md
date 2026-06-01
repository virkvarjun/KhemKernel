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
| `matmul_backward` | `dA = dC·Bᵀ`, `dB = Aᵀ·dC` | ✅ verified |
| `gelu` (fwd+bwd)  | tanh-approx gelu and its derivative | ✅ verified |
| `layer_norm_backward` | `dx`, `dγ`, `dβ` (Bessel-corrected) | ✅ verified |
| `softmax_backward` | `dz = p ⊙ (dy − Σ(dy⊙p))` (pure softmax, for attention) | ✅ verified |
| `cross_entropy` (fwd+bwd) | NLL loss; `(softmax − onehot)/n_valid`, pad-masked | ✅ verified |
| `embedding_backward` | scatter-add grad rows by token id (`atomicAdd`) | ✅ verified |
| `adam_update` | in-place Adam step over a flat buffer | ✅ verified |

**Phase 1 verified** on an RTX PRO 4000 Blackwell (cc 12.0) with CUDA 12.4 +
driver 580. All standalone self-tests pass (max errors 1e-7…1e-10) and all 17
`tests/test_cuda_bindings.py` parity cases pass. Because the 12.4 toolkit can't
target `sm_120`, the build emits `compute_90` PTX and the driver JITs it onto the
GPU at runtime — `setup.py` now does this fallback automatically.

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
