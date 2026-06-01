"""Build picochem_cuda.so: CUDA kernels exposed to Python via pybind11.

Usage (from repo root or this directory):
    python picochem/kernels/cuda/setup.py

Environment variables:
    CUDA_ARCH  GPU SM version, e.g. sm_120 (RTX 5090) or sm_89 (RTX 4090).
               Defaults to sm_120.
    CUDA_HOME  Path to CUDA installation. Defaults to /usr/local/cuda.
"""
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

try:
    import pybind11
except ImportError:
    sys.exit("pybind11 not found — run: pip install pybind11")


def _supported_virtual_archs():
    """Virtual archs the installed nvcc knows about, e.g. {50,...,90}."""
    try:
        out = subprocess.check_output(
            ["nvcc", "--list-gpu-arch"], stderr=subprocess.DEVNULL
        ).decode()
        return {int(tok.split("_")[1]) for tok in out.split() if tok.startswith("compute_")}
    except Exception:
        return set()


def detect_cuda_arch():
    """Pick an nvcc -arch for this GPU + toolkit.

    Native ``sm_NN`` when the toolkit supports the GPU's compute capability;
    otherwise fall back to the newest virtual ``compute_NN`` the toolkit knows
    (PTX only — the driver JITs it onto the newer GPU at runtime). This keeps
    builds working on, e.g., a Blackwell card (sm_120) with a CUDA 12.4 toolkit.
    """
    env_arch = os.environ.get("CUDA_ARCH")
    if env_arch:
        return env_arch
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().split("\n")[0]
        major, minor = out.strip().split(".")
        cc = int(f"{major}{minor}")
    except Exception:
        return "sm_120"

    supported = _supported_virtual_archs()
    if not supported or cc in supported:
        return f"sm_{cc}"  # toolkit can target the GPU natively
    newest = max(supported)
    print(f"warning: CUDA toolkit lacks sm_{cc}; building compute_{newest} PTX "
          f"(driver will JIT to sm_{cc} at runtime)")
    return f"compute_{newest}"


def run(cmd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd])


cuda_dir = Path(__file__).parent.resolve()
cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
cuda_arch = detect_cuda_arch()
ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
output = cuda_dir.parent / f"picochem_cuda{ext_suffix}"

print(f"CUDA arch  : {cuda_arch}")
print(f"CUDA home  : {cuda_home}")
print(f"Output     : {output}")

cu_sources = ["vector_add.cu", "matmul_naive.cu", "matmul_tiled.cu",
              "matmul_backward.cu", "softmax.cu", "softmax_backward.cu",
              "layer_norm.cu", "layer_norm_backward.cu", "gelu.cu",
              "cross_entropy.cu", "embedding.cu", "adam.cu"]
objects = []

for src in cu_sources:
    obj = cuda_dir / src.replace(".cu", ".o")
    run([
        "nvcc", "-O3", f"-arch={cuda_arch}", "-std=c++17",
        "-Xcompiler", "-fPIC",
        f"-I{cuda_dir}",
        "-c", cuda_dir / src, "-o", obj,
    ])
    objects.append(obj)

pybind11_include = pybind11.get_include()
python_include = sysconfig.get_path("include")

binding_obj = cuda_dir / "bindings.o"
run([
    "c++", "-O3", "-std=c++17", "-fPIC",
    f"-I{pybind11_include}",
    f"-I{python_include}",
    f"-I{cuda_dir}",
    f"-I{cuda_home / 'include'}",
    "-c", cuda_dir / "bindings.cpp", "-o", binding_obj,
])

run([
    "c++", "-shared", "-fPIC",
    binding_obj, *objects,
    f"-L{cuda_home / 'lib64'}", "-lcudart",
    "-Wl,-rpath," + str(cuda_home / "lib64"),
    "-o", output,
])

print(f"\nDone. Module written to: {output}")
print(f"Import: import sys; sys.path.insert(0, '{output.parent}'); import picochem_cuda")
