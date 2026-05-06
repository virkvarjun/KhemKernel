#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>

#include "vector_add.h"
#include "matmul_naive.h"
#include "matmul_tiled.h"
#include "softmax.h"
#include "layer_norm.h"

namespace py = pybind11;

// float32 C-contiguous array type (forcecast handles contiguity automatically).
using f32arr = py::array_t<float, py::array::c_style | py::array::forcecast>;

static void require_ndim(const f32arr& a, int ndim, const char* name){
    if (static_cast<int>(a.ndim()) != ndim)
        throw std::runtime_error(
            std::string(name) + ": expected " + std::to_string(ndim) +
            "-D array, got " + std::to_string(a.ndim()) + "-D");
}

py::array_t<float> py_vector_add(f32arr a, f32arr b){
    require_ndim(a, 1, "vector_add a");
    require_ndim(b, 1, "vector_add b");
    if (a.size() != b.size())
        throw std::runtime_error("vector_add: size mismatch");
    int N = static_cast<int>(a.size());
    auto out = py::array_t<float>(N);
    launch_vector_add(a.data(), b.data(), out.mutable_data(), N);
    return out;
}

py::array_t<float> py_matmul_naive(f32arr A, f32arr B){
    require_ndim(A, 2, "matmul_naive A");
    require_ndim(B, 2, "matmul_naive B");
    int M = static_cast<int>(A.shape(0));
    int K = static_cast<int>(A.shape(1));
    int N = static_cast<int>(B.shape(1));
    if (static_cast<int>(B.shape(0)) != K)
        throw std::runtime_error("matmul_naive: inner dimension mismatch");
    auto C = py::array_t<float>({M, N});
    launch_matmul_naive(A.data(), B.data(), C.mutable_data(), M, N, K);
    return C;
}

py::array_t<float> py_matmul_tiled(f32arr A, f32arr B){
    require_ndim(A, 2, "matmul_tiled A");
    require_ndim(B, 2, "matmul_tiled B");
    int M = static_cast<int>(A.shape(0));
    int K = static_cast<int>(A.shape(1));
    int N = static_cast<int>(B.shape(1));
    if (static_cast<int>(B.shape(0)) != K)
        throw std::runtime_error("matmul_tiled: inner dimension mismatch");
    auto C = py::array_t<float>({M, N});
    launch_matmul_tiled(A.data(), B.data(), C.mutable_data(), M, N, K);
    return C;
}

// Softmax along the last axis of any shape array.
// Internally flattens to (M, N) where N = last dimension.
py::array_t<float> py_softmax(f32arr x){
    if (x.ndim() < 1)
        throw std::runtime_error("softmax: input must be at least 1-D");
    int N = static_cast<int>(x.shape(x.ndim() - 1));
    int M = static_cast<int>(x.size()) / N;
    auto out = py::array_t<float>(x.size());
    launch_softmax(x.data(), out.mutable_data(), M, N);
    // Reshape output to match input shape.
    out.resize(x.request().shape);
    return out;
}

// Layer norm along the last axis of any shape array.
// gamma and beta must be 1-D with length == x.shape[-1].
py::array_t<float> py_layer_norm(f32arr x, f32arr gamma, f32arr beta){
    if (x.ndim() < 1)
        throw std::runtime_error("layer_norm: input must be at least 1-D");
    require_ndim(gamma, 1, "layer_norm gamma");
    require_ndim(beta,  1, "layer_norm beta");
    int N = static_cast<int>(x.shape(x.ndim() - 1));
    int M = static_cast<int>(x.size()) / N;
    if (static_cast<int>(gamma.size()) != N || static_cast<int>(beta.size()) != N)
        throw std::runtime_error("layer_norm: gamma/beta size must equal last dim of x");
    auto out = py::array_t<float>(x.size());
    launch_layer_norm(x.data(), gamma.data(), beta.data(), out.mutable_data(), M, N);
    out.resize(x.request().shape);
    return out;
}

PYBIND11_MODULE(picochem_cuda, m){
    m.doc() = "CUDA kernels for picochem (forward-pass only)";
    m.def("vector_add",   &py_vector_add,   "Element-wise float32 vector addition");
    m.def("matmul_naive", &py_matmul_naive, "Naive CUDA matmul (float32, 2-D inputs)");
    m.def("matmul_tiled", &py_matmul_tiled, "Tiled CUDA matmul (float32, 2-D inputs)");
    m.def("softmax",      &py_softmax,      "Row-wise softmax along last axis (float32)");
    m.def("layer_norm",   &py_layer_norm,   "Layer norm along last axis (float32)");
}
