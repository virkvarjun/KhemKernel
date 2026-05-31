#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>

#include "vector_add.h"
#include "matmul_naive.h"
#include "matmul_tiled.h"
#include "matmul_backward.h"
#include "softmax.h"
#include "softmax_backward.h"
#include "layer_norm.h"
#include "layer_norm_backward.h"
#include "gelu.h"

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

// Backward of C = A @ B.  Given dC and B, returns dA = dC @ Bᵀ, shape (M, K).
py::array_t<float> py_matmul_dA(f32arr dC, f32arr B){
    require_ndim(dC, 2, "matmul_dA dC");
    require_ndim(B,  2, "matmul_dA B");
    int M = static_cast<int>(dC.shape(0));
    int N = static_cast<int>(dC.shape(1));
    int K = static_cast<int>(B.shape(0));
    if (static_cast<int>(B.shape(1)) != N)
        throw std::runtime_error("matmul_dA: B.shape[1] must equal dC.shape[1]");
    auto dA = py::array_t<float>({M, K});
    launch_matmul_dA(dC.data(), B.data(), dA.mutable_data(), M, N, K);
    return dA;
}

// Backward of C = A @ B.  Given A and dC, returns dB = Aᵀ @ dC, shape (K, N).
py::array_t<float> py_matmul_dB(f32arr A, f32arr dC){
    require_ndim(A,  2, "matmul_dB A");
    require_ndim(dC, 2, "matmul_dB dC");
    int M = static_cast<int>(A.shape(0));
    int K = static_cast<int>(A.shape(1));
    int N = static_cast<int>(dC.shape(1));
    if (static_cast<int>(dC.shape(0)) != M)
        throw std::runtime_error("matmul_dB: dC.shape[0] must equal A.shape[0]");
    auto dB = py::array_t<float>({K, N});
    launch_matmul_dB(A.data(), dC.data(), dB.mutable_data(), M, N, K);
    return dB;
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

// GeLU forward, elementwise over any-shape input (returns same shape).
py::array_t<float> py_gelu_forward(f32arr x){
    int N = static_cast<int>(x.size());
    auto out = py::array_t<float>(x.size());
    launch_gelu_forward(x.data(), out.mutable_data(), N);
    out.resize(x.request().shape);
    return out;
}

// GeLU backward: grad_x = grad_y · dy/dx(x). Shapes must match.
py::array_t<float> py_gelu_backward(f32arr grad_y, f32arr x){
    if (grad_y.size() != x.size())
        throw std::runtime_error("gelu_backward: grad_y and x must have the same size");
    int N = static_cast<int>(x.size());
    auto grad_x = py::array_t<float>(x.size());
    launch_gelu_backward(grad_y.data(), x.data(), grad_x.mutable_data(), N);
    grad_x.resize(x.request().shape);
    return grad_x;
}

// Pure-softmax backward along the last axis (any leading dims).
py::array_t<float> py_softmax_backward(f32arr grad_out, f32arr probs){
    if (grad_out.size() != probs.size())
        throw std::runtime_error("softmax_backward: grad_out and probs must match");
    int N = static_cast<int>(probs.shape(probs.ndim() - 1));
    int M = static_cast<int>(probs.size()) / N;
    auto grad_in = py::array_t<float>(probs.size());
    launch_softmax_backward(grad_out.data(), probs.data(), grad_in.mutable_data(), M, N);
    grad_in.resize(probs.request().shape);
    return grad_in;
}

// Layer-norm backward. grad_y/x_hat: (..., N); gamma: (N,); inv_std: (M,) flattened.
// Returns (grad_x [same shape as grad_y], grad_gamma [N], grad_beta [N]).
py::tuple py_layer_norm_backward(f32arr grad_y, f32arr x_hat, f32arr gamma, f32arr inv_std){
    if (grad_y.size() != x_hat.size())
        throw std::runtime_error("layer_norm_backward: grad_y and x_hat must match");
    require_ndim(gamma, 1, "layer_norm_backward gamma");
    int N = static_cast<int>(x_hat.shape(x_hat.ndim() - 1));
    int M = static_cast<int>(x_hat.size()) / N;
    if (static_cast<int>(gamma.size()) != N)
        throw std::runtime_error("layer_norm_backward: gamma size must equal last dim");
    if (static_cast<int>(inv_std.size()) != M)
        throw std::runtime_error("layer_norm_backward: inv_std size must equal M (rows)");
    auto grad_x     = py::array_t<float>(x_hat.size());
    auto grad_gamma = py::array_t<float>(N);
    auto grad_beta  = py::array_t<float>(N);
    launch_layer_norm_backward(grad_y.data(), x_hat.data(), gamma.data(), inv_std.data(),
                               grad_x.mutable_data(), grad_gamma.mutable_data(),
                               grad_beta.mutable_data(), M, N);
    grad_x.resize(grad_y.request().shape);
    return py::make_tuple(grad_x, grad_gamma, grad_beta);
}

PYBIND11_MODULE(picochem_cuda, m){
    m.doc() = "CUDA kernels for picochem (forward-pass only)";
    m.def("vector_add",   &py_vector_add,   "Element-wise float32 vector addition");
    m.def("matmul_naive", &py_matmul_naive, "Naive CUDA matmul (float32, 2-D inputs)");
    m.def("matmul_tiled", &py_matmul_tiled, "Tiled CUDA matmul (float32, 2-D inputs)");
    m.def("matmul_dA",    &py_matmul_dA,    "Backward dA = dC @ Bᵀ for C = A @ B (float32)");
    m.def("matmul_dB",    &py_matmul_dB,    "Backward dB = Aᵀ @ dC for C = A @ B (float32)");
    m.def("softmax",      &py_softmax,      "Row-wise softmax along last axis (float32)");
    m.def("softmax_backward", &py_softmax_backward, "Pure-softmax backward along last axis (float32)");
    m.def("layer_norm",   &py_layer_norm,   "Layer norm along last axis (float32)");
    m.def("layer_norm_backward", &py_layer_norm_backward,
          "Layer-norm backward -> (grad_x, grad_gamma, grad_beta) (float32)");
    m.def("gelu_forward",  &py_gelu_forward,  "GeLU forward, tanh approximation (float32)");
    m.def("gelu_backward", &py_gelu_backward, "GeLU backward, grad_x = grad_y·dy/dx (float32)");
}
