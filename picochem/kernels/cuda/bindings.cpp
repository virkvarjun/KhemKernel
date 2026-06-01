#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>   // std::vector <-> list (DeviceTensor.shape)
#include <stdexcept>
#include <string>
#include <cstring>

#include "vector_add.h"
#include "matmul_naive.h"
#include "matmul_tiled.h"
#include "matmul_backward.h"
#include "softmax.h"
#include "softmax_backward.h"
#include "layer_norm.h"
#include "layer_norm_backward.h"
#include "gelu.h"
#include "cross_entropy.h"
#include "embedding.h"
#include "adam.h"
#include "bias.h"
#include "batched_matmul.h"
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "common.h"   // CUDA_CHECK

namespace py = pybind11;

// float32 C-contiguous array type (forcecast handles contiguity automatically).
using f32arr = py::array_t<float, py::array::c_style | py::array::forcecast>;
// int32 C-contiguous array type (forcecast casts int64 numpy arrays down to int).
using i32arr = py::array_t<int, py::array::c_style | py::array::forcecast>;

// ── DeviceTensor: a float32 buffer that lives on the GPU ─────────────────────
// The building block of the device-resident path: arrays stay on the GPU across
// many ops; we only copy in (construct from numpy) and out (.numpy()).
struct DeviceTensor {
    float* d = nullptr;
    std::vector<py::ssize_t> shape;
    size_t n = 0;

    explicit DeviceTensor(f32arr a){
        auto info = a.request();
        shape.assign(info.shape.begin(), info.shape.end());
        n = static_cast<size_t>(a.size());
        CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
    explicit DeviceTensor(std::vector<py::ssize_t> shp){
        shape = std::move(shp);
        n = 1; for (auto s : shape) n *= static_cast<size_t>(s);
        CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    }
    ~DeviceTensor(){ if (d) cudaFree(d); }
    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    py::array_t<float> numpy() const {
        py::array_t<float> out(shape);
        CUDA_CHECK(cudaMemcpy(out.mutable_data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
        return out;
    }
    std::vector<py::ssize_t> get_shape() const { return shape; }
};

// Device-resident matmul: C(M,N) = A(M,K) @ B(K,N), all on the GPU, no copies.
DeviceTensor* dt_matmul(const DeviceTensor& A, const DeviceTensor& B){
    if (A.shape.size() != 2 || B.shape.size() != 2)
        throw std::runtime_error("dt_matmul: both operands must be 2-D");
    int M = static_cast<int>(A.shape[0]);
    int K = static_cast<int>(A.shape[1]);
    int N = static_cast<int>(B.shape[1]);
    if (static_cast<int>(B.shape[0]) != K)
        throw std::runtime_error("dt_matmul: inner dimension mismatch");
    auto* C = new DeviceTensor(std::vector<py::ssize_t>{M, N});
    launch_matmul_tiled_device(A.d, B.d, C->d, M, N, K);
    return C;
}

// Device-resident Linear backward: grad_x = grad_y @ Wᵀ.
// dC = grad_y (M,N), B = W (K,N) -> dA = grad_x (M,K).
DeviceTensor* dt_matmul_dA(const DeviceTensor& dC, const DeviceTensor& B){
    if (dC.shape.size() != 2 || B.shape.size() != 2)
        throw std::runtime_error("dt_matmul_dA: operands must be 2-D");
    int M = static_cast<int>(dC.shape[0]);
    int N = static_cast<int>(dC.shape[1]);
    int K = static_cast<int>(B.shape[0]);
    if (static_cast<int>(B.shape[1]) != N)
        throw std::runtime_error("dt_matmul_dA: B.shape[1] must equal dC.shape[1]");
    auto* dA = new DeviceTensor(std::vector<py::ssize_t>{M, K});
    launch_matmul_dA_device(dC.d, B.d, dA->d, M, N, K);
    return dA;
}

// Device-resident Linear backward: grad_W = xᵀ @ grad_y.
// A = x (M,K), dC = grad_y (M,N) -> dB = grad_W (K,N).
DeviceTensor* dt_matmul_dB(const DeviceTensor& A, const DeviceTensor& dC){
    if (A.shape.size() != 2 || dC.shape.size() != 2)
        throw std::runtime_error("dt_matmul_dB: operands must be 2-D");
    int M = static_cast<int>(A.shape[0]);
    int K = static_cast<int>(A.shape[1]);
    int N = static_cast<int>(dC.shape[1]);
    if (static_cast<int>(dC.shape[0]) != M)
        throw std::runtime_error("dt_matmul_dB: dC.shape[0] must equal A.shape[0]");
    auto* dB = new DeviceTensor(std::vector<py::ssize_t>{K, N});
    launch_matmul_dB_device(A.d, dC.d, dB->d, M, N, K);
    return dB;
}

// Element-wise add of two equally-shaped device tensors.
DeviceTensor* dt_add(const DeviceTensor& a, const DeviceTensor& b){
    if (a.n != b.n)
        throw std::runtime_error("dt_add: size mismatch");
    auto* out = new DeviceTensor(a.shape);
    launch_vector_add_device(a.d, b.d, out->d, static_cast<int>(a.n));
    return out;
}

// GeLU forward / backward on resident tensors (same shape out).
DeviceTensor* dt_gelu_forward(const DeviceTensor& x){
    auto* out = new DeviceTensor(x.shape);
    launch_gelu_forward_device(x.d, out->d, static_cast<int>(x.n));
    return out;
}

DeviceTensor* dt_gelu_backward(const DeviceTensor& grad_y, const DeviceTensor& x){
    if (grad_y.n != x.n)
        throw std::runtime_error("dt_gelu_backward: size mismatch");
    auto* out = new DeviceTensor(x.shape);
    launch_gelu_backward_device(grad_y.d, x.d, out->d, static_cast<int>(x.n));
    return out;
}

// Broadcast bias add: x(M,N) + b(N) -> (M,N).
DeviceTensor* dt_add_bias(const DeviceTensor& x, const DeviceTensor& b){
    if (x.shape.size() != 2) throw std::runtime_error("dt_add_bias: x must be 2-D");
    int M = static_cast<int>(x.shape[0]);
    int N = static_cast<int>(x.shape[1]);
    if (static_cast<int>(b.n) != N)
        throw std::runtime_error("dt_add_bias: bias length must equal x.shape[1]");
    auto* out = new DeviceTensor(x.shape);
    launch_add_bias_device(x.d, b.d, out->d, M, N);
    return out;
}

// Column sum (bias gradient): x(M,N) -> (N,).
DeviceTensor* dt_colsum(const DeviceTensor& x){
    if (x.shape.size() != 2) throw std::runtime_error("dt_colsum: x must be 2-D");
    int M = static_cast<int>(x.shape[0]);
    int N = static_cast<int>(x.shape[1]);
    auto* out = new DeviceTensor(std::vector<py::ssize_t>{N});
    launch_colsum_device(x.d, out->d, M, N);
    return out;
}

// Batched matmul with optional transpose. A,B are 3-D (batch, ., .).
// Result C = (batch, Mr, Nr); Mr/Kc/Nr derived from shapes + the transpose flags.
DeviceTensor* dt_bmm(const DeviceTensor& A, const DeviceTensor& B,
                     bool transA, bool transB){
    if (A.shape.size() != 3 || B.shape.size() != 3)
        throw std::runtime_error("dt_bmm: operands must be 3-D (batch, ., .)");
    int batch = static_cast<int>(A.shape[0]);
    if (static_cast<int>(B.shape[0]) != batch)
        throw std::runtime_error("dt_bmm: batch mismatch");
    int Mr = static_cast<int>(transA ? A.shape[2] : A.shape[1]);
    int Kc = static_cast<int>(transA ? A.shape[1] : A.shape[2]);
    int Bk = static_cast<int>(transB ? B.shape[2] : B.shape[1]);
    int Nr = static_cast<int>(transB ? B.shape[1] : B.shape[2]);
    if (Bk != Kc)
        throw std::runtime_error("dt_bmm: contraction dimension mismatch");
    auto* C = new DeviceTensor(std::vector<py::ssize_t>{batch, Mr, Nr});
    launch_bmm_device(A.d, B.d, C->d, batch, Mr, Nr, Kc, transA ? 1 : 0, transB ? 1 : 0);
    return C;
}

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

// Softmax cross-entropy forward. logits (M,V), targets (M,) int.
// Returns (loss, n_valid); pass n_valid back into the backward call.
py::tuple py_cross_entropy_forward(f32arr logits, i32arr targets, int ignore_index){
    require_ndim(logits, 2, "cross_entropy_forward logits");
    require_ndim(targets, 1, "cross_entropy_forward targets");
    int M = static_cast<int>(logits.shape(0));
    int V = static_cast<int>(logits.shape(1));
    if (static_cast<int>(targets.size()) != M)
        throw std::runtime_error("cross_entropy_forward: targets length must equal logits rows");
    float n_valid = 0.0f;
    float loss = launch_cross_entropy_forward(logits.data(), targets.data(),
                                              M, V, ignore_index, &n_valid);
    return py::make_tuple(loss, n_valid);
}

// Softmax cross-entropy backward -> grad_logits (M,V).
py::array_t<float> py_cross_entropy_backward(f32arr logits, i32arr targets,
                                             int ignore_index, float n_valid,
                                             float grad_loss){
    require_ndim(logits, 2, "cross_entropy_backward logits");
    require_ndim(targets, 1, "cross_entropy_backward targets");
    int M = static_cast<int>(logits.shape(0));
    int V = static_cast<int>(logits.shape(1));
    if (static_cast<int>(targets.size()) != M)
        throw std::runtime_error("cross_entropy_backward: targets length must equal logits rows");
    auto grad = py::array_t<float>({M, V});
    launch_cross_entropy_backward(logits.data(), targets.data(), grad.mutable_data(),
                                  M, V, ignore_index, n_valid, grad_loss);
    return grad;
}

// Embedding backward (scatter-add). grad_out (M,D), ids (M,) int -> grad_table (V,D).
py::array_t<float> py_embedding_backward(f32arr grad_out, i32arr ids, int V){
    require_ndim(grad_out, 2, "embedding_backward grad_out");
    require_ndim(ids, 1, "embedding_backward ids");
    int M = static_cast<int>(grad_out.shape(0));
    int D = static_cast<int>(grad_out.shape(1));
    if (static_cast<int>(ids.size()) != M)
        throw std::runtime_error("embedding_backward: ids length must equal grad_out rows");
    auto grad_table = py::array_t<float>({V, D});
    launch_embedding_backward(grad_out.data(), ids.data(), grad_table.mutable_data(), M, D, V);
    return grad_table;
}

// Adam update over a flat buffer. Returns updated (param, m, v) as new arrays;
// inputs are not mutated. step is 1-indexed (used for bias correction).
py::tuple py_adam_update(f32arr param, f32arr grad, f32arr m, f32arr v,
                         int step, float lr, float b1, float b2, float eps){
    int n = static_cast<int>(param.size());
    if (grad.size() != n || m.size() != n || v.size() != n)
        throw std::runtime_error("adam_update: param/grad/m/v must have equal size");
    auto p_out = py::array_t<float>(param.request().shape);
    auto m_out = py::array_t<float>(m.request().shape);
    auto v_out = py::array_t<float>(v.request().shape);
    std::memcpy(p_out.mutable_data(), param.data(), (size_t)n * sizeof(float));
    std::memcpy(m_out.mutable_data(), m.data(),     (size_t)n * sizeof(float));
    std::memcpy(v_out.mutable_data(), v.data(),     (size_t)n * sizeof(float));
    float bc1 = 1.0f - std::pow(b1, step);
    float bc2 = 1.0f - std::pow(b2, step);
    launch_adam_update(p_out.mutable_data(), grad.data(),
                       m_out.mutable_data(), v_out.mutable_data(),
                       n, lr, b1, b2, eps, bc1, bc2);
    return py::make_tuple(p_out, m_out, v_out);
}

PYBIND11_MODULE(picochem_cuda, m){
    m.doc() = "CUDA kernels for picochem (forward-pass only)";

    py::class_<DeviceTensor>(m, "DeviceTensor")
        .def(py::init<f32arr>(), "Upload a float32 numpy array to the GPU")
        .def("numpy", &DeviceTensor::numpy, "Download back to a numpy array")
        .def_property_readonly("shape", &DeviceTensor::get_shape);
    m.def("dt_matmul", &dt_matmul, py::return_value_policy::take_ownership,
          "Device-resident matmul: DeviceTensor(M,K) @ DeviceTensor(K,N) -> DeviceTensor(M,N)");
    m.def("dt_matmul_dA", &dt_matmul_dA, py::return_value_policy::take_ownership,
          "Device-resident Linear backward grad_x = grad_y @ Wᵀ");
    m.def("dt_matmul_dB", &dt_matmul_dB, py::return_value_policy::take_ownership,
          "Device-resident Linear backward grad_W = xᵀ @ grad_y");
    m.def("dt_add", &dt_add, py::return_value_policy::take_ownership,
          "Device-resident element-wise add");
    m.def("dt_gelu_forward", &dt_gelu_forward, py::return_value_policy::take_ownership,
          "Device-resident GeLU forward");
    m.def("dt_gelu_backward", &dt_gelu_backward, py::return_value_policy::take_ownership,
          "Device-resident GeLU backward");
    m.def("dt_add_bias", &dt_add_bias, py::return_value_policy::take_ownership,
          "Device-resident broadcast bias add x(M,N)+b(N)");
    m.def("dt_colsum", &dt_colsum, py::return_value_policy::take_ownership,
          "Device-resident column sum (Linear bias gradient)");
    m.def("dt_bmm", &dt_bmm, py::return_value_policy::take_ownership,
          py::arg("A"), py::arg("B"), py::arg("transA") = false, py::arg("transB") = false,
          "Device-resident batched matmul with optional per-operand transpose");

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
    m.def("cross_entropy_forward", &py_cross_entropy_forward,
          "Softmax cross-entropy forward -> (loss, n_valid) (logits f32, targets int)");
    m.def("cross_entropy_backward", &py_cross_entropy_backward,
          "Softmax cross-entropy backward -> grad_logits (float32)");
    m.def("embedding_backward", &py_embedding_backward,
          "Embedding backward (scatter-add) -> grad_table (V,D) (float32)");
    m.def("adam_update", &py_adam_update,
          "Adam update -> (param, m, v) updated (float32)");
}
