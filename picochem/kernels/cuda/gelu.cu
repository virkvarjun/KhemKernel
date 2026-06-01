#include "common.h"
#include "gelu.h"
#include <cmath>
#include <vector>

constexpr int   THREADS = 256;
constexpr float GELU_C  = 0.7978845608028654f;  // sqrt(2/pi)
constexpr float GELU_A  = 0.044715f;

__global__ void gelu_forward_kernel(const float* x, float* out, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float xi    = x[i];
    float inner = GELU_C * (xi + GELU_A * xi * xi * xi);
    out[i] = 0.5f * xi * (1.0f + tanhf(inner));
}

__global__ void gelu_backward_kernel(const float* grad_y, const float* x,
                                     float* grad_x, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float xi      = x[i];
    float inner   = GELU_C * (xi + GELU_A * xi * xi * xi);
    float th      = tanhf(inner);
    float sech_sq = 1.0f - th * th;
    float d_inner = GELU_C * (1.0f + 3.0f * GELU_A * xi * xi);
    float d_gelu  = 0.5f * (1.0f + th) + 0.5f * xi * sech_sq * d_inner;
    grad_x[i] = grad_y[i] * d_gelu;
}

void launch_gelu_forward(const float* h_x, float* h_out, int N){
    size_t bytes = (size_t)N * sizeof(float);
    float *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    gelu_forward_kernel<<<(N + THREADS - 1) / THREADS, THREADS>>>(d_x, d_out, N);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out));
}

void launch_gelu_backward(const float* h_grad_y, const float* h_x,
                          float* h_grad_x, int N){
    size_t bytes = (size_t)N * sizeof(float);
    float *d_gy, *d_x, *d_gx;
    CUDA_CHECK(cudaMalloc(&d_gy, bytes));
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_gx, bytes));
    CUDA_CHECK(cudaMemcpy(d_gy, h_grad_y, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    gelu_backward_kernel<<<(N + THREADS - 1) / THREADS, THREADS>>>(d_gy, d_x, d_gx, N);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_grad_x, d_gx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_gy));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_gx));
}

// ── device-resident launchers (pointers already on the GPU, no copies) ───────

void launch_gelu_forward_device(const float* d_x, float* d_out, int N){
    gelu_forward_kernel<<<(N + THREADS - 1) / THREADS, THREADS>>>(d_x, d_out, N);
    CUDA_CHECK_KERNEL();
}

void launch_gelu_backward_device(const float* d_grad_y, const float* d_x,
                                 float* d_grad_x, int N){
    gelu_backward_kernel<<<(N + THREADS - 1) / THREADS, THREADS>>>(d_grad_y, d_x, d_grad_x, N);
    CUDA_CHECK_KERNEL();
}

#ifdef BUILD_STANDALONE
int main(){
    const int N = 4096;
    std::vector<float> x(N), gy(N), out(N), gx(N), out_ref(N), gx_ref(N);
    for (int i = 0; i < N; ++i){
        x[i]  = static_cast<float>(rand()) / RAND_MAX * 6.0f - 3.0f;
        gy[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < N; ++i){
        float xi = x[i];
        float inner = GELU_C * (xi + GELU_A * xi * xi * xi);
        float th = std::tanh(inner);
        out_ref[i] = 0.5f * xi * (1.0f + th);
        float sech_sq = 1.0f - th * th;
        float d_inner = GELU_C * (1.0f + 3.0f * GELU_A * xi * xi);
        gx_ref[i] = gy[i] * (0.5f * (1.0f + th) + 0.5f * xi * sech_sq * d_inner);
    }
    launch_gelu_forward(x.data(), out.data(), N);
    launch_gelu_backward(gy.data(), x.data(), gx.data(), N);
    float ef = 0.0f, eb = 0.0f;
    for (int i = 0; i < N; ++i){
        ef = std::max(ef, std::abs(out[i] - out_ref[i]));
        eb = std::max(eb, std::abs(gx[i] - gx_ref[i]));
    }
    printf("gelu: forward max error = %.6e, backward max error = %.6e\n", ef, eb);
    return 0;
}
#endif
