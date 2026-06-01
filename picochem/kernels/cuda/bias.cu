#include "common.h"
#include "bias.h"
#include <cmath>
#include <vector>

constexpr int THREADS = 256;

// out[i,j] = x[i,j] + b[j]
__global__ void add_bias_kernel(const float* x, const float* b, float* out, int M, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int j = idx % N;
    out[idx] = x[idx] + b[j];
}

// out[i] = x[i] * alpha
__global__ void scale_kernel(const float* x, float* out, float alpha, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * alpha;
}

// out[j] = Σ_i x[i,j]  (one thread per column)
__global__ void colsum_kernel(const float* x, float* out, int M, int N){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float s = 0.0f;
    for (int i = 0; i < M; ++i) s += x[i * N + j];
    out[j] = s;
}

// ── device-resident launchers ────────────────────────────────────────────────

void launch_add_bias_device(const float* d_x, const float* d_b, float* d_out, int M, int N){
    int total = M * N;
    add_bias_kernel<<<(total + THREADS - 1) / THREADS, THREADS>>>(d_x, d_b, d_out, M, N);
    CUDA_CHECK_KERNEL();
}

void launch_colsum_device(const float* d_x, float* d_out, int M, int N){
    colsum_kernel<<<(N + THREADS - 1) / THREADS, THREADS>>>(d_x, d_out, M, N);
    CUDA_CHECK_KERNEL();
}

void launch_scale_device(const float* d_x, float* d_out, float alpha, int n){
    scale_kernel<<<(n + THREADS - 1) / THREADS, THREADS>>>(d_x, d_out, alpha, n);
    CUDA_CHECK_KERNEL();
}

// ── host launchers (allocate + copy; used by the standalone self-test) ───────

void launch_add_bias(const float* h_x, const float* h_b, float* h_out, int M, int N){
    size_t mat = (size_t)M * N * sizeof(float);
    float *d_x, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, mat));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, mat));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, mat, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));
    launch_add_bias_device(d_x, d_b, d_out, M, N);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, mat, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_out));
}

void launch_colsum(const float* h_x, float* h_out, int M, int N){
    float *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, (size_t)M * N * sizeof(float), cudaMemcpyHostToDevice));
    launch_colsum_device(d_x, d_out, M, N);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_out));
}

#ifdef BUILD_STANDALONE
int main(){
    const int M = 40, N = 64;
    std::vector<float> x(M * N), b(N), out(M * N), cs(N);
    for (int i = 0; i < M * N; ++i) x[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    for (int j = 0; j < N; ++j) b[j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;

    launch_add_bias(x.data(), b.data(), out.data(), M, N);
    float eb = 0.0f;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            eb = std::max(eb, std::abs(out[i * N + j] - (x[i * N + j] + b[j])));

    launch_colsum(x.data(), cs.data(), M, N);
    float ec = 0.0f;
    for (int j = 0; j < N; ++j){
        float s = 0.0f; for (int i = 0; i < M; ++i) s += x[i * N + j];
        ec = std::max(ec, std::abs(cs[j] - s));
    }
    printf("bias: add_bias max error = %.6e, colsum max error = %.6e\n", eb, ec);
    return 0;
}
#endif
