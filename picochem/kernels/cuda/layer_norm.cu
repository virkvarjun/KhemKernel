#include "common.h"
#include "layer_norm.h"
#include <cmath>
#include <vector>

constexpr int THREADS = 256;
constexpr float EPS = 1e-5f;

// Layer norm along the last axis of an (M, N) matrix
__global__ void layer_norm_kernel(const float* x, const float* gamma, const float* beta,
                                  float* out, int M, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float shared[THREADS];

    // Step 1: Compute the mean
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += THREADS){
        local_sum += x[row * N + i];
    }
    shared[tid] = local_sum;
    __syncthreads();

    // Tree reduction to combine into one sum
    for (int s = THREADS / 2; s > 0; s /= 2){
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float mean = shared[0] / N;

    // Variance
    float local_sq = 0.0f;
    for (int i = tid; i < N; i += THREADS){
        float d = x[row * N + i] - mean;
        local_sq += d * d;
    }
    shared[tid] = local_sq;
    __syncthreads();

    for (int s = THREADS / 2; s > 0; s /= 2){
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float inv_std = rsqrtf(shared[0] / N + EPS);
    for (int i = tid; i < N; i += THREADS){
        float norm = (x[row * N + i] - mean) * inv_std;
        out[row * N + i] = gamma[i] * norm + beta[i];
    }
}

void launch_layer_norm(const float* h_x, const float* h_gamma, const float* h_beta,
                       float* h_out, int M, int N){
    float *d_x, *d_g, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x,   M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x,     M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, h_gamma, N * sizeof(float),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_beta,  N * sizeof(float),     cudaMemcpyHostToDevice));

    layer_norm_kernel<<<M, THREADS>>>(d_x, d_g, d_b, d_out, M, N);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_out));
}

// Device-resident forward that also emits x_hat and inv_std for the backward
// pass (one block per row). Avoids recomputing statistics in backward.
__global__ void ln_fwd_cache_kernel(const float* x, const float* gamma, const float* beta,
                                    float* y, float* xhat, float* invstd, int M, int N){
    int row = blockIdx.x, tid = threadIdx.x;
    if (row >= M) return;
    __shared__ float sh[THREADS];

    float ls = 0.0f;
    for (int j = tid; j < N; j += THREADS) ls += x[row * N + j];
    sh[tid] = ls; __syncthreads();
    for (int s = THREADS / 2; s > 0; s /= 2){ if (tid < s) sh[tid] += sh[tid + s]; __syncthreads(); }
    float mean = sh[0] / N; __syncthreads();

    float lv = 0.0f;
    for (int j = tid; j < N; j += THREADS){ float d = x[row * N + j] - mean; lv += d * d; }
    sh[tid] = lv; __syncthreads();
    for (int s = THREADS / 2; s > 0; s /= 2){ if (tid < s) sh[tid] += sh[tid + s]; __syncthreads(); }
    float istd = rsqrtf(sh[0] / N + EPS);
    if (tid == 0) invstd[row] = istd;

    for (int j = tid; j < N; j += THREADS){
        float xh = (x[row * N + j] - mean) * istd;
        xhat[row * N + j] = xh;
        y[row * N + j] = gamma[j] * xh + beta[j];
    }
}

void launch_layer_norm_fwd_device(const float* d_x, const float* d_gamma, const float* d_beta,
                                  float* d_y, float* d_xhat, float* d_invstd, int M, int N){
    ln_fwd_cache_kernel<<<M, THREADS>>>(d_x, d_gamma, d_beta, d_y, d_xhat, d_invstd, M, N);
    CUDA_CHECK_KERNEL();
}

#ifdef BUILD_STANDALONE
int main(){
    const int M = 64, N = 768;
    std::vector<float> h_x(M * N), h_gamma(N), h_beta(N), h_out(M * N), h_expected(M * N);

    for (int i = 0; i < M * N; ++i) h_x[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    for (int i = 0; i < N; ++i) h_gamma[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < N; ++i) h_beta[i] = static_cast<float>(rand()) / RAND_MAX;

    // CPU reference.
    for (int row = 0; row < M; ++row){
        float s = 0.0f;
        for (int j = 0; j < N; ++j) s += h_x[row * N + j];
        float m = s / N;
        float sq = 0.0f;
        for (int j = 0; j < N; ++j){ float d = h_x[row * N + j] - m; sq += d * d; }
        float inv = 1.0f / std::sqrt(sq / N + EPS);
        for (int j = 0; j < N; ++j){
            float norm = (h_x[row * N + j] - m) * inv;
            h_expected[row * N + j] = h_gamma[j] * norm + h_beta[j];
        }
    }

    launch_layer_norm(h_x.data(), h_gamma.data(), h_beta.data(), h_out.data(), M, N);

    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i)
        max_err = std::max(max_err, std::abs(h_out[i] - h_expected[i]));
    printf("layer_norm: max error = %.6e\n", max_err);
    return 0;
}
#endif
