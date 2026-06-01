#include "common.h"
#include "layer_norm_backward.h"
#include <cmath>
#include <vector>

constexpr int THREADS = 256;

// grad_x: one block per row. Reduces Σ dxhat and Σ (dxhat·x_hat) over the row.
__global__ void ln_grad_x_kernel(const float* grad_y, const float* x_hat,
                                 const float* gamma, const float* inv_std,
                                 float* grad_x, int M, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float shared[THREADS];

    float l1 = 0.0f, l2 = 0.0f;
    for (int j = tid; j < N; j += THREADS){
        float dxh = grad_y[row * N + j] * gamma[j];
        l1 += dxh;
        l2 += dxh * x_hat[row * N + j];
    }

    // reduce Σ dxhat
    shared[tid] = l1; __syncthreads();
    for (int s = THREADS / 2; s > 0; s /= 2){ if (tid < s) shared[tid] += shared[tid + s]; __syncthreads(); }
    float sum1 = shared[0]; __syncthreads();

    // reduce Σ (dxhat·x_hat)
    shared[tid] = l2; __syncthreads();
    for (int s = THREADS / 2; s > 0; s /= 2){ if (tid < s) shared[tid] += shared[tid + s]; __syncthreads(); }
    float sum2 = shared[0]; __syncthreads();

    float istd = inv_std[row];
    float invN = 1.0f / N;
    for (int j = tid; j < N; j += THREADS){
        float dxh = grad_y[row * N + j] * gamma[j];
        grad_x[row * N + j] = invN * istd * (N * dxh - sum1 - x_hat[row * N + j] * sum2);
    }
}

// grad_gamma / grad_beta: one thread per column, reducing down the M rows.
__global__ void ln_grad_params_kernel(const float* grad_y, const float* x_hat,
                                      float* grad_gamma, float* grad_beta,
                                      int M, int N){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    float gg = 0.0f, gb = 0.0f;
    for (int m = 0; m < M; ++m){
        float gy = grad_y[m * N + col];
        gg += gy * x_hat[m * N + col];
        gb += gy;
    }
    grad_gamma[col] = gg;
    grad_beta[col]  = gb;
}

void launch_layer_norm_backward(const float* h_grad_y, const float* h_x_hat,
                                const float* h_gamma, const float* h_inv_std,
                                float* h_grad_x, float* h_grad_gamma, float* h_grad_beta,
                                int M, int N){
    size_t mat = (size_t)M * N * sizeof(float);
    float *d_gy, *d_xh, *d_g, *d_istd, *d_gx, *d_gg, *d_gb;
    CUDA_CHECK(cudaMalloc(&d_gy, mat));
    CUDA_CHECK(cudaMalloc(&d_xh, mat));
    CUDA_CHECK(cudaMalloc(&d_g,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_istd, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gx, mat));
    CUDA_CHECK(cudaMalloc(&d_gg, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gb, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_gy, h_grad_y, mat, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xh, h_x_hat, mat, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g,  h_gamma, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_istd, h_inv_std, M * sizeof(float), cudaMemcpyHostToDevice));

    ln_grad_x_kernel<<<M, THREADS>>>(d_gy, d_xh, d_g, d_istd, d_gx, M, N);
    CUDA_CHECK_KERNEL();
    ln_grad_params_kernel<<<(N + THREADS - 1) / THREADS, THREADS>>>(d_gy, d_xh, d_gg, d_gb, M, N);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_grad_x, d_gx, mat, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_gamma, d_gg, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_beta, d_gb, N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_gy)); CUDA_CHECK(cudaFree(d_xh)); CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_istd)); CUDA_CHECK(cudaFree(d_gx));
    CUDA_CHECK(cudaFree(d_gg)); CUDA_CHECK(cudaFree(d_gb));
}

// Device-resident: all pointers already on the GPU, no copies.
void launch_layer_norm_backward_device(const float* d_grad_y, const float* d_x_hat,
                                       const float* d_gamma, const float* d_inv_std,
                                       float* d_grad_x, float* d_grad_gamma, float* d_grad_beta,
                                       int M, int N){
    ln_grad_x_kernel<<<M, THREADS>>>(d_grad_y, d_x_hat, d_gamma, d_inv_std, d_grad_x, M, N);
    CUDA_CHECK_KERNEL();
    ln_grad_params_kernel<<<(N + THREADS - 1) / THREADS, THREADS>>>(d_grad_y, d_x_hat, d_grad_gamma, d_grad_beta, M, N);
    CUDA_CHECK_KERNEL();
}

#ifdef BUILD_STANDALONE
int main(){
    const int M = 24, N = 64;
    std::vector<float> x(M * N), gamma(N), grad_y(M * N);
    std::vector<float> x_hat(M * N), inv_std(M);
    for (int i = 0; i < M * N; ++i){ x[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f; grad_y[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f; }
    for (int j = 0; j < N; ++j) gamma[j] = static_cast<float>(rand()) / RAND_MAX + 0.5f;

    // Build x_hat / inv_std the way the forward pass does.
    const float eps = 1e-5f;
    for (int r = 0; r < M; ++r){
        float mean = 0.0f; for (int j = 0; j < N; ++j) mean += x[r * N + j]; mean /= N;
        float var = 0.0f; for (int j = 0; j < N; ++j){ float d = x[r * N + j] - mean; var += d * d; } var /= N;
        float istd = 1.0f / std::sqrt(var + eps); inv_std[r] = istd;
        for (int j = 0; j < N; ++j) x_hat[r * N + j] = (x[r * N + j] - mean) * istd;
    }

    // CPU reference.
    std::vector<float> gx_ref(M * N), gg_ref(N, 0.0f), gb_ref(N, 0.0f);
    for (int j = 0; j < N; ++j)
        for (int m = 0; m < M; ++m){ gg_ref[j] += grad_y[m * N + j] * x_hat[m * N + j]; gb_ref[j] += grad_y[m * N + j]; }
    for (int r = 0; r < M; ++r){
        float s1 = 0.0f, s2 = 0.0f;
        for (int j = 0; j < N; ++j){ float dxh = grad_y[r * N + j] * gamma[j]; s1 += dxh; s2 += dxh * x_hat[r * N + j]; }
        for (int j = 0; j < N; ++j){
            float dxh = grad_y[r * N + j] * gamma[j];
            gx_ref[r * N + j] = (1.0f / N) * inv_std[r] * (N * dxh - s1 - x_hat[r * N + j] * s2);
        }
    }

    std::vector<float> gx(M * N), gg(N), gb(N);
    launch_layer_norm_backward(grad_y.data(), x_hat.data(), gamma.data(), inv_std.data(),
                               gx.data(), gg.data(), gb.data(), M, N);
    float ex = 0.0f, eg = 0.0f, eb = 0.0f;
    for (int i = 0; i < M * N; ++i) ex = std::max(ex, std::abs(gx[i] - gx_ref[i]));
    for (int j = 0; j < N; ++j){ eg = std::max(eg, std::abs(gg[j] - gg_ref[j])); eb = std::max(eb, std::abs(gb[j] - gb_ref[j])); }
    printf("layer_norm_backward: grad_x err = %.6e, grad_gamma err = %.6e, grad_beta err = %.6e\n", ex, eg, eb);
    return 0;
}
#endif
