#include "common.h"
#include "matmul_naive.h"
#include <cmath>
#include <vector>

__global__ void matmul_naive_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N){
        float acc = 0.0f;
        for (int k = 0; k < K; ++k){
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

void launch_matmul_naive(const float* h_A, const float* h_B, float* h_C,
                         int M, int N, int K){
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        (N + threads_per_block.x - 1) / threads_per_block.x,
        (M + threads_per_block.y - 1) / threads_per_block.y
    );
    matmul_naive_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

#ifdef BUILD_STANDALONE
int main(){
    const int M = 512, K = 256, N = 512;

    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);
    std::vector<float> h_expected(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j){
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += h_A[i * K + k] * h_B[k * N + j];
            h_expected[i * N + j] = acc;
        }

    launch_matmul_naive(h_A.data(), h_B.data(), h_C.data(), M, N, K);

    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i)
        max_err = std::max(max_err, std::abs(h_C[i] - h_expected[i]));
    printf("Naive matmul: max error = %.6e\n", max_err);
    return 0;
}
#endif
