#include "common.h"
#include "matmul_tiled.h"
#include <chrono>
#include <cmath>
#include <vector>

constexpr int TILE = 16;

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K){
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    int n_tile = (K + TILE - 1) / TILE;
    for (int i = 0; i < n_tile; ++i){
        int a_col = i * TILE + tx;  // column of A to load
        int b_row = i * TILE + ty;  // row of B to load
        sA[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        sB[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k){
            acc += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N){
        C[row * N + col] = acc;
    }
}

void launch_matmul_tiled(const float* h_A, const float* h_B, float* h_C,
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

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

#ifdef BUILD_STANDALONE
int main(){
    const int M = 1024, K = 1024, N = 1024;
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_tiled_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_KERNEL();

    const int n_runs = 20;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_runs; ++i){
        matmul_tiled_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_runs;
    double gflops = (2.0 * M * N * K) / (ms / 1000.0) / 1e9;
    printf("matmul_tiled (%dx%dx%d): %.2f ms, %.1f GFLOPS\n", M, K, N, ms, gflops);

    // Correctness check on a smaller problem.
    const int Ms = 64, Ks = 64, Ns = 64;
    std::vector<float> sA(Ms * Ks), sB(Ks * Ns), sC(Ms * Ns), expected(Ms * Ns);
    for (int i = 0; i < Ms * Ks; ++i) sA[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < Ks * Ns; ++i) sB[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < Ms; ++i)
        for (int j = 0; j < Ns; ++j){
            float a = 0.0f;
            for (int k = 0; k < Ks; ++k) a += sA[i * Ks + k] * sB[k * Ns + j];
            expected[i * Ns + j] = a;
        }
    launch_matmul_tiled(sA.data(), sB.data(), sC.data(), Ms, Ns, Ks);

    float max_err = 0.0f;
    for (int i = 0; i < Ms * Ns; ++i)
        max_err = std::max(max_err, std::abs(sC[i] - expected[i]));
    printf("Correctness: max error = %.6e\n", max_err);

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    return 0;
}
#endif
