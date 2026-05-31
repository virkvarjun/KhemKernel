#include "common.h"
#include "matmul_backward.h"
#include <cmath>
#include <vector>

constexpr int TILE = 16;

// P(Mr,Nr) = X(Mr,Kc) @ Y(Nr,Kc)ᵀ
//   P[r,c] = Σ_k X[r,k] * Y[c,k]
// Both X and Y are row-major; Y is logically transposed by the indexing below.
__global__ void matmul_nt_kernel(const float* X, const float* Y, float* P,
                                 int Mr, int Nr, int Kc){
    __shared__ float sX[TILE][TILE];
    __shared__ float sY[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;   // over Mr
    int col = blockIdx.x * TILE + tx;   // over Nr
    float acc = 0.0f;

    int n_tile = (Kc + TILE - 1) / TILE;
    for (int t = 0; t < n_tile; ++t){
        int xk = t * TILE + tx;         // k index for X
        int yk = t * TILE + ty;         // k index for Y
        sX[ty][tx] = (row < Mr && xk < Kc) ? X[row * Kc + xk] : 0.0f;
        sY[ty][tx] = (col < Nr && yk < Kc) ? Y[col * Kc + yk] : 0.0f;
        __syncthreads();
        for (int kk = 0; kk < TILE; ++kk){
            acc += sX[ty][kk] * sY[kk][tx];
        }
        __syncthreads();
    }
    if (row < Mr && col < Nr){
        P[row * Nr + col] = acc;
    }
}

// P(Mr,Nr) = X(Kc,Mr)ᵀ @ Y(Kc,Nr)
//   P[r,c] = Σ_k X[k,r] * Y[k,c]
// Both X and Y are row-major; X is logically transposed by the indexing below.
__global__ void matmul_tn_kernel(const float* X, const float* Y, float* P,
                                 int Mr, int Nr, int Kc){
    __shared__ float sX[TILE][TILE];
    __shared__ float sY[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;   // over Mr
    int col = blockIdx.x * TILE + tx;   // over Nr
    float acc = 0.0f;

    int n_tile = (Kc + TILE - 1) / TILE;
    for (int t = 0; t < n_tile; ++t){
        int xk = t * TILE + tx;         // contraction index for X
        int yk = t * TILE + ty;         // contraction index for Y
        sX[ty][tx] = (row < Mr && xk < Kc) ? X[xk * Mr + row] : 0.0f;
        sY[ty][tx] = (col < Nr && yk < Kc) ? Y[yk * Nr + col] : 0.0f;
        __syncthreads();
        for (int kk = 0; kk < TILE; ++kk){
            acc += sX[ty][kk] * sY[kk][tx];
        }
        __syncthreads();
    }
    if (row < Mr && col < Nr){
        P[row * Nr + col] = acc;
    }
}

// ── host launchers ──────────────────────────────────────────────────────────

// dA(M,K) = dC(M,N) @ B(K,N)ᵀ  → matmul_nt with Mr=M, Nr=K, Kc=N
void launch_matmul_dA(const float* h_dC, const float* h_B, float* h_dA,
                      int M, int N, int K){
    size_t size_dC = (size_t)M * N * sizeof(float);
    size_t size_B  = (size_t)K * N * sizeof(float);
    size_t size_dA = (size_t)M * K * sizeof(float);

    float *d_dC, *d_B, *d_dA;
    CUDA_CHECK(cudaMalloc(&d_dC, size_dC));
    CUDA_CHECK(cudaMalloc(&d_B,  size_B));
    CUDA_CHECK(cudaMalloc(&d_dA, size_dA));
    CUDA_CHECK(cudaMemcpy(d_dC, h_dC, size_dC, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B,  h_B,  size_B,  cudaMemcpyHostToDevice));

    dim3 threads(TILE, TILE);
    dim3 blocks((K + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_nt_kernel<<<blocks, threads>>>(d_dC, d_B, d_dA, M, K, N);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_dA, d_dA, size_dA, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dC));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_dA));
}

// dB(K,N) = A(M,K)ᵀ @ dC(M,N)  → matmul_tn with Mr=K, Nr=N, Kc=M
void launch_matmul_dB(const float* h_A, const float* h_dC, float* h_dB,
                      int M, int N, int K){
    size_t size_A  = (size_t)M * K * sizeof(float);
    size_t size_dC = (size_t)M * N * sizeof(float);
    size_t size_dB = (size_t)K * N * sizeof(float);

    float *d_A, *d_dC, *d_dB;
    CUDA_CHECK(cudaMalloc(&d_A,  size_A));
    CUDA_CHECK(cudaMalloc(&d_dC, size_dC));
    CUDA_CHECK(cudaMalloc(&d_dB, size_dB));
    CUDA_CHECK(cudaMemcpy(d_A,  h_A,  size_A,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dC, h_dC, size_dC, cudaMemcpyHostToDevice));

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (K + TILE - 1) / TILE);
    matmul_tn_kernel<<<blocks, threads>>>(d_A, d_dC, d_dB, K, N, M);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_dB, d_dB, size_dB, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_dC));
    CUDA_CHECK(cudaFree(d_dB));
}

#ifdef BUILD_STANDALONE
int main(){
    // Small problem with a CPU reference for both dA and dB.
    const int M = 40, K = 24, N = 56;
    std::vector<float> A(M * K), B(K * N), dC(M * N);
    for (int i = 0; i < M * K; ++i) A[i]  = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; ++i) B[i]  = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    for (int i = 0; i < M * N; ++i) dC[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;

    // CPU reference: dA[m,k] = Σ_n dC[m,n] * B[k,n]
    std::vector<float> dA_ref(M * K, 0.0f), dA(M * K);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k){
            float s = 0.0f;
            for (int n = 0; n < N; ++n) s += dC[m * N + n] * B[k * N + n];
            dA_ref[m * K + k] = s;
        }

    // CPU reference: dB[k,n] = Σ_m A[m,k] * dC[m,n]
    std::vector<float> dB_ref(K * N, 0.0f), dB(K * N);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n){
            float s = 0.0f;
            for (int m = 0; m < M; ++m) s += A[m * K + k] * dC[m * N + n];
            dB_ref[k * N + n] = s;
        }

    launch_matmul_dA(dC.data(), B.data(), dA.data(), M, N, K);
    launch_matmul_dB(A.data(), dC.data(), dB.data(), M, N, K);

    float err_dA = 0.0f, err_dB = 0.0f;
    for (int i = 0; i < M * K; ++i) err_dA = std::max(err_dA, std::abs(dA[i] - dA_ref[i]));
    for (int i = 0; i < K * N; ++i) err_dB = std::max(err_dB, std::abs(dB[i] - dB_ref[i]));
    printf("matmul_backward: dA max error = %.6e, dB max error = %.6e\n", err_dA, err_dB);
    return 0;
}
#endif
