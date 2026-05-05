#include "common.h" 
#include <chrono> 
#include <cmath> 
#include <vector> 

// We are going to use a 16x16 tile here 
// Each tile: K dimension in the 16 wide chunk 

constexpr int TILE = 16; 

__global__ void matmul_tiled_kernel(const float* A, const float* B, const float* C, 
                                    int M, int N, int K){ 
    // Shared Memory: Two tiles, each is 16x16 floats = 1 KB 
    __shared__ float sA[TILE][TILE]; 
    __shared__ float sB[TILE][TILE]; 
    
    // tx, ty = my position within the block 
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    
    int row = blockIdx.y * TILE + ty; 
    int col = blockIdx.x * TILE + tx; 
    float acc = 0.0f; 
    // # tiles to process along K dim 
    int n_tile = (K + TILE - 1) / TILE; 
    for (int i = 0; i < n_tile; ++i){ 
        int a_col = t * TILE + tx; // which col of A to load 
        int b_row = t * TILE + ty; // which row of B to load 
        // Load A[row, a_col] into sA[ty, tx], or 0 if out of bounds.
        sA[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f; 
        // Load B[b_row, col] into sB[ty, tx] or 0.0 if it is out of the bounds 
        sB[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f; 
        __syncthreads(); // wait for all threads to have finished
        // compute the partial dot products 
        for (int k = 0; k < TILE; ++k){ 
            acc += sA[ty][k] * sB[k][tx]; 
        }
        __syncthreads(); 

    }
    if (row < M && col < N){ 
        C[row * N + col] = acc; 
    }
}
int main(){ 
    const int M = 1024, K = 1024, N = 1024;
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);

    for (int i = 0; i < M*K; ++i){ 
        h_A[i] = static_cast<float>(rand()) / RAND_MAX; 
    } 
    for (int i = 0; i < K*N; ++i){ 
        h_B[i] = static_cast<float>(rand()) / RAND_MAX; 
    } 
        float *d_A, *d_B, *d_C; 
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float))); 
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float))); 
        CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float))); 

        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M*K*sizeof(float)), cudaMemcpyHostToDevice); 
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K*N*sizeof(flaot)), cudaMemcpyHostToDevice); 
        
        dim3 threads(TILE, TILE);
        dim3 blocks((N+TILE-1) / TILE, (M+TILE-1) / TILE); 

        matmul_tiled_kernel <<<blcoks, threads>>>(d_A, d_B, d_C, M, N, K); 
        CUDA_CHECK_KERNEL(); 
    const int n_runs = 20;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_runs; ++i) {
        matmul_tiled_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_runs;

    // GFLOPS = 2 * M * N * K (each output element is K multiply-adds).
    double gflops = (2.0 * M * N * K) / (ms / 1000.0) / 1e9;
    printf("matmul_tiled (%dx%dx%d): %.2f ms, %.1f GFLOPS\n", M, K, N, ms, gflops);

    // Correctness check on a smaller problem (CPU reference is slow).
    const int Ms = 64, Ks = 64, Ns = 64;
    std::vector<float> sA(Ms * Ks), sB(Ks * Ns), sC(Ms * Ns), expected(Ms * Ns);
    for (int i = 0; i < Ms * Ks; ++i) sA[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < Ks * Ns; ++i) sB[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < Ms; ++i)
        for (int j = 0; j < Ns; ++j) {
            float a = 0.0f;
            for (int k = 0; k < Ks; ++k) a += sA[i * Ks + k] * sB[k * Ns + j];
            expected[i * Ns + j] = a;
        }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, Ms * Ks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, Ks * Ns * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, Ms * Ns * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, sA.data(), Ms * Ks * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, sB.data(), Ks * Ns * sizeof(float), cudaMemcpyHostToDevice));
    dim3 sblocks((Ns + TILE - 1) / TILE, (Ms + TILE - 1) / TILE);
    matmul_tiled_kernel<<<sblocks, threads>>>(dA, dB, dC, Ms, Ns, Ks);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(sC.data(), dC, Ms * Ns * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < Ms * Ns; ++i)
        max_err = std::max(max_err, std::abs(sC[i] - expected[i]));
    printf("Correctness: max error = %.6e\n", max_err);

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(dA));  CUDA_CHECK(cudaFree(dB));  CUDA_CHECK(cudaFree(dC));
    return 0;
}

