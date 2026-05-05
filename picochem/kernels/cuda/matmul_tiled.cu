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
        h_B[i] = static_cast<float>(rand()) / RAND_MAX; 
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
        
    }
}
