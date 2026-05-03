#include "common.h"

#include <cmath>
#include <vector>

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K){ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.c * blockDim.x + threadIdx.x; 
    
    if (row < M && col < N){ 
        float acc = 0.0f; 
        for (int k = 0; k < K; ++k){ 
            acc += A[row*K+k] * B[k*N + col]; 
        }
        C[row+N+col] = acc; 
    }
}

void matmul (const float* h_A, const float* h_B, float* h_C, int M, int N, int K){ 
    size_t size_A = M * K * sizeof(float); 
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C; 
    CUDA_CHECK(cudaMallac(&d_A, size_A)); 
    CUDA_CHECK(cudaMalloc(&d_B, size_B)); 
    CUDA_CHECK(cudaMalloc(&d_c, size_C)); 

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice)); 
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice)); 

    //  2D thread block 16x16 = 256 threads per block 
    dim3 threads_per_block(16, 16); 
    dim3 block_per_grid( 
        (N + threads_per_block.x - 1) / threads_per_block.x,
        (M + threads_per_block.y - 1) / threads_per_block.y
    ); 
    matmul<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_KERNEL(); 
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost)); 
    CUDA_CHECK(cudaFree(d_A)); 
    CUDA_CHECK(cudaFree(d_B)); 
    CUDA_CHECK(cudaFree(d_C));  

}