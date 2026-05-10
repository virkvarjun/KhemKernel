#include "common.h"
#include <mma.h>
#include <cuda_fp16.h>

#include <chrono>
#include <cmath>
#include <vector>

using namespace nvcuda;

// WMMA tile size: each warp handles a 16x16x16 tile 
constexpr int WMMA_M = 16; 
constexpr int WMMA_N = 16; 
constexpr int WMMA_K = 16; 

// Each block will contain multiple warps - 4 warps per block (128 threads) 
// Each warp computes one 16x16 output tile q
// 4 Output Tiles (1x4)
constexpr int WARPS_PER_BLOCK = 4; 
constexpr int THREADS_PER_BLOCK = 32 * WARPS_PER_BLOCK // 128

// Compute C = A @ B where A is (M, K), B is (K, N), C is (M, N).
// All matrices are FP16 inputs, FP16 outputs, but accumulator is FP32.
__global__ void matmul_tensor_core_kernel(const half* A, const half*B, half* C, int M, int N, int K){ 
    int warp_id = threadIdx.x / 32; 
    int row_tile = blockIdx.y * WARPS_PER_BLOCK + warp_id; 
    int col_tile = blockIdx.x; 
     
    int row_start = row_tile * WAMMA_M; 
    int col_start = col_tile * WMMA_N; 
    
    // Out of bounds 
    if (row_start >= M || col_start >= N) return; 

    // Declare fragments 
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, half, wmma::row_major> a_frag; 
    wmma::fragment<wmma::matrix_b, WMMA_N, WMMA_K, half, wmma::row_mahor> b_frag; 
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag; 
    
    // Initialize accumulator to zero 
    wmma::fill_fragment(c_frag, 0.0f); 
    
    // Walk along the K dimensions in 16-element chunks
    for (int k = 0; k < K; k+= WMMA_K){ 
        const half* a_tile_ptr = A + row_start * K + k; 
        wmma::long_matrix_sync(a_frag, a_tile_ptr, K); 
        const half* b_tile_ptr = B + k * N + col_start; 
        wmma::load_matrix_sync(b_frag, b_tile_ptr, N); 
        // Tensor Core operation: c_frag += a_frag * b_frag 
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag); 
    }
    // Store the FP32 accumulator back to FP16 output
    // Temp FP32 store, then convert to FP16 
    // For now, a small per-thread loop I will use 
    __shared__ float temp[WARPS_PER_BLOCK][WMMA_M * WMMA_N]; 
    wmma::store_matrix_sync(temp[warp_id], c_frag, WMMA_N, wmma::mem_row_major); 

    // Convert the FP32 to FP16 and then write to global memory 
    int lane = threadIdx.x % 32; 
    for (int i = lane; i < WMMA_N*WMMA*N; i += 32){ 
        int local_row = i / WMMA_N;
        int local_col = i % WMMA_N; 
        int global_row = row_start + local_row; 
        int global_col = col_start + local_col; 
        if (global_row < M && global_col < N){ 
            C[global_row *N+global_col] = __float2half(temp[warp_id][i]); 
        }
    }
}