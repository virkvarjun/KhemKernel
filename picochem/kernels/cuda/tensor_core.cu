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
    return; 
}