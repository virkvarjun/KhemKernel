#include <cublas_v2.h>

#include <chrono>
#include <cmath>
#include <vector>

// Wrapper for cuBLAS error checking.
#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n",                     \
                    __FILE__, __LINE__, status);                               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Compute C = A @ B using cuBLAS, where A is (M, K), B is (K, N), C is (M, N).
// Inputs and outputs are row-major (standard C convention).
void matmul_cublas(const float* h_A, const float* h_B, float* h_C, int M, int N, int K){ 
    float *d_A, *d_B, *d_C; 
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(float))); 
    CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(float))); 
    CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float))); 

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice)); 
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N*K*sizeof(float), cudaMemcpyHostToDevice)); 

    // cuBLAS handle
    cublasHandle_t handle; 
    CUBLAS_CHECK(cublasCreate(&handle)); 

    const float alpha = 1.0f; 
    const float beta = 0.0f; 

}