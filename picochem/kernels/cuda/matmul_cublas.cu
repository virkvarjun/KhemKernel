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

    // Swap A and B in he argument oder, swap M and N in the size arguments, and resukts is in our row-major C 
    CUBLAS_CHECK(cublasSgemm( 
       handle,
        CUBLAS_OP_N, CUBLAS_OP_N,    // no transpose for either (cuBLAS sees them as transposed already)
        N, M, K,                      // dimensions of result (in cuBLAS's column-major view)
        &alpha,
        d_B, N,                       // first matrix in cuBLAS view (= B^T in row-major view), leading dim = N
        d_A, K,                       // second matrix in cuBLAS view (= A^T in row-major view), leading dim = K
        &beta,
        d_C, N                        // output, leading dim = N
    )); 
    CUDA_CHECK(cudaMemcpy(h_C, d_c, M*N*sizeof(float), cudaMemcpyDeviceToHost)); 
    CUBLAS_CHECK(cublasDestroy(handle)); 
    CUDA_CHECK(cudaFree(d_A)); 
    CUDA_CHECK(cudaFree(d_B)); 
    CUDA_CHECK(cudaFree(d_C)); 
}
int main() {
    const int M = 1024, K = 1024, N = 1024;
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), expected(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // CPU reference on a smaller subset for correctness (full-size CPU is slow).
    const int Ms = 64, Ks = 64, Ns = 64;
    std::vector<float> sA(Ms * Ks), sB(Ks * Ns), sC(Ms * Ns), sExpected(Ms * Ns, 0.0f);
    for (int i = 0; i < Ms * Ks; ++i) sA[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < Ks * Ns; ++i) sB[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < Ms; ++i)
        for (int j = 0; j < Ns; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < Ks; ++k) acc += sA[i * Ks + k] * sB[k * Ns + j];
            sExpected[i * Ns + j] = acc;
        }

    matmul_cublas(sA.data(), sB.data(), sC.data(), Ms, Ns, Ks);

    float max_err = 0.0f;
    for (int i = 0; i < Ms * Ns; ++i)
        max_err = std::max(max_err, std::abs(sC[i] - sExpected[i]));
    printf("matmul_cublas correctness: max error = %.6e\n", max_err);

    // Benchmark on the larger problem.
    matmul_cublas(h_A.data(), h_B.data(), h_C.data(), M, N, K);  // warmup

    const int n_runs = 50;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_runs; ++i) {
        matmul_cublas(h_A.data(), h_B.data(), h_C.data(), M, N, K);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_runs;
    double gflops = (2.0 * M * N * K) / (ms / 1000.0) / 1e9;
    printf("matmul_cublas (%dx%dx%d): %.3f ms, %.1f GFLOPS\n", M, K, N, ms, gflops);

    return 0;
}