#include "common.h"
#include "batched_matmul.h"
#include <cmath>
#include <vector>

constexpr int TILE = 16;

// blockIdx.z selects the batch element. Within a batch element this is the
// same tiled scheme as matmul_nt/tn, generalized to optional transpose on
// either operand. Per-batch strides: A has Mr*Kc elems, B has Kc*Nr, C Mr*Nr.
__global__ void bmm_kernel(const float* A, const float* B, float* C,
                           int Mr, int Nr, int Kc, int transA, int transB){
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int b  = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;   // over Mr
    int col = blockIdx.x * TILE + tx;   // over Nr

    const float* Ab = A + (size_t)b * Mr * Kc;
    const float* Bb = B + (size_t)b * Kc * Nr;
    float* Cb       = C + (size_t)b * Mr * Nr;

    float acc = 0.0f;
    int n_tile = (Kc + TILE - 1) / TILE;
    for (int t = 0; t < n_tile; ++t){
        int ka = t * TILE + tx;   // contraction index paired with row (A)
        int kb = t * TILE + ty;   // contraction index paired with col (B)
        // A element (row, ka): row-major (Mr,Kc) normally, or (Kc,Mr) if transA
        sA[ty][tx] = (row < Mr && ka < Kc)
            ? Ab[transA ? (ka * Mr + row) : (row * Kc + ka)] : 0.0f;
        // B element (kb, col): row-major (Kc,Nr) normally, or (Nr,Kc) if transB
        sB[ty][tx] = (col < Nr && kb < Kc)
            ? Bb[transB ? (col * Kc + kb) : (kb * Nr + col)] : 0.0f;
        __syncthreads();
        for (int kk = 0; kk < TILE; ++kk) acc += sA[ty][kk] * sB[kk][tx];
        __syncthreads();
    }
    if (row < Mr && col < Nr) Cb[row * Nr + col] = acc;
}

void launch_bmm_device(const float* d_A, const float* d_B, float* d_C,
                       int batch, int Mr, int Nr, int Kc, int transA, int transB){
    dim3 threads(TILE, TILE);
    dim3 blocks((Nr + TILE - 1) / TILE, (Mr + TILE - 1) / TILE, batch);
    bmm_kernel<<<blocks, threads>>>(d_A, d_B, d_C, Mr, Nr, Kc, transA, transB);
    CUDA_CHECK_KERNEL();
}

void launch_bmm(const float* h_A, const float* h_B, float* h_C,
                int batch, int Mr, int Nr, int Kc, int transA, int transB){
    size_t sA = (size_t)batch * Mr * Kc * sizeof(float);
    size_t sB = (size_t)batch * Kc * Nr * sizeof(float);
    size_t sC = (size_t)batch * Mr * Nr * sizeof(float);
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sA));
    CUDA_CHECK(cudaMalloc(&d_B, sB));
    CUDA_CHECK(cudaMalloc(&d_C, sC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sB, cudaMemcpyHostToDevice));
    launch_bmm_device(d_A, d_B, d_C, batch, Mr, Nr, Kc, transA, transB);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sC, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
}

#ifdef BUILD_STANDALONE
int main(){
    const int batch = 3, Mr = 20, Nr = 24, Kc = 16;
    // Test the nn case against a CPU reference: C[b] = A[b](Mr,Kc) @ B[b](Kc,Nr).
    std::vector<float> A(batch * Mr * Kc), B(batch * Kc * Nr), C(batch * Mr * Nr), ref(batch * Mr * Nr, 0.0f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    for (int b = 0; b < batch; ++b)
        for (int m = 0; m < Mr; ++m)
            for (int n = 0; n < Nr; ++n){
                float s = 0.0f;
                for (int k = 0; k < Kc; ++k) s += A[(b*Mr+m)*Kc+k] * B[(b*Kc+k)*Nr+n];
                ref[(b*Mr+m)*Nr+n] = s;
            }
    launch_bmm(A.data(), B.data(), C.data(), batch, Mr, Nr, Kc, 0, 0);
    float err = 0.0f;
    for (size_t i = 0; i < C.size(); ++i) err = std::max(err, std::abs(C[i] - ref[i]));
    printf("batched_matmul (nn): max error = %.6e\n", err);
    return 0;
}
#endif
