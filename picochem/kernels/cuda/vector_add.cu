#include "common.h"
#include "vector_add.h"
#include <cmath>
#include <vector>

__global__ void vecAdd(const float* a, const float* b, float* c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] + b[i];
    }
}

void launch_vector_add(const float* h_a, const float* h_b, float* h_c, int N){
    const size_t bytes = N * sizeof(float);
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

#ifdef BUILD_STANDALONE
int main(){
    const int N = 1'000'000;

    std::vector<float> h_a(N), h_b(N), h_c(N), h_expected(N);
    for (int i = 0; i < N; ++i){
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
        h_expected[i] = h_a[i] + h_b[i];
    }

    launch_vector_add(h_a.data(), h_b.data(), h_c.data(), N);

    float max_err = 0.0f;
    for (int i = 0; i < N; ++i)
        max_err = std::max(max_err, std::abs(h_c[i] - h_expected[i]));
    printf("Vector add: max error = %.6e\n", max_err);
    return 0;
}
#endif
