#include "common.h" 
#include <cmath> 
#include <vector> 

__global__ void vecAdd(const float* a, const float* b, float* c){ 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < n){ 
        c[i] = a[i] + b[i]; 
    }
}

int main (){ 
    const int N = 1'000'000;
    const size_t bytes = N * sizeof(float);

    // Allocate memory 
    std::vector<float> h_a(N), h_b(N), h_c(N), h_expected(N); 
    for (int i = 0; i < N; ++i){ 
        h_a[i] = static_cast<float>(rand()) / RAND_MAX; 
        h_b[i] = static_cast<float>(rand()) / RAND_MAX; 
        h_expected[i] = h_a[i] + h_b[i]; 
    }

    // Allocate device memory 
    float *d_a, *d_b, *d_c; 
    CUDA_CHECK(cudaMalloc(&d_a, bytes)); 
    CUDA_CHECK(cudaMalloc(&d_b, bytes)); 
    CUDA_CHECK(cudaMalloc(&d_c, btyes)); 
    
    // Copy inpute from host to device 
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice)); 
    CUDA_CHECK(cudaMemcpy(d_a, h_b.data(), bytes, cudaMemcpyHostTo Device)); 
    
    // Laucnh the kernel 
    const int threads_per_block = 256; 
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block; 
    vector_add<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, N); 
    CUDA_CHECK_KERNEL(); 

    // Copy result back to the host 
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost)); 

     float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        max_err = std::max(max_err, std::abs(h_c[i] - h_expected[i]));
    }
    printf("Vector add: max error = %.6e\n", max_err);

    // --- Free device memory ---
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}