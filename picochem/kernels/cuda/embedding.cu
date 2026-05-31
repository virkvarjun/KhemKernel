#include "common.h"
#include "embedding.h"
#include <cmath>
#include <vector>

constexpr int THREADS = 256;

// One thread per (row, d) element. Multiple rows may target the same table row,
// so accumulation must use atomicAdd.
__global__ void embedding_backward_kernel(const float* grad_out, const int* ids,
                                          float* grad_table, int M, int D){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * D) return;
    int m = idx / D;
    int d = idx % D;
    atomicAdd(&grad_table[ids[m] * D + d], grad_out[m * D + d]);
}

void launch_embedding_backward(const float* h_grad_out, const int* h_ids,
                               float* h_grad_table, int M, int D, int V){
    size_t grad_bytes  = (size_t)M * D * sizeof(float);
    size_t table_bytes = (size_t)V * D * sizeof(float);
    float *d_grad_out, *d_table;
    int *d_ids;
    CUDA_CHECK(cudaMalloc(&d_grad_out, grad_bytes));
    CUDA_CHECK(cudaMalloc(&d_ids, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_table, table_bytes));
    CUDA_CHECK(cudaMemcpy(d_grad_out, h_grad_out, grad_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ids, h_ids, M * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_table, 0, table_bytes));

    int total = M * D;
    embedding_backward_kernel<<<(total + THREADS - 1) / THREADS, THREADS>>>(
        d_grad_out, d_ids, d_table, M, D);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_grad_table, d_table, table_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_grad_out)); CUDA_CHECK(cudaFree(d_ids)); CUDA_CHECK(cudaFree(d_table));
}

#ifdef BUILD_STANDALONE
int main(){
    const int M = 128, D = 32, V = 20;
    std::vector<float> grad_out(M * D);
    std::vector<int> ids(M);
    for (int i = 0; i < M * D; ++i) grad_out[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    for (int m = 0; m < M; ++m) ids[m] = rand() % V;  // repeats exercise atomics

    std::vector<float> ref(V * D, 0.0f), out(V * D);
    for (int m = 0; m < M; ++m)
        for (int d = 0; d < D; ++d)
            ref[ids[m] * D + d] += grad_out[m * D + d];

    launch_embedding_backward(grad_out.data(), ids.data(), out.data(), M, D, V);

    float err = 0.0f;
    for (int i = 0; i < V * D; ++i) err = std::max(err, std::abs(out[i] - ref[i]));
    printf("embedding_backward: max error = %.6e\n", err);
    return 0;
}
#endif
