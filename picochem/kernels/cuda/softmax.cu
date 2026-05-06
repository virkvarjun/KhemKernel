#include "common.h"
#include "softmax.h"
#include <cfloat>
#include <cmath>
#include <vector>

// 256 threads per block. Each block handles ONE row of the input matrix.
constexpr int THREADS = 256;

// For each row: max, then exp(x - max), then divide by sum.
__global__ void softmax_kernel(const float* x, float* out, int M, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float shared[THREADS];

    // 1. Find the Max of the Row
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += THREADS){
        float v = x[row * N + i];
        if (v > local_max) local_max = v;
    }
    shared[tid] = local_max;
    __syncthreads();

    // Tree reduction: combine the 256 values into 1
    for (int s = THREADS / 2; s > 0; s /= 2){
        if (tid < s && shared[tid + s] > shared[tid]) shared[tid] = shared[tid + s];
        __syncthreads();
    }
    float row_max = shared[0];

    // 2. Compute the sum of exp(x - max)
    float local_s = 0.0f;
    for (int i = tid; i < N; i += THREADS){
        local_s += expf(x[row * N + i] - row_max);
    }
    shared[tid] = local_s;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s /= 2){
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float row_sum = shared[0];

    // 3. Normalized output
    for (int i = tid; i < N; i += THREADS){
        out[row * N + i] = expf(x[row * N + i] - row_max) / row_sum;
    }
}

void launch_softmax(const float* h_x, float* h_out, int M, int N){
    float *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, M * N * sizeof(float), cudaMemcpyHostToDevice));

    softmax_kernel<<<M, THREADS>>>(d_x, d_out, M, N);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out));
}

#ifdef BUILD_STANDALONE
int main(){
    const int M = 64, N = 1000;
    std::vector<float> h_x(M * N), h_out(M * N), h_expected(M * N);

    for (int i = 0; i < M * N; ++i) h_x[i] = static_cast<float>(rand()) / RAND_MAX;

    // CPU reference (numerically stable softmax).
    for (int row = 0; row < M; ++row){
        float m = -FLT_MAX;
        for (int j = 0; j < N; ++j) m = std::max(m, h_x[row * N + j]);
        float s = 0.0f;
        for (int j = 0; j < N; ++j) s += std::exp(h_x[row * N + j] - m);
        for (int j = 0; j < N; ++j) h_expected[row * N + j] = std::exp(h_x[row * N + j] - m) / s;
    }

    launch_softmax(h_x.data(), h_out.data(), M, N);

    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i)
        max_err = std::max(max_err, std::abs(h_out[i] - h_expected[i]));
    printf("softmax: max error = %.6e\n", max_err);
    return 0;
}
#endif
