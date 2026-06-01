#include "common.h"
#include "softmax_backward.h"
#include <cmath>
#include <vector>

constexpr int THREADS = 256;

// One block per row. dot = Σ_j grad_out·probs ; grad_in = probs·(grad_out − dot).
__global__ void softmax_backward_kernel(const float* grad_out, const float* probs,
                                        float* grad_in, int M, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float shared[THREADS];

    float local = 0.0f;
    for (int j = tid; j < N; j += THREADS){
        local += grad_out[row * N + j] * probs[row * N + j];
    }
    shared[tid] = local;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s /= 2){
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float dot = shared[0];

    for (int j = tid; j < N; j += THREADS){
        int idx = row * N + j;
        grad_in[idx] = probs[idx] * (grad_out[idx] - dot);
    }
}

void launch_softmax_backward(const float* h_grad_out, const float* h_probs,
                             float* h_grad_in, int M, int N){
    size_t bytes = (size_t)M * N * sizeof(float);
    float *d_go, *d_p, *d_gi;
    CUDA_CHECK(cudaMalloc(&d_go, bytes));
    CUDA_CHECK(cudaMalloc(&d_p, bytes));
    CUDA_CHECK(cudaMalloc(&d_gi, bytes));
    CUDA_CHECK(cudaMemcpy(d_go, h_grad_out, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p, h_probs, bytes, cudaMemcpyHostToDevice));
    softmax_backward_kernel<<<M, THREADS>>>(d_go, d_p, d_gi, M, N);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_grad_in, d_gi, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_go));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_gi));
}

// Device-resident: pointers already on the GPU, no copies.
void launch_softmax_backward_device(const float* d_grad_out, const float* d_probs,
                                    float* d_grad_in, int M, int N){
    softmax_backward_kernel<<<M, THREADS>>>(d_grad_out, d_probs, d_grad_in, M, N);
    CUDA_CHECK_KERNEL();
}

#ifdef BUILD_STANDALONE
int main(){
    const int M = 32, N = 100;
    std::vector<float> go(M * N), p(M * N), gi(M * N), gi_ref(M * N);
    for (int r = 0; r < M; ++r){
        // Build a valid probability row.
        float s = 0.0f;
        for (int j = 0; j < N; ++j){ p[r * N + j] = static_cast<float>(rand()) / RAND_MAX + 1e-3f; s += p[r * N + j]; }
        for (int j = 0; j < N; ++j) p[r * N + j] /= s;
        for (int j = 0; j < N; ++j) go[r * N + j] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    for (int r = 0; r < M; ++r){
        float dot = 0.0f;
        for (int j = 0; j < N; ++j) dot += go[r * N + j] * p[r * N + j];
        for (int j = 0; j < N; ++j) gi_ref[r * N + j] = p[r * N + j] * (go[r * N + j] - dot);
    }
    launch_softmax_backward(go.data(), p.data(), gi.data(), M, N);
    float err = 0.0f;
    for (int i = 0; i < M * N; ++i) err = std::max(err, std::abs(gi[i] - gi_ref[i]));
    printf("softmax_backward: max error = %.6e\n", err);
    return 0;
}
#endif
