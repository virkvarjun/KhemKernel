#include "common.h"
#include "cross_entropy.h"
#include <cfloat>
#include <cmath>
#include <vector>

constexpr int THREADS = 256;

// One block per row: stable row max, log-sum-exp, then the NLL of the target.
__global__ void ce_forward_kernel(const float* logits, const int* targets,
                                  float* nll_out, float* valid_out,
                                  int M, int V, int ignore_index){
    int row = blockIdx.x, tid = threadIdx.x;
    if (row >= M) return;
    __shared__ float sh[THREADS];

    float lmax = -FLT_MAX;
    for (int j = tid; j < V; j += THREADS){ float v = logits[row * V + j]; if (v > lmax) lmax = v; }
    sh[tid] = lmax; __syncthreads();
    for (int s = THREADS / 2; s > 0; s /= 2){ if (tid < s && sh[tid + s] > sh[tid]) sh[tid] = sh[tid + s]; __syncthreads(); }
    float rmax = sh[0]; __syncthreads();

    float lsum = 0.0f;
    for (int j = tid; j < V; j += THREADS) lsum += expf(logits[row * V + j] - rmax);
    sh[tid] = lsum; __syncthreads();
    for (int s = THREADS / 2; s > 0; s /= 2){ if (tid < s) sh[tid] += sh[tid + s]; __syncthreads(); }
    float rsum = sh[0];

    if (tid == 0){
        int t = targets[row];
        if (t == ignore_index){ nll_out[row] = 0.0f; valid_out[row] = 0.0f; }
        else {
            float logprob = (logits[row * V + t] - rmax) - logf(rsum);
            nll_out[row] = -logprob;
            valid_out[row] = 1.0f;
        }
    }
}

// One block per row: recompute softmax, write (p − onehot)·mask/n_valid·grad_loss.
__global__ void ce_backward_kernel(const float* logits, const int* targets,
                                   float* grad, int M, int V,
                                   int ignore_index, float n_valid, float grad_loss){
    int row = blockIdx.x, tid = threadIdx.x;
    if (row >= M) return;
    __shared__ float sh[THREADS];

    float lmax = -FLT_MAX;
    for (int j = tid; j < V; j += THREADS){ float v = logits[row * V + j]; if (v > lmax) lmax = v; }
    sh[tid] = lmax; __syncthreads();
    for (int s = THREADS / 2; s > 0; s /= 2){ if (tid < s && sh[tid + s] > sh[tid]) sh[tid] = sh[tid + s]; __syncthreads(); }
    float rmax = sh[0]; __syncthreads();

    float lsum = 0.0f;
    for (int j = tid; j < V; j += THREADS) lsum += expf(logits[row * V + j] - rmax);
    sh[tid] = lsum; __syncthreads();
    for (int s = THREADS / 2; s > 0; s /= 2){ if (tid < s) sh[tid] += sh[tid + s]; __syncthreads(); }
    float rsum = sh[0];

    int t = targets[row];
    float scale = ((t == ignore_index) ? 0.0f : 1.0f) / n_valid * grad_loss;
    for (int j = tid; j < V; j += THREADS){
        float p = expf(logits[row * V + j] - rmax) / rsum;
        float g = p - ((j == t) ? 1.0f : 0.0f);
        grad[row * V + j] = g * scale;
    }
}

float launch_cross_entropy_forward(const float* h_logits, const int* h_targets,
                                   int M, int V, int ignore_index, float* out_n_valid){
    size_t mat = (size_t)M * V * sizeof(float);
    float *d_logits, *d_nll, *d_valid;
    int *d_targets;
    CUDA_CHECK(cudaMalloc(&d_logits, mat));
    CUDA_CHECK(cudaMalloc(&d_targets, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nll, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_valid, M * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits, mat, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets, M * sizeof(int), cudaMemcpyHostToDevice));

    ce_forward_kernel<<<M, THREADS>>>(d_logits, d_targets, d_nll, d_valid, M, V, ignore_index);
    CUDA_CHECK_KERNEL();

    std::vector<float> nll(M), valid(M);
    CUDA_CHECK(cudaMemcpy(nll.data(), d_nll, M * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(valid.data(), d_valid, M * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_logits)); CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_nll)); CUDA_CHECK(cudaFree(d_valid));

    float sum_nll = 0.0f, n = 0.0f;
    for (int i = 0; i < M; ++i){ sum_nll += nll[i]; n += valid[i]; }
    if (n == 0.0f) n = 1.0f;
    *out_n_valid = n;
    return sum_nll / n;
}

void launch_cross_entropy_backward(const float* h_logits, const int* h_targets,
                                   float* h_grad_logits, int M, int V,
                                   int ignore_index, float n_valid, float grad_loss){
    size_t mat = (size_t)M * V * sizeof(float);
    float *d_logits, *d_grad;
    int *d_targets;
    CUDA_CHECK(cudaMalloc(&d_logits, mat));
    CUDA_CHECK(cudaMalloc(&d_targets, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grad, mat));
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits, mat, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets, M * sizeof(int), cudaMemcpyHostToDevice));

    ce_backward_kernel<<<M, THREADS>>>(d_logits, d_targets, d_grad, M, V,
                                       ignore_index, n_valid, grad_loss);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_grad_logits, d_grad, mat, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_logits)); CUDA_CHECK(cudaFree(d_targets)); CUDA_CHECK(cudaFree(d_grad));
}

#ifdef BUILD_STANDALONE
int main(){
    const int M = 16, V = 200, ignore = -1;
    std::vector<float> logits(M * V);
    std::vector<int> targets(M);
    for (int i = 0; i < M * V; ++i) logits[i] = static_cast<float>(rand()) / RAND_MAX * 4.0f - 2.0f;
    for (int m = 0; m < M; ++m) targets[m] = (m % 5 == 0) ? ignore : rand() % V;

    // CPU reference (stable).
    float sum_nll = 0.0f, n_valid = 0.0f;
    std::vector<float> grad_ref(M * V, 0.0f);
    for (int m = 0; m < M; ++m){
        float mx = -FLT_MAX; for (int j = 0; j < V; ++j) mx = std::max(mx, logits[m * V + j]);
        float s = 0.0f; for (int j = 0; j < V; ++j) s += std::exp(logits[m * V + j] - mx);
        int t = targets[m];
        if (t != ignore){ sum_nll += -((logits[m * V + t] - mx) - std::log(s)); n_valid += 1.0f; }
    }
    float nv = (n_valid == 0.0f) ? 1.0f : n_valid;
    float loss_ref = sum_nll / nv;
    for (int m = 0; m < M; ++m){
        float mx = -FLT_MAX; for (int j = 0; j < V; ++j) mx = std::max(mx, logits[m * V + j]);
        float s = 0.0f; for (int j = 0; j < V; ++j) s += std::exp(logits[m * V + j] - mx);
        int t = targets[m];
        float scale = ((t == ignore) ? 0.0f : 1.0f) / nv;  // grad_loss = 1
        for (int j = 0; j < V; ++j){
            float p = std::exp(logits[m * V + j] - mx) / s;
            grad_ref[m * V + j] = (p - ((j == t) ? 1.0f : 0.0f)) * scale;
        }
    }

    float nv_out = 0.0f;
    float loss = launch_cross_entropy_forward(logits.data(), targets.data(), M, V, ignore, &nv_out);
    std::vector<float> grad(M * V);
    launch_cross_entropy_backward(logits.data(), targets.data(), grad.data(), M, V, ignore, nv_out, 1.0f);

    float ge = 0.0f; for (int i = 0; i < M * V; ++i) ge = std::max(ge, std::abs(grad[i] - grad_ref[i]));
    printf("cross_entropy: loss err = %.6e (loss=%.6f), n_valid=%.0f, grad err = %.6e\n",
           std::abs(loss - loss_ref), loss, nv_out, ge);
    return 0;
}
#endif
