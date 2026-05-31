#include "common.h"
#include "adam.h"
#include <cmath>
#include <vector>

constexpr int THREADS = 256;

__global__ void adam_update_kernel(float* param, const float* grad, float* m, float* v,
                                   int n, float lr, float b1, float b2, float eps,
                                   float bc1, float bc2){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float mi = b1 * m[i] + (1.0f - b1) * g;
    float vi = b2 * v[i] + (1.0f - b2) * g * g;
    m[i] = mi;
    v[i] = vi;
    float m_hat = mi / bc1;
    float v_hat = vi / bc2;
    param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

void launch_adam_update(float* h_param, const float* h_grad, float* h_m, float* h_v,
                        int n, float lr, float b1, float b2, float eps,
                        float bc1, float bc2){
    size_t bytes = (size_t)n * sizeof(float);
    float *d_p, *d_g, *d_m, *d_v;
    CUDA_CHECK(cudaMalloc(&d_p, bytes));
    CUDA_CHECK(cudaMalloc(&d_g, bytes));
    CUDA_CHECK(cudaMalloc(&d_m, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMemcpy(d_p, h_param, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, h_grad, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m, h_m, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice));

    adam_update_kernel<<<(n + THREADS - 1) / THREADS, THREADS>>>(
        d_p, d_g, d_m, d_v, n, lr, b1, b2, eps, bc1, bc2);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_param, d_p, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m, d_m, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_p)); CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_m)); CUDA_CHECK(cudaFree(d_v));
}

#ifdef BUILD_STANDALONE
int main(){
    const int n = 4096;
    const float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    const int step = 7;
    float bc1 = 1.0f - std::pow(b1, step);
    float bc2 = 1.0f - std::pow(b2, step);

    std::vector<float> p(n), g(n), m(n), v(n);
    std::vector<float> p_ref(n), m_ref(n), v_ref(n);
    for (int i = 0; i < n; ++i){
        p[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        g[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        m[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f;
        v[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f;
        p_ref[i] = p[i]; m_ref[i] = m[i]; v_ref[i] = v[i];
    }
    // CPU reference.
    for (int i = 0; i < n; ++i){
        m_ref[i] = b1 * m_ref[i] + (1.0f - b1) * g[i];
        v_ref[i] = b2 * v_ref[i] + (1.0f - b2) * g[i] * g[i];
        float mh = m_ref[i] / bc1, vh = v_ref[i] / bc2;
        p_ref[i] -= lr * mh / (std::sqrt(vh) + eps);
    }

    launch_adam_update(p.data(), g.data(), m.data(), v.data(), n, lr, b1, b2, eps, bc1, bc2);

    float ep = 0.0f, em = 0.0f, ev = 0.0f;
    for (int i = 0; i < n; ++i){
        ep = std::max(ep, std::abs(p[i] - p_ref[i]));
        em = std::max(em, std::abs(m[i] - m_ref[i]));
        ev = std::max(ev, std::abs(v[i] - v_ref[i]));
    }
    printf("adam_update: param err = %.6e, m err = %.6e, v err = %.6e\n", ep, em, ev);
    return 0;
}
#endif
