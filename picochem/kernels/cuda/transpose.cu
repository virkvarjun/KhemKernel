#include "common.h"
#include "transpose.h"
#include <cmath>
#include <vector>

constexpr int THREADS = 256;

// out[b,h,s,dh] = x[b,s,h,dh].  One thread per output element.
__global__ void split_heads_kernel(const float* x, float* y, int B, int S, int H, int Dh){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * Dh;
    if (idx >= total) return;
    int dh = idx % Dh;
    int s  = (idx / Dh) % S;
    int h  = (idx / (Dh * S)) % H;
    int b  = idx / (Dh * S * H);
    int in_idx = ((b * S + s) * H + h) * Dh + dh;   // x is (B,S,H,Dh)
    y[idx] = x[in_idx];
}

// x[b,s,h,dh] = y[b,h,s,dh].  One thread per output (x) element.
__global__ void merge_heads_kernel(const float* y, float* x, int B, int S, int H, int Dh){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * H * Dh;
    if (idx >= total) return;
    int dh = idx % Dh;
    int h  = (idx / Dh) % H;
    int s  = (idx / (Dh * H)) % S;
    int b  = idx / (Dh * H * S);
    int y_idx = ((b * H + h) * S + s) * Dh + dh;     // y is (B,H,S,Dh)
    x[idx] = y[y_idx];
}

void launch_split_heads_device(const float* d_x, float* d_y, int B, int S, int H, int Dh){
    int total = B * H * S * Dh;
    split_heads_kernel<<<(total + THREADS - 1) / THREADS, THREADS>>>(d_x, d_y, B, S, H, Dh);
    CUDA_CHECK_KERNEL();
}

void launch_merge_heads_device(const float* d_y, float* d_x, int B, int S, int H, int Dh){
    int total = B * S * H * Dh;
    merge_heads_kernel<<<(total + THREADS - 1) / THREADS, THREADS>>>(d_y, d_x, B, S, H, Dh);
    CUDA_CHECK_KERNEL();
}

void launch_split_heads(const float* h_x, float* h_y, int B, int S, int H, int Dh){
    size_t bytes = (size_t)B * S * H * Dh * sizeof(float);
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes)); CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    launch_split_heads_device(d_x, d_y, B, S, H, Dh);
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
}

void launch_merge_heads(const float* h_y, float* h_x, int B, int S, int H, int Dh){
    size_t bytes = (size_t)B * S * H * Dh * sizeof(float);
    float *d_y, *d_x;
    CUDA_CHECK(cudaMalloc(&d_y, bytes)); CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));
    launch_merge_heads_device(d_y, d_x, B, S, H, Dh);
    CUDA_CHECK(cudaMemcpy(h_x, d_x, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_x));
}

#ifdef BUILD_STANDALONE
int main(){
    const int B = 2, S = 5, H = 4, Dh = 6;
    std::vector<float> x(B * S * H * Dh), y(B * S * H * Dh), x2(B * S * H * Dh);
    for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(rand()) / RAND_MAX;
    launch_split_heads(x.data(), y.data(), B, S, H, Dh);
    launch_merge_heads(y.data(), x2.data(), B, S, H, Dh);  // round-trip must equal x
    float err = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) err = std::max(err, std::abs(x2[i] - x[i]));
    printf("transpose: split/merge round-trip max error = %.6e\n", err);
    return 0;
}
#endif
