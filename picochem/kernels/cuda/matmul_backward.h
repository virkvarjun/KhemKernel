#pragma once

// Backward pass for C = A @ B, with A:(M,K), B:(K,N), C:(M,N).
//
//   dA = dC @ Bᵀ   -> (M, K)
//   dB = Aᵀ @ dC   -> (K, N)
//
// All arrays are row-major float32 on the host; launchers handle the device copy.

// dA(M,K) = dC(M,N) @ B(K,N)ᵀ
void launch_matmul_dA(const float* h_dC, const float* h_B, float* h_dA,
                      int M, int N, int K);

// dB(K,N) = A(M,K)ᵀ @ dC(M,N)
void launch_matmul_dB(const float* h_A, const float* h_dC, float* h_dB,
                      int M, int N, int K);
