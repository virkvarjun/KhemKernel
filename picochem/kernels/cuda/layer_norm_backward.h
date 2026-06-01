#pragma once

// Backward of layer norm along the last axis of an (M, N) matrix.
// Inputs come from the forward cache (x_hat, gamma, inv_std) plus the upstream
// grad_y. Produces grad_x (M,N) and the parameter grads grad_gamma/grad_beta (N).
//   grad_gamma = Σ_rows grad_y ⊙ x_hat
//   grad_beta  = Σ_rows grad_y
//   dxhat      = grad_y ⊙ gamma
//   grad_x     = (1/N)·inv_std·(N·dxhat − Σ dxhat − x_hat·Σ(dxhat⊙x_hat))   (per row)
void launch_layer_norm_backward(const float* h_grad_y, const float* h_x_hat,
                                const float* h_gamma, const float* h_inv_std,
                                float* h_grad_x, float* h_grad_gamma, float* h_grad_beta,
                                int M, int N);

// Device-resident variant (all pointers already on the GPU, no copies).
void launch_layer_norm_backward_device(const float* d_grad_y, const float* d_x_hat,
                                       const float* d_gamma, const float* d_inv_std,
                                       float* d_grad_x, float* d_grad_gamma, float* d_grad_beta,
                                       int M, int N);
