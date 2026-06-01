#pragma once

// Layer norm along the last axis of an (M, N) matrix.
void launch_layer_norm(const float* h_x, const float* h_gamma, const float* h_beta,
                       float* h_out, int M, int N);

// Device-resident forward; also writes x_hat (M,N) and inv_std (M,) for backward.
void launch_layer_norm_fwd_device(const float* d_x, const float* d_gamma, const float* d_beta,
                                  float* d_y, float* d_xhat, float* d_invstd, int M, int N);
