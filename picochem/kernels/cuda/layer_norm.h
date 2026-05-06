#pragma once

// Layer norm along the last axis of an (M, N) matrix.
void launch_layer_norm(const float* h_x, const float* h_gamma, const float* h_beta,
                       float* h_out, int M, int N);
