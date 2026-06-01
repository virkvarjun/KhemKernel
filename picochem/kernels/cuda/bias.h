#pragma once

// Broadcast bias add: out(M,N) = x(M,N) + b(N)   (b added to every row).
void launch_add_bias(const float* h_x, const float* h_b, float* h_out, int M, int N);
void launch_add_bias_device(const float* d_x, const float* d_b, float* d_out, int M, int N);

// Column sum: out(N) = Σ_rows x(M,N)   (the bias gradient for a Linear layer).
void launch_colsum(const float* h_x, float* h_out, int M, int N);
void launch_colsum_device(const float* d_x, float* d_out, int M, int N);

// Scalar multiply: out = x * alpha (device-resident).
void launch_scale_device(const float* d_x, float* d_out, float alpha, int n);
