#pragma once

// GeLU (tanh approximation, Hendrycks & Gimpel), elementwise over N values.
//   forward:  y = 0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))
//   backward: grad_x = grad_y · dy/dx
void launch_gelu_forward(const float* h_x, float* h_out, int N);
void launch_gelu_backward(const float* h_grad_y, const float* h_x, float* h_grad_x, int N);

// Device-resident variants (pointers already on the GPU, no copies).
void launch_gelu_forward_device(const float* d_x, float* d_out, int N);
void launch_gelu_backward_device(const float* d_grad_y, const float* d_x, float* d_grad_x, int N);
