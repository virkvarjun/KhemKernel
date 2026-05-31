#pragma once

// GeLU (tanh approximation, Hendrycks & Gimpel), elementwise over N values.
//   forward:  y = 0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))
//   backward: grad_x = grad_y · dy/dx
void launch_gelu_forward(const float* h_x, float* h_out, int N);
void launch_gelu_backward(const float* h_grad_y, const float* h_x, float* h_grad_x, int N);
