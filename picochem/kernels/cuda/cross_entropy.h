#pragma once

// Softmax cross-entropy over (M, V) logits with integer targets.
// Rows whose target == ignore_index are masked out (no loss, no gradient).
//
// Forward returns the mean NLL over non-ignored rows. The effective n_valid
// (number of non-ignored rows, floored at 1) is written to *out_n_valid so the
// backward pass can reuse the exact normaliser.
float launch_cross_entropy_forward(const float* h_logits, const int* h_targets,
                                   int M, int V, int ignore_index, float* out_n_valid);

// grad_logits = (softmax(logits) − onehot(target)) · mask / n_valid · grad_loss
void launch_cross_entropy_backward(const float* h_logits, const int* h_targets,
                                   float* h_grad_logits, int M, int V,
                                   int ignore_index, float n_valid, float grad_loss);

// Device-resident (logits and targets already on the GPU).
float launch_cross_entropy_forward_device(const float* d_logits, const int* d_targets,
                                          int M, int V, int ignore_index, float* out_n_valid);
void launch_cross_entropy_backward_device(const float* d_logits, const int* d_targets,
                                          float* d_grad, int M, int V, int ignore_index,
                                          float n_valid, float grad_loss);
