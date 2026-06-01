#pragma once

// In-place Adam update over a flat parameter buffer of length n.
//   m = b1·m + (1−b1)·g
//   v = b2·v + (1−b2)·g²
//   p -= lr · (m / bc1) / (sqrt(v / bc2) + eps)
// Bias-correction denominators bc1 = 1−b1^step and bc2 = 1−b2^step are computed
// on the host and passed in. Updated p, m, v are written back to the host buffers.
void launch_adam_update(float* h_param, const float* h_grad, float* h_m, float* h_v,
                        int n, float lr, float b1, float b2, float eps,
                        float bc1, float bc2);

// Device-resident, in-place (param/m/v already on the GPU).
void launch_adam_update_device(float* d_param, const float* d_grad, float* d_m, float* d_v,
                               int n, float lr, float b1, float b2, float eps,
                               float bc1, float bc2);
