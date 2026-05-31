#pragma once

// Backward of a pure softmax (no cross-entropy), row-wise over an (M, N) matrix.
//   grad_in = probs ⊙ (grad_out − Σ_j (grad_out ⊙ probs))
// `probs` are the softmax outputs from the forward pass.
void launch_softmax_backward(const float* h_grad_out, const float* h_probs,
                             float* h_grad_in, int M, int N);
