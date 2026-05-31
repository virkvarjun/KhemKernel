#pragma once

// Backward of an embedding lookup: scatter-add the upstream gradient rows into
// the embedding table by token id (the transpose of a gather).
//   grad_out:   (M, D)  one gradient row per token occurrence
//   ids:        (M,)    int token ids in [0, V)
//   grad_table: (V, D)  zeroed, then grad_table[ids[m]] += grad_out[m] via atomics
void launch_embedding_backward(const float* h_grad_out, const int* h_ids,
                               float* h_grad_table, int M, int D, int V);
