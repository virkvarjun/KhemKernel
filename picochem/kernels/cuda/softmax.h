#pragma once

// Row-wise softmax over an (M, N) matrix.
void launch_softmax(const float* h_x, float* h_out, int M, int N);
