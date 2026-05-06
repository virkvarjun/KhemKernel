#pragma once

void launch_matmul_tiled(const float* h_A, const float* h_B, float* h_C,
                         int M, int N, int K);
