#pragma once

// Host-pointer launcher: allocates device memory, copies in/out per call.
void launch_matmul_tiled(const float* h_A, const float* h_B, float* h_C,
                         int M, int N, int K);

// Device-pointer launcher: inputs/output already on the GPU; no copies.
// Used by the device-resident path (DeviceTensor).
void launch_matmul_tiled_device(const float* d_A, const float* d_B, float* d_C,
                                int M, int N, int K);
