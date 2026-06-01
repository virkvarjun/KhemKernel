#pragma once

// Host-side launcher: allocates device memory, copies in, launches, copies out.
void launch_vector_add(const float* h_a, const float* h_b, float* h_c, int N);

// Device-resident variant (pointers already on the GPU, no copies).
void launch_vector_add_device(const float* d_a, const float* d_b, float* d_c, int N);
