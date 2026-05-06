#pragma once

// Host-side launcher: allocates device memory, copies in, launches, copies out.
void launch_vector_add(const float* h_a, const float* h_b, float* h_c, int N);
