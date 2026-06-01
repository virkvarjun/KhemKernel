#pragma once

// Multi-head split/merge (the only data permutation attention needs).
// split: (B, S, D=H*Dh) -> (B*H, S, Dh)   [logically (B,H,S,Dh), contiguous]
// merge: (B*H, S, Dh)   -> (B, S, D)
void launch_split_heads(const float* h_x, float* h_y, int B, int S, int H, int Dh);
void launch_split_heads_device(const float* d_x, float* d_y, int B, int S, int H, int Dh);
void launch_merge_heads(const float* h_y, float* h_x, int B, int S, int H, int Dh);
void launch_merge_heads_device(const float* d_y, float* d_x, int B, int S, int H, int Dh);
