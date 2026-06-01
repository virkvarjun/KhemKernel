#pragma once

// Batched tiled matmul over a leading batch dim, with optional per-operand
// transpose. For each b in [0,batch):
//   C[b](Mr,Nr) = opA(A[b]) @ opB(B[b]),  contraction dim Kc
// where opA(A[b]) is A[b](Mr,Kc) if transA==0 else A[b](Kc,Mr) transposed,
// and   opB(B[b]) is B[b](Kc,Nr) if transB==0 else B[b](Nr,Kc) transposed.
//
// This one routine covers every attention matmul (QKᵀ, weights·V, and their
// backward forms) by choosing transA/transB.
void launch_bmm(const float* h_A, const float* h_B, float* h_C,
                int batch, int Mr, int Nr, int Kc, int transA, int transB);
void launch_bmm_device(const float* d_A, const float* d_B, float* d_C,
                       int batch, int Mr, int Nr, int Kc, int transA, int transB);
