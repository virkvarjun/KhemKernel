#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Error-checking macro: wrap any CUDA runtime call.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Check the last kernel launch for errors. Call after every kernel launch.
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "Kernel launch error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)