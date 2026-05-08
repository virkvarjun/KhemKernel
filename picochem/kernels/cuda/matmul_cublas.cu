#include <cublas_v2.h>

#include <chrono>
#include <cmath>
#include <vector>

// Wrapper for cuBLAS error checking.
#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n",                     \
                    __FILE__, __LINE__, status);                               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
