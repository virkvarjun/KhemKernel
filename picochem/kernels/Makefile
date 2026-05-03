NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_75 -std=c++17

# List of executables we'll build (one per .cu file)
TARGETS = vector_add matmul_naive matmul_tiled softmax layer_norm

all: $(TARGETS)

%: %.cu common.h
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean 