NVCC       = nvcc
CUDA_ARCH ?= sm_120
NVCC_FLAGS = -O3 -arch=$(CUDA_ARCH) -std=c++17

CUDA_DIR = cuda
TARGETS   = vector_add matmul_naive matmul_tiled matmul_backward \
            softmax softmax_backward layer_norm layer_norm_backward gelu \
            cross_entropy
SOURCES   = $(addprefix $(CUDA_DIR)/, $(addsuffix .cu, $(TARGETS)))
OBJECTS   = $(addprefix $(CUDA_DIR)/, $(addsuffix .o,  $(TARGETS)))

# Default: build the standalone test executables.
.PHONY: all standalone library clean
all: standalone

standalone: $(TARGETS)

$(TARGETS): %: $(CUDA_DIR)/%.cu $(CUDA_DIR)/common.h $(CUDA_DIR)/%.h
	$(NVCC) $(NVCC_FLAGS) -DBUILD_STANDALONE $< -o $(CUDA_DIR)/$@

# Compile position-independent object files for linking into the Python module.
library: $(OBJECTS)

$(CUDA_DIR)/%.o: $(CUDA_DIR)/%.cu $(CUDA_DIR)/common.h $(CUDA_DIR)/%.h
	$(NVCC) $(NVCC_FLAGS) -Xcompiler -fPIC -c $< -o $@

clean:
	rm -f $(addprefix $(CUDA_DIR)/, $(TARGETS)) $(OBJECTS)
