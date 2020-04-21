
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdlib.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

constexpr uint32_t THREADS_PER_BLOCK = 512;
