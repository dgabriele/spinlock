#include "error_handling.h"
#include <cuda_runtime.h>

namespace spinlock {
namespace cuda {

/**
 * @brief Simple dummy kernel for testing infrastructure
 *
 * Adds two arrays element-wise: c[i] = a[i] + b[i]
 * This kernel is only used to verify the build system works.
 */
__global__ void dummy_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * @brief Host function to launch dummy kernel
 *
 * This tests:
 * - CUDA compilation
 * - Memory allocation
 * - Kernel launch
 * - Error handling macros
 */
void dummy_add(const float* a, const float* b, float* c, int n) {
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    size_t bytes = n * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    dummy_add_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

} // namespace cuda
} // namespace spinlock
