#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
#include <string>

namespace spinlock {
namespace cuda {

/**
 * @brief CUDA error checking macro
 *
 * Checks the result of a CUDA API call and throws an exception if it failed.
 * Includes file name, line number, and CUDA error message.
 */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t error = call;                                             \
        if (error != cudaSuccess) {                                           \
            std::ostringstream oss;                                           \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "  \
                << cudaGetErrorString(error);                                 \
            throw std::runtime_error(oss.str());                              \
        }                                                                     \
    } while (0)

/**
 * @brief Kernel launch error checking macro
 *
 * Checks for errors after a kernel launch. Must be called after kernel<<<>>>().
 * Uses cudaGetLastError() to catch launch configuration errors, then
 * cudaDeviceSynchronize() to catch runtime errors.
 */
#define CUDA_CHECK_KERNEL()                                                   \
    do {                                                                      \
        cudaError_t error = cudaGetLastError();                               \
        if (error != cudaSuccess) {                                           \
            std::ostringstream oss;                                           \
            oss << "CUDA kernel launch error at " << __FILE__ << ":"         \
                << __LINE__ << " - " << cudaGetErrorString(error);            \
            throw std::runtime_error(oss.str());                              \
        }                                                                     \
        error = cudaDeviceSynchronize();                                      \
        if (error != cudaSuccess) {                                           \
            std::ostringstream oss;                                           \
            oss << "CUDA kernel execution error at " << __FILE__ << ":"      \
                << __LINE__ << " - " << cudaGetErrorString(error);            \
            throw std::runtime_error(oss.str());                              \
        }                                                                     \
    } while (0)

/**
 * @brief Device-side assertion macro
 *
 * Only active in debug builds. Causes kernel to fail if condition is false.
 */
#ifndef NDEBUG
#define CUDA_ASSERT(condition)                                                \
    do {                                                                      \
        if (!(condition)) {                                                   \
            printf("CUDA assertion failed: %s at %s:%d\n", #condition,        \
                   __FILE__, __LINE__);                                       \
            asm("trap;");                                                     \
        }                                                                     \
    } while (0)
#else
#define CUDA_ASSERT(condition) ((void)0)
#endif

/**
 * @brief Get device properties in a safe way
 */
inline cudaDeviceProp getDeviceProperties(int device = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop;
}

/**
 * @brief Print device information
 */
inline void printDeviceInfo(int device = 0) {
    cudaDeviceProp prop = getDeviceProperties(device);
    printf("CUDA Device %d: %s\n", device, prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total Global Memory: %.2f GB\n",
           prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("  Shared Memory per Block: %zu KB\n",
           prop.sharedMemPerBlock / 1024);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
}

} // namespace cuda
} // namespace spinlock
