#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

namespace spinlock {
namespace cuda {

/**
 * @brief GPU device capabilities and optimal launch configurations
 *
 * Queries GPU properties at runtime and determines optimal kernel parameters
 * based on actual hardware capabilities (not hard-coded).
 */
struct DeviceCapabilities {
    int device_id;
    int compute_capability_major;
    int compute_capability_minor;
    size_t total_global_mem;
    size_t shared_mem_per_block;
    size_t shared_mem_per_sm;
    int max_threads_per_block;
    int max_threads_per_sm;
    int warp_size;
    int num_sms;

    DeviceCapabilities(int device = 0) : device_id(device) {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to get CUDA device properties");
        }

        compute_capability_major = prop.major;
        compute_capability_minor = prop.minor;
        total_global_mem = prop.totalGlobalMem;
        shared_mem_per_block = prop.sharedMemPerBlock;
        shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
        max_threads_per_block = prop.maxThreadsPerBlock;
        max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
        warp_size = prop.warpSize;
        num_sms = prop.multiProcessorCount;
    }
};

/**
 * @brief Compute optimal tile size for convolution kernels
 *
 * Determines largest tile that fits in available shared memory for:
 * - Input tile with halo: (TILE_H + 2*halo) × (TILE_W + 2*halo) × IN_CHANNELS
 * - Conv output buffer: TILE_H × TILE_W × OUT_CHANNELS (if fused)
 * - Statistics buffer: Small overhead for normalization
 *
 * @param device_caps GPU capabilities
 * @param kernel_size Convolution kernel size (e.g., 3 for 3×3)
 * @param in_channels Input channel count
 * @param out_channels Output channel count (for fused kernels)
 * @param fused Whether this is a fused kernel (needs conv output buffer)
 * @return Optimal square tile size (e.g., 8, 12, 16)
 */
inline int compute_optimal_tile_size(
    const DeviceCapabilities& caps,
    int kernel_size,
    int in_channels,
    int out_channels = 0,
    bool fused = false
) {
    const int halo = kernel_size / 2;
    const size_t bytes_per_float = sizeof(float);

    // Reserve some shared memory for statistics and other overhead (2KB)
    const size_t reserved_smem = 2048;
    const size_t usable_smem = caps.shared_mem_per_block - reserved_smem;

    // Try tile sizes that we actually compile: 8, 4
    // (Larger tiles like 12×12 and 16×16 exceed 48KB shared memory on most GPUs)
    for (int tile_size : {8, 4}) {
        // Input tile with halo
        int input_tile_h = tile_size + 2 * halo;
        int input_tile_w = tile_size + 2 * halo;
        size_t input_smem = input_tile_h * input_tile_w * in_channels * bytes_per_float;

        // Conv output buffer (only for fused kernels)
        size_t output_smem = 0;
        if (fused && out_channels > 0) {
            output_smem = tile_size * tile_size * out_channels * bytes_per_float;
        }

        size_t total_smem = input_smem + output_smem + reserved_smem;

        if (total_smem <= usable_smem) {
            return tile_size;
        }
    }

    // Fallback to smallest tile (should always fit)
    return 4;
}

/**
 * @brief Compute optimal block size for 1D kernels (e.g., normalization)
 *
 * Selects block size that maximizes occupancy while respecting limits.
 *
 * @param caps GPU capabilities
 * @param shared_mem_per_thread Shared memory bytes per thread
 * @return Optimal block size (multiple of warp size)
 */
inline int compute_optimal_block_size_1d(
    const DeviceCapabilities& caps,
    size_t shared_mem_per_thread = 0
) {
    // Try common block sizes: 256, 512, 1024
    for (int block_size : {1024, 512, 256, 128}) {
        if (block_size > caps.max_threads_per_block) continue;

        size_t required_smem = block_size * shared_mem_per_thread;
        if (required_smem <= caps.shared_mem_per_block) {
            return block_size;
        }
    }

    // Fallback
    return 128;
}

/**
 * @brief Global device capabilities cache
 *
 * Initialized once per device, reused across kernel launches.
 */
inline DeviceCapabilities& get_device_capabilities(int device = 0) {
    static DeviceCapabilities caps(device);
    return caps;
}

} // namespace cuda
} // namespace spinlock
