#pragma once

#include <cuda_runtime.h>

namespace spinlock {
namespace cuda {

/**
 * @brief Warp-level reduction primitives
 *
 * Efficient parallel reductions within a warp (32 threads) using
 * shuffle instructions. No shared memory needed!
 */

/**
 * @brief Warp-level sum reduction
 *
 * Each thread in the warp holds a value. After this function,
 * all threads have the sum of all values.
 *
 * @param val Value from this thread
 * @return Sum across all threads in warp
 */
__device__ inline float warp_reduce_sum(float val) {
    // Butterfly reduction using shuffle
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    // Broadcast result to all threads
    return __shfl_sync(0xffffffff, val, 0);
}

/**
 * @brief Warp-level maximum reduction
 */
__device__ inline float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);
}

/**
 * @brief Warp-level minimum reduction
 */
__device__ inline float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);
}

/**
 * @brief Block-level sum reduction using warp reductions
 *
 * Reduces across all threads in a thread block. Uses shared memory
 * to combine warp-level reductions.
 *
 * @param val Value from this thread
 * @param shared Shared memory buffer (size: num_warps in block)
 * @return Sum across all threads in block (valid in all threads)
 */
template <int BLOCK_SIZE>
__device__ inline float block_reduce_sum(float val, float* shared) {
    constexpr int WARP_SIZE = 32;  // CUDA warp size is always 32
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    constexpr int num_warps = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;

    // Reduce within warp
    val = warp_reduce_sum(val);

    // Warp leaders write to shared memory
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // Final reduction across warps (using first warp)
    if (warp_id == 0) {
        val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
        val = warp_reduce_sum(val);
    }

    // Broadcast result to all threads
    if (threadIdx.x == 0) {
        shared[0] = val;
    }
    __syncthreads();

    return shared[0];
}

/**
 * @brief Welford's online algorithm for mean and variance
 *
 * Numerically stable computation of running mean and variance.
 * Used for batch/instance normalization.
 */
struct WelfordAccumulator {
    float mean;
    float m2;    // Sum of squared deviations
    int count;

    __device__ WelfordAccumulator() : mean(0.0f), m2(0.0f), count(0) {}

    __device__ inline void update(float value) {
        count++;
        float delta = value - mean;
        mean += delta / count;
        float delta2 = value - mean;
        m2 += delta * delta2;
    }

    __device__ inline float variance() const {
        return (count > 1) ? m2 / count : 0.0f;
    }

    __device__ inline float std() const {
        return sqrtf(variance());
    }
};

/**
 * @brief Combine two Welford accumulators (for parallel reduction)
 */
__device__ inline WelfordAccumulator combine_welford(
    const WelfordAccumulator& a,
    const WelfordAccumulator& b
) {
    if (a.count == 0) return b;
    if (b.count == 0) return a;

    WelfordAccumulator result;
    result.count = a.count + b.count;

    float delta = b.mean - a.mean;
    result.mean = a.mean + delta * b.count / result.count;

    result.m2 = a.m2 + b.m2 + delta * delta * a.count * b.count / result.count;

    return result;
}

} // namespace cuda
} // namespace spinlock
