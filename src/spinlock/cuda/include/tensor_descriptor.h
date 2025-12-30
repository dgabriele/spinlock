#pragma once

#include <cuda_runtime.h>

namespace spinlock {
namespace cuda {

/**
 * @brief Lightweight tensor descriptor for kernel arguments
 *
 * Describes tensor layout (NCHW format) without storing the data pointer.
 * Used to pass shape and stride information to kernels efficiently.
 */
struct TensorDescriptor {
    int N;  // Batch size
    int C;  // Channels
    int H;  // Height
    int W;  // Width

    // Strides (number of elements to skip for each dimension)
    int stride_N;
    int stride_C;
    int stride_H;
    int stride_W;

    // Default constructor
    __host__ __device__ TensorDescriptor()
        : N(0), C(0), H(0), W(0),
          stride_N(0), stride_C(0), stride_H(0), stride_W(0) {}

    // Constructor for contiguous NCHW tensor
    __host__ __device__ TensorDescriptor(int n, int c, int h, int w)
        : N(n), C(c), H(h), W(w) {
        // Contiguous strides: [C*H*W, H*W, W, 1]
        stride_W = 1;
        stride_H = W;
        stride_C = H * W;
        stride_N = C * H * W;
    }

    // Get linear index for [n, c, h, w] access
    __host__ __device__ inline int index(int n, int c, int h, int w) const {
        return n * stride_N + c * stride_C + h * stride_H + w * stride_W;
    }

    // Total number of elements
    __host__ __device__ inline int numel() const {
        return N * C * H * W;
    }

    // Check if tensor is contiguous
    __host__ __device__ inline bool is_contiguous() const {
        return (stride_W == 1 &&
                stride_H == W &&
                stride_C == H * W &&
                stride_N == C * H * W);
    }
};

/**
 * @brief 4D convolution parameters
 */
struct Conv2dParams {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int dilation;

    __host__ __device__ Conv2dParams()
        : in_channels(0), out_channels(0), kernel_size(0),
          stride(1), padding(0), dilation(1) {}

    __host__ __device__ Conv2dParams(int ic, int oc, int k,
                                      int s = 1, int p = 0, int d = 1)
        : in_channels(ic), out_channels(oc), kernel_size(k),
          stride(s), padding(p), dilation(d) {}

    // Compute output spatial dimensions
    __host__ __device__ inline int output_size(int input_size) const {
        int effective_kernel = dilation * (kernel_size - 1) + 1;
        return (input_size + 2 * padding - effective_kernel) / stride + 1;
    }
};

} // namespace cuda
} // namespace spinlock
