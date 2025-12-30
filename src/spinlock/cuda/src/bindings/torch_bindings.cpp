#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations of CUDA functions
namespace spinlock {
namespace cuda {

void dummy_add(const float* a, const float* b, float* c, int n);

} // namespace cuda
} // namespace spinlock

/**
 * @brief PyTorch wrapper for dummy_add kernel
 *
 * Takes PyTorch tensors, extracts raw pointers, and calls CUDA kernel.
 * This tests the full Python → C++ → CUDA pipeline.
 */
torch::Tensor dummy_add_torch(torch::Tensor a, torch::Tensor b) {
    // Input validation
    TORCH_CHECK(a.device().is_cuda(), "Tensor a must be on CUDA");
    TORCH_CHECK(b.device().is_cuda(), "Tensor b must be on CUDA");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Tensor a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Tensor b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensors must have same shape");
    TORCH_CHECK(a.is_contiguous(), "Tensor a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Tensor b must be contiguous");

    // Allocate output tensor
    auto c = torch::empty_like(a);

    // Get data pointers
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* c_ptr = c.data_ptr<float>();

    int n = a.numel();

    // Call CUDA kernel
    spinlock::cuda::dummy_add(a_ptr, b_ptr, c_ptr, n);

    return c;
}

/**
 * @brief Get CUDA device information
 */
py::dict get_device_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    py::dict info;
    info["name"] = prop.name;
    info["compute_capability"] = std::to_string(prop.major) + "." +
                                  std::to_string(prop.minor);
    info["total_memory_gb"] = prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0;
    info["shared_memory_per_block_kb"] = prop.sharedMemPerBlock / 1024;
    info["max_threads_per_block"] = prop.maxThreadsPerBlock;
    info["warp_size"] = prop.warpSize;
    info["multiprocessor_count"] = prop.multiProcessorCount;

    return info;
}

/**
 * @brief pybind11 module definition
 *
 * Exposes CUDA functions to Python as spinlock.cuda._C module
 *
 * NOTE: Fused ConvBlock CUDA kernels removed due to 80× performance regression.
 * Pivoting to torch.compile() optimization strategy instead.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Spinlock CUDA utilities (custom kernels abandoned, using torch.compile instead)";

    m.def("dummy_add", &dummy_add_torch,
          "Element-wise addition of two tensors (testing infrastructure)",
          py::arg("a"), py::arg("b"));

    m.def("get_device_info", &get_device_info,
          "Get CUDA device properties");
}
