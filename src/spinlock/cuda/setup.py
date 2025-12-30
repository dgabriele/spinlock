"""
Setup script for Spinlock CUDA extensions.

This uses PyTorch's C++ extension builder to compile CUDA kernels.
Can be used standalone or integrated into the main Poetry build.
"""

import os
from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

# Get current directory
CUDA_DIR = Path(__file__).parent.resolve()

# Source files (CUDA kernel approach abandoned - pivoting to torch.compile)
sources = [
    str(CUDA_DIR / "src" / "bindings" / "torch_bindings.cpp"),
    str(CUDA_DIR / "src" / "common" / "dummy_kernel.cu"),
]

# Include directories
include_dirs = [
    str(CUDA_DIR / "include"),
]

# CUDA architectures
# RTX 3060 Ti uses sm_86 (Ampere)
cuda_architectures = [
    "-gencode=arch=compute_75,code=sm_75",  # Turing (RTX 20xx)
    "-gencode=arch=compute_80,code=sm_80",  # Ampere (A100)
    "-gencode=arch=compute_86,code=sm_86",  # Ampere (RTX 30xx)
    "-gencode=arch=compute_89,code=sm_89",  # Ada (RTX 40xx)
]

# Compiler flags
cxx_flags = ["-O3", "-Wall", "-Wextra"]
nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "-lineinfo",
    "--extended-lambda",
] + cuda_architectures

# Debug mode
if os.environ.get("DEBUG", "0") == "1":
    nvcc_flags.extend(["-G", "-g"])

# Create extension
ext_modules = [
    CUDAExtension(
        name="spinlock.cuda._C",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
        libraries=["cufft"],  # For spectral features (Phase 2)
    )
]

setup(
    name="spinlock-cuda",
    version="0.1.0",
    description="CUDA kernels for Spinlock dataset generation",
    author="Spinlock Team",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0.0"],
)
