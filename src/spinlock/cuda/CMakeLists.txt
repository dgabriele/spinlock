cmake_minimum_required(VERSION 3.18)
project(spinlock_cuda LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures (Turing, Ampere, Ada, Hopper)
# sm_75: RTX 20xx, Tesla T4
# sm_80: A100
# sm_86: RTX 30xx (user's GPU)
# sm_89: RTX 40xx
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

# Find packages
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

# Collect source files
file(GLOB_RECURSE CUDA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fused_ops/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/feature_extraction/*.cu
)

file(GLOB_RECURSE CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/*.cpp
)

# Create shared library
add_library(spinlock_cuda SHARED
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

# Link libraries
target_link_libraries(spinlock_cuda PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cufft  # For spectral features (Phase 2)
)

# Compiler flags
target_compile_options(spinlock_cuda PRIVATE
    # C++ flags
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -Wall
        -Wextra
    >
    # CUDA flags
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --expt-relaxed-constexpr
        -Xptxas=-v
        -lineinfo
        --extended-lambda
    >
)

# Debug flags
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(spinlock_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -G
            -g
        >
    )
endif()

# Installation (optional, for now just build in-place)
install(TARGETS spinlock_cuda
    LIBRARY DESTINATION lib
)

# Print configuration
message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "PyTorch Version: ${Torch_VERSION}")
message(STATUS "PyTorch Include Dirs: ${TORCH_INCLUDE_DIRS}")
