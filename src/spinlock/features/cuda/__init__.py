"""
CUDA kernel implementations for high-performance feature extraction.

Provides modular CUDA kernels organized by computation pattern:
- reduction: Parallel reduction operations (sum, mean, std, min, max)
- spatial: Spatial derivatives (gradient, Laplacian, Sobel)
- windowed: Windowed operations (local variance, autocorrelation)
- histogram: Histogram construction and quantile extraction
- moments: Statistical moments (skewness, kurtosis)

All kernels use JIT compilation with PyTorch fallbacks for robustness.
"""

__version__ = "1.0.0"
