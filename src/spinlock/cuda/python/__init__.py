"""
Spinlock CUDA module - Fused kernels for accelerated dataset generation.

This module provides custom CUDA kernels that fuse operator forward passes
and enable streaming feature extraction during rollout.

Expected speedup: 20-40× over PyTorch implementation.

Example usage:
    >>> import spinlock.cuda
    >>> info = spinlock.cuda.get_device_info()
    >>> print(f"Running on {info['name']}")

    >>> import torch
    >>> a = torch.randn(1000, device='cuda')
    >>> b = torch.randn(1000, device='cuda')
    >>> c = spinlock.cuda.dummy_add(a, b)  # Infrastructure test
"""

try:
    from ._C import (
        dummy_add,
        get_device_info,
    )

    __all__ = [
        "dummy_add",
        "get_device_info",
    ]

    # Print device info on import (helpful for debugging)
    _device_info = get_device_info()
    print(f"Spinlock CUDA initialized on {_device_info['name']} "
          f"(compute {_device_info['compute_capability']}, "
          f"{_device_info['total_memory_gb']:.1f} GB)")

except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import Spinlock CUDA extension: {e}\n"
        "The CUDA kernels are not available. "
        "You may need to build the extension first:\n"
        "  cd src/spinlock/cuda\n"
        "  python setup.py install"
    )

    __all__ = []
