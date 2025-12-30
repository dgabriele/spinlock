"""
Test script for Phase 0: Infrastructure verification.

This script tests that:
1. CUDA extension compiles successfully
2. Python can import the module
3. Basic kernel execution works
4. Device information is accessible
"""

import sys
from pathlib import Path

# Add parent directory to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))


def test_import():
    """Test that the CUDA module can be imported."""
    print("Test 1: Importing spinlock.cuda...")
    try:
        import spinlock.cuda as cuda_module
        print("✓ Import successful")
        return cuda_module
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return None


def test_device_info(cuda_module):
    """Test getting device information."""
    print("\nTest 2: Getting device info...")
    try:
        info = cuda_module.get_device_info()
        print("✓ Device info retrieved:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        return True
    except Exception as e:
        print(f"✗ Failed to get device info: {e}")
        return False


def test_dummy_kernel(cuda_module):
    """Test the dummy addition kernel."""
    print("\nTest 3: Testing dummy_add kernel...")
    try:
        import torch

        # Create test tensors
        n = 10000
        a = torch.randn(n, device='cuda', dtype=torch.float32)
        b = torch.randn(n, device='cuda', dtype=torch.float32)

        # Run CUDA kernel
        c_cuda = cuda_module.dummy_add(a, b)

        # Compare with PyTorch
        c_torch = a + b

        # Check correctness
        max_error = (c_cuda - c_torch).abs().max().item()
        print(f"  Max error vs PyTorch: {max_error:.2e}")

        if max_error < 1e-5:
            print("✓ Dummy kernel works correctly")
            return True
        else:
            print(f"✗ Dummy kernel has errors (max error: {max_error})")
            return False

    except Exception as e:
        print(f"✗ Dummy kernel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance(cuda_module):
    """Benchmark the dummy kernel vs PyTorch."""
    print("\nTest 4: Performance benchmark...")
    try:
        import torch
        import time

        # Large tensor for benchmarking
        n = 10_000_000
        a = torch.randn(n, device='cuda', dtype=torch.float32)
        b = torch.randn(n, device='cuda', dtype=torch.float32)

        # Warmup
        for _ in range(10):
            _ = cuda_module.dummy_add(a, b)
        torch.cuda.synchronize()

        # Benchmark CUDA kernel
        start = time.perf_counter()
        for _ in range(100):
            c = cuda_module.dummy_add(a, b)
        torch.cuda.synchronize()
        cuda_time = time.perf_counter() - start

        # Benchmark PyTorch
        start = time.perf_counter()
        for _ in range(100):
            c = a + b
        torch.cuda.synchronize()
        pytorch_time = time.perf_counter() - start

        print(f"  CUDA kernel: {cuda_time*10:.2f} ms/call")
        print(f"  PyTorch:     {pytorch_time*10:.2f} ms/call")
        print(f"  Speedup:     {pytorch_time/cuda_time:.2f}×")
        print("✓ Benchmark complete")
        return True

    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 0: Infrastructure Test")
    print("=" * 60)

    # Run tests
    cuda_module = test_import()
    if cuda_module is None:
        print("\n" + "=" * 60)
        print("FAILED: Could not import CUDA module")
        print("Build the extension first:")
        print("  cd src/spinlock/cuda")
        print("  python setup.py build_ext --inplace")
        print("=" * 60)
        sys.exit(1)

    results = [
        test_device_info(cuda_module),
        test_dummy_kernel(cuda_module),
        test_performance(cuda_module),
    ]

    print("\n" + "=" * 60)
    if all(results):
        print("SUCCESS: All infrastructure tests passed!")
        print("Phase 0 complete - CUDA build system is working.")
    else:
        print("FAILED: Some tests did not pass")
        print(f"Passed: {sum(results)}/{len(results)}")
    print("=" * 60)

    sys.exit(0 if all(results) else 1)
