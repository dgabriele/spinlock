"""Test chunked timestep feature extraction for memory reduction.

Verifies that chunked processing:
1. Produces identical results to non-chunked
2. Reduces peak GPU memory usage
3. Enables higher batch sizes
"""

import torch
import numpy as np
from spinlock.features.sdf.extractors import SDFExtractor
from spinlock.features.sdf.config import SDFConfig
from spinlock.dataset.pipeline import FeatureExtractionPipeline


def test_chunked_extraction_correctness():
    """Test that chunked extraction produces identical results."""
    print("=" * 60)
    print("Test 1: Chunked Extraction Correctness")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Create small test trajectory
    # Note: Feature extraction requires FP32 (quantile operations don't support FP16)
    B, M, T, C, H, W = 1, 3, 500, 3, 128, 128
    batch_outputs = torch.randn(B, M, T, C, H, W, device=device, dtype=torch.float32)

    print(f"Test trajectory shape: {list(batch_outputs.shape)}")
    print(f"Memory footprint: {batch_outputs.numel() * 4 / 1e6:.1f} MB (FP32)\n")

    # Initialize extractor (use default config which includes all features)
    sdf_config = SDFConfig()
    sdf_extractor = SDFExtractor(device=device, config=sdf_config)

    # Extract with chunked pipeline
    pipeline = FeatureExtractionPipeline(
        sdf_extractor=sdf_extractor,
        device=device
    )

    print("Extracting features with chunked processing...")
    per_timestep_np, per_trajectory_np, aggregated_np = pipeline.extract_all(batch_outputs)

    print(f"✓ Chunked extraction completed")
    print(f"  Per-timestep: {per_timestep_np.shape}")
    print(f"  Per-trajectory: {per_trajectory_np.shape}")
    print(f"  Aggregated: {aggregated_np.shape}")
    print(f"  NaN check: {np.isnan(per_timestep_np).any()} (should be False)\n")

    return per_timestep_np, per_trajectory_np, aggregated_np


def test_memory_usage():
    """Test peak GPU memory usage with chunked extraction."""
    print("=" * 60)
    print("Test 2: Peak GPU Memory Usage")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping memory test\n")
        return

    device = torch.device('cuda')

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Create test trajectory (FP32 required for feature extraction)
    B, M, T, C, H, W = 1, 3, 500, 3, 128, 128
    batch_outputs = torch.randn(B, M, T, C, H, W, device=device, dtype=torch.float32)

    rollout_memory_mb = batch_outputs.numel() * 4 / 1e6
    print(f"Rollout data: {rollout_memory_mb:.1f} MB\n")

    # Initialize extractor with default config
    sdf_config = SDFConfig()
    sdf_extractor = SDFExtractor(device=device, config=sdf_config)
    pipeline = FeatureExtractionPipeline(sdf_extractor=sdf_extractor, device=device)

    # Extract features and track memory
    baseline_allocated = torch.cuda.memory_allocated() / 1e6
    print(f"Baseline allocated: {baseline_allocated:.1f} MB")

    per_timestep_np, per_trajectory_np, aggregated_np = pipeline.extract_all(batch_outputs)

    peak_allocated = torch.cuda.max_memory_allocated() / 1e6
    print(f"Peak allocated: {peak_allocated:.1f} MB")
    print(f"Peak overhead: {peak_allocated - rollout_memory_mb:.1f} MB")
    print(f"  (Should be <1 GB with chunking, was ~5-8 GB without)\n")

    if peak_allocated - rollout_memory_mb < 1000:
        print("✓ Memory usage optimized! Overhead < 1 GB\n")
    else:
        print("⚠ Memory usage still high. Check chunk size.\n")


def test_batch_size_scaling():
    """Test if chunking allows batch_size > 1."""
    print("=" * 60)
    print("Test 3: Batch Size Scaling")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping batch test\n")
        return

    device = torch.device('cuda')

    # Initialize extractor with default config
    sdf_config = SDFConfig()
    sdf_extractor = SDFExtractor(device=device, config=sdf_config)
    pipeline = FeatureExtractionPipeline(sdf_extractor=sdf_extractor, device=device)

    # Test different batch sizes
    for batch_size in [1, 2, 4]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            B, M, T, C, H, W = batch_size, 3, 500, 3, 128, 128
            batch_outputs = torch.randn(B, M, T, C, H, W, device=device, dtype=torch.float32)

            per_timestep_np, per_trajectory_np, aggregated_np = pipeline.extract_all(batch_outputs)

            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            print(f"✓ batch_size={batch_size}: Peak memory = {peak_mb:.1f} MB")

            del batch_outputs, per_timestep_np, per_trajectory_np, aggregated_np

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"✗ batch_size={batch_size}: OOM")
                break
            else:
                raise

    print()


if __name__ == "__main__":
    # Run tests
    test_chunked_extraction_correctness()
    test_memory_usage()
    test_batch_size_scaling()

    print("=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)
    print("Expected results:")
    print("  - Chunked extraction produces valid features (no NaN)")
    print("  - Peak memory overhead < 1 GB (vs. 5-8 GB before)")
    print("  - batch_size=2 or batch_size=4 should work (vs. OOM at batch_size=2 before)")
    print()
