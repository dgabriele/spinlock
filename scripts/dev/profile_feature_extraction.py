#!/usr/bin/env python3
"""
Profile feature extraction during NO dataset generation.

Measures runtime of individual features/categories to identify bottlenecks.

Usage:
    # Profile 4 operators (category-level, low overhead)
    poetry run python scripts/dev/profile_feature_extraction.py --num-operators 4

    # Fine-grained per-feature profiling (higher overhead)
    poetry run python scripts/dev/profile_feature_extraction.py --num-operators 4 --level feature

    # Custom timesteps/realizations
    poetry run python scripts/dev/profile_feature_extraction.py \\
        --num-operators 4 --timesteps 250 --realizations 3
"""

import argparse
import torch
from pathlib import Path

from spinlock.features.summary import SummaryExtractor, SummaryConfig
from spinlock.profiling import FeatureProfilingContext


def run_profiling(
    num_operators: int = 4,
    level: str = "category",
    timesteps: int = 500,
    realizations: int = 5,
    grid_size: int = 64,
    output_dir: Path = Path("profiling_results"),
):
    """
    Run feature extraction profiling.

    Args:
        num_operators: Number of operators to profile
        level: 'category' or 'feature' profiling granularity
        timesteps: Temporal rollout length
        realizations: Stochastic realizations per operator
        grid_size: Spatial grid size (default: 64x64)
        output_dir: Directory for profiling reports
    """

    # Setup
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("FEATURE EXTRACTION PROFILING")
    print("=" * 80)
    print(f"Operators: {num_operators}")
    print(f"Timesteps: {timesteps}")
    print(f"Realizations: {realizations}")
    print(f"Grid size: {grid_size}×{grid_size}")
    print(f"Profiling level: {level}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("=" * 80)
    print()

    # Create profiling context
    profiling_context = FeatureProfilingContext(
        device=device, level=level, enabled=True
    )

    # Initialize feature extractor with profiling
    print("Initializing feature extractor with profiling...")
    summary_config = SummaryConfig()  # All features enabled
    summary_extractor = SummaryExtractor(
        device=device, config=summary_config, profiling_context=profiling_context
    )

    # Generate test trajectories
    print(f"Generating {num_operators} test trajectories...")
    B = num_operators
    M = realizations
    T = timesteps
    C = 3  # channels
    H = W = grid_size

    # Simulate realistic trajectories (warm up GPU)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    trajectories = torch.randn(B, M, T, C, H, W, device=device) * 0.1

    # Add temporal evolution (simple diffusion-like dynamics)
    for t in range(1, T):
        trajectories[:, :, t] = 0.95 * trajectories[:, :, t - 1] + 0.05 * torch.randn(
            B, M, C, H, W, device=device
        )

    print(f"Trajectory shape: {trajectories.shape}")
    if device.type == "cuda":
        print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    print()

    # Run extraction with profiling
    print("Extracting features (profiling enabled)...")
    if device.type == "cuda":
        torch.cuda.synchronize()

    features = summary_extractor.extract_all(trajectories)

    if device.type == "cuda":
        torch.cuda.synchronize()
    print("Feature extraction complete!")
    print()

    # Generate reports
    print("Generating profiling report...")
    report = profiling_context.generate_report()

    # Print to console
    report.print_summary()

    # Save reports
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = (
        f"profile_{num_operators}ops_{timesteps}t_{realizations}r_{level}_{timestamp}"
    )

    report.save_json(output_dir / f"{base_name}.json")
    report.save_csv(output_dir / f"{base_name}.csv")

    print(f"\nProfiling results saved to: {output_dir}/")
    print(f"  - {base_name}.json")
    print(f"  - {base_name}.csv")
    print()

    # Cleanup
    del trajectories, features
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile feature extraction during NO dataset generation"
    )
    parser.add_argument(
        "--num-operators",
        type=int,
        default=4,
        help="Number of operators to profile (default: 4)",
    )
    parser.add_argument(
        "--level",
        choices=["category", "feature"],
        default="category",
        help="Profiling granularity: 'category' (low overhead) or 'feature' (fine-grained)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500,
        help="Temporal rollout length (default: 500)",
    )
    parser.add_argument(
        "--realizations",
        type=int,
        default=5,
        help="Stochastic realizations per operator (default: 5)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=64,
        help="Spatial grid size (default: 64 for 64×64)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profiling_results"),
        help="Output directory for reports (default: profiling_results/)",
    )

    args = parser.parse_args()

    run_profiling(
        num_operators=args.num_operators,
        level=args.level,
        timesteps=args.timesteps,
        realizations=args.realizations,
        grid_size=args.grid_size,
        output_dir=args.output_dir,
    )
