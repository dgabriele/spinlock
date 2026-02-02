#!/usr/bin/env python3
"""Measure actual speedup from torch.compile for your specific model."""

import sys
import time
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def measure_epoch_time(config_path: str, use_compile: bool, num_epochs: int = 5):
    """Measure average epoch time with/without compilation."""
    from spinlock.cli.train_vqvae import TrainVQVAECommand

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override settings for quick measurement
    config['use_torch_compile'] = use_compile
    config['epochs'] = num_epochs
    config['val_every_n_epochs'] = num_epochs + 1  # Skip validation for speed
    config['verbose'] = False

    # Run training
    cmd = TrainVQVAECommand()
    start = time.time()

    try:
        # Suppress output
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                cmd._run_training(config, type('Args', (), {'config': config_path})())
    except Exception as e:
        print(f"Training failed: {e}")
        return None

    elapsed = time.time() - start
    return elapsed / num_epochs  # Average time per epoch


def main():
    """Compare compilation speedup."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config file to benchmark')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to measure')
    args = parser.parse_args()

    print("=" * 70)
    print("MEASURING ACTUAL COMPILATION SPEEDUP")
    print("=" * 70)
    print(f"\nConfig: {args.config}")
    print(f"Epochs: {args.epochs}")

    # Measure without compilation
    print("\n1. Measuring WITHOUT torch.compile...")
    time_without = measure_epoch_time(args.config, use_compile=False, num_epochs=args.epochs)
    if time_without is None:
        print("Failed to measure without compilation")
        return 1
    print(f"   Average epoch time: {time_without:.2f}s")

    # Measure with compilation
    print("\n2. Measuring WITH torch.compile...")
    time_with = measure_epoch_time(args.config, use_compile=True, num_epochs=args.epochs)
    if time_with is None:
        print("Failed to measure with compilation")
        return 1
    print(f"   Average epoch time: {time_with:.2f}s")

    # Calculate speedup
    if time_without > 0:
        speedup_pct = (1 - time_with / time_without) * 100
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Without compilation: {time_without:.2f}s per epoch")
        print(f"With compilation:    {time_with:.2f}s per epoch")
        print(f"\nSpeedup: {speedup_pct:+.1f}%")

        if speedup_pct < 5:
            print("\n⚠️  Minimal speedup detected. Possible reasons:")
            print("  - Running on CPU (compile works best on CUDA)")
            print("  - Small model (overhead dominates)")
            print("  - Limited GPU resources (see warning above)")
            print("  - Encoding step dominates compute (check profiling)")
        elif speedup_pct < 15:
            print("\n✓ Moderate speedup (expected for variable-length models)")
        else:
            print("\n✓ Good speedup!")

        print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
