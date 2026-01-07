#!/usr/bin/env python
"""Test different BPTT window sizes to find the optimal tradeoff."""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spinlock.noa import NOABackbone

def test_bptt_window(checkpoint_path: str, window_size: int, device: str = "cuda"):
    """Test gradient stability with a given BPTT window size.

    Returns:
        dict with gradient statistics
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint['args']

    # Create NOA
    noa = NOABackbone(
        in_channels=1,
        out_channels=1,
        base_channels=args['base_channels'],
        encoder_levels=args['encoder_levels'],
        modes=args['modes'],
        afno_blocks=args['afno_blocks'],
    ).to(device)

    noa.load_state_dict(checkpoint['model_state_dict'])
    noa.train()

    # Generate test trajectory
    u0 = torch.randn(1, 1, 64, 64, device=device)
    target = torch.randn(1, window_size, 1, 64, 64, device=device) * 0.5  # Realistic target

    # Warmup phase (no grad)
    warmup_steps = 256 - window_size
    x = u0
    with torch.no_grad():
        for _ in range(warmup_steps):
            x = noa.single_step(x)

    # Supervised phase (with grad)
    trajectory = [x]
    for _ in range(window_size):
        x = noa.single_step(x)
        trajectory.append(x)

    pred = torch.stack(trajectory[1:], dim=1)  # [1, window_size, 1, 64, 64]

    # Compute loss and backward
    loss = F.mse_loss(pred, target)
    noa.zero_grad()

    try:
        loss.backward()

        # Analyze gradients
        grad_norms = []
        nan_count = 0
        inf_count = 0

        for name, param in noa.named_parameters():
            if param.grad is not None:
                grad = param.grad

                if torch.isnan(grad).any():
                    nan_count += 1
                elif torch.isinf(grad).any():
                    inf_count += 1
                else:
                    grad_norms.append(grad.norm().item())

        if grad_norms:
            return {
                "window": window_size,
                "status": "success" if (nan_count == 0 and inf_count == 0) else "unstable",
                "nan_params": nan_count,
                "inf_params": inf_count,
                "min_grad_norm": min(grad_norms),
                "max_grad_norm": max(grad_norms),
                "mean_grad_norm": np.mean(grad_norms),
                "median_grad_norm": np.median(grad_norms),
                "p95_grad_norm": np.percentile(grad_norms, 95),
                "p99_grad_norm": np.percentile(grad_norms, 99),
            }
        else:
            return {
                "window": window_size,
                "status": "failed",
                "nan_params": nan_count,
                "inf_params": inf_count,
            }

    except RuntimeError as e:
        return {
            "window": window_size,
            "status": "error",
            "error": str(e),
        }


def main():
    checkpoint = "checkpoints/noa/step_500.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("TESTING BPTT WINDOW SIZES")
    print("=" * 80)
    print(f"\nCheckpoint: {checkpoint}")
    print(f"Device: {device}")
    print(f"\nTesting windows: 8, 16, 32, 64, 128")
    print("\n" + "=" * 80)

    windows = [8, 16, 32, 64, 128]
    results = []

    for window in windows:
        print(f"\n### Window = {window} steps")

        # Clear GPU cache before each test
        if device == "cuda":
            torch.cuda.empty_cache()

        try:
            result = test_bptt_window(checkpoint, window, device)
            results.append(result)
        except Exception as e:
            result = {"window": window, "status": "error", "error": str(e)[:100]}
            results.append(result)

        # Clear cache after test
        if device == "cuda":
            torch.cuda.empty_cache()

        if result["status"] == "success":
            print(f"  Status: ✅ {result['status'].upper()}")
            print(f"  NaN params: {result['nan_params']}")
            print(f"  Inf params: {result['inf_params']}")
            print(f"  Gradient norms:")
            print(f"    Min:    {result['min_grad_norm']:.2e}")
            print(f"    Max:    {result['max_grad_norm']:.2e}")
            print(f"    Mean:   {result['mean_grad_norm']:.2e}")
            print(f"    Median: {result['median_grad_norm']:.2e}")
            print(f"    95th:   {result['p95_grad_norm']:.2e}")
            print(f"    99th:   {result['p99_grad_norm']:.2e}")
        elif result["status"] == "unstable":
            print(f"  Status: ⚠️  {result['status'].upper()}")
            print(f"  NaN params: {result['nan_params']}")
            print(f"  Inf params: {result['inf_params']}")
            if 'max_grad_norm' in result:
                print(f"  Max grad norm: {result['max_grad_norm']:.2e}")
        else:
            print(f"  Status: 🔴 {result['status'].upper()}")
            if 'error' in result:
                print(f"  Error: {result['error']}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n| Window | Status | NaN Params | Inf Params | Max Grad Norm |")
    print("|--------|--------|------------|------------|---------------|")

    for r in results:
        status_emoji = {
            "success": "✅",
            "unstable": "⚠️",
            "failed": "🔴",
            "error": "🔴"
        }.get(r["status"], "❓")

        max_norm = f"{r['max_grad_norm']:.2e}" if 'max_grad_norm' in r else "N/A"

        print(f"| {r['window']:6d} | {status_emoji} {r['status']:8s} | "
              f"{r.get('nan_params', 0):10d} | {r.get('inf_params', 0):10d} | "
              f"{max_norm:13s} |")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Find largest successful window
    successful = [r for r in results if r["status"] == "success"]

    if successful:
        max_safe = max(r["window"] for r in successful)
        print(f"\n✅ Maximum stable window: {max_safe} steps")

        # Find optimal (largest safe window with reasonable gradients)
        optimal_candidates = [r for r in successful if r['max_grad_norm'] < 1e8]
        if optimal_candidates:
            optimal = max(optimal_candidates, key=lambda r: r['window'])
            print(f"✅ Recommended window: {optimal['window']} steps")
            print(f"   (Max gradient norm: {optimal['max_grad_norm']:.2e})")
        else:
            print(f"✅ Recommended window: {max_safe} steps")
            print(f"   (All stable windows have large gradients - use gradient clipping)")
    else:
        print("\n🔴 No stable windows found!")
        print("   Consider:")
        print("   - Smaller model")
        print("   - Stronger gradient clipping")
        print("   - Residual connections with smaller scale")


if __name__ == "__main__":
    main()
