"""Fix checkpoint by reformatting val_metrics to final_metrics format."""
import torch
from pathlib import Path

def fix_checkpoint_metrics(checkpoint_path: Path):
    """Reformat validation metrics to final_metrics format for visualization."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get last validation metrics
    if 'metadata' not in ckpt or 'training_history' not in ckpt['metadata']:
        print("ERROR: No training_history found in checkpoint")
        return

    history = ckpt['metadata']['training_history']
    if 'val_metrics' not in history or not history['val_metrics']:
        print("ERROR: No val_metrics found in training_history")
        return

    last_val = history['val_metrics'][-1]
    print(f"Found last validation metrics from epoch {len(history['val_metrics'])}")

    # Extract per-quantizer utilization and per-category MSE
    per_quantizer_util = last_val.get('per_quantizer_utilization', {})
    per_category_mse = last_val.get('per_category_mse', {})

    print(f"  Per-quantizer utilization: {len(per_quantizer_util)} quantizers")
    print(f"  Per-category MSE: {len(per_category_mse)} categories")

    # Format for visualization compatibility
    final_metrics = {}

    # Parse quantizer keys like "temporal_group_1_L0" → category + level
    for quantizer_name, utilization in per_quantizer_util.items():
        # Try to parse hierarchical format
        parts = quantizer_name.rsplit('_L', 1)
        if len(parts) == 2:
            category = parts[0]
            level = parts[1]
            try:
                level_num = int(level)

                # Format: "{category}/level_{level}/utilization"
                # Convert from percentage (0-100) to fraction (0-1)
                key = f"{category}/level_{level_num}/utilization"
                final_metrics[key] = utilization / 100.0
            except ValueError:
                # Fallback for non-hierarchical quantizers
                final_metrics[f"{quantizer_name}/utilization"] = utilization / 100.0
        else:
            # Fallback for non-hierarchical quantizers
            final_metrics[f"{quantizer_name}/utilization"] = utilization / 100.0

    # Add per-category reconstruction MSE
    for category, mse in per_category_mse.items():
        final_metrics[f"{category}/reconstruction_mse"] = mse

    print(f"\nFormatted {len(final_metrics)} metrics for visualization")

    # Show sample metrics
    print("\nSample utilization metrics:")
    util_keys = [k for k in sorted(final_metrics.keys()) if 'utilization' in k]
    for k in util_keys[:5]:
        print(f"  {k}: {final_metrics[k]:.4f}")

    print("\nSample MSE metrics:")
    mse_keys = [k for k in sorted(final_metrics.keys()) if 'mse' in k]
    for k in mse_keys[:3]:
        print(f"  {k}: {final_metrics[k]:.6f}")

    # Add to training history
    history['final_metrics'] = final_metrics
    ckpt['metadata']['training_history'] = history

    # Save updated checkpoint
    print(f"\nSaving updated checkpoint to: {checkpoint_path}")
    torch.save(ckpt, checkpoint_path)
    print("✓ Checkpoint updated successfully!")

if __name__ == "__main__":
    checkpoint_path = Path("checkpoints/v2/vqvae/vq_tokenizer_best.pt")
    fix_checkpoint_metrics(checkpoint_path)
