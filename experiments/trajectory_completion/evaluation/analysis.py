#!/usr/bin/env python
"""
Analyze trajectory completion results.

Usage:
    python -m experiments.trajectory_completion.evaluation.analysis \
        --results_dir experiments/trajectory_completion/results/baseline
"""

import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def plot_training_history(results_dir: Path):
    """Plot training curves."""
    with open(results_dir / "training_history.json", 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    train_acc = [m['accuracy'] for m in history['train_metrics']]
    val_acc = [m['accuracy'] for m in history['val_metrics']]
    axes[1].plot(train_acc, label='Train')
    axes[1].plot(val_acc, label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Token Accuracy')
    axes[1].set_title('Token Prediction Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # Reconstruction error
    recon_errors = [m['reconstruction_error'] for m in history['val_metrics']]
    axes[2].plot(recon_errors)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MSE')
    axes[2].set_title('Feature Reconstruction Error')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    print(f"Saved: {results_dir / 'training_curves.png'}")


def summarize_results(results_dir: Path):
    """Print summary statistics."""
    with open(results_dir / "training_history.json", 'r') as f:
        history = json.load(f)

    print("=" * 60)
    print("TRAJECTORY COMPLETION RESULTS")
    print("=" * 60)

    final_metrics = history['val_metrics'][-1]

    print(f"\nFinal Validation Metrics:")
    print(f"  Token Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Reconstruction Error (MSE): {final_metrics['reconstruction_error']:.6f}")

    # Best epoch
    best_epoch = np.argmin(history['val_loss'])
    print(f"\nBest Epoch: {best_epoch + 1}")
    print(f"  Val Loss: {history['val_loss'][best_epoch]:.4f}")
    print(f"  Token Accuracy: {history['val_metrics'][best_epoch]['accuracy']:.4f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=Path, required=True)
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    summarize_results(args.results_dir)
    plot_training_history(args.results_dir)


if __name__ == '__main__':
    main()
