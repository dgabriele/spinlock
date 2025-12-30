"""Normalization utilities for VQ-VAE token training.

Provides:
- Standard normalization (zero-mean, unit-variance)
- Robust normalization (MAD-based, outlier-resistant)
- L2 normalization (unit vector)
- Statistics management (save/load)

Ported from unisim.system.data.normalization (100% generic, no domain-specific code).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch


@dataclass
class NormalizationStats:
    """Statistics for standard normalization."""

    mean: np.ndarray
    std: np.ndarray

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, data: dict):
        """Load from dictionary."""
        return cls(mean=data["mean"], std=data["std"])


def standard_normalize(
    x: Union[np.ndarray, torch.Tensor], eps: float = 1e-8
) -> Union[np.ndarray, torch.Tensor]:
    """Apply standard normalization (zero-mean, unit-variance).

    Args:
        x: Input data [N, D]
        eps: Small constant to avoid division by zero

    Returns:
        Normalized data [N, D] with mean=0, std=1 per dimension
    """
    if isinstance(x, torch.Tensor):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        # Avoid division by zero for constant dimensions
        std = torch.where(std < eps, torch.ones_like(std), std)
        return (x - mean) / std
    else:
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        # Avoid division by zero for constant dimensions
        std = np.where(std < eps, 1.0, std)
        return (x - mean) / std


def robust_normalize(
    x: Union[np.ndarray, torch.Tensor], eps: float = 1e-8
) -> Union[np.ndarray, torch.Tensor]:
    """Apply robust normalization using MAD (Median Absolute Deviation).

    Robust to outliers. Uses median instead of mean, MAD instead of std.

    Args:
        x: Input data [N, D]
        eps: Small constant to avoid division by zero

    Returns:
        Normalized data [N, D] with median=0, MAD≈1 per dimension

    Note:
        MAD = median(|x - median(x)|) * 1.4826
        The 1.4826 factor makes MAD comparable to std for normal distributions.
    """
    if isinstance(x, torch.Tensor):
        median = x.median(dim=0, keepdim=True).values
        mad = torch.median(torch.abs(x - median), dim=0, keepdim=True).values * 1.4826
        mad = torch.where(mad < eps, torch.ones_like(mad), mad)
        return (x - median) / mad
    else:
        median = np.median(x, axis=0, keepdims=True)
        mad = np.median(np.abs(x - median), axis=0, keepdims=True) * 1.4826
        mad = np.where(mad < eps, 1.0, mad)
        return (x - median) / mad


def l2_normalize(
    x: Union[np.ndarray, torch.Tensor], eps: float = 1e-8
) -> Union[np.ndarray, torch.Tensor]:
    """Apply L2 normalization (unit vector).

    Args:
        x: Input data [N, D]
        eps: Small constant to avoid division by zero

    Returns:
        L2-normalized data [N, D] with ||x||_2 = 1 per sample
    """
    if isinstance(x, torch.Tensor):
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        norm = torch.clamp(norm, min=eps)
        return x / norm
    else:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm = np.maximum(norm, eps)
        return x / norm


def compute_normalization_stats(
    data: Union[np.ndarray, torch.Tensor]
) -> NormalizationStats:
    """Compute normalization statistics for later use.

    Args:
        data: Input data [N, D]

    Returns:
        NormalizationStats with mean and std per dimension
    """
    if isinstance(data, torch.Tensor):
        mean = data.mean(dim=0).cpu().numpy()
        std = data.std(dim=0).cpu().numpy()
    else:
        mean = data.mean(axis=0)
        std = data.std(axis=0)

    # Avoid division by zero for constant dimensions
    std = np.where(std < 1e-8, 1.0, std)

    return NormalizationStats(mean=mean, std=std)


def apply_standard_normalization(
    data: Union[np.ndarray, torch.Tensor], stats: NormalizationStats
) -> Union[np.ndarray, torch.Tensor]:
    """Apply pre-computed normalization statistics.

    Args:
        data: Input data [N, D]
        stats: Pre-computed normalization statistics

    Returns:
        Normalized data [N, D]
    """
    if isinstance(data, torch.Tensor):
        mean_t = torch.from_numpy(stats.mean).to(data.device).float()
        std_t = torch.from_numpy(stats.std).to(data.device).float()
        return (data - mean_t) / std_t
    else:
        return (data - stats.mean) / stats.std


def save_normalization_stats(stats: NormalizationStats, path: Path):
    """Save normalization statistics to disk.

    Args:
        stats: Normalization statistics
        path: Output path (.npz format)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **stats.to_dict())


def load_normalization_stats(path: Path) -> NormalizationStats:
    """Load normalization statistics from disk.

    Args:
        path: Path to .npz file

    Returns:
        NormalizationStats instance
    """
    data = np.load(path)
    return NormalizationStats.from_dict(
        {"mean": data["mean"], "std": data["std"]}
    )
