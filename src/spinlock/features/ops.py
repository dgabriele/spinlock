"""Gradient-safe feature computation operations.

Provides two implementations of common numerical operations used by feature extractors:

- ``StandardOps``: Zero-overhead wrappers around native PyTorch ops. Used for
  inference and dataset-generation paths where gradients are not needed.

- ``DifferentiableOps``: Same forward values, bounded backward gradients. Uses
  custom ``autograd.Function`` subclasses for division and logarithm, and soft
  continuous approximations for non-differentiable ops (argmax, median, step
  functions, bincount/histograms). Used during MNO discrete-token training
  where gradients must flow back through the feature extractors.

**Design invariant:** Forward outputs are identical (or negligibly different)
between StandardOps and DifferentiableOps. Feature values don't change — only
backward gradients are bounded. VQ token assignments are therefore unaffected.

Usage::

    from spinlock.features.ops import create_ops

    ops = create_ops(differentiable=True)
    result = ops.safe_div(numerator, denominator)  # grad clamped to [-100, 100]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class FeatureOps(ABC):
    """Interface for feature computation operations."""

    # --- Division & log ---------------------------------------------------

    @abstractmethod
    def safe_div(self, a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
        """Division with bounded gradient."""
        ...

    @abstractmethod
    def safe_log(self, x: Tensor, eps: float = 1e-10) -> Tensor:
        """Logarithm with bounded gradient."""
        ...

    # --- Higher-order moments ---------------------------------------------

    @abstractmethod
    def standardize(self, x: Tensor, dim: int | Tuple[int, ...], eps: float = 1e-4) -> Tuple[Tensor, Tensor]:
        """Return (z-scores, std)."""
        ...

    @abstractmethod
    def kurtosis(self, x: Tensor, dim: int | Tuple[int, ...]) -> Tensor:
        """Excess kurtosis (= E[z^4] - 3)."""
        ...

    @abstractmethod
    def skewness(self, x: Tensor, dim: int | Tuple[int, ...]) -> Tensor:
        """Skewness (= E[z^3])."""
        ...

    # --- Non-differentiable → soft approximations -------------------------

    @abstractmethod
    def soft_argmax(self, x: Tensor, dim: int = -1, tau: float = 0.1) -> Tensor:
        """Soft or hard argmax depending on mode."""
        ...

    @abstractmethod
    def soft_median(self, x: Tensor, dim: int = -1) -> Tensor:
        """Soft or hard median depending on mode."""
        ...

    @abstractmethod
    def soft_step(self, x: Tensor, threshold: float | Tensor, sharpness: float = 10.0) -> Tensor:
        """Soft or hard step function depending on mode."""
        ...

    # --- Histogram / discretisation ---------------------------------------

    @abstractmethod
    def soft_histogram(
        self, x: Tensor, num_bins: int, x_min: Tensor, x_max: Tensor,
    ) -> Tensor:
        """Soft or hard histogram depending on mode."""
        ...

    # --- Linear algebra ---------------------------------------------------

    @abstractmethod
    def safe_eigvalsh(self, x: Tensor) -> Tensor:
        """Eigenvalues of symmetric matrix with stable backward."""
        ...

    @abstractmethod
    def safe_svd(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """SVD with stable backward."""
        ...

    # --- Outlier clipping (centralises 3× duplicated logic) ---------------

    def outlier_clip(self, values: Tensor, iqr_multiplier: float = 10.0) -> Tensor:
        """Adaptive IQR-based outlier clipping.

        Computes bounds from non-NaN values and clips to
        ``[Q1 - k*IQR, Q3 + k*IQR]``.
        """
        if not values.is_floating_point():
            values = values.float()
        values_flat = values.flatten()
        valid_mask = ~torch.isnan(values_flat)
        valid_values = values_flat[valid_mask]
        if valid_values.numel() < 4:
            return values
        q1 = torch.quantile(valid_values, 0.25)
        q3 = torch.quantile(valid_values, 0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        return torch.clamp(values, min=lower, max=upper)


# ---------------------------------------------------------------------------
# StandardOps — zero overhead, delegates to native PyTorch
# ---------------------------------------------------------------------------

class StandardOps(FeatureOps):
    """Native PyTorch ops — identical to current codebase behaviour."""

    # Division / log -------------------------------------------------------

    def safe_div(self, a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
        return a / (b + eps)

    def safe_log(self, x: Tensor, eps: float = 1e-10) -> Tensor:
        return torch.log(x + eps)

    # Moments --------------------------------------------------------------

    def standardize(self, x: Tensor, dim, eps: float = 1e-4):
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True).clamp(min=eps)
        z = (x - mean) / std
        return z, std

    def kurtosis(self, x: Tensor, dim):
        z, _ = self.standardize(x, dim)
        return (z ** 4).mean(dim=dim) - 3.0

    def skewness(self, x: Tensor, dim):
        z, _ = self.standardize(x, dim)
        return (z ** 3).mean(dim=dim)

    # Non-differentiable ops — hard versions -------------------------------

    def soft_argmax(self, x: Tensor, dim: int = -1, tau: float = 0.1) -> Tensor:
        return torch.argmax(x, dim=dim).float()

    def soft_median(self, x: Tensor, dim: int = -1) -> Tensor:
        return torch.median(x, dim=dim).values

    def soft_step(self, x: Tensor, threshold, sharpness: float = 10.0) -> Tensor:
        return (x > threshold).float()

    # Histogram ------------------------------------------------------------

    def soft_histogram(self, x: Tensor, num_bins: int, x_min: Tensor, x_max: Tensor) -> Tensor:
        # Hard histogram: normalise → discretise → bincount
        x_norm = (x - x_min) / (x_max - x_min + 1e-10) * (num_bins - 1e-6)
        x_idx = x_norm.clamp(0, num_bins - 1).long()
        # One-hot accumulate
        one_hot = F.one_hot(x_idx, num_classes=num_bins).float()  # [..., N, num_bins]
        hist = one_hot.sum(dim=-2)  # [..., num_bins]
        return hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)

    # Linear algebra -------------------------------------------------------

    def safe_eigvalsh(self, x: Tensor) -> Tensor:
        return torch.linalg.eigvalsh(x)

    def safe_svd(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return torch.linalg.svd(x, full_matrices=False)


# ---------------------------------------------------------------------------
# Custom autograd functions for DifferentiableOps
# ---------------------------------------------------------------------------

class _SafeDiv(torch.autograd.Function):
    """Division with gradient clamping."""

    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor, eps: float, max_grad: float) -> Tensor:
        safe_b = b.abs().clamp(min=eps)
        sign_b = b.sign()
        # Where b == 0, treat sign as +1 to avoid 0 * inf
        sign_b = torch.where(sign_b == 0, torch.ones_like(sign_b), sign_b)
        result = a / (safe_b * sign_b)
        ctx.save_for_backward(a, safe_b)
        ctx.max_grad = max_grad
        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        a, safe_b = ctx.saved_tensors
        mg = ctx.max_grad
        grad_a = (grad_output / safe_b).clamp(-mg, mg)
        grad_b = (-grad_output * a / safe_b.square()).clamp(-mg, mg)
        return grad_a, grad_b, None, None


class _SafeLog(torch.autograd.Function):
    """Logarithm with gradient clamping."""

    @staticmethod
    def forward(ctx, x: Tensor, eps: float, max_grad: float) -> Tensor:
        safe_x = x.abs().clamp(min=eps)
        ctx.save_for_backward(safe_x)
        ctx.max_grad = max_grad
        return torch.log(safe_x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (safe_x,) = ctx.saved_tensors
        mg = ctx.max_grad
        grad = (grad_output / safe_x).clamp(-mg, mg)
        return grad, None, None


# ---------------------------------------------------------------------------
# DifferentiableOps — bounded gradients for training
# ---------------------------------------------------------------------------

class DifferentiableOps(FeatureOps):
    """Same forward values as StandardOps, but with bounded backward gradients.

    Custom ``autograd.Function`` subclasses clamp division/log gradients.
    Non-differentiable ops (argmax, median, step, histogram) are replaced with
    soft continuous approximations that provide meaningful gradient signal.

    Args:
        max_grad: Maximum absolute gradient for div/log (default 100.0).
        z_clamp: Maximum z-score for moment computations (default 6.0).
        tau: Temperature for soft-argmax (default 0.1).
        step_sharpness: Sharpness for sigmoid soft-step (default 10.0).
    """

    def __init__(
        self,
        max_grad: float = 100.0,
        z_clamp: float = 6.0,
        tau: float = 0.1,
        step_sharpness: float = 10.0,
    ):
        self.max_grad = max_grad
        self.z_clamp = z_clamp
        self.tau = tau
        self.step_sharpness = step_sharpness

    # Division / log -------------------------------------------------------

    def safe_div(self, a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
        return _SafeDiv.apply(a, b, eps, self.max_grad)

    def safe_log(self, x: Tensor, eps: float = 1e-10) -> Tensor:
        return _SafeLog.apply(x, eps, self.max_grad)

    # Moments (z-score clamped) --------------------------------------------

    def standardize(self, x: Tensor, dim, eps: float = 1e-4):
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True).clamp(min=eps)
        z = self.safe_div(x - mean, std)
        return z, std

    def kurtosis(self, x: Tensor, dim):
        z, _ = self.standardize(x, dim)
        z = z.clamp(-self.z_clamp, self.z_clamp)
        return (z ** 4).mean(dim=dim) - 3.0

    def skewness(self, x: Tensor, dim):
        z, _ = self.standardize(x, dim)
        z = z.clamp(-self.z_clamp, self.z_clamp)
        return (z ** 3).mean(dim=dim)

    # Soft approximations --------------------------------------------------

    def soft_argmax(self, x: Tensor, dim: int = -1, tau: float | None = None) -> Tensor:
        """Weighted-sum with sharpened softmax (continuous relaxation of argmax)."""
        _tau = tau if tau is not None else self.tau
        weights = F.softmax(x / _tau, dim=dim)
        indices = torch.arange(x.shape[dim], device=x.device, dtype=x.dtype)
        # Broadcast indices to match x shape
        shape = [1] * x.ndim
        shape[dim] = x.shape[dim]
        indices = indices.view(shape).expand_as(x)
        return (weights * indices).sum(dim=dim)

    def soft_median(self, x: Tensor, dim: int = -1) -> Tensor:
        """Differentiable median via torch.quantile (sorted interpolation backward)."""
        return torch.quantile(x.float(), 0.5, dim=dim)

    def soft_step(self, x: Tensor, threshold, sharpness: float | None = None) -> Tensor:
        """Sigmoid approximation of step function."""
        _s = sharpness if sharpness is not None else self.step_sharpness
        return torch.sigmoid(_s * (x - threshold))

    # Soft histogram -------------------------------------------------------

    def soft_histogram(self, x: Tensor, num_bins: int, x_min: Tensor, x_max: Tensor) -> Tensor:
        """Gaussian-kernel soft assignment instead of hard binning."""
        centers = torch.linspace(0, 1, num_bins, device=x.device, dtype=x.dtype)
        # Normalise x to [0, 1]
        x_norm = (x - x_min) / (x_max - x_min + 1e-10)
        bin_width = 1.0 / max(num_bins - 1, 1)
        # x_norm: [..., N], centers: [num_bins] → diffs: [..., N, num_bins]
        diffs = (x_norm.unsqueeze(-1) - centers) / (bin_width * 0.5)
        weights = torch.exp(-0.5 * diffs.square())
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        hist = weights.sum(dim=-2)  # [..., num_bins]
        return hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)

    # Linear algebra (with jitter for stability) ---------------------------

    def safe_eigvalsh(self, x: Tensor) -> Tensor:
        # Add small jitter to diagonal for repeated-eigenvalue stability
        jitter = 1e-6 * torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
        return torch.linalg.eigvalsh(x + jitter)

    def safe_svd(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Add small jitter to prevent degenerate singular values
        jitter = 1e-6 * torch.randn_like(x)
        return torch.linalg.svd(x + jitter, full_matrices=False)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_ops(differentiable: bool = False) -> FeatureOps:
    """Create a FeatureOps instance.

    Args:
        differentiable: If True, return DifferentiableOps with bounded
            gradients. If False, return StandardOps (zero overhead).

    Returns:
        FeatureOps instance.
    """
    return DifferentiableOps() if differentiable else StandardOps()
