"""QBM-specific renderers for quantum observables."""
import torch
from typing import Optional, Literal
from spinlock.visualization.core.renderer import RenderStrategy
from spinlock.visualization.colormaps import GPUColormap


class QBMRenderer(RenderStrategy):
    """Render probability density |ψ|² from complex wavefunctions.

    Supports both pre-computed probability densities and on-the-fly
    conversion from complex [real, imag] representation.

    Extends base RenderStrategy with QBM-specific normalization modes
    optimized for quantum probability distributions.
    """

    def __init__(
        self,
        colormap: str = "viridis",
        normalize_mode: Literal["percentile", "global", "per_frame"] = "percentile",
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        device: torch.device = torch.device("cuda"),
    ):
        """Initialize QBM renderer.

        Args:
            colormap: Matplotlib colormap name
            normalize_mode: Normalization strategy
                - "percentile": Robust clipping via percentiles
                - "global": Use dataset-wide statistics (requires global_stats)
                - "per_frame": Normalize each frame independently
            percentile_low: Low percentile for clipping (0-100)
            percentile_high: High percentile for clipping (0-100)
            device: Torch device for colormap
        """
        super().__init__()
        self.colormap = GPUColormap(colormap, device=device)
        self.normalize_mode = normalize_mode
        self.percentile_low = percentile_low / 100.0
        self.percentile_high = percentile_high / 100.0
        self.device = device

    def supports_channels(self, num_channels: int) -> bool:
        """Support 1 channel (probability) or 2 channels (real/imag)."""
        return num_channels in [1, 2]

    def render(
        self,
        data: torch.Tensor,
        global_stats: Optional[dict] = None
    ) -> torch.Tensor:
        """Render probability density to RGB image.

        Args:
            data: Either [B, H, W], [B, 1, H, W] (probability),
                  or [B, 2, H, W] (real/imag)
            global_stats: Optional normalization statistics with 'vmin', 'vmax'

        Returns:
            RGB image [B, H, W, 3] in range [0, 1]
        """
        # Extract probability density
        prob_density = self._extract_probability(data)  # [B, H, W]

        # Normalize based on mode
        if self.normalize_mode == "global" and global_stats is not None:
            normalized = self._normalize(
                prob_density,
                vmin=global_stats['vmin'],
                vmax=global_stats['vmax']
            )
        elif self.normalize_mode == "percentile":
            normalized = self._normalize(
                prob_density,
                percentile_clip=(self.percentile_low, self.percentile_high)
            )
        else:  # per_frame
            normalized = self._normalize(prob_density)

        # Apply colormap [B, H, W] → [B, 3, H, W]
        rgb = self.colormap.apply(normalized)

        # Permute to [B, H, W, 3] for easier grid stacking
        rgb = rgb.permute(0, 2, 3, 1)

        return rgb

    def compute_global_stats(self, data: torch.Tensor) -> dict:
        """Compute statistics across entire dataset for consistent normalization.

        Args:
            data: Full rollout [N, M, T, C, H, W] or [N, M, T, H, W]

        Returns:
            Dict with 'vmin', 'vmax' for normalization
        """
        # Extract probability density
        prob_density = self._extract_probability(data)

        # Sample a subset if tensor is too large (> 10M elements)
        flat = prob_density.flatten()
        if flat.numel() > 10_000_000:
            # Sample 10M random elements for statistics
            indices = torch.randperm(flat.numel())[:10_000_000]
            flat = flat[indices]

        # Compute percentile-based bounds
        vmin = torch.quantile(flat, self.percentile_low).item()
        vmax = torch.quantile(flat, self.percentile_high).item()

        return {'vmin': vmin, 'vmax': vmax}

    def _extract_probability(self, data: torch.Tensor) -> torch.Tensor:
        """Extract probability density from various input formats.

        Args:
            data: Input tensor in various formats

        Returns:
            Probability density [B, H, W] or [N, M, T, H, W]
        """
        # Handle different input shapes
        if data.ndim == 3:
            # Already [B, H, W]
            return data
        elif data.ndim == 4:
            if data.shape[1] == 2:
                # [B, 2, H, W] - complex representation
                real, imag = data[:, 0], data[:, 1]
                return real**2 + imag**2
            elif data.shape[1] == 1:
                # [B, 1, H, W] - squeeze channel dim
                return data.squeeze(1)
            else:
                raise ValueError(f"Unexpected shape for 4D input: {data.shape}")
        elif data.ndim == 6:
            # Full rollout [N, M, T, C, H, W]
            if data.shape[3] == 2:
                real, imag = data[..., 0, :, :], data[..., 1, :, :]
                return real**2 + imag**2
            elif data.shape[3] == 1:
                return data.squeeze(3)
            else:
                raise ValueError(f"Unexpected channels in 6D input: {data.shape[3]}")
        elif data.ndim == 5:
            # Already probability [N, M, T, H, W]
            return data
        else:
            raise ValueError(f"Unsupported input shape: {data.shape}")

    def _normalize(
        self,
        x: torch.Tensor,
        percentile_clip: Optional[tuple] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ) -> torch.Tensor:
        """Normalize tensor to [0, 1] range.

        Reuses base class normalization logic with QBM-specific defaults.

        Args:
            x: Tensor to normalize
            percentile_clip: Optional (low, high) percentile clipping
            vmin: Optional explicit minimum value
            vmax: Optional explicit maximum value

        Returns:
            Normalized tensor in [0, 1]
        """
        if vmin is not None and vmax is not None:
            # Use explicit bounds (for global normalization)
            pass
        elif percentile_clip:
            low, high = percentile_clip
            vmin = torch.quantile(x.flatten(), low).item()
            vmax = torch.quantile(x.flatten(), high).item()
        else:
            # Per-frame normalization
            vmin = x.min().item()
            vmax = x.max().item()

        # Avoid division by zero
        if abs(vmax - vmin) < 1e-8:
            return torch.zeros_like(x)

        return (x - vmin) / (vmax - vmin)
