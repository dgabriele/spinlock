"""
Aggregate renderers for ensemble statistics across realizations.

Provides visualization strategies for summarizing multiple stochastic
realizations: mean field, variance maps, etc.
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional
from .renderer import RenderStrategy, HeatmapRenderer


class AggregateRenderer(ABC):
    """
    Abstract base for ensemble aggregation strategies.

    Aggregates M stochastic realizations into summary visualizations
    (e.g., mean, variance, entropy).
    """

    @abstractmethod
    def aggregate(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Aggregate M realizations to single summary.

        Args:
            realizations: [M, C, H, W] - M realizations

        Returns:
            Aggregated result [C, H, W] or [1, H, W]
        """
        pass

    @abstractmethod
    def render(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Aggregate and render to RGB.

        Args:
            realizations: [M, C, H, W]

        Returns:
            RGB image [3, H, W]
        """
        pass


class MeanFieldRenderer(AggregateRenderer):
    """
    Mean field across realizations.

    Computes pixel-wise mean across all M realizations,
    visualizing the expected/average state.

    Example:
        ```python
        renderer = MeanFieldRenderer(base_renderer=rgb_renderer)
        realizations = torch.randn(10, 3, 64, 64)  # 10 realizations
        mean_rgb = renderer.render(realizations)  # [3, 64, 64]
        ```
    """

    def __init__(
        self,
        base_renderer: RenderStrategy,
        device: torch.device = torch.device("cuda")
    ):
        """
        Initialize mean field renderer.

        Args:
            base_renderer: Renderer for the mean field
            device: Torch device
        """
        self.base_renderer = base_renderer
        self.device = device

    def aggregate(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Compute mean field.

        Args:
            realizations: [M, C, H, W]

        Returns:
            Mean field [C, H, W]
        """
        return realizations.mean(dim=0)  # [M, C, H, W] -> [C, H, W]

    def render(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Render mean field to RGB.

        Args:
            realizations: [M, C, H, W]

        Returns:
            RGB image [3, H, W]
        """
        # Compute mean
        mean_field = self.aggregate(realizations)  # [C, H, W]

        # Add batch dimension for renderer
        mean_field = mean_field.unsqueeze(0)  # [1, C, H, W]

        # Render
        rgb = self.base_renderer.render(mean_field)  # [1, 3, H, W]

        # Remove batch dimension
        return rgb.squeeze(0)  # [3, H, W]


class VarianceMapRenderer(AggregateRenderer):
    """
    Variance/uncertainty visualization across realizations.

    Computes spatial variance to show where realizations diverge,
    indicating regions of high uncertainty or stochasticity.

    Example:
        ```python
        renderer = VarianceMapRenderer(colormap="hot")
        realizations = torch.randn(10, 3, 64, 64)  # 10 realizations
        var_rgb = renderer.render(realizations)  # [3, 64, 64]
        ```
    """

    def __init__(
        self,
        colormap: str = "hot",
        device: torch.device = torch.device("cuda")
    ):
        """
        Initialize variance map renderer.

        Args:
            colormap: Colormap for variance visualization
            device: Torch device
        """
        self.heatmap = HeatmapRenderer(colormap=colormap, device=device)
        self.device = device

    def aggregate(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Compute variance magnitude across realizations.

        Args:
            realizations: [M, C, H, W]

        Returns:
            Variance magnitude [1, H, W]
        """
        # Compute variance per channel: [M, C, H, W] -> [C, H, W]
        var_per_channel = realizations.var(dim=0)

        # Compute magnitude (L2 norm across channels): [C, H, W] -> [1, H, W]
        variance_magnitude = var_per_channel.norm(dim=0, keepdim=True)

        return variance_magnitude

    def render(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Render variance map to RGB.

        Args:
            realizations: [M, C, H, W]

        Returns:
            RGB image [3, H, W]
        """
        # Compute variance magnitude
        var_map = self.aggregate(realizations)  # [1, H, W]

        # Add batch dimension
        var_map = var_map.unsqueeze(0)  # [1, 1, H, W]

        # Render as heatmap
        rgb = self.heatmap.render(var_map)  # [1, 3, H, W]

        # Remove batch dimension
        return rgb.squeeze(0)  # [3, H, W]


class StdDevMapRenderer(AggregateRenderer):
    """
    Standard deviation map across realizations.

    Similar to variance but uses standard deviation (square root of variance)
    for more interpretable magnitude visualization.

    Example:
        ```python
        renderer = StdDevMapRenderer(colormap="plasma")
        realizations = torch.randn(10, 3, 64, 64)
        std_rgb = renderer.render(realizations)  # [3, 64, 64]
        ```
    """

    def __init__(
        self,
        colormap: str = "plasma",
        device: torch.device = torch.device("cuda")
    ):
        """
        Initialize standard deviation map renderer.

        Args:
            colormap: Colormap for std dev visualization
            device: Torch device
        """
        self.heatmap = HeatmapRenderer(colormap=colormap, device=device)
        self.device = device

    def aggregate(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Compute standard deviation magnitude.

        Args:
            realizations: [M, C, H, W]

        Returns:
            Std dev magnitude [1, H, W]
        """
        # Compute std dev per channel: [M, C, H, W] -> [C, H, W]
        std_per_channel = realizations.std(dim=0)

        # Compute magnitude: [C, H, W] -> [1, H, W]
        std_magnitude = std_per_channel.norm(dim=0, keepdim=True)

        return std_magnitude

    def render(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Render std dev map to RGB.

        Args:
            realizations: [M, C, H, W]

        Returns:
            RGB image [3, H, W]
        """
        # Compute std dev magnitude
        std_map = self.aggregate(realizations)  # [1, H, W]

        # Add batch dimension
        std_map = std_map.unsqueeze(0)  # [1, 1, H, W]

        # Render as heatmap
        rgb = self.heatmap.render(std_map)  # [1, 3, H, W]

        # Remove batch dimension
        return rgb.squeeze(0)  # [3, H, W]


class EnvelopeRenderer(AggregateRenderer):
    """
    Min/Max envelope visualization.

    Shows the range of values across realizations by overlaying
    min (blue channel) and max (red channel) bounds.

    Example:
        ```python
        renderer = EnvelopeRenderer()
        realizations = torch.randn(10, 3, 64, 64)
        envelope_rgb = renderer.render(realizations)  # [3, 64, 64]
        ```
    """

    def __init__(
        self,
        colormap: str = "coolwarm",
        device: torch.device = torch.device("cuda")
    ):
        """
        Initialize envelope renderer.

        Args:
            colormap: Colormap for rendering (default: coolwarm for min/max)
            device: Torch device
        """
        self.heatmap = HeatmapRenderer(colormap=colormap, device=device)
        self.device = device

    def aggregate(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Compute min/max envelope.

        Args:
            realizations: [M, C, H, W]

        Returns:
            Range magnitude [1, H, W]
        """
        # Min and max per channel
        min_vals = realizations.min(dim=0)[0]  # [C, H, W]
        max_vals = realizations.max(dim=0)[0]  # [C, H, W]

        # Range magnitude (L2 norm across channels)
        range_magnitude = (max_vals - min_vals).norm(dim=0, keepdim=True)

        return range_magnitude

    def render(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Render min/max envelope to RGB.

        Args:
            realizations: [M, C, H, W]

        Returns:
            RGB image [3, H, W]
        """
        range_map = self.aggregate(realizations)  # [1, H, W]
        range_map = range_map.unsqueeze(0)  # [1, 1, H, W]

        rgb = self.heatmap.render(range_map)  # [1, 3, H, W]
        return rgb.squeeze(0)  # [3, H, W]


class TrajectoryOverlayRenderer(AggregateRenderer):
    """
    Overlay all realizations with transparency.

    Creates a "spaghetti plot" by blending all realizations together,
    showing the distribution of outcomes visually.

    Example:
        ```python
        renderer = TrajectoryOverlayRenderer(alpha=0.3)
        realizations = torch.randn(10, 3, 64, 64)
        overlay_rgb = renderer.render(realizations)  # [3, 64, 64]
        ```
    """

    def __init__(
        self,
        base_renderer: RenderStrategy,
        alpha: float = 0.3,
        device: torch.device = torch.device("cuda")
    ):
        """
        Initialize trajectory overlay renderer.

        Args:
            base_renderer: Renderer for individual realizations
            alpha: Transparency for each realization (0-1)
            device: Torch device
        """
        self.base_renderer = base_renderer
        self.alpha = alpha
        self.device = device

    def aggregate(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Overlay is done in render() - returns identity.

        Args:
            realizations: [M, C, H, W]

        Returns:
            Same as input
        """
        return realizations

    def render(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Render overlay of all realizations.

        Args:
            realizations: [M, C, H, W]

        Returns:
            RGB composite [3, H, W]
        """
        M = realizations.shape[0]

        # Render each realization
        rgb_all = self.base_renderer.render(realizations)  # [M, 3, H, W]

        # Alpha blend: weighted average
        composite = rgb_all.mean(dim=0)  # [3, H, W]

        return composite


class EntropyMapRenderer(AggregateRenderer):
    """
    Spatial entropy map showing structural uncertainty.

    Computes Shannon entropy per pixel across realizations,
    revealing where outcomes are most uncertain or multimodal.
    High entropy indicates high variability in *structure*.

    Example:
        ```python
        renderer = EntropyMapRenderer(num_bins=32)
        realizations = torch.randn(10, 3, 64, 64)
        entropy_rgb = renderer.render(realizations)  # [3, 64, 64]
        ```
    """

    def __init__(
        self,
        num_bins: int = 32,
        colormap: str = "inferno",
        device: torch.device = torch.device("cuda")
    ):
        """
        Initialize entropy map renderer.

        Args:
            num_bins: Number of bins for histogram (more = finer resolution)
            colormap: Colormap for entropy visualization
            device: Torch device
        """
        self.num_bins = num_bins
        self.heatmap = HeatmapRenderer(colormap=colormap, device=device)
        self.device = device

    def aggregate(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial entropy map.

        Args:
            realizations: [M, C, H, W]

        Returns:
            Entropy map [1, H, W]
        """
        M, C, H, W = realizations.shape

        # Compute per-pixel histograms across realizations
        # Flatten channels into realizations for simplicity
        values = realizations.permute(2, 3, 0, 1).reshape(H, W, M * C)  # [H, W, M*C]

        # Compute histogram per pixel
        eps = 1e-10
        entropies = torch.zeros(H, W, device=self.device)

        for i in range(H):
            for j in range(W):
                pixel_vals = values[i, j]  # [M*C]

                # Histogram
                hist = torch.histc(pixel_vals, bins=self.num_bins, min=pixel_vals.min(), max=pixel_vals.max())

                # Normalize to probability
                probs = hist / (hist.sum() + eps)

                # Shannon entropy
                entropy = -(probs * torch.log2(probs + eps)).sum()
                entropies[i, j] = entropy

        return entropies.unsqueeze(0)  # [1, H, W]

    def render(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Render entropy map to RGB.

        Args:
            realizations: [M, C, H, W]

        Returns:
            RGB image [3, H, W]
        """
        entropy_map = self.aggregate(realizations)  # [1, H, W]
        entropy_map = entropy_map.unsqueeze(0)  # [1, 1, H, W]

        rgb = self.heatmap.render(entropy_map)  # [1, 3, H, W]
        return rgb.squeeze(0)  # [3, H, W]


class PCAModeRenderer(AggregateRenderer):
    """
    PCA mode visualization showing principal components of variation.

    Performs PCA across realizations and visualizes the first 3 modes
    as RGB channels, showing *how* realizations differ.

    Example:
        ```python
        renderer = PCAModeRenderer()
        realizations = torch.randn(10, 3, 64, 64)
        pca_rgb = renderer.render(realizations)  # [3, 64, 64]
        ```
    """

    def __init__(
        self,
        n_components: int = 3,
        device: torch.device = torch.device("cuda")
    ):
        """
        Initialize PCA mode renderer.

        Args:
            n_components: Number of PCA components to visualize (default: 3 for RGB)
            device: Torch device
        """
        self.n_components = n_components
        self.device = device

    def aggregate(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Compute PCA modes.

        Args:
            realizations: [M, C, H, W]

        Returns:
            PCA modes [n_components, H, W]
        """
        M, C, H, W = realizations.shape

        # Reshape to [M, C*H*W] for PCA
        data = realizations.reshape(M, -1)  # [M, C*H*W]

        # Center data
        mean = data.mean(dim=0, keepdim=True)
        centered = data - mean

        # SVD: X = U @ S @ V^T
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

        # Principal components: V[:, :n_components]
        # Project back to spatial domain
        pcs = Vt[:self.n_components, :]  # [n_components, C*H*W]
        pcs = pcs.reshape(self.n_components, C, H, W)

        # Aggregate across channels for visualization
        pcs_spatial = pcs.norm(dim=1)  # [n_components, H, W]

        return pcs_spatial

    def render(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Render PCA modes as RGB.

        Args:
            realizations: [M, C, H, W]

        Returns:
            RGB image [3, H, W] (PC1=R, PC2=G, PC3=B)
        """
        pcs = self.aggregate(realizations)  # [n_components, H, W]

        # Take first 3 components for RGB
        if pcs.shape[0] < 3:
            # Pad with zeros if fewer than 3 components
            padding = torch.zeros(3 - pcs.shape[0], *pcs.shape[1:], device=self.device)
            pcs = torch.cat([pcs, padding], dim=0)
        else:
            pcs = pcs[:3]

        # Normalize each channel independently
        rgb = torch.zeros_like(pcs)
        for i in range(3):
            pc = pcs[i]
            pc_min, pc_max = pc.min(), pc.max()
            if (pc_max - pc_min).abs() > 1e-8:
                rgb[i] = (pc - pc_min) / (pc_max - pc_min)

        return rgb  # [3, H, W]


class SSIMMapRenderer(AggregateRenderer):
    """
    Structural Similarity (SSIM) map across realizations.

    Computes average pairwise SSIM to show structural consistency.
    High SSIM = realizations have similar structure despite differences.

    Example:
        ```python
        renderer = SSIMMapRenderer()
        realizations = torch.randn(10, 3, 64, 64)
        ssim_rgb = renderer.render(realizations)  # [3, 64, 64]
        ```
    """

    def __init__(
        self,
        window_size: int = 11,
        colormap: str = "viridis",
        device: torch.device = torch.device("cuda")
    ):
        """
        Initialize SSIM map renderer.

        Args:
            window_size: Window size for SSIM computation
            colormap: Colormap for SSIM visualization
            device: Torch device
        """
        self.window_size = window_size
        self.heatmap = HeatmapRenderer(colormap=colormap, device=device)
        self.device = device

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM between two images.

        Args:
            x, y: [C, H, W]

        Returns:
            SSIM map [H, W]
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Simple SSIM without gaussian window (using uniform window)
        mu_x = x.mean(dim=0)
        mu_y = y.mean(dim=0)

        sigma_x = x.var(dim=0)
        sigma_y = y.var(dim=0)
        sigma_xy = ((x - x.mean(dim=0, keepdim=True)) * (y - y.mean(dim=0, keepdim=True))).mean(dim=0)

        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

        return ssim

    def aggregate(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Compute average pairwise SSIM.

        Args:
            realizations: [M, C, H, W]

        Returns:
            Average SSIM map [1, H, W]
        """
        M = realizations.shape[0]
        H, W = realizations.shape[2:]

        ssim_sum = torch.zeros(H, W, device=self.device)
        count = 0

        # Pairwise SSIM
        for i in range(M):
            for j in range(i + 1, M):
                ssim_map = self._ssim(realizations[i], realizations[j])
                ssim_sum += ssim_map
                count += 1

        avg_ssim = ssim_sum / count if count > 0 else ssim_sum

        return avg_ssim.unsqueeze(0)  # [1, H, W]

    def render(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Render SSIM map to RGB.

        Args:
            realizations: [M, C, H, W]

        Returns:
            RGB image [3, H, W]
        """
        ssim_map = self.aggregate(realizations)  # [1, H, W]
        ssim_map = ssim_map.unsqueeze(0)  # [1, 1, H, W]

        rgb = self.heatmap.render(ssim_map)  # [1, 3, H, W]
        return rgb.squeeze(0)  # [3, H, W]


class SpectralAggregateRenderer(AggregateRenderer):
    """
    Spectral/FFT analysis showing dominant spatial frequencies.

    Computes 2D FFT of each realization and aggregates power spectra,
    revealing characteristic length scales and periodic patterns.

    Example:
        ```python
        renderer = SpectralAggregateRenderer()
        realizations = torch.randn(10, 3, 64, 64)
        spectral_rgb = renderer.render(realizations)  # [3, 64, 64]
        ```
    """

    def __init__(
        self,
        colormap: str = "magma",
        device: torch.device = torch.device("cuda")
    ):
        """
        Initialize spectral renderer.

        Args:
            colormap: Colormap for power spectrum visualization
            device: Torch device
        """
        self.heatmap = HeatmapRenderer(colormap=colormap, device=device)
        self.device = device

    def aggregate(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Compute average power spectrum.

        Args:
            realizations: [M, C, H, W]

        Returns:
            Average power spectrum [1, H, W]
        """
        M, C, H, W = realizations.shape

        # FFT of each realization
        power_spectra = []

        for m in range(M):
            # Average across channels
            field = realizations[m].mean(dim=0)  # [H, W]

            # 2D FFT
            fft = torch.fft.fft2(field)
            fft_shifted = torch.fft.fftshift(fft)

            # Power spectrum
            power = torch.abs(fft_shifted) ** 2
            power_spectra.append(power)

        # Average power spectrum
        avg_power = torch.stack(power_spectra).mean(dim=0)  # [H, W]

        # Log scale for visualization
        log_power = torch.log10(avg_power + 1e-10)

        return log_power.unsqueeze(0)  # [1, H, W]

    def render(self, realizations: torch.Tensor) -> torch.Tensor:
        """
        Render power spectrum to RGB.

        Args:
            realizations: [M, C, H, W]

        Returns:
            RGB image [3, H, W]
        """
        spectrum = self.aggregate(realizations)  # [1, H, W]
        spectrum = spectrum.unsqueeze(0)  # [1, 1, H, W]

        rgb = self.heatmap.render(spectrum)  # [1, 3, H, W]
        return rgb.squeeze(0)  # [3, H, W]


def create_aggregate_renderer(
    aggregate_type: str,
    base_renderer: Optional[RenderStrategy] = None,
    colormap: str = "hot",
    device: Optional[torch.device] = None
) -> AggregateRenderer:
    """
    Factory function for creating aggregate renderers.

    Args:
        aggregate_type: Aggregate type:
            - "mean": Mean field across realizations
            - "variance": Variance magnitude map
            - "stddev": Standard deviation map
            - "envelope": Min/max range map
            - "overlay": Trajectory overlay with transparency
            - "entropy": Spatial entropy map (structural uncertainty)
            - "pca": PCA mode visualization
            - "ssim": Structural similarity map
            - "spectral": FFT power spectrum
        base_renderer: Base renderer (required for "mean" and "overlay")
        colormap: Colormap for single-channel visualizations
        device: Torch device

    Returns:
        AggregateRenderer instance

    Example:
        ```python
        # Mean field
        renderer = create_aggregate_renderer("mean", base_renderer=RGBRenderer())

        # Entropy map
        renderer = create_aggregate_renderer("entropy", colormap="inferno")

        # PCA modes
        renderer = create_aggregate_renderer("pca")
        ```
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if aggregate_type == "mean":
        if base_renderer is None:
            raise ValueError("base_renderer required for 'mean' aggregate type")
        return MeanFieldRenderer(base_renderer=base_renderer, device=device)

    elif aggregate_type == "variance":
        return VarianceMapRenderer(colormap=colormap, device=device)

    elif aggregate_type == "stddev":
        return StdDevMapRenderer(colormap=colormap, device=device)

    elif aggregate_type == "envelope":
        return EnvelopeRenderer(colormap=colormap, device=device)

    elif aggregate_type == "overlay":
        if base_renderer is None:
            raise ValueError("base_renderer required for 'overlay' aggregate type")
        return TrajectoryOverlayRenderer(base_renderer=base_renderer, device=device)

    elif aggregate_type == "entropy":
        return EntropyMapRenderer(colormap=colormap, device=device)

    elif aggregate_type == "pca":
        return PCAModeRenderer(device=device)

    elif aggregate_type == "ssim":
        return SSIMMapRenderer(colormap=colormap, device=device)

    elif aggregate_type == "spectral":
        return SpectralAggregateRenderer(colormap=colormap, device=device)

    else:
        raise ValueError(
            f"Unknown aggregate type: {aggregate_type}. "
            f"Must be one of: mean, variance, stddev, envelope, overlay, "
            f"entropy, pca, ssim, spectral"
        )
