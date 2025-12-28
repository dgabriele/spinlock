"""
Modular CNN building blocks for stochastic neural operators.

Provides composable, reusable components:
- ConvBlock: Basic conv + norm + activation
- ResidualBlock: Residual connections with configurable depth
- StochasticBlock: Noise injection (Gaussian, dropout, multiplicative)
- DownsampleBlock, UpsampleBlock: Resolution changes

Design principles:
- DRY: Shared base class with common factories (norm, activation)
- Composition: Complex blocks built from simple ones
- Flexibility: YAML-configurable architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Callable
from abc import ABC


class BaseBlock(nn.Module, ABC):
    """
    Abstract base with shared functionality.

    All blocks inherit:
    - Normalization factory (DRY)
    - Activation factory (DRY)
    - Channel tracking (in_channels, out_channels)
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def _make_norm(self, norm_type: str, channels: int) -> nn.Module:
        """
        DRY normalization factory (reused across all blocks).

        Args:
            norm_type: Type of normalization
            channels: Number of channels

        Returns:
            Normalization module
        """
        registry = {
            "batch": lambda: nn.BatchNorm2d(channels),
            "instance": lambda: nn.InstanceNorm2d(channels, affine=True),
            "group": lambda: nn.GroupNorm(min(32, channels), channels),
            "layer": lambda: nn.GroupNorm(1, channels),
            "none": lambda: nn.Identity(),
        }

        if norm_type not in registry:
            raise ValueError(f"Unknown normalization: {norm_type}")

        return registry[norm_type]()

    def _make_activation(self, act_type: str) -> nn.Module:
        """
        DRY activation factory (reused across all blocks).

        Args:
            act_type: Type of activation

        Returns:
            Activation module
        """
        registry = {
            "relu": lambda: nn.ReLU(inplace=True),
            "gelu": lambda: nn.GELU(),
            "silu": lambda: nn.SiLU(inplace=True),
            "swish": lambda: nn.SiLU(inplace=True),  # Alias
            "tanh": lambda: nn.Tanh(),
            "none": lambda: nn.Identity(),
        }

        if act_type not in registry:
            raise ValueError(f"Unknown activation: {act_type}")

        return registry[act_type]()


class ConvBlock(BaseBlock):
    """
    Standard conv + norm + activation block.

    Composable building block used throughout the architecture.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        normalization: Normalization type
        activation: Activation type
        padding_mode: Padding mode (circular for periodic boundaries)
        dropout: Dropout probability
        use_bias: Whether to use bias (typically False when using normalization)

    Example:
        ```python
        block = ConvBlock(64, 128, kernel_size=3, normalization="instance", activation="gelu")
        x = torch.randn(8, 64, 32, 32)
        out = block(x)  # Shape: (8, 128, 32, 32)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        normalization: str = "instance",
        activation: str = "gelu",
        padding_mode: str = "circular",
        dropout: float = 0.0,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels)

        padding = kernel_size // 2

        # Build block layers
        layers = []

        # Convolution
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                bias=use_bias and normalization == "none",
            )
        )

        # Normalization
        if normalization != "none":
            layers.append(self._make_norm(normalization, out_channels))

        # Activation
        layers.append(self._make_activation(activation))

        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(BaseBlock):
    """
    Residual block with configurable depth.

    Implements skip connections with optional projection for channel matching.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        num_convs: Number of convolution blocks in residual path
        kernel_size: Convolution kernel size
        stochastic_depth: Probability of dropping residual branch
        **kwargs: Additional arguments passed to ConvBlock

    Example:
        ```python
        block = ResidualBlock(64, 128, num_convs=2, kernel_size=3)
        x = torch.randn(8, 64, 32, 32)
        out = block(x)  # Shape: (8, 128, 32, 32)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        kernel_size: int = 3,
        stochastic_depth: float = 0.0,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels)

        self.stochastic_depth = stochastic_depth

        # Residual path: compose multiple ConvBlocks (DRY)
        self.blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    **kwargs,
                )
                for i in range(num_convs)
            ]
        )

        # Skip connection projection (if channels change)
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Apply projection if needed
        if self.projection is not None:
            identity = self.projection(identity)

        # Residual path
        residual = x
        for block in self.blocks:
            # Stochastic depth: randomly drop residual branch during training
            if self.training and self.stochastic_depth > 0:
                if torch.rand(1).item() < self.stochastic_depth:
                    continue  # Skip this block

            residual = block(residual)

        # Add skip connection
        return residual + identity


class StochasticBlock(BaseBlock):
    """
    Block that adds stochastic elements (noise, dropout, etc.).

    Supports multiple noise types and schedules for exploring diverse dynamics.

    Args:
        in_channels: Number of channels (passthrough)
        noise_type: Type of stochastic perturbation
        noise_scale: Scale/intensity of noise
        learnable_scale: Whether noise scale is learnable parameter
        always_active: If True, apply noise even during eval()
        noise_schedule: Temporal noise schedule ("constant", "annealing", "periodic")
        schedule_period: Period for periodic schedule (in steps)
        spatial_correlation: Spatial correlation length (0.0 = uncorrelated)

    Example:
        ```python
        # Simple Gaussian noise
        block = StochasticBlock(64, noise_type="gaussian", noise_scale=0.1)
        x = torch.randn(8, 64, 32, 32)
        out = block(x)  # x + Gaussian noise

        # Annealing noise schedule
        block = StochasticBlock(64, noise_scale=0.1, noise_schedule="annealing")
        out = block(x, step=50, max_steps=100)  # Noise decays with time

        # Spatially correlated noise
        block = StochasticBlock(64, noise_scale=0.1, spatial_correlation=0.1)
        out = block(x)  # Smooth, correlated noise pattern
        ```
    """

    def __init__(
        self,
        in_channels: int,
        noise_type: Literal["gaussian", "dropout", "multiplicative", "laplace"] = "gaussian",
        noise_scale: float = 0.1,
        learnable_scale: bool = False,
        always_active: bool = True,  # For dataset generation
        noise_schedule: Literal["constant", "annealing", "periodic"] = "constant",
        schedule_period: int = 100,
        spatial_correlation: float = 0.0,
        **kwargs,
    ):
        super().__init__(in_channels, in_channels)  # Passthrough channels

        self.noise_type = noise_type
        self.always_active = always_active
        self.noise_schedule = noise_schedule
        self.schedule_period = schedule_period
        self.spatial_correlation = spatial_correlation

        # Noise scale: learnable parameter or fixed buffer
        if learnable_scale:
            self.noise_scale = nn.Parameter(torch.tensor(noise_scale))
        else:
            self.register_buffer("noise_scale", torch.tensor(noise_scale))

        # Precompute Fourier grid for spatial correlation (lazy init in forward)
        self._k_squared = None

    def _get_scheduled_scale(self, step: Optional[int] = None, max_steps: Optional[int] = None) -> torch.Tensor:
        """
        Compute noise scale based on schedule.

        Args:
            step: Current timestep (required for annealing/periodic)
            max_steps: Maximum timesteps (required for annealing)

        Returns:
            Scheduled noise scale
        """
        if self.noise_schedule == "constant":
            return self.noise_scale

        elif self.noise_schedule == "annealing":
            if step is None or max_steps is None:
                return self.noise_scale
            # Linear annealing: scale * (1 - step / max_steps)
            progress = min(step / max_steps, 1.0)
            return self.noise_scale * (1.0 - progress)

        elif self.noise_schedule == "periodic":
            if step is None:
                return self.noise_scale
            # Sinusoidal modulation: scale * (1 + 0.5 * sin(2π * step / period))
            import math
            phase = 2.0 * math.pi * step / self.schedule_period
            modulation = 1.0 + 0.5 * math.sin(phase)
            return self.noise_scale * modulation

        return self.noise_scale

    def _generate_correlated_noise(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """
        Generate spatially correlated noise using Fourier filtering.

        Args:
            shape: Noise tensor shape (B, C, H, W)
            device: Device for tensor

        Returns:
            Correlated noise tensor
        """
        B, C, H, W = shape

        # Lazy initialization of Fourier grid
        if self._k_squared is None or self._k_squared.shape[-1] != H:
            kx = torch.fft.fftfreq(H, d=1.0, device=device).view(H, 1)
            ky = torch.fft.fftfreq(W, d=1.0, device=device).view(1, W)
            self._k_squared = kx**2 + ky**2

        # Generate white noise in Fourier space
        noise_freq = torch.randn(B, C, H, W, dtype=torch.complex64, device=device)

        # Apply Gaussian filter: P(k) = exp(-k^2 * correlation^2 / 2)
        filter_kernel = torch.exp(-self._k_squared * (self.spatial_correlation * H) ** 2 / 2.0)
        noise_freq = noise_freq * filter_kernel.unsqueeze(0).unsqueeze(0)

        # Transform back to real space
        noise_real = torch.fft.ifft2(noise_freq).real

        # Normalize to unit variance
        noise_real = noise_real / (noise_real.std() + 1e-8)

        return noise_real

    def forward(self, x: torch.Tensor, step: Optional[int] = None, max_steps: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass with optional step information for scheduling.

        Args:
            x: Input tensor [B, C, H, W]
            step: Current timestep (for noise scheduling)
            max_steps: Maximum timesteps (for annealing schedule)

        Returns:
            Noisy output tensor
        """
        # Apply stochasticity during training OR if always_active
        if not self.training and not self.always_active:
            return x

        # Get scheduled noise scale
        scale = self._get_scheduled_scale(step, max_steps)

        if self.noise_type == "gaussian":
            # Generate noise (correlated or uncorrelated)
            if self.spatial_correlation > 0:
                noise = self._generate_correlated_noise(x.shape, x.device) * scale
            else:
                noise = torch.randn_like(x) * scale
            return x + noise

        elif self.noise_type == "laplace":
            # Laplace distribution: exponential with random sign
            if self.spatial_correlation > 0:
                noise = self._generate_correlated_noise(x.shape, x.device)
                noise = noise * torch.sign(torch.randn_like(x))  # Make it Laplace-like
            else:
                noise = torch.empty_like(x).exponential_() * torch.sign(torch.randn_like(x))
            return x + noise * scale

        elif self.noise_type == "dropout":
            return F.dropout2d(x, p=float(scale), training=True)

        elif self.noise_type == "multiplicative":
            # Multiplicative noise
            if self.spatial_correlation > 0:
                noise = 1.0 + self._generate_correlated_noise(x.shape, x.device) * scale
            else:
                noise = 1.0 + torch.randn_like(x) * scale
            return x * noise

        return x


class DownsampleBlock(BaseBlock):
    """
    Downsampling block using strided convolution or pooling.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        downsample_factor: Downsampling factor (typically 2)
        mode: "conv" (strided conv) or "pool" (avg pooling + conv)
        **kwargs: Additional arguments for ConvBlock

    Example:
        ```python
        block = DownsampleBlock(64, 128, downsample_factor=2, mode="conv")
        x = torch.randn(8, 64, 64, 64)
        out = block(x)  # Shape: (8, 128, 32, 32)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample_factor: int = 2,
        mode: str = "conv",
        **kwargs,
    ):
        super().__init__(in_channels, out_channels)

        if mode == "conv":
            # Strided convolution
            self.downsample = ConvBlock(
                in_channels, out_channels, stride=downsample_factor, **kwargs
            )
        elif mode == "pool":
            # Average pooling + convolution
            self.downsample = nn.Sequential(
                nn.AvgPool2d(downsample_factor), ConvBlock(in_channels, out_channels, **kwargs)
            )
        else:
            raise ValueError(f"Unknown downsample mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)


class UpsampleBlock(BaseBlock):
    """
    Upsampling block with optional skip connections.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        upsample_factor: Upsampling factor (typically 2)
        mode: "bilinear", "nearest", or "transpose"
        skip_channels: Channels from skip connection (if any)
        fusion_mode: How to fuse skip connection ("concat", "add")
        **kwargs: Additional arguments for ConvBlock

    Example:
        ```python
        block = UpsampleBlock(128, 64, upsample_factor=2, mode="bilinear")
        x = torch.randn(8, 128, 16, 16)
        out = block(x)  # Shape: (8, 64, 32, 32)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_factor: int = 2,
        mode: str = "bilinear",
        skip_channels: Optional[int] = None,
        fusion_mode: str = "concat",
        **kwargs,
    ):
        super().__init__(in_channels, out_channels)

        self.fusion_mode = fusion_mode

        # Upsample layer
        if mode == "transpose":
            self.upsample = nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size=upsample_factor, stride=upsample_factor
            )
        elif mode in ["bilinear", "nearest"]:
            self.upsample = nn.Upsample(
                scale_factor=upsample_factor,
                mode=mode,
                align_corners=False if mode == "bilinear" else None,
            )
        else:
            raise ValueError(f"Unknown upsample mode: {mode}")

        # Compute conv input channels based on fusion
        conv_in_channels = in_channels
        if skip_channels is not None:
            if fusion_mode == "concat":
                conv_in_channels = in_channels + skip_channels
            elif fusion_mode == "add":
                if in_channels != skip_channels:
                    raise ValueError(
                        f"For add fusion, in_channels ({in_channels}) must equal "
                        f"skip_channels ({skip_channels})"
                    )
            else:
                raise ValueError(f"Unknown fusion mode: {fusion_mode}")

        # Refinement convolution
        self.conv = ConvBlock(conv_in_channels, out_channels, **kwargs)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)

        # Fuse skip connection
        if skip is not None:
            if self.fusion_mode == "concat":
                x = torch.cat([x, skip], dim=1)
            elif self.fusion_mode == "add":
                x = x + skip

        return self.conv(x)


class OutputLayer(BaseBlock):
    """
    Final output layer (typically 1x1 conv).

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size (typically 1)
        activation: Output activation ("none" for raw outputs)

    Example:
        ```python
        output = OutputLayer(64, 3, activation="none")
        x = torch.randn(8, 64, 32, 32)
        out = output(x)  # Shape: (8, 3, 32, 32)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        activation: str = "none",
        **kwargs,
    ):
        super().__init__(in_channels, out_channels)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

        self.activation = self._make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))
