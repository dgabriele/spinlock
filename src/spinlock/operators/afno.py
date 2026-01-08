"""
Adaptive Fourier Neural Operator (AFNO) blocks for spectral mixing.

Implements spectral convolution via FFT for global receptive field
with O(N log N) complexity. Key component of U-AFNO architecture.

Design principles:
- DRY: Inherits from BaseBlock for shared functionality
- Global mixing: FFT enables full spatial receptive field
- Efficiency: Mode truncation for compression (keep low frequencies)
- Compatibility: Works with existing rollout and feature extraction
"""

import torch
import torch.nn as nn
from typing import Optional

from .blocks import BaseBlock


class SpectralMixingBlock(BaseBlock):
    """
    Spectral mixing via FFT-based learned convolution.

    Applies learned weights in Fourier space for global receptive field.
    Operates on multi-channel latent representations.

    The spectral mixing process:
    1. FFT to frequency domain
    2. Truncate to `modes` lowest frequencies (compression)
    3. Apply learned complex-valued mixing weights
    4. Inverse FFT back to spatial domain
    5. MLP for channel mixing

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (typically same as in_channels)
        modes: Number of Fourier modes to keep (lower = more compression)
        hidden_dim: Hidden dimension for MLP mixing (default: 2x in_channels)
        activation: Activation function for MLP
        use_residual: Whether to add residual connection

    Example:
        ```python
        block = SpectralMixingBlock(64, 64, modes=16)
        x = torch.randn(8, 64, 32, 32)
        out = block(x)  # Shape: (8, 64, 32, 32)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 32,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
        use_residual: bool = True,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels)

        self.modes = modes
        self.hidden_dim = hidden_dim or in_channels * 2
        self.use_residual = use_residual

        # Complex-valued mixing weights for each mode
        # Shape: [in_channels, out_channels, modes, modes]
        # Using separate real/imag for better optimization
        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )
        self.weights_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes)
        )

        # MLP for channel mixing after spectral operation
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, self.hidden_dim),
            self._make_activation(activation),
            nn.Linear(self.hidden_dim, out_channels),
        )

        # Projection for residual if channels change
        self.projection = None
        if in_channels != out_channels and use_residual:
            self.projection = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral mixing.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Spectrally mixed output [B, C, H, W]
        """
        B, C_in, H, W = x.shape
        identity = x

        # FFT (use rfft2 for efficiency - exploits conjugate symmetry)
        x_fft = torch.fft.rfft2(x, norm="ortho")  # [B, C, H, W//2+1]

        # Determine actual modes to use (clamp to available frequencies)
        modes_h = min(self.modes, H)
        modes_w = min(self.modes, W // 2 + 1)

        # Apply learned spectral mixing to low-frequency modes
        out_fft = self._spectral_multiply(x_fft, modes_h, modes_w, H, W)

        # Inverse FFT back to spatial domain
        out = torch.fft.irfft2(out_fft, s=(H, W), norm="ortho")  # [B, C_out, H, W]

        # MLP mixing per spatial location
        out = out.permute(0, 2, 3, 1)  # [B, H, W, C_out]
        out = self.mlp(out)
        out = out.permute(0, 3, 1, 2)  # [B, C_out, H, W]

        # Residual connection
        if self.use_residual:
            if self.projection is not None:
                identity = self.projection(identity)
            out = out + identity

        return out

    def _spectral_multiply(
        self,
        x_fft: torch.Tensor,
        modes_h: int,
        modes_w: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Complex multiplication in frequency domain.

        Applies learned weights to low-frequency modes, zeros high-frequency.

        Args:
            x_fft: Input FFT [B, C_in, H, W//2+1]
            modes_h: Number of modes in height dimension
            modes_w: Number of modes in width dimension
            H, W: Original spatial dimensions

        Returns:
            Mixed FFT [B, C_out, H, W//2+1]
        """
        B = x_fft.shape[0]
        C_out = self.out_channels

        # Initialize output with zeros
        out_fft = torch.zeros(
            B, C_out, H, W // 2 + 1, dtype=x_fft.dtype, device=x_fft.device
        )

        # Get weights for available modes
        weights_real = self.weights_real[:, :, :modes_h, :modes_w]
        weights_imag = self.weights_imag[:, :, :modes_h, :modes_w]

        # Extract low-frequency modes from input
        # Handle both positive and negative frequencies in height
        x_modes_pos = x_fft[:, :, :modes_h, :modes_w]  # [B, C_in, modes_h, modes_w]

        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        # einsum: contract over input channels
        # x_modes: [B, C_in, modes_h, modes_w] (complex)
        # weights: [C_in, C_out, modes_h, modes_w] (real and imag separate)

        x_real = x_modes_pos.real
        x_imag = x_modes_pos.imag

        # Compute real part: x_real * w_real - x_imag * w_imag
        out_real = torch.einsum("bihw,iohw->bohw", x_real, weights_real) - torch.einsum(
            "bihw,iohw->bohw", x_imag, weights_imag
        )
        # Compute imag part: x_real * w_imag + x_imag * w_real
        out_imag = torch.einsum("bihw,iohw->bohw", x_real, weights_imag) + torch.einsum(
            "bihw,iohw->bohw", x_imag, weights_real
        )

        # Place in output (low-frequency region)
        # Ensure float32 for complex (ComplexHalf not fully supported)
        out_fft[:, :, :modes_h, :modes_w] = torch.complex(out_real.float(), out_imag.float())

        return out_fft


class AFNOBlock(BaseBlock):
    """
    Full AFNO block with pre-norm, spectral mixing, and feedforward.

    Architecture (following transformer conventions):
        LayerNorm -> SpectralMixing -> Residual
        LayerNorm -> FeedForward -> Residual

    This is the core building block for the AFNO bottleneck in U-AFNO.
    Stacking multiple AFNOBlocks enables deep spectral processing.

    Args:
        channels: Number of channels (input == output)
        modes: Number of Fourier modes for spectral mixing
        mlp_ratio: Ratio of hidden dim to channels in feedforward
        activation: Activation function
        dropout: Dropout probability

    Example:
        ```python
        block = AFNOBlock(64, modes=16, mlp_ratio=4.0)
        x = torch.randn(8, 64, 32, 32)
        out = block(x)  # Shape: (8, 64, 32, 32)
        ```
    """

    def __init__(
        self,
        channels: int,
        modes: int = 32,
        mlp_ratio: float = 4.0,
        activation: str = "gelu",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(channels, channels)

        self.channels = channels
        self.modes = modes

        # Pre-norm for spectral mixing path
        # Use GroupNorm(1, C) as LayerNorm equivalent for spatial data
        self.norm1 = nn.GroupNorm(1, channels)

        # Spectral mixing (no internal residual - we add it here)
        self.spectral = SpectralMixingBlock(
            channels,
            channels,
            modes=modes,
            activation=activation,
            use_residual=False,  # Residual added at block level
        )
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Pre-norm for feedforward path
        self.norm2 = nn.GroupNorm(1, channels)

        # Feedforward MLP (pointwise convolutions for spatial data)
        hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1),
            self._make_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden_dim, channels, 1),
        )
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-norm and residual connections.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, C, H, W]
        """
        # Spectral mixing path: Norm -> Spectral -> Dropout -> Residual
        x = x + self.drop1(self.spectral(self.norm1(x)))

        # Feedforward path: Norm -> MLP -> Dropout -> Residual
        x = x + self.drop2(self.mlp(self.norm2(x)))

        return x
