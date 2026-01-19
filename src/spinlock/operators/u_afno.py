"""
U-AFNO: U-Net encoder/decoder with AFNO bottleneck.

Combines multi-scale spatial hierarchy with global spectral mixing
for learning complex spatiotemporal operator dynamics.

Architecture:
    Input [B, C, H, W]
        |
        v
    U-Net Encoder (progressive downsampling)
        |-- Level 0: H×W → H/2×W/2
        |-- Level 1: H/2×W/2 → H/4×W/4
        |-- ...
        |-- Level N: Bottleneck resolution
        |
        v
    AFNO Bottleneck (spectral mixing via FFT)
        |-- Multiple stacked AFNOBlocks
        |-- Global receptive field
        |
        v
    U-Net Decoder (upsampling with skip connections)
        |-- Level N-1: Upsample + skip + conv
        |-- ...
        |-- Level 0: Output resolution
        |
        v
    Output [B, C, H, W]

Design principles:
- Multi-scale: U-Net captures hierarchical features
- Global mixing: AFNO bottleneck has full receptive field
- Compatibility: Works with existing rollout policies and features
- Stochasticity: Reuses StochasticBlock for noise injection
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Tuple

from .blocks import (
    ConvBlock,
    ResidualBlock,
    FiLMResidualBlock,
    StochasticBlock,
    DownsampleBlock,
    UpsampleBlock,
    OutputLayer,
)
from .afno import AFNOBlock, FiLMAFNOBlock
from .film import (
    FiLMConfig,
    FiLMLayerSpec,
    FiLMLayer,
    FiLMGenerator,
    build_film_layer_specs,
)


class UNetEncoder(nn.Module):
    """
    U-Net encoder: progressive downsampling with feature extraction.

    Creates multi-scale feature hierarchy while tracking skip connections
    for the decoder. Each level doubles the channel count and halves
    the spatial resolution.

    Args:
        in_channels: Input channels
        base_channels: Base channel count (doubles each level)
        num_levels: Number of downsampling levels
        blocks_per_level: Residual blocks per level
        normalization: Normalization type
        activation: Activation function

    Example:
        ```python
        encoder = UNetEncoder(3, base_channels=32, num_levels=3)
        x = torch.randn(8, 3, 64, 64)
        bottleneck, skips = encoder(x)
        # bottleneck: [8, 256, 8, 8]
        # skips: list of skip features at each level
        ```
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        num_levels: int = 3,
        blocks_per_level: int = 2,
        normalization: str = "instance",
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__()

        self.num_levels = num_levels
        self.skip_channels: List[int] = []  # Track skip connection channels

        # Initial projection to base_channels
        self.stem = ConvBlock(
            in_channels,
            base_channels,
            kernel_size=3,
            normalization=normalization,
            activation=activation,
        )

        # Encoder levels - split into feature extraction and downsampling
        # so we can capture skips before downsampling
        self.feature_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        channels = base_channels

        for level in range(num_levels):
            # Double channels at each level (cap at 8x base)
            out_channels = min(channels * 2, base_channels * 8)

            # Residual blocks for feature extraction
            feature_block = nn.Sequential(
                *[
                    ResidualBlock(
                        channels if i == 0 else out_channels,
                        out_channels,
                        num_convs=2,
                        normalization=normalization,
                        activation=activation,
                    )
                    for i in range(blocks_per_level)
                ]
            )

            # Downsample (halve spatial dimensions)
            downsample = DownsampleBlock(
                out_channels,
                out_channels,
                downsample_factor=2,
                mode="conv",
                normalization=normalization,
                activation=activation,
            )

            self.feature_blocks.append(feature_block)
            self.downsample_blocks.append(downsample)
            self.skip_channels.append(out_channels)
            channels = out_channels

        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode with skip connections.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Tuple of:
                - bottleneck: Encoded features at lowest resolution
                - skips: List of skip features (before downsampling, in encoder order)
        """
        skips = []

        # Initial projection
        x = self.stem(x)

        # Encoder levels - capture skip BEFORE downsampling
        for feature_block, downsample in zip(self.feature_blocks, self.downsample_blocks):
            x = feature_block(x)
            skips.append(x)  # Capture skip at full resolution for this level
            x = downsample(x)

        # Return bottleneck (after all downsamples) and skips
        return x, skips


class UNetDecoder(nn.Module):
    """
    U-Net decoder: progressive upsampling with skip connections.

    Reconstructs spatial resolution while integrating skip connections
    from the encoder. Uses concatenation fusion for skip integration.

    Args:
        in_channels: Input channels (from bottleneck/AFNO)
        out_channels: Final output channels
        skip_channels: List of channel counts for skip connections
        blocks_per_level: Residual blocks per level
        normalization: Normalization type
        activation: Activation function

    Example:
        ```python
        decoder = UNetDecoder(
            in_channels=256,
            out_channels=3,
            skip_channels=[32, 64, 128],
        )
        x = torch.randn(8, 256, 8, 8)
        skips = [torch.randn(8, 32, 64, 64), torch.randn(8, 64, 32, 32), ...]
        out = decoder(x, skips)  # [8, 3, 64, 64]
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: List[int],
        blocks_per_level: int = 2,
        normalization: str = "instance",
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__()

        self.levels = nn.ModuleList()
        channels = in_channels

        # Process in reverse order (bottom-up)
        for skip_ch in reversed(skip_channels):
            # Target channels after this level
            level_out_ch = skip_ch

            # Upsample block with skip fusion
            upsample = UpsampleBlock(
                channels,
                level_out_ch,
                upsample_factor=2,
                mode="bilinear",
                skip_channels=skip_ch,
                fusion_mode="concat",
                normalization=normalization,
                activation=activation,
            )

            # Residual blocks to refine fused features
            # UpsampleBlock already handles concat and outputs level_out_ch
            refinement = nn.Sequential(
                *[
                    ResidualBlock(
                        level_out_ch,
                        level_out_ch,
                        num_convs=2,
                        normalization=normalization,
                        activation=activation,
                    )
                    for i in range(blocks_per_level)
                ]
            )

            self.levels.append(nn.ModuleDict({"upsample": upsample, "refine": refinement}))
            channels = level_out_ch

        # Final output projection
        self.output = OutputLayer(channels, out_channels, activation="none")

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode with skip connections.

        Args:
            x: Bottleneck features [B, C, H, W]
            skips: List of skip features from encoder (in encoder order)

        Returns:
            Output tensor [B, out_channels, H_out, W_out]
        """
        # Process skips in reverse order (to match decoder levels)
        for i, level in enumerate(self.levels):
            skip = skips[-(i + 1)]  # Get corresponding skip from end
            x = level["upsample"](x, skip)
            x = level["refine"](x)

        return self.output(x)


class UAFNOOperator(nn.Module):
    """
    U-AFNO: Complete neural operator architecture.

    Combines:
    - U-Net encoder for multi-scale feature extraction
    - AFNO bottleneck for global spectral mixing
    - U-Net decoder with skip connections for reconstruction
    - Optional stochastic block for noise injection

    This operator is designed for learning complex spatiotemporal dynamics
    with both local (convolution) and global (spectral) processing.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        base_channels: Base channel count for U-Net
        encoder_levels: Number of U-Net encoder levels
        modes: AFNO Fourier modes to keep
        afno_blocks: Number of AFNO blocks in bottleneck
        hidden_dim: AFNO hidden dimension (for MLP in spectral mixing)
        blocks_per_level: Residual blocks per U-Net level
        normalization: Normalization type
        activation: Activation function
        noise_type: Stochastic noise type (optional, e.g., "gaussian")
        noise_scale: Noise scale (optional)
        noise_schedule: Noise schedule ("constant", "annealing", "periodic")
        spatial_correlation: Spatial correlation for noise (0.0 = uncorrelated)

    Example:
        ```python
        op = UAFNOOperator(
            in_channels=3,
            out_channels=3,
            base_channels=32,
            encoder_levels=3,
            modes=16,
            afno_blocks=4,
        )
        x = torch.randn(8, 3, 64, 64)
        out = op(x)  # Shape: (8, 3, 64, 64)
        ```
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        encoder_levels: int = 3,
        modes: int = 32,
        afno_blocks: int = 4,
        hidden_dim: Optional[int] = None,
        blocks_per_level: int = 2,
        normalization: str = "instance",
        activation: str = "gelu",
        noise_type: Optional[str] = None,
        noise_scale: Optional[float] = None,
        noise_schedule: str = "constant",
        spatial_correlation: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        # Store config for serialization/logging
        self.config: Dict[str, Any] = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "base_channels": base_channels,
            "encoder_levels": encoder_levels,
            "modes": modes,
            "afno_blocks": afno_blocks,
            "hidden_dim": hidden_dim,
            "blocks_per_level": blocks_per_level,
            "normalization": normalization,
            "activation": activation,
            "noise_type": noise_type,
            "noise_scale": noise_scale,
        }

        # U-Net encoder
        self.encoder = UNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_levels=encoder_levels,
            blocks_per_level=blocks_per_level,
            normalization=normalization,
            activation=activation,
        )

        bottleneck_channels = self.encoder.out_channels

        # Determine AFNO hidden dim
        afno_hidden = hidden_dim or bottleneck_channels * 2

        # AFNO bottleneck: stack of AFNO blocks for global spectral mixing
        self.bottleneck = nn.Sequential(
            *[
                AFNOBlock(
                    channels=bottleneck_channels,
                    modes=modes,
                    mlp_ratio=afno_hidden / bottleneck_channels,
                    activation=activation,
                )
                for _ in range(afno_blocks)
            ]
        )

        # U-Net decoder
        self.decoder = UNetDecoder(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            skip_channels=self.encoder.skip_channels,
            blocks_per_level=blocks_per_level,
            normalization=normalization,
            activation=activation,
        )

        # Optional stochastic block (reuses existing infrastructure)
        self.stochastic: Optional[StochasticBlock] = None
        if noise_type and noise_scale and noise_scale > 0:
            self.stochastic = StochasticBlock(
                out_channels,
                noise_type=noise_type,
                noise_scale=noise_scale,
                always_active=True,  # Always inject noise for dataset generation
                noise_schedule=noise_schedule,
                spatial_correlation=spatial_correlation,
            )

    def forward(
        self,
        x: torch.Tensor,
        step: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]
            step: Current timestep (for noise scheduling)
            max_steps: Max timesteps (for noise scheduling)

        Returns:
            Output tensor [B, C, H, W]
        """
        # Encode: extract multi-scale features
        bottleneck, skips = self.encoder(x)

        # AFNO bottleneck: global spectral mixing
        bottleneck = self.bottleneck(bottleneck)

        # Decode: reconstruct with skip connections
        out = self.decoder(bottleneck, skips)

        # Optional stochasticity
        if self.stochastic is not None:
            out = self.stochastic(out, step=step, max_steps=max_steps)

        return out

    def get_intermediate_features(
        self,
        x: torch.Tensor,
        extract_from: str = "bottleneck",
        skip_levels: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate features for learned SUMMARY descriptors.

        This method runs the encoder and bottleneck to extract latent
        representations without running the full decoder. Used for
        learned feature extraction in Phase 2.

        Args:
            x: Input tensor [B, C, H, W]
            extract_from: What to extract:
                - "bottleneck": Only AFNO bottleneck output (default)
                - "skips": Only encoder skip connections
                - "all": Both bottleneck and skips
            skip_levels: Which skip levels to extract (default: all available).
                Level 0 is shallowest (highest resolution), higher levels are
                deeper (lower resolution).

        Returns:
            Dictionary with extracted features:
                - 'bottleneck': [B, C_bottleneck, H', W'] (after AFNO blocks)
                - 'skip_0': [B, C_0, H_0, W_0] (encoder level 0)
                - 'skip_1': [B, C_1, H_1, W_1] (encoder level 1)
                - ...

        Example:
            ```python
            operator = UAFNOOperator(...)
            x = torch.randn(8, 3, 64, 64)

            # Extract bottleneck only (compact global features)
            features = operator.get_intermediate_features(x, extract_from="bottleneck")
            bottleneck = features['bottleneck']  # [8, 256, 4, 4]

            # Extract all latents for rich multi-scale representation
            features = operator.get_intermediate_features(x, extract_from="all")
            bottleneck = features['bottleneck']  # [8, 256, 4, 4]
            skip_0 = features['skip_0']          # [8, 64, 32, 32]
            skip_1 = features['skip_1']          # [8, 128, 16, 16]
            skip_2 = features['skip_2']          # [8, 256, 8, 8]
            ```
        """
        features: Dict[str, torch.Tensor] = {}

        # Run encoder to get raw bottleneck and skips
        bottleneck_raw, skips = self.encoder(x)

        # Run AFNO bottleneck for global spectral mixing
        bottleneck_processed = self.bottleneck(bottleneck_raw)

        # Extract bottleneck if requested
        if extract_from in ("bottleneck", "all"):
            features["bottleneck"] = bottleneck_processed

        # Extract skips if requested
        if extract_from in ("skips", "all"):
            levels_to_extract = (
                skip_levels if skip_levels is not None
                else list(range(len(skips)))
            )
            for level in levels_to_extract:
                if 0 <= level < len(skips):
                    features[f"skip_{level}"] = skips[level]

        return features

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        return self.config.copy()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "UAFNOOperator":
        """Create instance from configuration dictionary."""
        return cls(**config)


class FiLMUNetEncoder(nn.Module):
    """U-Net encoder with FiLM modulation support.

    Extends UNetEncoder to use FiLMResidualBlock at specified levels,
    enabling theta-conditioned feature modulation during encoding.

    Args:
        in_channels: Input channels
        base_channels: Base channel count (doubles each level)
        num_levels: Number of downsampling levels
        blocks_per_level: Residual blocks per level
        normalization: Normalization type
        activation: Activation function
        film_levels: Which levels to use FiLMResidualBlock (None = all)
        post_norm: Apply LayerNorm after FiLM modulation

    Example:
        >>> encoder = FiLMUNetEncoder(3, base_channels=32, num_levels=3)
        >>> x = torch.randn(8, 3, 64, 64)
        >>> film_params = {"encoder_0": (gamma0, beta0), "encoder_1": (gamma1, beta1)}
        >>> bottleneck, skips = encoder(x, film_params)
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        num_levels: int = 3,
        blocks_per_level: int = 2,
        normalization: str = "instance",
        activation: str = "gelu",
        film_levels: Optional[List[int]] = None,
        post_norm: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.num_levels = num_levels
        self.skip_channels: List[int] = []
        self.film_levels = film_levels or list(range(num_levels))

        # Initial projection to base_channels
        self.stem = ConvBlock(
            in_channels,
            base_channels,
            kernel_size=3,
            normalization=normalization,
            activation=activation,
        )

        # Encoder levels
        self.feature_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        channels = base_channels

        for level in range(num_levels):
            out_channels = min(channels * 2, base_channels * 8)

            # Use FiLMResidualBlock for specified levels
            use_film = level in self.film_levels
            block_class = FiLMResidualBlock if use_film else ResidualBlock

            feature_block = nn.ModuleList([
                block_class(
                    channels if i == 0 else out_channels,
                    out_channels,
                    num_convs=2,
                    normalization=normalization,
                    activation=activation,
                    post_norm=post_norm if use_film else True,
                )
                for i in range(blocks_per_level)
            ])

            downsample = DownsampleBlock(
                out_channels,
                out_channels,
                downsample_factor=2,
                mode="conv",
                normalization=normalization,
                activation=activation,
            )

            self.feature_blocks.append(feature_block)
            self.downsample_blocks.append(downsample)
            self.skip_channels.append(out_channels)
            channels = out_channels

        self.out_channels = channels

    def forward(
        self,
        x: torch.Tensor,
        film_params: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode with optional FiLM modulation.

        Args:
            x: Input tensor [B, C, H, W]
            film_params: Optional dict mapping "encoder_{level}" to (gamma, beta)

        Returns:
            Tuple of:
                - bottleneck: Encoded features at lowest resolution
                - skips: List of skip features
        """
        skips = []

        x = self.stem(x)

        for level, (feature_block, downsample) in enumerate(
            zip(self.feature_blocks, self.downsample_blocks)
        ):
            # Get FiLM params for this level if available
            gamma, beta = None, None
            if film_params and f"encoder_{level}" in film_params:
                gamma, beta = film_params[f"encoder_{level}"]

            # Apply feature blocks
            for block in feature_block:
                if isinstance(block, FiLMResidualBlock) and gamma is not None:
                    x = block(x, gamma=gamma, beta=beta)
                else:
                    x = block(x)

            skips.append(x)
            x = downsample(x)

        return x, skips


class FiLMUNetDecoder(nn.Module):
    """U-Net decoder with FiLM modulation support.

    Extends UNetDecoder to use FiLMResidualBlock at specified levels.

    Args:
        in_channels: Input channels (from bottleneck/AFNO)
        out_channels: Final output channels
        skip_channels: List of channel counts for skip connections
        blocks_per_level: Residual blocks per level
        normalization: Normalization type
        activation: Activation function
        film_levels: Which levels to use FiLMResidualBlock (None = all)
        post_norm: Apply LayerNorm after FiLM modulation

    Example:
        >>> decoder = FiLMUNetDecoder(256, 3, [32, 64, 128])
        >>> film_params = {"decoder_0": (gamma0, beta0)}
        >>> out = decoder(bottleneck, skips, film_params)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: List[int],
        blocks_per_level: int = 2,
        normalization: str = "instance",
        activation: str = "gelu",
        film_levels: Optional[List[int]] = None,
        post_norm: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.film_levels = film_levels or list(range(len(skip_channels)))
        self.levels = nn.ModuleList()
        channels = in_channels

        # Process in reverse order (bottom-up)
        # level_idx 0 = deepest (fuses with last encoder level)
        # level_idx N-1 = shallowest (fuses with first encoder level)
        for level_idx, skip_ch in enumerate(reversed(skip_channels)):
            level_out_ch = skip_ch

            # Upsample block with skip fusion
            upsample = UpsampleBlock(
                channels,
                level_out_ch,
                upsample_factor=2,
                mode="bilinear",
                skip_channels=skip_ch,
                fusion_mode="concat",
                normalization=normalization,
                activation=activation,
            )

            # Use processing order for film_levels check (level_idx)
            # This matches the decoder_channels ordering in FiLM spec generation
            use_film = level_idx in self.film_levels
            block_class = FiLMResidualBlock if use_film else ResidualBlock

            refinement = nn.ModuleList([
                block_class(
                    level_out_ch,
                    level_out_ch,
                    num_convs=2,
                    normalization=normalization,
                    activation=activation,
                    post_norm=post_norm if use_film else True,
                )
                for _ in range(blocks_per_level)
            ])

            self.levels.append(nn.ModuleDict({
                "upsample": upsample,
                "refine": refinement,
            }))
            channels = level_out_ch

        self.output = OutputLayer(channels, out_channels, activation="none")

    def forward(
        self,
        x: torch.Tensor,
        skips: List[torch.Tensor],
        film_params: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Decode with optional FiLM modulation.

        Args:
            x: Bottleneck features [B, C, H, W]
            skips: List of skip features from encoder
            film_params: Optional dict mapping "decoder_{level}" to (gamma, beta)

        Returns:
            Output tensor [B, out_channels, H_out, W_out]
        """
        for i, level in enumerate(self.levels):
            skip = skips[-(i + 1)]
            x = level["upsample"](x, skip)

            # Get FiLM params for this level if available
            # Use processing order index (i) which matches decoder_channels ordering
            gamma, beta = None, None
            if film_params and f"decoder_{i}" in film_params:
                gamma, beta = film_params[f"decoder_{i}"]

            # Apply refinement blocks
            for block in level["refine"]:
                if isinstance(block, FiLMResidualBlock) and gamma is not None:
                    x = block(x, gamma=gamma, beta=beta)
                else:
                    x = block(x)

        return self.output(x)


class FiLMUAFNOOperator(nn.Module):
    """U-AFNO with FiLM modulation for parameter conditioning.

    Combines:
    - FiLM-enabled U-Net encoder/decoder for spatial modulation (default)
    - Optional FiLM AFNO blocks for post-spectral modulation
    - FiLMGenerator to produce gamma/beta from conditioning embedding

    Key design principles:
    - Spatial-only modulation by default (encoder/decoder)
    - POST-spectral AFNO modulation only (never inside FFT path)
    - Post-FiLM LayerNorm to prevent drift on long rollouts

    Args:
        in_channels: Input channels (without conditioning channels)
        out_channels: Output channels
        base_channels: Base channel count for U-Net
        encoder_levels: Number of U-Net encoder levels
        modes: AFNO Fourier modes to keep
        afno_blocks: Number of AFNO blocks in bottleneck
        hidden_dim: AFNO hidden dimension
        blocks_per_level: Residual blocks per U-Net level
        normalization: Normalization type
        activation: Activation function
        film_config: FiLMConfig instance or dict with FiLM settings
        noise_type: Stochastic noise type (optional)
        noise_scale: Noise scale (optional)

    Example:
        >>> film_cfg = FiLMConfig(enabled=True, embed_dim=64, hidden_dim=128)
        >>> op = FiLMUAFNOOperator(
        ...     in_channels=1, out_channels=1, film_config=film_cfg
        ... )
        >>> x = torch.randn(8, 1, 64, 64)
        >>> conditioning = torch.randn(8, 64)  # param embedding
        >>> out = op(x, conditioning=conditioning)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        encoder_levels: int = 3,
        modes: int = 32,
        afno_blocks: int = 4,
        hidden_dim: Optional[int] = None,
        blocks_per_level: int = 2,
        normalization: str = "instance",
        activation: str = "gelu",
        film_config: Optional[Dict[str, Any]] = None,
        noise_type: Optional[str] = None,
        noise_scale: Optional[float] = None,
        noise_schedule: str = "constant",
        spatial_correlation: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        # Parse FiLM config
        if isinstance(film_config, dict):
            self.film_config = FiLMConfig.from_dict(film_config)
        elif isinstance(film_config, FiLMConfig):
            self.film_config = film_config
        else:
            self.film_config = FiLMConfig(enabled=False)

        self.film_enabled = self.film_config.enabled

        # Validate and warn if needed
        if self.film_enabled:
            self.film_config.validate()

        # Store config for serialization
        self.config: Dict[str, Any] = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "base_channels": base_channels,
            "encoder_levels": encoder_levels,
            "modes": modes,
            "afno_blocks": afno_blocks,
            "hidden_dim": hidden_dim,
            "blocks_per_level": blocks_per_level,
            "normalization": normalization,
            "activation": activation,
            "film_config": self.film_config.to_dict() if self.film_enabled else None,
            "noise_type": noise_type,
            "noise_scale": noise_scale,
        }

        # Build encoder (use FiLM version if enabled)
        if self.film_enabled and self.film_config.modulate_encoder:
            self.encoder = FiLMUNetEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
                num_levels=encoder_levels,
                blocks_per_level=blocks_per_level,
                normalization=normalization,
                activation=activation,
                film_levels=self.film_config.encoder_levels,
                post_norm=self.film_config.post_norm,
            )
        else:
            self.encoder = UNetEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
                num_levels=encoder_levels,
                blocks_per_level=blocks_per_level,
                normalization=normalization,
                activation=activation,
            )

        bottleneck_channels = self.encoder.out_channels
        afno_hidden = hidden_dim or bottleneck_channels * 2

        # Build bottleneck (use FiLM AFNO blocks if configured)
        if self.film_enabled and self.film_config.modulate_afno_post:
            afno_block_indices = self.film_config.afno_blocks or list(range(afno_blocks))
            self.bottleneck = nn.ModuleList([
                FiLMAFNOBlock(
                    channels=bottleneck_channels,
                    modes=modes,
                    mlp_ratio=afno_hidden / bottleneck_channels,
                    activation=activation,
                    post_norm=self.film_config.post_norm,
                )
                if i in afno_block_indices
                else AFNOBlock(
                    channels=bottleneck_channels,
                    modes=modes,
                    mlp_ratio=afno_hidden / bottleneck_channels,
                    activation=activation,
                )
                for i in range(afno_blocks)
            ])
            self._afno_film_indices = afno_block_indices
        else:
            self.bottleneck = nn.ModuleList([
                AFNOBlock(
                    channels=bottleneck_channels,
                    modes=modes,
                    mlp_ratio=afno_hidden / bottleneck_channels,
                    activation=activation,
                )
                for _ in range(afno_blocks)
            ])
            self._afno_film_indices = []

        # Build decoder (use FiLM version if enabled)
        if self.film_enabled and self.film_config.modulate_decoder:
            self.decoder = FiLMUNetDecoder(
                in_channels=bottleneck_channels,
                out_channels=out_channels,
                skip_channels=self.encoder.skip_channels,
                blocks_per_level=blocks_per_level,
                normalization=normalization,
                activation=activation,
                film_levels=self.film_config.decoder_levels,
                post_norm=self.film_config.post_norm,
            )
        else:
            self.decoder = UNetDecoder(
                in_channels=bottleneck_channels,
                out_channels=out_channels,
                skip_channels=self.encoder.skip_channels,
                blocks_per_level=blocks_per_level,
                normalization=normalization,
                activation=activation,
            )

        # Build FiLM generator if enabled
        if self.film_enabled:
            # Build layer specs from config
            encoder_channels = self.encoder.skip_channels
            decoder_channels = list(reversed(self.encoder.skip_channels))
            layer_specs = build_film_layer_specs(
                self.film_config,
                encoder_channels=encoder_channels,
                decoder_channels=decoder_channels,
                afno_channels=bottleneck_channels,
                num_afno_blocks=afno_blocks,
            )

            self.film_generator = FiLMGenerator(
                layer_specs=layer_specs,
                embed_dim=self.film_config.embed_dim,
                hidden_dim=self.film_config.hidden_dim,
                init_gamma=self.film_config.init_gamma,
                init_beta=self.film_config.init_beta,
            )
        else:
            self.film_generator = None

        # Optional stochastic block
        self.stochastic: Optional[StochasticBlock] = None
        if noise_type and noise_scale and noise_scale > 0:
            self.stochastic = StochasticBlock(
                out_channels,
                noise_type=noise_type,
                noise_scale=noise_scale,
                always_active=True,
                noise_schedule=noise_schedule,
                spatial_correlation=spatial_correlation,
            )

    def forward(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass with optional FiLM conditioning.

        Args:
            x: Input tensor [B, C, H, W]
            conditioning: Optional conditioning embedding [B, embed_dim]
            step: Current timestep (for noise scheduling)
            max_steps: Max timesteps (for noise scheduling)

        Returns:
            Output tensor [B, out_channels, H, W]
        """
        # Generate FiLM parameters from conditioning
        film_params = None
        if self.film_enabled and conditioning is not None:
            film_params = self.film_generator(conditioning)

        # Encode with optional FiLM modulation
        if isinstance(self.encoder, FiLMUNetEncoder):
            bottleneck, skips = self.encoder(x, film_params)
        else:
            bottleneck, skips = self.encoder(x)

        # AFNO bottleneck with optional post-spectral FiLM
        for i, block in enumerate(self.bottleneck):
            if isinstance(block, FiLMAFNOBlock) and film_params:
                key = f"afno_post_{i}"
                if key in film_params:
                    gamma, beta = film_params[key]
                    bottleneck = block(bottleneck, gamma=gamma, beta=beta)
                else:
                    bottleneck = block(bottleneck)
            else:
                bottleneck = block(bottleneck)

        # Decode with optional FiLM modulation
        if isinstance(self.decoder, FiLMUNetDecoder):
            out = self.decoder(bottleneck, skips, film_params)
        else:
            out = self.decoder(bottleneck, skips)

        # Optional stochasticity
        if self.stochastic is not None:
            out = self.stochastic(out, step=step, max_steps=max_steps)

        return out

    def get_intermediate_features(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        extract_from: str = "bottleneck",
        skip_levels: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract intermediate features for learned SUMMARY descriptors.

        Args:
            x: Input tensor [B, C, H, W]
            conditioning: Optional conditioning embedding [B, embed_dim]
            extract_from: What to extract ("bottleneck", "skips", "all")
            skip_levels: Which skip levels to extract

        Returns:
            Dictionary with extracted features
        """
        features: Dict[str, torch.Tensor] = {}

        # Generate FiLM parameters
        film_params = None
        if self.film_enabled and conditioning is not None:
            film_params = self.film_generator(conditioning)

        # Run encoder
        if isinstance(self.encoder, FiLMUNetEncoder):
            bottleneck_raw, skips = self.encoder(x, film_params)
        else:
            bottleneck_raw, skips = self.encoder(x)

        # Run AFNO bottleneck
        bottleneck_processed = bottleneck_raw
        for i, block in enumerate(self.bottleneck):
            if isinstance(block, FiLMAFNOBlock) and film_params:
                key = f"afno_post_{i}"
                if key in film_params:
                    gamma, beta = film_params[key]
                    bottleneck_processed = block(bottleneck_processed, gamma=gamma, beta=beta)
                else:
                    bottleneck_processed = block(bottleneck_processed)
            else:
                bottleneck_processed = block(bottleneck_processed)

        # Extract features
        if extract_from in ("bottleneck", "all"):
            features["bottleneck"] = bottleneck_processed

        if extract_from in ("skips", "all"):
            levels_to_extract = skip_levels or list(range(len(skips)))
            for level in levels_to_extract:
                if 0 <= level < len(skips):
                    features[f"skip_{level}"] = skips[level]

        return features

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        return self.config.copy()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FiLMUAFNOOperator":
        """Create instance from configuration dictionary."""
        return cls(**config)

    def reinit_film_layers(self) -> None:
        """Re-initialize FiLM layers to identity transformation.

        Useful when loading incompatible checkpoints.
        """
        if self.film_generator is not None:
            self.film_generator.reinit_all_identity()

    def get_film_parameter_count(self) -> int:
        """Return number of parameters in FiLM generator."""
        if self.film_generator is not None:
            return self.film_generator.count_parameters()
        return 0
