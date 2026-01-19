"""Feature-wise Linear Modulation (FiLM) for parameter conditioning.

FiLM applies affine transformations (gamma * features + beta) to feature maps,
enabling fine-grained theta-conditioned control without inflating input channels.

Key design principles:
- Spatial-only modulation by default (encoder/decoder conv layers)
- POST-spectral AFNO modulation only (never inside FFT path)
- Post-FiLM LayerNorm to prevent drift on long rollouts (T=256)
- Identity initialization (gamma=1, beta=0) for stable training

Literature reference: UFNO-FiLM (2025-2026) reports 15-25% error reduction
vs concatenation-based conditioning.

Documentation:
    - FiLM implementation plan: docs/film-modulation-plan.md
    - MNO architecture spec: docs/MNO_ARCHITECTURE.md
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import warnings

import torch
import torch.nn as nn


@dataclass
class FiLMLayerSpec:
    """Specification for a single FiLM layer.

    Attributes:
        name: Unique identifier (e.g., "encoder_0", "decoder_1", "afno_post_0")
        channels: Number of channels to modulate at this layer
        location: Where in the network ("encoder" | "decoder" | "afno_post")
        level: Index within location (e.g., 0, 1, 2 for encoder levels)
    """

    name: str
    channels: int
    location: str  # "encoder" | "decoder" | "afno_post"
    level: int


@dataclass
class FiLMConfig:
    """Configuration for FiLM modulation.

    Loaded from YAML config and controls which layers receive FiLM modulation.

    Example YAML:
        film:
          enabled: true
          embed_dim: 64
          hidden_dim: 128
          post_norm: true
          modulate_encoder: true
          modulate_decoder: true
          modulate_afno_post: false

    Attributes:
        enabled: Whether FiLM modulation is enabled
        embed_dim: Embedding dimension (must match param_embed_dim)
        hidden_dim: Hidden dimension in projection heads
        init_gamma: Initial gamma value (default 1.0 for identity)
        init_beta: Initial beta value (default 0.0 for identity)
        post_norm: Apply LayerNorm after FiLM (recommended for T=256 stability)
        modulate_encoder: Apply FiLM to encoder conv outputs (spatial, safe)
        modulate_decoder: Apply FiLM to decoder conv outputs (spatial, safe)
        encoder_levels: Which encoder levels to modulate (None = all)
        decoder_levels: Which decoder levels to modulate (None = all)
        modulate_afno_post: Apply FiLM after AFNO blocks (post-spectral only)
        afno_blocks: Which AFNO blocks to modulate if enabled (None = all)
        _warn_afno: Emit warning if AFNO modulation is enabled
    """

    enabled: bool = False
    embed_dim: int = 64
    hidden_dim: int = 128
    init_gamma: float = 1.0
    init_beta: float = 0.0
    post_norm: bool = True

    # Spatial modulation (DEFAULT - safe for long rollouts)
    modulate_encoder: bool = True
    modulate_decoder: bool = True
    encoder_levels: Optional[List[int]] = None
    decoder_levels: Optional[List[int]] = None

    # AFNO modulation (USE WITH CAUTION - post-spectral only)
    modulate_afno_post: bool = False
    afno_blocks: Optional[List[int]] = None
    _warn_afno: bool = True

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "FiLMConfig":
        """Create FiLMConfig from dictionary (e.g., from YAML).

        Args:
            config: Configuration dictionary

        Returns:
            FiLMConfig instance
        """
        return cls(
            enabled=config.get("enabled", False),
            embed_dim=config.get("embed_dim", 64),
            hidden_dim=config.get("hidden_dim", 128),
            init_gamma=config.get("init_gamma", 1.0),
            init_beta=config.get("init_beta", 0.0),
            post_norm=config.get("post_norm", True),
            modulate_encoder=config.get("modulate_encoder", True),
            modulate_decoder=config.get("modulate_decoder", True),
            encoder_levels=config.get("encoder_levels"),
            decoder_levels=config.get("decoder_levels"),
            modulate_afno_post=config.get("modulate_afno_post", False),
            afno_blocks=config.get("afno_blocks"),
            _warn_afno=config.get("_warn_afno", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "init_gamma": self.init_gamma,
            "init_beta": self.init_beta,
            "post_norm": self.post_norm,
            "modulate_encoder": self.modulate_encoder,
            "modulate_decoder": self.modulate_decoder,
            "encoder_levels": self.encoder_levels,
            "decoder_levels": self.decoder_levels,
            "modulate_afno_post": self.modulate_afno_post,
            "afno_blocks": self.afno_blocks,
        }

    def validate(self) -> None:
        """Validate configuration and emit warnings."""
        if self.modulate_afno_post and self._warn_afno:
            warnings.warn(
                "FiLM: AFNO post-spectral modulation enabled. "
                "This may cause instability on long rollouts (T>64). "
                "Monitor gamma/beta statistics and consider spatial-only "
                "modulation if training becomes unstable.",
                UserWarning,
            )


class FiLMLayer(nn.Module):
    """Apply FiLM modulation: output = gamma * features + beta.

    With optional post-normalization for stability on long rollouts.

    The modulation is applied element-wise after broadcasting gamma/beta
    to spatial dimensions.

    Args:
        channels: Number of channels to modulate
        post_norm: Apply LayerNorm after modulation (recommended)

    Example:
        >>> layer = FiLMLayer(64, post_norm=True)
        >>> features = torch.randn(8, 64, 32, 32)
        >>> gamma = torch.ones(8, 64)
        >>> beta = torch.zeros(8, 64)
        >>> out = layer(features, gamma, beta)  # Shape: (8, 64, 32, 32)
    """

    def __init__(self, channels: int, post_norm: bool = True):
        super().__init__()
        self.channels = channels
        self.post_norm = post_norm

        if post_norm:
            # Channel-wise normalization
            # Note: Using GroupNorm(1, channels) as LayerNorm equivalent for 2D
            self.norm = nn.GroupNorm(1, channels)

    def forward(
        self, features: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            features: Feature tensor [B, C, H, W]
            gamma: Scale parameters [B, C]
            beta: Shift parameters [B, C]

        Returns:
            Modulated features [B, C, H, W]
        """
        # Reshape gamma/beta for broadcasting: [B, C] -> [B, C, 1, 1]
        # Use reshape instead of view for potential non-contiguous tensors
        gamma = gamma.reshape(-1, self.channels, 1, 1)
        beta = beta.reshape(-1, self.channels, 1, 1)

        # Apply affine transformation
        out = gamma * features + beta

        # Optional post-normalization
        if self.post_norm:
            out = self.norm(out)

        return out


class FiLMProjectionHead(nn.Module):
    """Per-layer projection head: embed_dim -> hidden -> 2*channels.

    Generates gamma and beta for a single layer from the conditioning embedding.
    Uses identity initialization (gamma=1, beta=0) for stable training start.

    Args:
        embed_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension
        out_channels: Output channels (generates gamma and beta for this many channels)
        init_gamma: Initial gamma value (default 1.0)
        init_beta: Initial beta value (default 0.0)

    Example:
        >>> head = FiLMProjectionHead(64, 128, 256)
        >>> embedding = torch.randn(8, 64)
        >>> gamma, beta = head(embedding)
        >>> gamma.shape, beta.shape  # (8, 256), (8, 256)
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        out_channels: int,
        init_gamma: float = 1.0,
        init_beta: float = 0.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.init_gamma = init_gamma
        self.init_beta = init_beta

        # MLP: embed_dim -> hidden -> 2 * out_channels
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * out_channels),
        )

        # Initialize for identity transformation
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for identity transformation at start."""
        # Zero the final layer's weights
        final_layer = self.mlp[-1]
        nn.init.zeros_(final_layer.weight)

        # Set bias to produce gamma=init_gamma, beta=init_beta
        with torch.no_grad():
            bias = final_layer.bias
            bias[:self.out_channels] = self.init_gamma  # gamma
            bias[self.out_channels:] = self.init_beta  # beta

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate gamma and beta from embedding.

        Args:
            embedding: Conditioning embedding [B, embed_dim]

        Returns:
            Tuple of (gamma, beta), each [B, out_channels]
        """
        out = self.mlp(embedding)  # [B, 2 * out_channels]
        gamma = out[:, :self.out_channels]
        beta = out[:, self.out_channels:]
        return gamma, beta

    def reinit_identity(self) -> None:
        """Re-initialize to identity transformation.

        Useful when loading incompatible checkpoints.
        """
        self._init_weights()


class FiLMGenerator(nn.Module):
    """Generate per-layer gamma/beta from conditioning embedding.

    Creates projection heads for each FiLM layer spec and generates
    all gamma/beta values in a single forward pass.

    Args:
        layer_specs: List of FiLMLayerSpec defining which layers to modulate
        embed_dim: Conditioning embedding dimension
        hidden_dim: Hidden dimension for projection heads
        init_gamma: Initial gamma value
        init_beta: Initial beta value

    Example:
        >>> specs = [
        ...     FiLMLayerSpec("encoder_0", 64, "encoder", 0),
        ...     FiLMLayerSpec("encoder_1", 128, "encoder", 1),
        ... ]
        >>> generator = FiLMGenerator(specs, embed_dim=64, hidden_dim=128)
        >>> embedding = torch.randn(8, 64)
        >>> film_params = generator(embedding)
        >>> film_params["encoder_0"]  # (gamma, beta) tuple
    """

    def __init__(
        self,
        layer_specs: List[FiLMLayerSpec],
        embed_dim: int,
        hidden_dim: int,
        init_gamma: float = 1.0,
        init_beta: float = 0.0,
    ):
        super().__init__()
        self.layer_specs = layer_specs
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Create projection head for each layer
        self.projection_heads = nn.ModuleDict()
        for spec in layer_specs:
            self.projection_heads[spec.name] = FiLMProjectionHead(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                out_channels=spec.channels,
                init_gamma=init_gamma,
                init_beta=init_beta,
            )

    def forward(
        self, embedding: torch.Tensor
    ) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Generate FiLM parameters for all layers.

        Args:
            embedding: Conditioning embedding [B, embed_dim]

        Returns:
            Dictionary mapping layer names to (gamma, beta) tuples
        """
        film_params = {}
        for spec in self.layer_specs:
            gamma, beta = self.projection_heads[spec.name](embedding)
            film_params[spec.name] = (gamma, beta)
        return film_params

    def reinit_all_identity(self) -> None:
        """Re-initialize all projection heads to identity.

        Useful when loading incompatible checkpoints.
        """
        for head in self.projection_heads.values():
            head.reinit_identity()

    def get_layer_names(self) -> List[str]:
        """Return list of all layer names."""
        return [spec.name for spec in self.layer_specs]

    def count_parameters(self) -> int:
        """Count total parameters in generator."""
        return sum(p.numel() for p in self.parameters())


def build_film_layer_specs(
    config: FiLMConfig,
    encoder_channels: List[int],
    decoder_channels: List[int],
    afno_channels: int,
    num_afno_blocks: int,
) -> List[FiLMLayerSpec]:
    """Build layer specs from FiLMConfig and architecture info.

    Args:
        config: FiLM configuration
        encoder_channels: Channel counts for each encoder level
        decoder_channels: Channel counts for each decoder level
        afno_channels: Channel count for AFNO bottleneck
        num_afno_blocks: Number of AFNO blocks in bottleneck

    Returns:
        List of FiLMLayerSpec for all layers to modulate
    """
    specs = []

    # Encoder layers (spatial modulation - safe)
    if config.modulate_encoder:
        levels = config.encoder_levels or list(range(len(encoder_channels)))
        for level in levels:
            if 0 <= level < len(encoder_channels):
                specs.append(
                    FiLMLayerSpec(
                        name=f"encoder_{level}",
                        channels=encoder_channels[level],
                        location="encoder",
                        level=level,
                    )
                )

    # Decoder layers (spatial modulation - safe)
    if config.modulate_decoder:
        levels = config.decoder_levels or list(range(len(decoder_channels)))
        for level in levels:
            if 0 <= level < len(decoder_channels):
                specs.append(
                    FiLMLayerSpec(
                        name=f"decoder_{level}",
                        channels=decoder_channels[level],
                        location="decoder",
                        level=level,
                    )
                )

    # AFNO post-spectral layers (use with caution)
    if config.modulate_afno_post:
        blocks = config.afno_blocks or list(range(num_afno_blocks))
        for block_idx in blocks:
            if 0 <= block_idx < num_afno_blocks:
                specs.append(
                    FiLMLayerSpec(
                        name=f"afno_post_{block_idx}",
                        channels=afno_channels,
                        location="afno_post",
                        level=block_idx,
                    )
                )

    return specs


def estimate_film_parameter_overhead(
    config: FiLMConfig,
    encoder_channels: List[int],
    decoder_channels: List[int],
    afno_channels: int,
    num_afno_blocks: int,
    base_model_params: int,
) -> Dict[str, Any]:
    """Estimate parameter overhead from FiLM modulation.

    Args:
        config: FiLM configuration
        encoder_channels: Channel counts for each encoder level
        decoder_channels: Channel counts for each decoder level
        afno_channels: Channel count for AFNO bottleneck
        num_afno_blocks: Number of AFNO blocks
        base_model_params: Total parameters in base model

    Returns:
        Dictionary with overhead statistics
    """
    specs = build_film_layer_specs(
        config, encoder_channels, decoder_channels, afno_channels, num_afno_blocks
    )

    total_film_params = 0
    breakdown = {}

    for spec in specs:
        # Each projection head: embed_dim * hidden_dim + hidden_dim + hidden_dim * 2*channels + 2*channels
        head_params = (
            config.embed_dim * config.hidden_dim  # First linear weight
            + config.hidden_dim  # First linear bias
            + config.hidden_dim * 2 * spec.channels  # Second linear weight
            + 2 * spec.channels  # Second linear bias
        )
        total_film_params += head_params
        breakdown[spec.name] = head_params

    overhead_pct = 100 * total_film_params / base_model_params if base_model_params > 0 else 0

    return {
        "total_film_params": total_film_params,
        "overhead_percent": overhead_pct,
        "num_layers": len(specs),
        "breakdown": breakdown,
    }
