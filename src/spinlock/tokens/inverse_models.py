"""
Inverse decoders for VQTokenizer reconstruction.

This module provides neural network decoders that map from encoded feature spaces
back to the original continuous spaces:
  - ThetaInverseMLP: Encoded theta [B, 32] → actual parameters [B, 14]
  - InitialInverseCNN: Encoded initial [B, D] → spatial grids [B, C, 64, 64]
  - TemporalInverseMLP: Encoded temporal [B, 1920] → synthetic CNN features [B, T_rt, 240]

These inverse models enable the VQTokenizer to properly decode tokens back to
(theta, ICs, temporal features) for roundtrip symbolic self-consistency:
tokens → decode → re-encode → same tokens.
"""

import math

import torch
import torch.nn as nn
from typing import Tuple


class ThetaInverseMLP(nn.Module):
    """
    Inverse decoder: encoded theta features → actual operator parameters.

    Maps from the encoded theta space [B, encoded_dim] back to actual
    continuous parameters [B, param_dim] in [0,1] range.

    Architecture:
        Input [B, encoded_dim] → Linear(encoded_dim, 64) → LayerNorm → ReLU → Dropout
                                → Linear(64, param_dim) → Sigmoid
                                → Output [B, param_dim] in [0,1]

    Args:
        encoded_dim: Dimension of encoded theta features (default: 32)
        param_dim: Dimension of actual parameters (default: 14)
        hidden_dim: Hidden layer size (default: 64)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        encoded_dim: int = 32,
        param_dim: int = 14,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim

        # Build MLP: encoded_dim → hidden_dim → param_dim
        self.net = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, param_dim),
            nn.Sigmoid(),  # Ensure output in [0,1]
        )

    def forward(self, theta_encoded: torch.Tensor) -> torch.Tensor:
        """
        Decode encoded theta to actual parameters.

        Args:
            theta_encoded: [B, encoded_dim] encoded theta features

        Returns:
            [B, param_dim] actual parameters in [0,1]
        """
        return self.net(theta_encoded)

    def __repr__(self) -> str:
        return (
            f"ThetaInverseMLP(encoded_dim={self.encoded_dim}, "
            f"param_dim={self.param_dim}, hidden_dim={self.hidden_dim})"
        )


class _ResBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN residual block (pre-activation style)."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class InitialInverseCNN(nn.Module):
    """
    Inverse decoder: encoded initial features → spatial initial conditions.

    Maps from the encoded initial space [B, encoded_dim] back to spatial
    initial condition grids [B, channels, spatial_size, spatial_size].

    Architecture:
        Input [B, encoded_dim] → Linear → Reshape [B, 128, 16, 16]
                               → ResBlock(128)
                               → Upsample(2×) + Conv 128→64 → ResBlock(64)
                               → Upsample(2×) + Conv 64→channels
                               → Sigmoid
                               → Output [B, channels, 64, 64]

    Improvements over the original ConvTranspose2d decoder:
      - Bilinear upsample + conv eliminates checkerboard artifacts
      - Residual blocks add refinement capacity at each resolution
      - 16×16 starting resolution reduces upsampling burden (2 stages vs 3)
      - Sigmoid output matches [0,1] target range (Lenia ICs/states)

    Args:
        encoded_dim: Dimension of encoded initial features (default: 426)
        channels: Number of output channels (default: 3)
        spatial_size: Spatial dimension of output grid (default: 64)
    """

    def __init__(
        self,
        encoded_dim: int = 426,
        channels: int = 3,
        spatial_size: int = 64,
    ):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.channels = channels
        self.spatial_size = spatial_size

        # Validate spatial_size is power of 2 and >= 32
        assert spatial_size in [32, 64, 128], f"spatial_size must be 32, 64, or 128, got {spatial_size}"

        # Start from 16×16 (spatial_size // 4) instead of 8×8
        self.start_size = spatial_size // 4
        self.start_channels = 128

        # Project encoded features to spatial starting point via bottleneck
        spatial_dim = self.start_channels * self.start_size * self.start_size
        bottleneck = min(512, encoded_dim)
        self.project = nn.Sequential(
            nn.Linear(encoded_dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, spatial_dim),
            nn.ReLU(inplace=True),
        )

        # 16×16: residual refinement at starting resolution
        self.res0 = _ResBlock(128)

        # 16×16 → 32×32: bilinear upsample + conv + residual
        self.up1_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.res1 = _ResBlock(64)

        # 32×32 → 64×64: bilinear upsample + conv to output channels
        self.up2_conv = nn.Sequential(
            nn.Conv2d(64, channels, 3, padding=1),
        )

    def forward(self, initial_encoded: torch.Tensor) -> torch.Tensor:
        """
        Decode encoded initial features to spatial grids.

        Args:
            initial_encoded: [B, encoded_dim] encoded initial features

        Returns:
            [B, channels, spatial_size, spatial_size] spatial initial conditions
        """
        B = initial_encoded.shape[0]

        # Project to spatial features [B, 128, 16, 16]
        x = self.project(initial_encoded)
        x = x.view(B, self.start_channels, self.start_size, self.start_size)

        # Refine at 16×16
        x = self.res0(x)

        # 16×16 → 32×32
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up1_conv(x)
        x = self.res1(x)

        # 32×32 → 64×64
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up2_conv(x)

        # Sigmoid: ICs are in [0, 1]
        return torch.sigmoid(x)

    def __repr__(self) -> str:
        return (
            f"InitialInverseCNN(encoded_dim={self.encoded_dim}, "
            f"channels={self.channels}, spatial_size={self.spatial_size})"
        )


class TemporalInverseMLP(nn.Module):
    """
    Inverse decoder: encoded temporal features → synthetic CNN feature space.

    Maps from the concatenated pyramid-encoded temporal features [B, encoded_dim]
    to a synthetic trajectory in CNN feature space [B, T_rt, cnn_dim]. The output
    is re-encoded through the REAL PyramidTemporalEncoders during roundtrip loss,
    testing the full decode → re-encode → quantize cycle for symbolic self-consistency.

    The bottleneck hidden layer forces cross-group information sharing: all 30
    temporal groups must jointly reconstruct a coherent CNN feature trajectory.

    Architecture:
        Input [B, encoded_dim] → Linear → LayerNorm → ReLU → Dropout
                                → Linear (bottleneck) → LayerNorm → ReLU → Dropout
                                → Linear → reshape → Output [B, T_rt, cnn_dim]

    Args:
        encoded_dim: Total encoded temporal dimension (e.g. 1920 = 30 groups × 64)
        cnn_dim: CNN output dimension per frame (e.g. 240 = 30 groups × 8)
        roundtrip_timesteps: Number of synthetic timesteps to generate (default: 8)
        hidden_dim: Hidden layer size (default: 512)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        encoded_dim: int,
        cnn_dim: int,
        roundtrip_timesteps: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.cnn_dim = cnn_dim
        self.roundtrip_timesteps = roundtrip_timesteps
        self.hidden_dim = hidden_dim

        output_dim = roundtrip_timesteps * cnn_dim

        self.net = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, temporal_encoded: torch.Tensor) -> torch.Tensor:
        """
        Decode encoded temporal features to synthetic CNN feature trajectory.

        Args:
            temporal_encoded: [B, encoded_dim] concatenated pyramid-encoded temporal

        Returns:
            [B, T_rt, cnn_dim] synthetic CNN features for roundtrip re-encoding
        """
        B = temporal_encoded.shape[0]
        flat = self.net(temporal_encoded)
        return flat.view(B, self.roundtrip_timesteps, self.cnn_dim)

    def __repr__(self) -> str:
        return (
            f"TemporalInverseMLP(encoded_dim={self.encoded_dim}, "
            f"cnn_dim={self.cnn_dim}, T_rt={self.roundtrip_timesteps}, "
            f"hidden_dim={self.hidden_dim})"
        )


class InitialSpectralInverse(nn.Module):
    """Decode quantized IC features to grids via Fourier synthesis.

    MLP: encoded_dim → 2*C*K*K → reshape to [B, C, K, K] complex → hermitian_ifft2.
    Counterpart of SpectralICEncoder: the encoder extracts low-frequency Fourier
    coefficients deterministically; this inverse reconstructs the grid from
    quantized coefficients via iFFT with Hermitian symmetry.

    Args:
        encoded_dim: Input dimension from the shared decoder / quantized space.
        channels: Number of output channels (e.g. 3 for Lenia).
        spatial_size: Target grid size H=W.
        num_modes: Fourier modes per spatial dimension.
        hidden_dims: MLP hidden layer dimensions.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        encoded_dim: int,
        channels: int,
        spatial_size: int,
        num_modes: int = 16,
        hidden_dims: list[int] = [256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.channels = channels
        self.spatial_size = spatial_size
        self.num_modes = num_modes

        if num_modes > spatial_size // 2:
            raise ValueError(
                f"num_modes ({num_modes}) must be <= spatial_size//2 = {spatial_size // 2}"
            )

        # MLP: encoded_dim → C * K * K * 2 (real + imaginary)
        coeff_size = channels * num_modes * num_modes * 2
        layers = []
        prev_dim = encoded_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, coeff_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Decode quantized IC features to spatial grids.

        Args:
            encoded: Quantized/reconstructed IC features [B, encoded_dim].

        Returns:
            Reconstructed grids [B, C, H, W].
        """
        from .spectral_utils import hermitian_ifft2

        B = encoded.shape[0]
        C = self.channels
        K = self.num_modes

        raw = self.mlp(encoded)  # [B, C * K * K * 2]
        raw = raw.view(B, C, K, K, 2)
        coeffs = torch.complex(raw[..., 0], raw[..., 1])  # [B, C, K, K]

        return hermitian_ifft2(coeffs, self.spatial_size, self.spatial_size)

    def __repr__(self) -> str:
        return (
            f"InitialSpectralInverse(encoded_dim={self.encoded_dim}, "
            f"channels={self.channels}, spatial_size={self.spatial_size}, "
            f"num_modes={self.num_modes})"
        )


def _sinusoidal_embedding(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding for temporal positions in [0, 1].

    Uses the standard Transformer positional encoding formula adapted for
    continuous positions. Each position generates a dim-dimensional vector
    of alternating sin/cos frequencies at geometrically increasing wavelengths.

    Args:
        positions: [K] tensor of temporal positions in [0, 1].
        dim: Embedding dimension (must be even).

    Returns:
        [K, dim] sinusoidal embeddings.
    """
    half_dim = dim // 2
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=positions.device, dtype=positions.dtype) / half_dim
    )
    # [K, half_dim]: outer product of positions and frequencies
    args = positions.unsqueeze(1) * freq.unsqueeze(0) * math.pi
    return torch.cat([args.sin(), args.cos()], dim=-1)  # [K, dim]


class TrajectoryPrototypeHead(nn.Module):
    """Decode quantized latents to K keyframe images using FiLM-conditioned spatial decoder.

    V2 architecture: wider latent (512D), wider spatial decoder (128 base channels),
    FiLM time conditioning at each upsample stage for selective feature activation.

    Architecture:
        [B, encoded_dim] → pre_project → [B, latent_dim=512]
        → project_spatial → [B, base_ch, 4, 4]
        → ResBlock
        → {Upsample(2×) → Conv → FiLM(t) → ResBlock} × N_stages
        → Conv → Sigmoid → [B, C, H, W]

    Time conditioning is generated per-keyframe via sinusoidal embedding →
    time_mlp → FiLMGenerator, producing per-stage (gamma, beta) pairs.
    Identity initialization (gamma=1, beta=0) ensures stable training start.

    Starts at 4×4 spatial resolution; total ~4.3M params for 128×128 output
    (vs ~1.4M in v1). Channel schedule halves every 2 stages:
    128→128→64→64→32 for spatial_size=128.

    Args:
        encoded_dim: Total quantized latent dimension (e.g. 2880 for 30 groups × 3 levels).
        channels: Number of output image channels (e.g. 3 for Lenia).
        spatial_size: Spatial resolution of decoded keyframes (must be a power of 2, >= 8).
        num_keyframes: Number of temporal keyframes to decode (default: 16).
        time_embed_dim: Dimension of raw sinusoidal time embedding (default: 64).
        latent_dim: Compressed latent dimension (default: 512, v1 was 256).
        base_ch: Base channel width for spatial decoder (default: 128, v1 was 64).
    """

    def __init__(
        self,
        encoded_dim: int,
        channels: int = 3,
        spatial_size: int = 128,
        num_keyframes: int = 16,
        time_embed_dim: int = 64,
        latent_dim: int = 512,
        base_ch: int = 128,
    ):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.channels = channels
        self.spatial_size = spatial_size
        self.num_keyframes = num_keyframes
        self.time_embed_dim = time_embed_dim
        self._base_ch = base_ch

        from spinlock.operators.film import FiLMLayer, FiLMGenerator, FiLMLayerSpec

        # Pre-projection: compress high-dim quantized vector to latent_dim
        self.pre_project = nn.Sequential(
            nn.Linear(encoded_dim, latent_dim),
            nn.ReLU(inplace=True),
        )

        # Time embedding MLP: sinusoidal(k/K) → conditioning embedding for FiLM
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.GELU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        # Spatial decoder: progressive upsampling from 4×4
        n_stages = int(math.log2(spatial_size // 4))
        assert 4 * (2 ** n_stages) == spatial_size, \
            f"spatial_size must be 4 * 2^N, got {spatial_size}"

        # Flat → spatial: latent_dim → base_ch × 4 × 4
        self.project_spatial = nn.Sequential(
            nn.Linear(latent_dim, base_ch * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.res0 = _ResBlock(base_ch)

        # Build channel schedule: halve every 2 stages (after stage 0)
        stage_channels = []
        in_ch = base_ch
        for i in range(n_stages):
            out_ch = max(32, in_ch // 2) if (i > 0 and i % 2 == 0) else in_ch
            stage_channels.append(out_ch)
            in_ch = out_ch

        # Progressive upsample stages
        self.up_convs = nn.ModuleList()
        self.up_res = nn.ModuleList()
        prev_ch = base_ch
        for out_ch in stage_channels:
            self.up_convs.append(nn.Sequential(
                nn.Conv2d(prev_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            self.up_res.append(_ResBlock(out_ch))
            prev_ch = out_ch

        # FiLM conditioning: per-stage modulation from time embedding
        film_specs = [
            FiLMLayerSpec(name=f"up_{i}", channels=ch, location="decoder", level=i)
            for i, ch in enumerate(stage_channels)
        ]
        self.film_generator = FiLMGenerator(
            layer_specs=film_specs,
            embed_dim=time_embed_dim,
            hidden_dim=time_embed_dim,
            init_gamma=1.0,
            init_beta=0.0,
        )
        self.film_layers = nn.ModuleList([
            FiLMLayer(ch, post_norm=True) for ch in stage_channels
        ])

        # Final conv to output channels
        self.final_conv = nn.Conv2d(stage_channels[-1], channels, 3, padding=1)

    def _decode_spatial(self, z: torch.Tensor, film_params: dict) -> torch.Tensor:
        """Decode latent vectors to spatial images with FiLM time conditioning.

        Uses gradient checkpointing on the upsample stages to trade compute
        for memory: intermediate activations are recomputed during backward
        instead of stored (~5× activation memory reduction).

        Args:
            z: Pre-projected latent vectors [N, latent_dim].
            film_params: Dict mapping stage name → (gamma, beta) from FiLMGenerator.

        Returns:
            [N, C, H, W] decoded images in [0, 1].
        """
        x = self.project_spatial(z)
        x = x.view(-1, self._base_ch, 4, 4)
        x = self.res0(x)

        for i, (up_conv, up_res, film_layer) in enumerate(
            zip(self.up_convs, self.up_res, self.film_layers)
        ):
            gamma, beta = film_params[f"up_{i}"]
            x = torch.utils.checkpoint.checkpoint(
                self._upsample_stage, x, up_conv, up_res, film_layer, gamma, beta,
                use_reentrant=False,
            )

        return torch.sigmoid(self.final_conv(x))

    @staticmethod
    def _upsample_stage(x, up_conv, up_res, film_layer, gamma, beta):
        """Single upsample stage with FiLM modulation (checkpointed to save memory)."""
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = up_conv(x)
        x = film_layer(x, gamma, beta)
        x = up_res(x)
        return x

    def forward(self, quantized_latent: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents to K keyframe prototype images.

        Decodes keyframes in chunks to bound peak GPU memory. With B=24
        and K=16, a full batched decode would push B*K=384 samples through
        the spatial CNN simultaneously. Chunking (default: 2 keyframes at
        a time) reduces peak memory by K/chunk_size while autograd still
        computes correct gradients across all chunks.

        Args:
            quantized_latent: [B, encoded_dim] concatenated quantized VQ codes.

        Returns:
            [B, K, C, H, W] prototype trajectory keyframes in [0, 1].
        """
        B = quantized_latent.shape[0]
        K = self.num_keyframes
        device = quantized_latent.device

        # Compress to latent_dim
        z = self.pre_project(quantized_latent)  # [B, latent_dim]

        # Generate sinusoidal time embeddings for K evenly-spaced positions
        t_positions = torch.linspace(0.0, 1.0, K, device=device)  # [K]
        t_raw = _sinusoidal_embedding(t_positions, self.time_embed_dim)  # [K, time_embed_dim]
        t_embeds = self.time_mlp(t_raw)  # [K, time_embed_dim]

        # Chunked keyframe decode to bound GPU memory.
        # Each chunk processes B * chunk_size samples through the spatial decoder.
        chunk_size = min(2, K)
        frame_chunks = []

        for i in range(0, K, chunk_size):
            chunk_t = t_embeds[i:i + chunk_size]  # [C_k, time_embed_dim]
            C_k = chunk_t.shape[0]

            # Tile z for chunk: [B, latent_dim] → [B*C_k, latent_dim]
            z_tiled = z.unsqueeze(1).expand(-1, C_k, -1).reshape(B * C_k, -1)

            # Tile t for batch: [C_k, D_t] → [B*C_k, D_t]
            t_tiled = chunk_t.unsqueeze(0).expand(B, -1, -1).reshape(B * C_k, -1)

            # Generate per-stage FiLM params from time embeddings
            film_params = self.film_generator(t_tiled)

            # Decode with FiLM-conditioned spatial decoder
            chunk_frames = self._decode_spatial(z_tiled, film_params)  # [B*C_k, C, H, W]
            frame_chunks.append(chunk_frames.reshape(B, C_k, self.channels,
                                                      self.spatial_size, self.spatial_size))

        return torch.cat(frame_chunks, dim=1)  # [B, K, C, H, W]

    def __repr__(self) -> str:
        return (
            f"TrajectoryPrototypeHead(encoded_dim={self.encoded_dim}, "
            f"channels={self.channels}, spatial={self.spatial_size}, "
            f"K={self.num_keyframes}, base_ch={self._base_ch})"
        )
