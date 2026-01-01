"""
CNN encoder for learned IC representations.

Architecture: ResNet-3 (3 residual blocks)
- Lighter than ResNet-18, optimized for 128×128 single-channel inputs
- Adapted from unisim's IC encoder but with richer 28D output

Input: [B*M, 1, 128, 128]
Output: [B*M, 28] learned embeddings

Note: Designed for generative extension - can add decoder for VAE/autoencoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Basic residual block with optional downsampling.

    Structure:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+) -> ReLU
        |                                                |
        +------------> Downsample (if stride > 1) ------+
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Args:
            in_channels: Input channel count
            out_channels: Output channel count
            stride: Stride for first convolution (for downsampling)
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample for skip connection if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]

        Returns:
            [B, C_out, H//stride, W//stride]
        """
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection with optional downsampling
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ICCNNEncoder(nn.Module):
    """
    ResNet-3 CNN encoder for IC embeddings.

    Architecture:
        Stage 0: Conv(1→32, k=7, s=2) + BN + ReLU + MaxPool
        Stage 1: ResBlock(32→64, s=2)
        Stage 2: ResBlock(64→128, s=2)
        Stage 3: ResBlock(128→256, s=2)
        Output: AdaptiveAvgPool(1×1) → Linear(256→embedding_dim) → BatchNorm1d

    Spatial resolution progression (for 128×128 input):
        128×128 → 64×64 (conv1) → 32×32 (maxpool) →
        16×16 (stage1) → 8×8 (stage2) → 4×4 (stage3) →
        1×1 (global pool) → embedding_dim

    Input: [B*M, 1, H, W] where H=W=128
    Output: [B*M, embedding_dim] learned features
    """

    def __init__(self, embedding_dim: int = 28):
        """
        Args:
            embedding_dim: Output embedding dimensionality
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # Stage 0: Initial convolution
        self.conv1 = nn.Conv2d(
            1, 32,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 1: 32→64 (downsample)
        self.stage1 = ResidualBlock(32, 64, stride=2)

        # Stage 2: 64→128 (downsample)
        self.stage2 = ResidualBlock(64, 128, stride=2)

        # Stage 3: 128→256 (downsample)
        self.stage3 = ResidualBlock(128, 256, stride=2)

        # Global pooling and embedding projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, embedding_dim)
        self.bn_out = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B*M, 1, H, W] initial conditions (typically H=W=128)

        Returns:
            [B*M, embedding_dim] learned feature embeddings
        """
        # Stage 0: Initial convolution
        x = self.conv1(x)           # [B*M, 32, 64, 64]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # [B*M, 32, 32, 32]

        # Residual stages with downsampling
        x = self.stage1(x)          # [B*M, 64, 16, 16]
        x = self.stage2(x)          # [B*M, 128, 8, 8]
        x = self.stage3(x)          # [B*M, 256, 4, 4]

        # Global average pooling
        x = self.avgpool(x)         # [B*M, 256, 1, 1]
        x = torch.flatten(x, 1)     # [B*M, 256]

        # Embedding projection with normalization
        x = self.fc(x)              # [B*M, embedding_dim]
        x = self.bn_out(x)          # Normalize embeddings

        return x

    def get_intermediate_features(self, x: torch.Tensor) -> dict:
        """
        Get intermediate feature maps for visualization/analysis.

        Args:
            x: [B*M, 1, H, W]

        Returns:
            Dictionary with intermediate activations:
                'conv1': [B*M, 32, 64, 64]
                'stage1': [B*M, 64, 16, 16]
                'stage2': [B*M, 128, 8, 8]
                'stage3': [B*M, 256, 4, 4]
                'embedding': [B*M, embedding_dim]
        """
        features = {}

        # Stage 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['conv1'] = x
        x = self.maxpool(x)

        # Residual stages
        x = self.stage1(x)
        features['stage1'] = x

        x = self.stage2(x)
        features['stage2'] = x

        x = self.stage3(x)
        features['stage3'] = x

        # Global pooling + embedding
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bn_out(x)
        features['embedding'] = x

        return features


class ICCNNDecoder(nn.Module):
    """
    Decoder for generative IC reconstruction (VAE/autoencoder extension).

    Mirrors the encoder architecture in reverse:
        embedding_dim → 256 (FC) → 4×4 (reshape) →
        256→128 (upsample) → 128→64 (upsample) → 64→32 (upsample) →
        32→1 (final conv) → 128×128

    Note: This is for future generative capability when NOA needs to
          construct ICs from embeddings.
    """

    def __init__(self, embedding_dim: int = 28):
        """
        Args:
            embedding_dim: Input embedding dimensionality
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # Project embedding to spatial features
        self.fc = nn.Linear(embedding_dim, 256 * 4 * 4)
        self.bn_in = nn.BatchNorm1d(256 * 4 * 4)

        # Upsampling stages (transpose convolutions)
        self.stage1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.stage2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Final upsampling to original resolution
        self.stage4 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Final 1×1 conv to get single channel
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode embedding to IC reconstruction.

        Args:
            z: [B*M, embedding_dim] latent codes

        Returns:
            [B*M, 1, 128, 128] reconstructed ICs
        """
        # Project and reshape
        x = self.fc(z)              # [B*M, 256*4*4]
        x = self.bn_in(x)
        x = x.view(-1, 256, 4, 4)   # [B*M, 256, 4, 4]

        # Upsample through stages
        x = self.stage1(x)          # [B*M, 128, 8, 8]
        x = self.stage2(x)          # [B*M, 64, 16, 16]
        x = self.stage3(x)          # [B*M, 32, 32, 32]
        x = self.stage4(x)          # [B*M, 32, 64, 64]

        # Actually we need one more upsampling to get to 128×128
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)

        # Final convolution
        x = self.final_conv(x)      # [B*M, 1, 128, 128]

        return x


class ICVAE(nn.Module):
    """
    Variational Autoencoder for IC representation learning.

    Combines encoder + decoder with reparameterization trick for
    generative capability. Enables NOA to sample and construct ICs.

    Architecture:
        IC [B, 1, 128, 128] → Encoder → μ, log_σ² [B, embedding_dim]
                                          ↓ (sample z ~ N(μ, σ²))
        IC_recon [B, 1, 128, 128] ← Decoder ← z [B, embedding_dim]

    Loss: reconstruction + KL divergence
    """

    def __init__(self, embedding_dim: int = 28):
        """
        Args:
            embedding_dim: Latent code dimensionality
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # Encoder (deterministic backbone)
        self.encoder_backbone = ICCNNEncoder(embedding_dim=256)

        # VAE heads for μ and log_σ²
        self.fc_mu = nn.Linear(256, embedding_dim)
        self.fc_logvar = nn.Linear(256, embedding_dim)

        # Decoder
        self.decoder = ICCNNDecoder(embedding_dim=embedding_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode IC to latent distribution parameters.

        Args:
            x: [B, 1, 128, 128] ICs

        Returns:
            mu: [B, embedding_dim] mean
            logvar: [B, embedding_dim] log variance
        """
        h = self.encoder_backbone(x)  # [B, 256]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε where ε ~ N(0, 1).

        Args:
            mu: [B, embedding_dim]
            logvar: [B, embedding_dim]

        Returns:
            z: [B, embedding_dim] sampled latent codes
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to IC reconstruction.

        Args:
            z: [B, embedding_dim]

        Returns:
            [B, 1, 128, 128] reconstructed ICs
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass.

        Args:
            x: [B, 1, 128, 128] input ICs

        Returns:
            recon: [B, 1, 128, 128] reconstructions
            mu: [B, embedding_dim] latent means
            logvar: [B, embedding_dim] latent log variances
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample ICs from prior N(0, I).

        Args:
            num_samples: Number of ICs to generate
            device: Device to generate on

        Returns:
            [num_samples, 1, 128, 128] generated ICs
        """
        z = torch.randn(num_samples, self.embedding_dim, device=device)
        ics = self.decode(z)
        return ics
