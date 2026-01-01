"""IC feature extraction configuration."""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class ICManualConfig(BaseModel):
    """
    Manual (hand-crafted) feature extraction configuration.

    Controls which of the 14 manual features to extract:
    - Spatial (4): clustering, localization, autocorrelation
    - Spectral (3): frequency content, power laws
    - Information (4): entropy, complexity, predictability
    - Morphological (3): density, gradients, symmetry
    """
    enabled: bool = True

    # Spatial features (4)
    include_spatial_cluster_count: bool = True
    include_spatial_largest_cluster_frac: bool = True
    include_spatial_autocorr: bool = True
    include_spatial_centroid_dist: bool = True

    # Spectral features (3)
    include_spectral_dominant_freq: bool = True
    include_spectral_centroid: bool = True
    include_spectral_power_law_exp: bool = True

    # Information features (4)
    include_info_entropy: bool = True
    include_info_local_entropy_var: bool = True
    include_info_lz_complexity: bool = True
    include_info_predictability: bool = True

    # Morphological features (3)
    include_morph_density: bool = True
    include_morph_radial_gradient: bool = True
    include_morph_symmetry: bool = True


class ICCNNConfig(BaseModel):
    """
    CNN encoder configuration for learned IC features.

    The CNN can operate in multiple modes:
    - Encoder-only: Extract features for analysis
    - VAE: Generative model for IC construction (NOA thoughts → ICs)
    """
    enabled: bool = True

    # Architecture
    embedding_dim: int = Field(
        default=28,
        ge=8,
        le=128,
        description="Dimensionality of learned embedding"
    )
    architecture: Literal['resnet3'] = Field(
        default='resnet3',
        description="CNN backbone architecture"
    )

    # Training/loading
    pretrained: bool = Field(
        default=False,
        description="Load pretrained weights (if available)"
    )
    freeze_encoder: bool = Field(
        default=False,
        description="Freeze encoder during VQ-VAE training"
    )

    # Generative capability
    use_vae: bool = Field(
        default=False,
        description="Use VAE instead of deterministic encoder (enables generation)"
    )
    vae_beta: float = Field(
        default=1.0,
        ge=0.0,
        description="Beta weight for KL divergence in VAE loss"
    )


class ICConfig(BaseModel):
    """
    IC feature family configuration.

    Extracts hybrid features from initial conditions [N, M, C, H, W]:
    - Manual: 14D hand-crafted features (spatial/spectral/info/morph)
    - CNN: 28D learned embeddings (ResNet-3)
    - Combined: 42D total (14 + 28)

    Output shape: [N, M, D_ic] where M = num_realizations per operator

    Design Notes:
    -------------
    The IC representation is designed for bidirectional use:
    1. **Analysis**: Extract features from existing ICs
    2. **Generation**: Construct ICs from embeddings (NOA thoughts → ICs)

    For generative capability, set `cnn.use_vae = True` to enable
    the VAE architecture with encoder + decoder. This allows the NOA
    to sample from the learned latent space and construct ICs that
    embody its "thoughts" when paired with neural operator parameters.
    """
    version: str = Field(
        default="1.0.0",
        description="IC feature family version"
    )

    # Feature extraction modes
    manual: ICManualConfig = Field(
        default_factory=ICManualConfig,
        description="Manual feature extraction config"
    )
    cnn: ICCNNConfig = Field(
        default_factory=ICCNNConfig,
        description="CNN encoder/decoder config"
    )

    # Storage options
    store_manual_separate: bool = Field(
        default=True,
        description="Store manual features separately in /features/ic/manual_features"
    )
    store_cnn_separate: bool = Field(
        default=True,
        description="Store CNN features separately in /features/ic/cnn_features"
    )
    store_combined: bool = Field(
        default=True,
        description="Store concatenated features in /features/ic/combined_features"
    )

    # Tokenization (future)
    tokenization_enabled: bool = Field(
        default=False,
        description="Enable VQ-VAE tokenization (Phase 8, not yet implemented)"
    )
    num_tokens_per_category: int = Field(
        default=128,
        ge=32,
        le=512,
        description="Codebook size per category for VQ-VAE"
    )

    def estimate_feature_count(self) -> int:
        """Estimate total number of features that will be extracted."""
        count = 0

        if self.manual.enabled:
            # Count enabled manual features
            manual_features = [
                self.manual.include_spatial_cluster_count,
                self.manual.include_spatial_largest_cluster_frac,
                self.manual.include_spatial_autocorr,
                self.manual.include_spatial_centroid_dist,
                self.manual.include_spectral_dominant_freq,
                self.manual.include_spectral_centroid,
                self.manual.include_spectral_power_law_exp,
                self.manual.include_info_entropy,
                self.manual.include_info_local_entropy_var,
                self.manual.include_info_lz_complexity,
                self.manual.include_info_predictability,
                self.manual.include_morph_density,
                self.manual.include_morph_radial_gradient,
                self.manual.include_morph_symmetry,
            ]
            count += sum(manual_features)

        if self.cnn.enabled:
            count += self.cnn.embedding_dim

        return count
