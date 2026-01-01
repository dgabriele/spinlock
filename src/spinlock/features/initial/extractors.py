"""
Hybrid IC feature extractor.

Combines manual and CNN-based features for comprehensive IC representation.
Builds feature registry and coordinates extraction pipeline.
"""

import torch
from typing import Dict, Tuple, Optional
from .manual_extractors import ICManualExtractor
from .cnn_encoder import InitialCNNEncoder, InitialVAE
from .config import InitialConfig
from ..registry import FeatureRegistry


class InitialExtractor:
    """
    Hybrid IC feature extractor.

    Extracts 42D features per IC (14D manual + 28D CNN):
    - Manual features: Interpretable, domain-driven
    - CNN features: Learned, high representation power

    Output shape: [B, M, 42] for B operators, M realizations each

    Design Philosophy:
    ------------------
    This extractor serves dual purposes:
    1. **Analysis**: Extract features from existing ICs for understanding
    2. **Generation**: When using VAE mode, enables IC construction from embeddings
                      (critical for NOA to construct IC+NO pairs embodying thoughts)

    The hybrid architecture ensures both interpretability (manual features) and
    representational power (learned features), while the optional VAE mode
    enables the generative capability needed for the NOA's "thought→IC" process.
    """

    def __init__(self, config: InitialConfig, device: torch.device):
        """
        Initialize hybrid IC extractor.

        Args:
            config: IC feature extraction configuration
            device: Computation device (cpu or cuda)
        """
        self.config = config
        self.device = device

        # Initialize manual extractor
        self.manual_extractor: Optional[ICManualExtractor] = None
        if config.manual.enabled:
            self.manual_extractor = ICManualExtractor(device=device)

        # Initialize CNN encoder or VAE
        self.cnn_encoder: Optional[torch.nn.Module] = None
        if config.cnn.enabled:
            if config.cnn.use_vae:
                # Generative mode: VAE for bidirectional encoding/decoding
                self.cnn_encoder = InitialVAE(embedding_dim=config.cnn.embedding_dim)
            else:
                # Analysis mode: Encoder only
                self.cnn_encoder = InitialCNNEncoder(embedding_dim=config.cnn.embedding_dim)

            self.cnn_encoder = self.cnn_encoder.to(device)

            # Freeze encoder if specified
            if config.cnn.freeze_encoder:
                for param in self.cnn_encoder.parameters():
                    param.requires_grad = False

        # Build feature registry
        self.registry = self._build_registry()

    def _build_registry(self) -> FeatureRegistry:
        """
        Build IC feature registry.

        Creates registry with metadata for all enabled features.
        """
        registry = FeatureRegistry(family_name="initial")

        # Manual features (14 total when all enabled)
        if self.config.manual.enabled:
            # Spatial (4)
            if self.config.manual.include_spatial_cluster_count:
                registry.register(
                    "ic_spatial_cluster_count",
                    "spatial",
                    description="Number of spatial clusters (log-scaled)"
                )
            if self.config.manual.include_spatial_largest_cluster_frac:
                registry.register(
                    "ic_spatial_largest_cluster_frac",
                    "spatial",
                    description="Fraction of largest cluster to total active area"
                )
            if self.config.manual.include_spatial_autocorr:
                registry.register(
                    "ic_spatial_autocorr",
                    "spatial",
                    description="Spatial autocorrelation (Moran's I)"
                )
            if self.config.manual.include_spatial_centroid_dist:
                registry.register(
                    "ic_spatial_centroid_dist",
                    "spatial",
                    description="Distance of centroid from grid center"
                )

            # Spectral (3)
            if self.config.manual.include_spectral_dominant_freq:
                registry.register(
                    "ic_spectral_dominant_freq",
                    "spectral",
                    description="Dominant frequency in power spectrum"
                )
            if self.config.manual.include_spectral_centroid:
                registry.register(
                    "ic_spectral_centroid",
                    "spectral",
                    description="Spectral centroid (weighted mean frequency)"
                )
            if self.config.manual.include_spectral_power_law_exp:
                registry.register(
                    "ic_spectral_power_law_exp",
                    "spectral",
                    description="Power law exponent (1/f^β behavior)"
                )

            # Information (4)
            if self.config.manual.include_info_entropy:
                registry.register(
                    "ic_info_entropy",
                    "information",
                    description="Shannon entropy (histogram-based)"
                )
            if self.config.manual.include_info_local_entropy_var:
                registry.register(
                    "ic_info_local_entropy_var",
                    "information",
                    description="Variance of local patch entropies"
                )
            if self.config.manual.include_info_lz_complexity:
                registry.register(
                    "ic_info_lz_complexity",
                    "information",
                    description="Approximate Lempel-Ziv complexity"
                )
            if self.config.manual.include_info_predictability:
                registry.register(
                    "ic_info_predictability",
                    "information",
                    description="Predictability via autocorrelation"
                )

            # Morphological (3)
            if self.config.manual.include_morph_density:
                registry.register(
                    "ic_morph_density",
                    "morphological",
                    description="Density (fraction of high-valued pixels)"
                )
            if self.config.manual.include_morph_radial_gradient:
                registry.register(
                    "ic_morph_radial_gradient",
                    "morphological",
                    description="Average gradient magnitude (edge strength)"
                )
            if self.config.manual.include_morph_symmetry:
                registry.register(
                    "ic_morph_symmetry",
                    "morphological",
                    description="4-fold rotational symmetry score"
                )

        # CNN features (embedding_dim total, typically 28)
        if self.config.cnn.enabled:
            for i in range(self.config.cnn.embedding_dim):
                registry.register(
                    f"ic_cnn_{i:02d}",
                    "cnn_learned",
                    description=f"CNN learned embedding dimension {i}"
                )

        return registry

    def extract_all(
        self,
        ics: torch.Tensor,
        return_vae_outputs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all IC features (manual + CNN).

        Args:
            ics: [B, M, C, H, W] initial conditions
            return_vae_outputs: If True and using VAE, return (mu, logvar) as well

        Returns:
            Dictionary with:
                'manual': [B, M, D_manual] manual features (if enabled)
                'cnn': [B, M, D_cnn] CNN features (if enabled)
                'combined': [B, M, D_total] concatenated features

            If return_vae_outputs=True and using VAE, also includes:
                'vae_mu': [B*M, D_cnn] latent means
                'vae_logvar': [B*M, D_cnn] latent log variances
        """
        B, M, C, H, W = ics.shape

        outputs = {}

        # Extract manual features [B, M, D_manual]
        manual_features = None
        if self.manual_extractor is not None:
            manual_features = self.manual_extractor.extract_all(ics)
            outputs['manual'] = manual_features

        # Extract CNN features
        cnn_features = None
        if self.cnn_encoder is not None:
            # Reshape for CNN: [B*M, C, H, W]
            ics_flat = ics.view(B * M, C, H, W)

            # Extract features
            if isinstance(self.cnn_encoder, InitialVAE):
                # VAE mode: Get latent distribution
                mu, logvar = self.cnn_encoder.encode(ics_flat)

                # Sample latent code (or use mean for deterministic extraction)
                z = self.cnn_encoder.reparameterize(mu, logvar) if self.training else mu

                # Store VAE outputs if requested
                if return_vae_outputs:
                    outputs['vae_mu'] = mu
                    outputs['vae_logvar'] = logvar

                cnn_features_flat = z  # [B*M, D_cnn]
            else:
                # Encoder-only mode
                cnn_features_flat = self.cnn_encoder(ics_flat)  # [B*M, D_cnn]

            # Reshape back to [B, M, D_cnn]
            cnn_features = cnn_features_flat.view(B, M, -1)
            outputs['cnn'] = cnn_features

        # Combine features
        combined = None
        if manual_features is not None and cnn_features is not None:
            combined = torch.cat([manual_features, cnn_features], dim=-1)  # [B, M, D_total]
        elif manual_features is not None:
            combined = manual_features
        elif cnn_features is not None:
            combined = cnn_features

        outputs['combined'] = combined

        return outputs

    def generate_ics(
        self,
        latent_codes: torch.Tensor,
        num_realizations: int = 1
    ) -> torch.Tensor:
        """
        Generate ICs from latent codes (VAE mode only).

        This method enables the NOA to construct ICs from thought embeddings.

        Args:
            latent_codes: [N, D_cnn] latent codes to decode
            num_realizations: Number of realizations per latent code

        Returns:
            [N, num_realizations, 1, H, W] generated ICs

        Raises:
            ValueError: If not using VAE mode
        """
        if not isinstance(self.cnn_encoder, InitialVAE):
            raise ValueError(
                "IC generation requires VAE mode (config.cnn.use_vae=True). "
                "Current encoder is not a VAE."
            )

        N, D = latent_codes.shape

        # Replicate latent codes for multiple realizations
        # [N, D] → [N, M, D] → [N*M, D]
        z_expanded = latent_codes.unsqueeze(1).expand(N, num_realizations, D)
        z_flat = z_expanded.reshape(N * num_realizations, D)

        # Decode
        with torch.no_grad():
            ics_flat = self.cnn_encoder.decode(z_flat)  # [N*M, 1, H, W]

        # Reshape to [N, M, 1, H, W]
        _, C, H, W = ics_flat.shape
        ics = ics_flat.view(N, num_realizations, C, H, W)

        return ics

    def sample_ics(
        self,
        num_operators: int,
        num_realizations: int = 1
    ) -> torch.Tensor:
        """
        Sample ICs from prior distribution (VAE mode only).

        Enables the NOA to create novel ICs without conditioning on existing ones.

        Args:
            num_operators: Number of operators to generate ICs for
            num_realizations: Number of realizations per operator

        Returns:
            [num_operators, num_realizations, 1, H, W] sampled ICs

        Raises:
            ValueError: If not using VAE mode
        """
        if not isinstance(self.cnn_encoder, InitialVAE):
            raise ValueError(
                "IC sampling requires VAE mode (config.cnn.use_vae=True). "
                "Current encoder is not a VAE."
            )

        # Sample from prior N(0, I)
        total_samples = num_operators * num_realizations
        ics_flat = self.cnn_encoder.sample(total_samples, self.device)  # [N*M, 1, H, W]

        # Reshape to [N, M, 1, H, W]
        _, C, H, W = ics_flat.shape
        ics = ics_flat.view(num_operators, num_realizations, C, H, W)

        return ics

    def get_feature_registry(self) -> FeatureRegistry:
        """Get IC feature registry."""
        return self.registry

    @property
    def is_generative(self) -> bool:
        """Check if extractor is in generative mode (VAE)."""
        return isinstance(self.cnn_encoder, InitialVAE)

    @property
    def training(self) -> bool:
        """Check if CNN encoder is in training mode."""
        if self.cnn_encoder is not None:
            return self.cnn_encoder.training
        return False

    def train(self, mode: bool = True):
        """Set training mode for CNN encoder."""
        if self.cnn_encoder is not None:
            self.cnn_encoder.train(mode)
        return self

    def eval(self):
        """Set evaluation mode for CNN encoder."""
        return self.train(False)
