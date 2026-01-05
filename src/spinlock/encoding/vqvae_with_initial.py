"""VQ-VAE wrapper with end-to-end INITIAL CNN training.

Wraps CategoricalHierarchicalVQVAE to support hybrid INITIAL encoding where:
- Manual features (14D) are pre-extracted
- CNN features (28D) are learned end-to-end from raw ICs

During training:
1. Raw ICs + manual features are passed in
2. InitialHybridEncoder encodes them to embeddings
3. Combined with other pre-encoded features
4. Fed to VQ-VAE for tokenization
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from .categorical_vqvae import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig
from .encoders.initial_hybrid import InitialHybridEncoder


class VQVAEWithInitial(nn.Module):
    """VQ-VAE with end-to-end INITIAL CNN training.

    This wrapper enables joint training of:
    - InitialHybridEncoder: CNN learned from raw ICs + manual features pass-through
    - CategoricalHierarchicalVQVAE: Feature tokenization

    The CNN portion of InitialHybridEncoder is trained via backprop from VQ-VAE losses.

    Args:
        vqvae_config: Configuration for the underlying VQ-VAE
        initial_manual_dim: Dimension of pre-extracted manual features (default: 14)
        initial_cnn_dim: Output dimension of CNN encoder (default: 28)
        initial_feature_offset: Starting index of INITIAL features in concatenated input
        initial_feature_count: Number of INITIAL features expected (manual_dim)
        in_channels: Number of channels in raw ICs (default: 1)

    Example:
        >>> model = VQVAEWithInitial(
        ...     vqvae_config=config,
        ...     initial_manual_dim=14,
        ...     initial_cnn_dim=28,
        ...     initial_feature_offset=0,
        ...     initial_feature_count=14,
        ... )
        >>> # During training:
        >>> features = torch.randn(32, 268)  # Other features + 14D manual INITIAL
        >>> raw_ics = torch.randn(32, 1, 128, 128)  # Raw initial conditions
        >>> outputs = model(features, raw_ics=raw_ics)
    """

    def __init__(
        self,
        vqvae_config: CategoricalVQVAEConfig,
        initial_manual_dim: int = 14,
        initial_cnn_dim: int = 28,
        initial_feature_offset: int = 0,
        initial_feature_count: int = 14,
        in_channels: int = 1,
    ):
        super().__init__()

        self.initial_manual_dim = initial_manual_dim
        self.initial_cnn_dim = initial_cnn_dim
        self.initial_feature_offset = initial_feature_offset
        self.initial_feature_count = initial_feature_count

        # Hybrid encoder for INITIAL features (trainable CNN)
        self.initial_encoder = InitialHybridEncoder(
            manual_dim=initial_manual_dim,
            cnn_embedding_dim=initial_cnn_dim,
            encode_manual=False,  # Pass through manual features
            in_channels=in_channels,
        )

        # Adjust VQ-VAE input dimension to account for CNN features
        # Original: features with manual_dim for INITIAL
        # New: features with (manual_dim + cnn_dim) for INITIAL
        adjusted_input_dim = (
            vqvae_config.input_dim
            - initial_feature_count  # Remove manual-only INITIAL
            + self.initial_encoder.output_dim  # Add hybrid INITIAL
        )

        # Update config with adjusted input dimension
        adjusted_config = CategoricalVQVAEConfig(
            input_dim=adjusted_input_dim,
            group_indices=self._adjust_group_indices(
                vqvae_config.group_indices,
                initial_feature_offset,
                initial_feature_count,
                self.initial_encoder.output_dim,
            ),
            group_embedding_dim=vqvae_config.group_embedding_dim,
            group_hidden_dim=vqvae_config.group_hidden_dim,
            levels=vqvae_config.levels,
            commitment_cost=vqvae_config.commitment_cost,
            orthogonality_weight=vqvae_config.orthogonality_weight,
            informativeness_weight=vqvae_config.informativeness_weight,
            use_ema=vqvae_config.use_ema,
            decay=vqvae_config.decay,
            dropout=vqvae_config.dropout,
            compression_ratios=vqvae_config.compression_ratios,
        )

        # Underlying VQ-VAE
        self.vqvae = CategoricalHierarchicalVQVAE(adjusted_config)

    def _adjust_group_indices(
        self,
        original_indices: Dict[str, list],
        initial_offset: int,
        initial_count: int,
        new_initial_count: int,
    ) -> Dict[str, list]:
        """Adjust group indices to account for expanded INITIAL features.

        The INITIAL features are expanded from manual_dim to (manual_dim + cnn_dim).
        All feature indices after INITIAL need to be shifted.
        """
        adjusted = {}
        shift = new_initial_count - initial_count

        for cat, indices in original_indices.items():
            new_indices = []
            for idx in indices:
                if idx < initial_offset:
                    # Before INITIAL: unchanged
                    new_indices.append(idx)
                elif idx < initial_offset + initial_count:
                    # Within original INITIAL: map to expanded range
                    # Keep manual features at same relative position
                    new_indices.append(idx)
                else:
                    # After INITIAL: shift by expansion amount
                    new_indices.append(idx + shift)
            adjusted[cat] = new_indices

        # Add new CNN feature indices to INITIAL category
        # Find which category contains INITIAL features
        for cat, indices in adjusted.items():
            if any(initial_offset <= idx < initial_offset + initial_count for idx in indices):
                # Add CNN feature indices (after manual features)
                cnn_indices = list(range(
                    initial_offset + self.initial_manual_dim,
                    initial_offset + new_initial_count
                ))
                adjusted[cat] = sorted(set(indices) | set(cnn_indices))
                break

        return adjusted

    def forward(
        self,
        features: torch.Tensor,
        raw_ics: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass with optional raw IC encoding.

        Args:
            features: Pre-encoded features [batch_size, feature_dim]
                      Contains manual INITIAL features at initial_feature_offset
            raw_ics: Raw initial conditions [batch_size, in_channels, H, W]
                     If None, assumes INITIAL CNN features are already in features

        Returns:
            VQ-VAE output dictionary with reconstructions, tokens, losses, etc.
        """
        if raw_ics is not None:
            # Extract manual INITIAL features
            manual_start = self.initial_feature_offset
            manual_end = manual_start + self.initial_manual_dim
            manual_features = features[:, manual_start:manual_end]

            # Encode with hybrid encoder (CNN is trainable)
            initial_embeddings = self.initial_encoder(manual_features, raw_ics)

            # Replace manual-only INITIAL with hybrid embeddings
            features_before = features[:, :manual_start]
            features_after = features[:, manual_end:]

            # Concatenate: [before_initial | hybrid_initial | after_initial]
            combined_features = torch.cat([
                features_before,
                initial_embeddings,
                features_after,
            ], dim=1)
        else:
            combined_features = features

        # Forward through VQ-VAE
        outputs = self.vqvae(combined_features)

        # Include combined_features as the reconstruction target
        # This is needed because the hybrid encoder expands features from
        # original_dim to original_dim + cnn_dim, so the reconstruction
        # target must match the expanded dimension.
        outputs["input_features"] = combined_features

        return outputs

    def get_tokens(self, features: torch.Tensor, raw_ics: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get discrete tokens for input features.

        Args:
            features: Input features [batch_size, feature_dim]
            raw_ics: Raw initial conditions (optional)

        Returns:
            Token indices [batch_size, num_categories * num_levels]
        """
        outputs = self.forward(features, raw_ics)
        return outputs["tokens"]

    @property
    def config(self) -> CategoricalVQVAEConfig:
        """Get underlying VQ-VAE config."""
        return self.vqvae.config

    @property
    def quantizers(self):
        """Delegate to underlying VQ-VAE quantizers (for loss computation)."""
        return self.vqvae.quantizers

    @property
    def encoders(self):
        """Delegate to underlying VQ-VAE encoders."""
        return self.vqvae.encoders

    @property
    def decoders(self):
        """Delegate to underlying VQ-VAE decoders."""
        return self.vqvae.decoders

    @property
    def category_names(self):
        """Delegate to underlying VQ-VAE category names."""
        return self.vqvae.category_names

    def reset_dead_codes(self, training_batch, threshold, raw_ics=None):
        """Reset dead codes in hybrid VQ-VAE model.

        For hybrid models, we need raw_ics to produce CNN embeddings before
        delegating to the underlying VQ-VAE's reset_dead_codes.

        Args:
            training_batch: Training features [batch, original_input_dim]
            threshold: Percentile threshold for dead code detection
            raw_ics: Raw initial conditions [batch, C, H, W] (required for hybrid)

        Returns:
            Number of codes reset
        """
        if raw_ics is None:
            # Can't do proper reset without raw_ics for CNN encoding
            return 0

        # Extract manual INITIAL features
        manual_start = self.initial_feature_offset
        manual_end = manual_start + self.initial_manual_dim
        manual_features = training_batch[:, manual_start:manual_end]

        # Encode with hybrid encoder (CNN needs raw_ics)
        with torch.no_grad():
            initial_embeddings = self.initial_encoder(manual_features, raw_ics)

        # Expand features: [before | hybrid_initial | after]
        features_before = training_batch[:, :manual_start]
        features_after = training_batch[:, manual_end:]
        combined_features = torch.cat([
            features_before,
            initial_embeddings,
            features_after,
        ], dim=1)

        # Delegate to underlying VQ-VAE
        return self.vqvae.reset_dead_codes(combined_features, threshold)
