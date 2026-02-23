"""VQ Coherence Adapter — connects MNO trajectories to frozen VQTokenizer.

Replaces the broken VQVAEAlignmentLoss (which depends on deleted
CategoricalHierarchicalVQVAE) with the modern VQTokenizer/JointHierarchicalVQVAE
architecture.

Responsibilities:
1. Load frozen VQTokenizer from checkpoint
2. Extract temporal features from MNO trajectories on-the-fly
3. Apply feature cleaning mask (kept_feature_indices from checkpoint)
4. Apply normalization stats from checkpoint
5. Run VQ forward pass with available families (zero-pad unavailable ones)
6. Provide cleaned features for contrastive loss (same feature space as tokenizer)

Architecture:
    MNO trajectory [B, T, C, H, W]
        → MNOFeatureExtractor.extract() → raw features [B, T, D_raw]
        → kept_feature_indices mask → cleaned features [B, T, D_clean]
        → normalization → normalized features [B, T, D_clean]
        → VQ forward (encode available families, zero-pad unavailable)
        → VQ losses (L_recon, L_commit) + token_indices (diversity metric)

Note: The VQ model's group_indices are cross-family — each quantizer group
can index features from any family's encoded portion. This means we cannot
simply skip families. Instead, unavailable families (e.g., initial) are
zero-padded in the encoded space.

Gradient flow: All loss paths flow through feature extraction → trajectory → MNO.
The VQ model is frozen (requires_grad=False) but passes input gradients via STE.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _GradientSanitizer(torch.autograd.Function):
    """Replace NaN/Inf in backward gradient with zero.

    Used as a gradient barrier between feature extractors (which can produce
    NaN Jacobians from volatile ops like kurtosis x⁴/σ⁴ and Lyapunov log)
    and the MNO trajectory tensor (which must receive clean gradients).

    Forward: identity (no-op).
    Backward: nan_to_num on incoming gradient.
    """

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return torch.nan_to_num(grad_output, nan=0.0, posinf=0.0, neginf=0.0)


class VQCoherenceAdapter(nn.Module):
    """Connects MNO trajectories to frozen VQTokenizer for coherence losses.

    The adapter owns a frozen VQTokenizer and an MNOFeatureExtractor. Given
    an MNO-predicted trajectory, it:
    1. Extracts temporal features (same pipeline as dataset generation)
    2. Applies the checkpoint's feature cleaning mask
    3. Applies the checkpoint's normalization statistics
    4. Runs VQ forward with zero-padded unavailable families
    5. Exposes cleaned features for contrastive loss computation

    All VQ model parameters are frozen — only input gradients propagate
    back to the MNO.

    Attributes:
        model: Frozen JointHierarchicalVQVAE
        extractor: MNOFeatureExtractor for temporal feature extraction
        kept_feature_indices: Feature cleaning mask from checkpoint
        normalization_stats: Per-category normalization statistics
        cleaned_feature_dim: Dimension of cleaned temporal features (D_clean)
    """

    def __init__(
        self,
        model: nn.Module,
        extractor,
        kept_feature_indices: list,
        normalization_stats: Optional[Dict[str, Any]],
        group_indices: Dict[str, list],
        config: Any,
        feature_prefix_len: Optional[int] = None,
    ):
        """Initialize adapter with pre-loaded components.

        Use `from_checkpoint()` classmethod for standard construction.

        Args:
            model: Frozen JointHierarchicalVQVAE
            extractor: MNOFeatureExtractor for temporal feature extraction
            kept_feature_indices: Feature indices kept after cleaning
            normalization_stats: Per-category normalization statistics (or None)
            group_indices: VQ category → feature index mapping
            config: TokenizerConfig from checkpoint
            feature_prefix_len: If not None, slice raw temporal features to
                [:, :, :feature_prefix_len] before applying kept_feature_indices.
                Used when the extractor was extended after tokenizer training
                (new sub-extractors appended, original features unchanged).
        """
        super().__init__()

        self.model = model
        self.extractor = extractor
        self._kept_feature_indices = kept_feature_indices
        self._normalization_stats = normalization_stats
        self._group_indices = group_indices
        self._config = config
        self._feature_prefix_len = feature_prefix_len  # None = use full raw dim

        # Register kept_feature_indices as buffer for device tracking
        self.register_buffer(
            'kept_indices',
            torch.tensor(kept_feature_indices, dtype=torch.long),
        )

        # Pre-compute normalization tensors for temporal features
        self._norm_means = {}
        self._norm_stds = {}
        if normalization_stats is not None:
            for cat_name, stats in normalization_stats.items():
                if cat_name.startswith('temporal_'):
                    mean = stats.mean if isinstance(stats.mean, torch.Tensor) else torch.tensor(stats.mean, dtype=torch.float32)
                    std = stats.std if isinstance(stats.std, torch.Tensor) else torch.tensor(stats.std, dtype=torch.float32)
                    self.register_buffer(f'norm_mean_{cat_name}', mean)
                    self.register_buffer(f'norm_std_{cat_name}', std)
                    self._norm_means[cat_name] = f'norm_mean_{cat_name}'
                    self._norm_stds[cat_name] = f'norm_std_{cat_name}'

        # Build temporal category → cleaned-feature-index mapping
        self._temporal_cat_cleaned_indices = {}
        kept_set = set(kept_feature_indices)
        orig_to_cleaned = {orig: i for i, orig in enumerate(kept_feature_indices)}
        for cat_name, orig_indices in group_indices.items():
            if cat_name.startswith('temporal_'):
                cleaned = [orig_to_cleaned[idx] for idx in orig_indices if idx in kept_set]
                if cleaned:
                    self._temporal_cat_cleaned_indices[cat_name] = cleaned

        # Pre-compute family offset and dim in the concatenated encoded vector
        # Families are sorted alphabetically in forward()
        self._family_offsets = {}
        self._family_dims = {}
        offset = 0
        for family in sorted(model.families):
            if family == 'temporal':
                dim = model.temporal_dim
            elif family == 'initial':
                dim = model.initial_dim
            elif family == 'theta':
                dim = model.theta_dim
            else:
                dim = 0
            self._family_offsets[family] = offset
            self._family_dims[family] = dim
            offset += dim

        self._total_encoded_dim = offset

        # Detect learned CNN mode from model
        self._learned_mode = (
            hasattr(model, 'temporal_cnn_encoder')
            and model.temporal_cnn_encoder is not None
        )
        if self._learned_mode:
            logger.info("Learned CNN temporal mode detected — using frozen CNN for feature extraction")

        # Detect pyramid-first mode (spatio-temporal pyramid → per-group embeddings)
        self._pyramid_first_mode = (
            hasattr(model, 'pyramid_first_encoder')
            and model.pyramid_first_encoder is not None
        )
        if self._pyramid_first_mode:
            logger.info("Pyramid-first temporal mode detected — bypassing extract_features()")

        logger.info(
            f"Encoded space: {self._total_encoded_dim}D "
            f"({', '.join(f'{f}={self._family_dims[f]}' for f in sorted(model.families))})"
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: str = "cuda",
    ) -> "VQCoherenceAdapter":
        """Load from VQTokenizer checkpoint file or directory.

        Loads the model, feature_metadata (kept_indices), normalization stats,
        and group_indices. Validates feature dimensions at construction time.

        Args:
            checkpoint_path: Path to VQTokenizer checkpoint (.pt file or directory)
            device: Device to load model onto

        Returns:
            Initialized VQCoherenceAdapter with frozen VQ model

        Raises:
            ValueError: If checkpoint missing required metadata
            ValueError: If feature dimensions don't match extractor output
        """
        from spinlock.tokens.tokenizer import VQTokenizer

        # Resolve directory → file (checkpoint dirs contain best_model.pt)
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            best_model = checkpoint_path / "best_model.pt"
            if best_model.exists():
                checkpoint_path = best_model
            else:
                pt_files = list(checkpoint_path.glob("*.pt"))
                if pt_files:
                    checkpoint_path = pt_files[0]
                else:
                    raise FileNotFoundError(
                        f"No .pt checkpoint file found in {checkpoint_path}"
                    )

        logger.info(f"Loading VQTokenizer from: {checkpoint_path}")
        tokenizer = VQTokenizer.from_checkpoint(checkpoint_path)

        # Extract components
        model = tokenizer.model
        if model is None:
            raise ValueError(f"Checkpoint at {checkpoint_path} has no model")

        # Freeze VQ model
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        model = model.to(device)

        # Detect learned CNN mode
        is_learned = (
            hasattr(model, 'temporal_cnn_encoder')
            and model.temporal_cnn_encoder is not None
        )

        if is_learned:
            # Learned mode: CNN is the feature extractor — no MNOFeatureExtractor needed
            logger.info(
                "Learned CNN mode detected — bypassing MNOFeatureExtractor. "
                "CNN provides dense spatial gradients directly."
            )
            # Use empty kept_feature_indices (entire CNN output is used)
            learned_cfg = tokenizer.config.encoder.temporal.learned
            kept_feature_indices = list(range(learned_cfg.embedding_dim))

            adapter = cls(
                model=model,
                extractor=None,  # Not used in learned mode
                kept_feature_indices=kept_feature_indices,
                normalization_stats=tokenizer.normalization_stats,
                group_indices=tokenizer.group_indices,
                config=tokenizer.config,
                feature_prefix_len=None,
            )
            return adapter

        # Detect pyramid-first mode
        is_pyramid_first = (
            hasattr(model, 'pyramid_first_encoder')
            and model.pyramid_first_encoder is not None
        )

        if is_pyramid_first:
            logger.info(
                "Pyramid-first mode detected — bypassing MNOFeatureExtractor. "
                "Raw trajectory is encoded directly by spatio-temporal pyramid."
            )
            pf_encoder = model.pyramid_first_encoder
            kept_feature_indices = list(range(pf_encoder.d_group))

            adapter = cls(
                model=model,
                extractor=None,  # Not used in pyramid-first mode
                kept_feature_indices=kept_feature_indices,
                normalization_stats=tokenizer.normalization_stats,
                group_indices=tokenizer.group_indices,
                config=tokenizer.config,
                feature_prefix_len=None,
            )
            return adapter

        # ── Manual mode: standard feature extractor setup ──
        feature_metadata = tokenizer.feature_metadata
        if feature_metadata is None:
            raise ValueError(
                f"Checkpoint at {checkpoint_path} has no feature_metadata. "
                f"Retrain VQTokenizer to generate v2.1+ checkpoint."
            )

        # Get kept_feature_indices for temporal family
        if 'temporal' not in feature_metadata.families:
            raise ValueError("Checkpoint has no temporal family metadata")

        temporal_meta = feature_metadata.families['temporal']
        kept_feature_indices = temporal_meta.kept_feature_indices

        logger.info(
            f"Temporal features: {temporal_meta.original_feature_count} original → "
            f"{temporal_meta.cleaned_feature_count} cleaned"
        )

        # Create feature extractor with differentiable=True so gradient
        # flows from contrastive loss through features back to the MNO
        from spinlock.mno.feature_extraction import MNOFeatureExtractor
        extractor = MNOFeatureExtractor(device=device, differentiable=True)

        # Auto-detect channel count by probing until we match the tokenizer's
        # expected feature dimension — no hardcoded channel counts.
        #
        # Pass 1: exact match (extractor unchanged since training)
        expected_raw_dim = temporal_meta.original_feature_count
        detected_channels = None
        feature_prefix_len = None

        for test_channels in range(1, 9):
            probe = extractor.probe_dimensions(
                timesteps=32, channels=test_channels, height=64, width=64
            )
            if probe['temporal_dim'] == expected_raw_dim:
                detected_channels = test_channels
                break

        # Pass 2: extended-extractor fallback (new sub-extractors appended after training)
        if detected_channels is None:
            best = None
            for test_channels in range(1, 9):
                actual = extractor.probe_dimensions(
                    timesteps=32, channels=test_channels, height=64, width=64
                )['temporal_dim']
                if actual >= expected_raw_dim:
                    excess = actual - expected_raw_dim
                    if best is None or excess < best[0]:
                        best = (excess, test_channels, actual)

            if best is not None:
                _, detected_channels, actual_raw_dim = best
                feature_prefix_len = expected_raw_dim
                logger.warning(
                    f"Feature extractor extended since tokenizer training: "
                    f"{actual_raw_dim} features produced for {detected_channels} channels, "
                    f"tokenizer expects {expected_raw_dim}. Using first {expected_raw_dim} "
                    f"features (assumes new features appended, not inserted). "
                    f"Retrain VQTokenizer to remove this assumption."
                )

        if detected_channels is None:
            raise ValueError(
                f"VQTokenizer checkpoint expects {expected_raw_dim} temporal features "
                f"but no channel count in [1..8] produces >= {expected_raw_dim} features "
                f"with the current extractor. Cannot use this checkpoint."
            )

        logger.info(
            f"Feature dimension auto-detected: channels={detected_channels}, "
            f"{expected_raw_dim} raw → {len(kept_feature_indices)} cleaned"
            + (f" (prefix_len={feature_prefix_len})" if feature_prefix_len else "")
        )

        adapter = cls(
            model=model,
            extractor=extractor,
            kept_feature_indices=kept_feature_indices,
            normalization_stats=tokenizer.normalization_stats,
            group_indices=tokenizer.group_indices,
            config=tokenizer.config,
            feature_prefix_len=feature_prefix_len,
        )

        return adapter

    def extract_and_encode(
        self,
        trajectory: torch.Tensor,
        ic: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Full pipeline: trajectory → features → VQ forward pass.

        Encodes available families (temporal, theta) normally and zero-pads
        unavailable families (initial) in the concatenated encoded space.
        The VQ model's cross-family group indices index into this combined
        vector, so all quantizers can operate normally.

        Args:
            trajectory: MNO output [B, T, C, H, W]
            ic: Initial condition [B, C, H, W] (unused — initial family zero-padded)
            params: QBM parameters [B, param_dim] (for theta family)

        Returns:
            Dict with:
                'cleaned_features': [B, T, D_clean] — for contrastive loss
                'reconstructed': [B, total_encoded_dim]
                'vq_loss': scalar — commitment loss
                'recon_loss': scalar — MSE(reconstructed, original_encoded)
                'token_indices': Dict[str, [B]] — per-quantizer indices
                'latents': Dict[str, [B, latent_dim]] — pre-quantization
                'quantized': [B, total_latent_dim]
        """
        # Convenience wrapper: calls extract_features() then encode_and_quantize()
        cleaned = self.extract_features(trajectory)

        vq_output = self.encode_and_quantize(cleaned, params=params)

        return {
            'cleaned_features': cleaned,  # Un-normalized for contrastive
            'reconstructed': vq_output['reconstructed'],
            'vq_loss': vq_output['vq_loss'],
            'recon_loss': vq_output['recon_loss'],
            'token_indices': vq_output['token_indices'],
            'latents': vq_output.get('latents', {}),
            'quantized': vq_output['quantized'],
        }

    def extract_features(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """Extract cleaned temporal features WITH gradient to trajectory.

        This is the first half of the extract_and_encode pipeline, split out
        so contrastive loss can backprop through feature extraction → MNO
        while VQ encoding runs separately without gradient.

        In learned CNN mode, the frozen CNN provides dense per-frame features
        with spatially-specific gradients. In manual mode, the hand-crafted
        MNOFeatureExtractor is used with gradient sanitization.

        Args:
            trajectory: MNO output [B, T, C, H, W] (should already be sanitized)

        Returns:
            cleaned_features: [B, T, D_clean] with gradient to trajectory
        """
        # Learned CNN mode: use frozen CNN directly
        if self._learned_mode:
            # Sanitize input
            nan_mask = torch.isnan(trajectory) | torch.isinf(trajectory)
            if nan_mask.any():
                trajectory = torch.nan_to_num(
                    trajectory, nan=0.0, posinf=1e6, neginf=-1e6
                )
            # CNN: [B, T, C, H, W] → [B, T, D_per_frame]
            return self.model.temporal_cnn_encoder(trajectory)

        # ── Manual mode: hand-crafted feature extraction ──
        # 1. Sanitize — same logic as extract_and_encode
        nan_mask = torch.isnan(trajectory) | torch.isinf(trajectory)
        if nan_mask.any():
            nan_frac = nan_mask.float().mean().item()
            if nan_frac > 0.5:
                logger.warning(
                    f"VQCoherenceAdapter.extract_features: {nan_frac:.1%} of trajectory "
                    f"values are NaN/Inf — MNO may be diverging"
                )
            trajectory = torch.nan_to_num(
                trajectory, nan=0.0, posinf=1e6, neginf=-1e6
            )

        # 2. Gradient barrier: wrap trajectory so NaN gradients produced by
        # volatile feature extractors (kurtosis, Lyapunov) are sanitized to
        # zero before reaching the MNO parameters.
        if trajectory.requires_grad:
            trajectory = _GradientSanitizer.apply(trajectory)

        # 3. Extract temporal features
        feat_output = self.extractor.extract(trajectory)
        raw_temporal = feat_output['temporal']  # [B, T, D_raw_current]

        # Prefix-slice if extractor was extended after tokenizer was trained
        # (new sub-extractors appended — original features unchanged at [0:N])
        if self._feature_prefix_len is not None:
            raw_temporal = raw_temporal[:, :, :self._feature_prefix_len]

        # Sanitize extracted features (FFT, skewness/kurtosis can produce NaN)
        raw_temporal = torch.nan_to_num(raw_temporal, nan=0.0, posinf=1e6, neginf=-1e6)

        # 4. Apply feature cleaning mask
        cleaned = raw_temporal[:, :, self.kept_indices]  # [B, T, D_clean]

        # 5. Clamp extreme values (tight bound for gradient stability)
        cleaned = torch.clamp(cleaned, min=-100, max=100)

        return cleaned

    def clean_raw_features(
        self,
        raw_features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cleaning pipeline to pre-extracted raw temporal features from HDF5.

        Equivalent to the cleaning steps in extract_features() but skips the
        actual feature extraction (extractor.extract() call). Use this when GT
        raw features are loaded from dataset HDF5 ('features/temporal/features').

        Args:
            raw_features: [B, T, D_raw] raw temporal features (pre-extracted,
                e.g. loaded from 'features/temporal/features' in HDF5)

        Returns:
            cleaned: [B, T, D_clean] cleaned features in the same space as
                extract_features() output — ready for MSE comparison.
        """
        # Prefix-slice if extractor was extended after tokenizer was trained
        if self._feature_prefix_len is not None:
            raw_features = raw_features[:, :, :self._feature_prefix_len]

        # Sanitize
        raw_features = torch.nan_to_num(raw_features, nan=0.0, posinf=1e6, neginf=-1e6)

        # Apply feature cleaning mask
        cleaned = raw_features[:, :, self.kept_indices]  # [B, T, D_clean]

        # Clamp extreme values (match extract_features bound)
        cleaned = torch.clamp(cleaned, min=-100, max=100)

        return cleaned

    def encode_trajectory(
        self,
        trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Extract cleaned features and compute soft VQ logits for a trajectory slice.

        Factored entry point for multi-scale callers: given a trajectory of any
        temporal length T (the pyramid encoder handles variable T), returns both
        the cleaned features (with gradient to the MNO) and the per-quantizer
        soft logits (differentiable through the frozen encoder via STE).

        Args:
            trajectory: [B, T_any, C, H, W] — any temporal length accepted

        Returns:
            cleaned_features: [B, T_any, D_clean] — gradients flow through
            soft_logits:       Dict[base_key → [B, V]] — keys without trunc suffix
        """
        cleaned_features = self.extract_features(trajectory)
        vq_out = self.extract_soft_logits_and_hard_tokens(cleaned_features)
        return cleaned_features, vq_out['soft_logits']

    def encode_and_quantize(
        self,
        cleaned_features: torch.Tensor,
        params: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run VQ pipeline on pre-extracted features.

        This is the second half of the extract_and_encode pipeline. Takes
        already-cleaned features and runs normalization → VQ encoding →
        quantization → decoding.

        Can be called with gradient-carrying features (for commit loss training)
        or with detached features (for monitoring only). The VQ model is frozen
        but passes input gradients through its MLPs.

        Args:
            cleaned_features: [B, T, D_clean] — pre-extracted temporal features
            params: QBM parameters [B, param_dim] (for theta family)

        Returns:
            Dict with recon_loss, vq_loss, token_indices, reconstructed, etc.
        """
        # Normalize (match training-time normalization)
        normalized = self._normalize_temporal(cleaned_features)

        # Run VQ forward with zero-padded unavailable families
        vq_output = self._forward_with_padding(normalized, params)

        return {
            'recon_loss': vq_output['recon_loss'],
            'vq_loss': vq_output['vq_loss'],
            'token_indices': vq_output['token_indices'],
            'reconstructed': vq_output['reconstructed'],
            'latents': vq_output.get('latents', {}),
            'quantized': vq_output['quantized'],
        }

    def extract_soft_logits_and_hard_tokens(
        self,
        cleaned_features: torch.Tensor,
        params: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Extract per-quantizer soft logits (differentiable) and hard token indices.

        For each quantizer, computes squared L2 distances from the projected
        latent to all codebook entries. Returns:
        - soft_logits: -distances/temperature (differentiable through encoder)
        - hard_tokens: argmin(distances) per quantizer (non-differentiable)

        This is the core of the roundtrip consistency loss gradient path:
        features → encoder → projector → distances → soft_logits → CrossEntropy

        Args:
            cleaned_features: [B, T, D_clean] pre-extracted temporal features
            params: [B, param_dim] QBM parameters (for theta family)
            temperature: Softmax temperature for soft logits

        Returns:
            Dict with:
                'soft_logits': Dict[str, [B, K]] per-quantizer log-probabilities
                'hard_tokens': Dict[str, [B]] per-quantizer hard indices
        """
        # Normalize and encode (same path as encode_and_quantize)
        normalized = self._normalize_temporal(cleaned_features)

        model = self.model
        batch_size = normalized.shape[0]
        device = normalized.device

        # Build all_encoded (for decoder) AND per-family dict (for projectors)
        all_encoded, family_encoded = self._build_all_encoded(
            model, normalized, params, batch_size, device,
        )

        # Per-quantizer: project → compute distances → soft logits + hard tokens
        soft_logits = {}
        hard_tokens = {}

        for family_cat, indices in model.group_indices.items():
            family, _ = family_cat.split('_', 1)
            # Per-group paths store encodings keyed by family_cat (preferred).
            # Legacy path stores only a family-level key; slice by indices instead.
            if family_cat in family_encoded:
                cat_features = family_encoded[family_cat]
            else:
                cat_features = all_encoded[:, indices]
            projector = model.projectors[family_cat]
            latents = projector(cat_features)

            num_levels = model.config.hierarchy.num_levels
            for level_idx, latent in enumerate(latents):
                quantizer_key = f"{family_cat}_L{level_idx}"
                quantizer = model.quantizers[quantizer_key]
                codebook = quantizer.embedding.weight  # [K, D]

                # Squared L2 distances: [B, K]
                distances = torch.cdist(
                    latent.unsqueeze(1), codebook.unsqueeze(0), p=2.0
                ).squeeze(1).pow(2)

                soft_logits[quantizer_key] = -distances / temperature
                hard_tokens[quantizer_key] = distances.argmin(dim=1)

        return {
            'soft_logits': soft_logits,
            'hard_tokens': hard_tokens,
        }

    def extract_soft_logits_from_trajectory(
        self,
        trajectory: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Unified pipeline: raw trajectory → soft logits + hard tokens.

        Handles all encoder modes automatically:
        - Pyramid-first: runs encoder directly on raw trajectory [B, T, C, H, W],
          then per-group projectors and quantizers. Bypasses extract_features().
        - Other modes: delegates to extract_features() → extract_soft_logits_and_hard_tokens().

        Args:
            trajectory: Raw trajectory [B, T, C, H, W].
            temperature: Softmax temperature for soft logits.

        Returns:
            Dict with 'soft_logits' and 'hard_tokens' dicts keyed by quantizer key.
        """
        if self._pyramid_first_mode:
            # Sanitize input
            nan_mask = torch.isnan(trajectory) | torch.isinf(trajectory)
            if nan_mask.any():
                trajectory = torch.nan_to_num(
                    trajectory, nan=0.0, posinf=1e6, neginf=-1e6
                )

            # Pyramid-first: [B, T, C, H, W] → [B, G, D_group]
            per_group, _ = self.model.pyramid_first_encoder(trajectory)

            model = self.model
            soft_logits = {}
            hard_tokens = {}
            group_idx = 0

            for family_cat in sorted(model.group_indices.keys()):
                if not family_cat.startswith('temporal_'):
                    continue
                cat_features = per_group[:, group_idx, :]  # [B, D_group]
                projector = model.projectors[family_cat]
                latents = projector(cat_features)

                num_levels = model.config.hierarchy.num_levels
                for level_idx, latent in enumerate(latents):
                    qkey = f"{family_cat}_L{level_idx}"
                    codebook = model.quantizers[qkey].embedding.weight  # [K, D]
                    dists = torch.cdist(
                        latent.unsqueeze(1), codebook.unsqueeze(0), p=2.0
                    ).squeeze(1).pow(2)
                    soft_logits[qkey] = -dists / temperature
                    hard_tokens[qkey] = dists.argmin(dim=1)
                group_idx += 1

            return {'soft_logits': soft_logits, 'hard_tokens': hard_tokens}
        else:
            # Standard path: extract_features → extract_soft_logits_and_hard_tokens
            cleaned = self.extract_features(trajectory)
            return self.extract_soft_logits_and_hard_tokens(
                cleaned, temperature=temperature,
            )

    def get_gate_values(self) -> Optional[torch.Tensor]:
        """Return per-group gate activations [num_groups] in [0,1], or None.

        Only available in pyramid-first mode with gated groups enabled.
        Gate values indicate how much each VQ group contributes to the
        tokenization — useful for weighting per-group losses.
        """
        if self._pyramid_first_mode:
            return self.model.pyramid_first_encoder.group_proj.get_gate_values()
        return None

    @torch.no_grad()
    def decode_tokens_to_params(
        self,
        tokens_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Decode hard token indices to physics parameters via frozen VQ pipeline.

        Looks up codebook embeddings for each token, concatenates, passes through
        the shared decoder, then applies inverse heads (theta_inverse, initial_inverse)
        to recover physics parameters.

        This is used by Mode B of the roundtrip consistency loss: decode MNO's
        predicted tokens → (θ', IC') → re-simulate with QBM.

        Requires inverse heads to be loaded via load_inverse_heads() first.

        Args:
            tokens_dict: Dict mapping quantizer key -> [B] token indices

        Returns:
            Dict with:
                'theta': [B, param_dim] decoded Sobol parameters (if theta_inverse loaded)
                'u0': [B, C, H, W] decoded initial conditions (if initial_inverse loaded)

        Raises:
            RuntimeError: If no inverse heads are loaded
        """
        # Resolve inverse heads: prefer separately-loaded ones, fall back
        # to the model's built-in inverse heads (trained end-to-end in VQTokenizer)
        theta_inv = getattr(self, '_theta_inverse', None) or getattr(self.model, 'theta_inverse', None)
        initial_inv = getattr(self, '_initial_inverse', None) or getattr(self.model, 'initial_inverse', None)

        if theta_inv is None and initial_inv is None:
            raise RuntimeError(
                "No inverse heads available. Either the VQTokenizer checkpoint "
                "was trained without inverse_heads config, or call "
                "load_inverse_heads() to load them separately."
            )

        model = self.model

        # Lookup codebook embeddings
        embeddings = []
        for family_cat, indices in model.group_indices.items():
            num_levels = model.config.hierarchy.num_levels
            for level_idx in range(num_levels):
                quantizer_key = f"{family_cat}_L{level_idx}"
                if quantizer_key not in tokens_dict:
                    continue
                quantizer = model.quantizers[quantizer_key]
                emb = quantizer.embedding(tokens_dict[quantizer_key])  # [B, D]
                embeddings.append(emb)

        latent = torch.cat(embeddings, dim=-1)
        reconstructed = model.decoder(latent)

        # Split by family and apply inverse heads
        result = {}
        for family in sorted(model.families):
            offset = self._family_offsets[family]
            dim = self._family_dims[family]
            if dim == 0:
                continue
            family_recon = reconstructed[:, offset:offset + dim]

            if family == 'theta' and theta_inv is not None:
                result['theta'] = theta_inv(family_recon)
            elif family == 'initial' and initial_inv is not None:
                result['u0'] = initial_inv(family_recon)

        return result

    def load_inverse_heads(
        self,
        theta_inverse_path: Optional[str] = None,
        initial_inverse_path: Optional[str] = None,
    ) -> None:
        """Load trained inverse heads for Mode B token decoding.

        Args:
            theta_inverse_path: Path to theta inverse head checkpoint
            initial_inverse_path: Path to initial condition inverse head checkpoint
        """
        device = next(self.model.parameters()).device

        if theta_inverse_path is not None:
            checkpoint = torch.load(theta_inverse_path, map_location=device, weights_only=False)
            self._theta_inverse = checkpoint['model'] if 'model' in checkpoint else checkpoint
            self._theta_inverse.eval()
            for p in self._theta_inverse.parameters():
                p.requires_grad_(False)
            logger.info(f"Loaded theta inverse head from {theta_inverse_path}")

        if initial_inverse_path is not None:
            checkpoint = torch.load(initial_inverse_path, map_location=device, weights_only=False)
            self._initial_inverse = checkpoint['model'] if 'model' in checkpoint else checkpoint
            self._initial_inverse.eval()
            for p in self._initial_inverse.parameters():
                p.requires_grad_(False)
            logger.info(f"Loaded initial inverse head from {initial_inverse_path}")

    def _forward_with_padding(
        self,
        temporal_features: torch.Tensor,
        params: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Run VQ model forward with zero-padded unavailable families.

        Constructs the full all_encoded vector by:
        - Encoding temporal and theta families normally
        - Zero-padding the initial family's positions
        Then runs the standard quantization and decoding pipeline.

        Args:
            temporal_features: Normalized cleaned temporal [B, T, D_clean]
            params: QBM parameters [B, param_dim] (or None)

        Returns:
            Dict with reconstructed, vq_loss, recon_loss, token_indices, etc.
        """
        model = self.model
        batch_size = temporal_features.shape[0]
        device = temporal_features.device

        # Build all_encoded (for decoder) AND per-family dict (for projectors)
        all_encoded, family_encoded = self._build_all_encoded(
            model, temporal_features, params, batch_size, device,
        )

        # ── Quantize per-category (standard VQ pipeline) ─────────────
        all_quantized = []
        vq_losses = []
        encodings_dict = {}
        latents_dict = {}

        for family_cat, indices in model.group_indices.items():
            family, _ = family_cat.split('_', 1)
            # Per-group paths store encodings keyed by family_cat (preferred).
            # Legacy path stores only a family-level key; slice by indices instead.
            if family_cat in family_encoded:
                cat_features = family_encoded[family_cat]
            else:
                cat_features = all_encoded[:, indices]

            # Project to hierarchical latents
            projector = model.projectors[family_cat]
            latents = projector(cat_features)

            # Quantize each level
            num_levels = model.config.hierarchy.num_levels
            for level_idx, latent in enumerate(latents):
                quantizer_key = f"{family_cat}_L{level_idx}"
                quantizer = model.quantizers[quantizer_key]

                latents_dict[quantizer_key] = latent
                quantized, encodings, losses = quantizer(latent)

                all_quantized.append(quantized)
                vq_losses.append(losses['loss'])
                encodings_dict[quantizer_key] = quantized

        all_quantized_cat = torch.cat(all_quantized, dim=1)

        # ── Decode ───────────────────────────────────────────────────
        reconstructed = model.decoder(all_quantized_cat)

        # ── Losses ───────────────────────────────────────────────────
        total_vq_loss = torch.stack(vq_losses).mean()

        # Reconstruction loss: compare only the available family portions
        # (skip initial's zero-padded region to avoid penalizing zero→zero)
        recon_parts = []
        encoded_parts = []

        for family in ['temporal', 'theta']:
            if family in self._family_offsets:
                offset = self._family_offsets[family]
                dim = self._family_dims[family]
                if dim > 0:
                    recon_parts.append(reconstructed[:, offset:offset + dim])
                    encoded_parts.append(all_encoded[:, offset:offset + dim])

        if recon_parts:
            recon_cat = torch.cat(recon_parts, dim=1)
            encoded_cat = torch.cat(encoded_parts, dim=1)
            recon_loss = F.mse_loss(recon_cat, encoded_cat)
        else:
            recon_loss = torch.tensor(0.0, device=device)

        # Token indices
        token_indices = {}
        for quantizer_key, quantizer in model.quantizers.items():
            if quantizer_key in encodings_dict:
                quantized = encodings_dict[quantizer_key]
                distances = torch.cdist(
                    quantized, quantizer.embedding.weight, p=2.0
                )
                token_indices[quantizer_key] = distances.argmin(dim=1)

        return {
            'reconstructed': reconstructed,
            'vq_loss': total_vq_loss,
            'recon_loss': recon_loss,
            'token_indices': token_indices,
            'latents': latents_dict,
            'quantized': all_quantized_cat,
        }

    def _build_all_encoded(
        self,
        model,
        temporal_features: torch.Tensor,
        params: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> tuple:
        """Build concatenated encoded vector AND per-family/per-group dict.

        Supports three encoder paths matching JointHierarchicalVQVAE.forward():
        - Per-group pyramid (pca_raw + pyramid variant): per-group PyramidTemporalEncoder
        - Per-group MLP (pca_striped / opq, mean variant): mean+std summary + per-group MLP
        - Legacy (correlation grouping, shared encoder): shared temporal encoder

        Returns both the concatenated tensor (for the decoder/reconstruction target)
        and a dict that includes both family-level ('temporal') and group-level
        ('temporal_groupX') entries. Downstream quantization loops should prefer
        the group-level entries so per-group projectors receive the correct input.

        Uses torch.cat (not in-place assignment) to avoid CopySlices autograd issues.

        Families are ordered alphabetically: initial, temporal, theta.
        Missing families are zero-padded.

        Returns:
            Tuple of (all_encoded [B, total_dim], family_encoded {key: [B, dim]})
        """
        parts = []
        family_encoded = {}

        for family in sorted(model.families):
            dim = self._family_dims[family]
            if dim == 0:
                continue

            if family == 'temporal' and 'temporal' in model.families:
                if model.per_group_pyramid_encoders is not None:
                    # pca_raw + pyramid: slice per-group features, run pyramid encoder
                    group_parts = []
                    for family_cat, indices in model.group_indices.items():
                        if not family_cat.startswith('temporal_'):
                            continue
                        idx_t = torch.tensor(indices, device=device, dtype=torch.long)
                        group_temporal = temporal_features[:, :, idx_t]  # [B, T, G_k]
                        enc = model.per_group_pyramid_encoders[family_cat](group_temporal)
                        family_encoded[family_cat] = enc
                        group_parts.append(enc)
                    enc = torch.cat(group_parts, dim=1)
                    family_encoded[family] = enc
                    parts.append(enc)

                elif model.per_group_temporal_encoders is not None:
                    # pca_striped / opq: mean+std summary → rotation → per-group MLP
                    t_mean = temporal_features.mean(dim=1)                  # [B, D_t]
                    t_std  = temporal_features.std(dim=1)                   # [B, D_t]
                    t_summary = torch.cat([t_mean, t_std], dim=1)           # [B, 2*D_t]

                    if model.temporal_rotation_matrix is not None:
                        t_summary = (
                            (t_summary - model.temporal_rotation_mean)
                            @ model.temporal_rotation_matrix.T
                        )

                    group_parts = []
                    for family_cat, indices in model.group_indices.items():
                        if not family_cat.startswith('temporal_'):
                            continue
                        group_feats = t_summary[:, indices]                 # [B, G_k]
                        enc = model.per_group_temporal_encoders[family_cat](group_feats)
                        family_encoded[family_cat] = enc                    # [B, embedding_dim]
                        group_parts.append(enc)
                    enc = torch.cat(group_parts, dim=1)
                    family_encoded[family] = enc
                    parts.append(enc)

                else:
                    # Legacy: shared temporal encoder
                    # all_encoded[:, indices] logic is handled in the quantization loop
                    enc = model.temporal_encoder(temporal_features)
                    family_encoded[family] = enc
                    parts.append(enc)

            elif family == 'theta' and 'theta' in model.families and params is not None:
                enc = model.theta_encoder(params)
                parts.append(enc)
                family_encoded[family] = enc
            else:
                # Zero-fill for unavailable families (e.g., 'initial' at MNO time)
                zeros = torch.zeros(
                    batch_size, dim,
                    device=device, dtype=temporal_features.dtype,
                )
                parts.append(zeros)
                family_encoded[family] = zeros

        return torch.cat(parts, dim=1), family_encoded

    def _normalize_temporal(self, cleaned: torch.Tensor) -> torch.Tensor:
        """Apply checkpoint normalization to cleaned temporal features.

        Replicates the per-category normalization applied during VQTokenizer
        training, using stored normalization stats.

        Args:
            cleaned: Cleaned temporal features [B, T, D_clean]

        Returns:
            Normalized features [B, T, D_clean]
        """
        if not self._norm_means:
            # No normalization stats — pass through
            return cleaned

        normalized = cleaned.clone()

        for cat_name, cleaned_indices in self._temporal_cat_cleaned_indices.items():
            mean_key = self._norm_means.get(cat_name)
            std_key = self._norm_stds.get(cat_name)

            if mean_key is None or std_key is None:
                continue

            mean = getattr(self, mean_key).to(cleaned.device)
            std = getattr(self, std_key).to(cleaned.device)

            idx = torch.tensor(cleaned_indices, device=cleaned.device, dtype=torch.long)
            normalized[:, :, idx] = (cleaned[:, :, idx] - mean) / std

        clip_val = getattr(self._config.normalization, 'clip_std_multiplier', None)
        if clip_val is not None:
            normalized = torch.clamp(normalized, -clip_val, clip_val)

        return normalized

    @property
    def cleaned_feature_dim(self) -> int:
        """D_clean — dimension of cleaned temporal features."""
        return len(self._kept_feature_indices)

    def __repr__(self) -> str:
        return (
            f"VQCoherenceAdapter("
            f"cleaned_dim={self.cleaned_feature_dim}, "
            f"families={sorted(self.model.families)}, "
            f"frozen=True)"
        )
