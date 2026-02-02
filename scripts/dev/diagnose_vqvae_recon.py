#!/usr/bin/env python
"""VQ-VAE Reconstruction Error Diagnostic Analysis.

Provides comprehensive diagnostic analysis for VQ-VAE reconstruction errors:
1. Per-family reconstruction analysis (INITIAL, SUMMARY, TEMPORAL)
2. Per-dimension error analysis (identify high-error features)
3. Codebook utilization analysis (dead codes, entropy, usage patterns)
4. Latent space visualization (t-SNE embeddings)

Usage:
    poetry run python scripts/dev/diagnose_vqvae_recon.py \
        --checkpoint checkpoints/production/100k_3family_v1 \
        --dataset datasets/100k_full_features.h5 \
        --num-samples 10000 \
        --output-dir diagnostics/vqvae_recon
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score

from spinlock.encoding import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig
from spinlock.noa.vqvae_alignment import AlignedFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VQVAEDiagnostics:
    """Main diagnostic orchestrator for VQ-VAE analysis."""

    def __init__(
        self,
        checkpoint_path: Path,
        dataset_path: Path,
        output_dir: Path,
        num_samples: int = 10000,
        device: str = 'cuda'
    ):
        """Initialize diagnostics.

        Args:
            checkpoint_path: Path to VQ-VAE checkpoint directory
            dataset_path: Path to HDF5 dataset
            output_dir: Output directory for results
            num_samples: Number of samples to analyze
            device: Computation device
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'visualizations' / 'family_reconstruction').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'visualizations' / 'dimension_errors').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'visualizations' / 'codebook').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'visualizations' / 'latent_space').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'raw_data').mkdir(parents=True, exist_ok=True)

        # Load VQ-VAE and feature extractor
        logger.info(f"Loading VQ-VAE from {checkpoint_path}")
        self.vqvae, self.feature_extractor, self.config, self.checkpoint = self._load_vqvae(checkpoint_path)
        self.vqvae.eval()

        # Get family configuration
        # IMPORTANT: Use group_indices from model_config, not checkpoint level
        # The checkpoint-level group_indices may be stale/incorrect
        if 'model_config' in self.checkpoint and 'group_indices' in self.checkpoint['model_config']:
            self.group_indices = self.checkpoint['model_config']['group_indices']
            logger.info("Using group_indices from model_config (authoritative)")
        else:
            self.group_indices = self.config.get('group_indices', {})
            logger.warning("Using group_indices from config (fallback - may not match model)")
        logger.info(f"Loaded VQ-VAE with {len(self.group_indices)} families: {list(self.group_indices.keys())}")

    def _load_vqvae(
        self, checkpoint_path: Path
    ) -> Tuple[nn.Module, Optional[nn.Module], Dict, Dict]:
        """Load VQ-VAE model and feature extractor from checkpoint.

        Returns:
            Tuple of (vqvae, feature_extractor, config_dict, checkpoint)
        """
        # Load checkpoint (config is stored in the checkpoint)
        checkpoint_file = checkpoint_path / 'best_model.pt'
        if not checkpoint_file.exists():
            checkpoint_file = checkpoint_path / 'final_model.pt'

        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)

        # Get config from checkpoint (preferred) or fall back to YAML
        if 'model_config' in checkpoint:
            # Use model_config if available (new format with full metadata)
            checkpoint_config = checkpoint['model_config']
            vqvae_config = CategoricalVQVAEConfig(**checkpoint_config)
            config_dict = {
                'group_indices': checkpoint_config.get('group_indices', {}),
                'model': vqvae_config,
                'families': checkpoint.get('config', {}).get('families', {}) if 'config' in checkpoint else {}
            }
            logger.info("Loaded config from checkpoint (model_config)")
        elif 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            # Config might be a CategoricalVQVAEConfig object or a dict
            if hasattr(checkpoint_config, 'group_indices'):
                # It's an object
                vqvae_config = checkpoint_config
                config_dict = {
                    'group_indices': vqvae_config.group_indices,
                    'model': vqvae_config,
                    'families': {}
                }
            else:
                # It's a dict - extract only model parameters
                model_params = [
                    'input_dim', 'group_indices', 'group_embedding_dim', 'group_hidden_dim',
                    'levels', 'commitment_cost', 'orthogonality_weight', 'informativeness_weight',
                    'use_ema', 'decay', 'dropout', 'compression_ratios', 'uniform_codebook_init'
                ]
                model_config_dict = {k: v for k, v in checkpoint_config.items() if k in model_params}
                vqvae_config = CategoricalVQVAEConfig(**model_config_dict)
                config_dict = {
                    'group_indices': checkpoint_config.get('group_indices', {}),
                    'model': vqvae_config,
                    'families': checkpoint_config.get('families', {})
                }
            logger.info("Loaded config from checkpoint")
        else:
            # Fall back to loading from YAML
            config_path = checkpoint_path / 'config.yaml'
            yaml.SafeLoader.add_constructor(
                'tag:yaml.org,2002:python/object/apply:pathlib._local.PosixPath',
                lambda loader, node: str(Path(*loader.construct_sequence(node)))
            )
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)

            if 'model' in config_dict:
                model_config_dict = config_dict['model']
            else:
                model_params = [
                    'input_dim', 'group_indices', 'group_embedding_dim', 'group_hidden_dim',
                    'levels', 'commitment_cost', 'orthogonality_weight', 'informativeness_weight',
                    'use_ema', 'decay', 'dropout', 'compression_ratios', 'uniform_codebook_init'
                ]
                model_config_dict = {k: v for k, v in config_dict.items() if k in model_params}

            vqvae_config = CategoricalVQVAEConfig(**model_config_dict)
            logger.info("Loaded config from YAML")

        # Create model from config
        vqvae = CategoricalHierarchicalVQVAE(vqvae_config).to(self.device)

        # Load weights (handle torch.compile() wrapped models and hybrid architectures)
        state_dict = checkpoint['model_state_dict']

        # Strip _orig_mod. prefix if present (from torch.compile)
        new_state_dict = {}
        for key, value in state_dict.items():
            # Handle different wrapper patterns
            if key.startswith('_orig_mod.vqvae.'):
                # Hybrid model with vqvae component
                new_key = key.replace('_orig_mod.vqvae.', '')
            elif key.startswith('_orig_mod.'):
                # Standard compiled model
                new_key = key.replace('_orig_mod.', '')
            else:
                # No prefix
                new_key = key

            # Only include keys that belong to the VQ-VAE
            if not new_key.startswith('initial_encoder.'):
                new_state_dict[new_key] = value

        # Load with strict=False to handle any remaining mismatches
        missing_keys, unexpected_keys = vqvae.load_state_dict(new_state_dict, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {len(missing_keys)} keys")
            logger.debug(f"First 5 missing keys: {missing_keys[:5]}")

        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            logger.debug(f"First 5 unexpected keys: {unexpected_keys[:5]}")

        # Add families config if available (stored at checkpoint level, not in model config)
        if 'families' in checkpoint:
            config_dict['families'] = checkpoint['families']
            logger.info(f"Loaded families configuration: {list(checkpoint['families'].keys())}")

        # Try to load feature extractor
        feature_extractor = None
        try:
            feature_extractor = AlignedFeatureExtractor.from_checkpoint(
                str(checkpoint_path), device=str(self.device)
            )
            feature_extractor.eval()
            logger.info("Loaded AlignedFeatureExtractor")
        except Exception as e:
            logger.warning(f"Could not load AlignedFeatureExtractor: {e}")

        return vqvae, feature_extractor, config_dict, checkpoint

    def load_dataset_features(self) -> torch.Tensor:
        """Load and encode features from HDF5 dataset.

        Loads raw features from multi-family dataset structure and applies
        per-family encoders to create the feature vector expected by VQ-VAE.

        Follows the same pipeline as training:
        1. Load raw features (initial, summary, temporal)
        2. Apply per-family encoders (using checkpoint encoders)
        3. Concatenate encoded features
        4. Apply feature cleaning (if needed to match VQ-VAE input_dim)
        """
        logger.info(f"Loading {self.num_samples} samples from {self.dataset_path}")

        # Try to load from simple flat structure first
        with h5py.File(self.dataset_path, 'r') as f:
            # Check if 'features' is a single dataset (already concatenated & encoded)
            if 'features' in f and isinstance(f['features'], h5py.Dataset):
                features = f['features'][:self.num_samples]
                logger.info(f"Loaded pre-encoded features: {features.shape}")

                # Check if dimensions match
                if features.shape[1] == self.vqvae.config.input_dim:
                    return torch.from_numpy(features).float().to(self.device)
                else:
                    logger.warning(
                        f"Feature dim mismatch: dataset has {features.shape[1]}D, "
                        f"VQ-VAE expects {self.vqvae.config.input_dim}D"
                    )

            # Otherwise, load from multi-family structure
            elif 'features' in f and isinstance(f['features'], h5py.Group):
                logger.info("Loading multi-family features for encoding...")
                features = self._load_and_encode_multi_family(f)
                return features

            # Try other common locations
            elif 'cleaned_features' in f:
                features = f['cleaned_features'][:self.num_samples]
                logger.info(f"Loaded cleaned features: {features.shape}")
            elif 'raw_features' in f:
                features = f['raw_features'][:self.num_samples]
                logger.info(f"Loaded raw features: {features.shape}")
            else:
                raise ValueError(
                    f"Could not find features in dataset. "
                    f"Available keys: {list(f.keys())}"
                )

        features = torch.from_numpy(features).float().to(self.device)
        logger.info(f"Loaded features: {features.shape}")
        return features

    def _load_and_encode_multi_family(self, hdf5_file: h5py.File) -> torch.Tensor:
        """Load raw features from multi-family structure and encode them.

        Args:
            hdf5_file: Open HDF5 file with features/{family} structure

        Returns:
            Encoded and concatenated feature tensor matching VQ-VAE input_dim
        """
        import numpy as np
        from spinlock.encoding.encoders import get_encoder
        from spinlock.encoding import FeatureProcessor

        # Get families config from checkpoint
        families_config = self.config.get('families', {})
        if not families_config:
            raise ValueError(
                "No families configuration found in checkpoint. "
                "Cannot encode multi-family features."
            )

        all_features = []
        feature_families = list(families_config.keys())

        logger.info(f"Encoding {len(feature_families)} feature families: {feature_families}")

        for family_name in feature_families:
            family_config = families_config[family_name]
            encoder_name = family_config.get('encoder')
            encoder_params = family_config.get('encoder_params', {})

            # Load raw features for this family
            if family_name == 'temporal':
                # Temporal uses /features/temporal/features directly
                features_path = f'/features/{family_name}/features'
            else:
                # Other families use /features/{family}/aggregated/features
                features_path = f'/features/{family_name}/aggregated/features'

            if features_path not in hdf5_file:
                logger.warning(f"Features not found at {features_path}, skipping {family_name}")
                continue

            family_features = np.array(hdf5_file[features_path][:self.num_samples])
            logger.info(f"  {family_name}: loaded raw shape {family_features.shape}")

            # Handle NaN values
            nan_count = np.isnan(family_features).sum()
            if nan_count > 0:
                family_features = np.nan_to_num(family_features, nan=0.0)
                logger.info(f"  {family_name}: replaced {nan_count} NaN values")

            # Apply encoder if configured
            if encoder_name and encoder_name not in ['identity', 'IdentityEncoder', 'initial_hybrid']:
                # Get input dimension
                if len(family_features.shape) == 3:
                    # Temporal: [N, T, D]
                    input_dim = family_features.shape[2]
                else:
                    # Others: [N, D]
                    input_dim = family_features.shape[1]

                # Create encoder
                encoder = get_encoder(
                    encoder_name,
                    input_dim=input_dim,
                    **encoder_params
                ).to(self.device)
                encoder.eval()

                # Encode features
                with torch.no_grad():
                    family_tensor = torch.from_numpy(family_features).float().to(self.device)
                    encoded = encoder(family_tensor)
                    family_features = encoded.cpu().numpy()

                logger.info(f"  {family_name}: encoded to shape {family_features.shape}")

            elif encoder_name == 'initial_hybrid':
                # For hybrid initial encoder, just use manual features (14D)
                # The CNN part would need raw ICs, which we skip for diagnostics
                logger.info(f"  {family_name}: using manual features only (hybrid CNN skipped)")

            all_features.append(family_features)

        # Concatenate all encoded features
        concatenated = np.concatenate(all_features, axis=1)
        logger.info(f"Concatenated features: {concatenated.shape}")

        # Apply feature cleaning to match VQ-VAE input_dim
        expected_dim = self.vqvae.config.input_dim
        if concatenated.shape[1] > expected_dim:
            # Check if we have saved feature_mask from checkpoint (for reproducibility)
            feature_mask_worked = False
            if 'feature_mask' in self.checkpoint:
                logger.info(
                    f"Found feature_mask in checkpoint - attempting to use it"
                )
                feature_mask = self.checkpoint['feature_mask']

                # Verify feature_mask is valid
                if isinstance(feature_mask, np.ndarray) and len(feature_mask) == concatenated.shape[1]:
                    concatenated_cleaned = concatenated[:, feature_mask]
                    logger.info(f"After applying feature_mask: {concatenated_cleaned.shape}")

                    if concatenated_cleaned.shape[1] == expected_dim:
                        logger.info("✓ Feature dimensions match VQ-VAE input_dim using saved feature_mask")
                        concatenated = concatenated_cleaned
                        feature_mask_worked = True
                    else:
                        logger.warning(
                            f"feature_mask produced {concatenated_cleaned.shape[1]}D but expected {expected_dim}D. "
                            f"Feature pipeline may have expanded dimensions after cleaning. "
                            f"Will skip feature_mask and use features as-is if they match expected dimension."
                        )
                else:
                    logger.warning(
                        f"feature_mask is invalid (type={type(feature_mask)}, "
                        f"len={len(feature_mask) if hasattr(feature_mask, '__len__') else 'N/A'}, "
                        f"expected={concatenated.shape[1]}). "
                    )

            if not feature_mask_worked:
                # Check if concatenated features already match expected dimension
                if concatenated.shape[1] == expected_dim:
                    logger.info(
                        f"✓ Feature dimensions already match VQ-VAE input_dim ({expected_dim}D) "
                        f"without applying feature_mask. Using features as-is."
                    )
                    # Use concatenated features directly - no cleaning needed
                    feature_mask = None
                elif 'feature_mask' not in self.checkpoint:
                    logger.warning(
                        "feature_mask not found in checkpoint - will re-run feature cleaning. "
                        "Diagnostics may not exactly match training metrics."
                    )
                    feature_mask = None
                else:
                    # feature_mask exists but didn't work
                    logger.warning(
                        "Saved feature_mask didn't produce correct dimension. "
                        "Will re-run feature cleaning (may not match training)."
                    )
                    feature_mask = None

            # If feature_mask didn't work or wasn't available, fall back to re-cleaning
            if feature_mask is None:
                logger.info(
                    f"Applying feature cleaning to reduce {concatenated.shape[1]}D → {expected_dim}D"
                )

                # Use same cleaning parameters as training (conservative defaults)
                processor = FeatureProcessor(
                    variance_threshold=1e-10,
                    deduplicate_threshold=0.99,
                    use_intelligent_dedup=True,
                    outlier_method='percentile',
                    percentile_range=[0.5, 99.5],
                    mad_threshold=3.0,
                    verbose=True,
                )

                cleaned_features, feature_mask, _ = processor.clean(concatenated)
                logger.info(f"After cleaning: {cleaned_features.shape}")

                # Check if we match expected dimension
                if cleaned_features.shape[1] == expected_dim:
                    logger.info("✓ Feature dimensions match VQ-VAE input_dim")
                    concatenated = cleaned_features
                elif cleaned_features.shape[1] < expected_dim:
                    # Pad if necessary
                    logger.warning(
                        f"After cleaning: {cleaned_features.shape[1]}D < {expected_dim}D. "
                        f"Padding with zeros."
                    )
                    padding = np.zeros((cleaned_features.shape[0], expected_dim - cleaned_features.shape[1]))
                    concatenated = np.concatenate([cleaned_features, padding], axis=1)
                else:
                    # Truncate if still too large
                    logger.warning(
                        f"After cleaning: {cleaned_features.shape[1]}D > {expected_dim}D. "
                        f"Truncating to {expected_dim}D."
                    )
                    concatenated = cleaned_features[:, :expected_dim]

        elif concatenated.shape[1] < expected_dim:
            logger.warning(
                f"Concatenated features {concatenated.shape[1]}D < expected {expected_dim}D. "
                f"Padding with zeros."
            )
            padding = np.zeros((concatenated.shape[0], expected_dim - concatenated.shape[1]))
            concatenated = np.concatenate([concatenated, padding], axis=1)

        # Convert to tensor
        features = torch.from_numpy(concatenated).float().to(self.device)
        logger.info(f"Final encoded features: {features.shape}")
        return features

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run all diagnostic analyses."""
        logger.info("Starting full diagnostic analysis...")

        # Load features
        features = self.load_dataset_features()

        results = {}

        # 1. Per-family reconstruction analysis
        logger.info("=== Phase 1: Per-Family Reconstruction Analysis ===")
        results['per_family'] = self.analyze_per_family_reconstruction(features)

        # 2. Per-dimension error analysis
        logger.info("=== Phase 2: Per-Dimension Error Analysis ===")
        results['per_dimension'] = self.analyze_per_dimension_errors(features)

        # 3. Codebook utilization analysis
        logger.info("=== Phase 3: Codebook Utilization Analysis ===")
        results['codebook'] = self.analyze_codebook_utilization(features)

        # 4. Latent space visualization
        logger.info("=== Phase 4: Latent Space Visualization ===")
        results['latent_space'] = self.visualize_latent_space(features)

        # 5. Generate recommendations
        logger.info("=== Phase 5: Generating Recommendations ===")
        results['recommendations'] = self.generate_recommendations(results)

        # 6. Save reports
        self.save_reports(results)

        logger.info(f"Analysis complete! Results saved to {self.output_dir}")
        return results

    def analyze_per_family_reconstruction(self, features: torch.Tensor) -> Dict:
        """Analyze reconstruction quality per feature family."""
        logger.info("Analyzing per-family reconstruction...")

        with torch.no_grad():
            # Forward pass through VQ-VAE
            outputs = self.vqvae(features)
            reconstructed = outputs['reconstruction']['features']

        # Convert to numpy
        features_np = features.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()

        family_results = {}

        # Split by family using group_indices
        for family_name, indices in self.group_indices.items():
            family_original = features_np[:, indices]
            family_recon = reconstructed_np[:, indices]

            # Compute MSE
            mse = np.mean((family_original - family_recon) ** 2)
            mae = np.mean(np.abs(family_original - family_recon))

            family_results[family_name] = {
                'mse': float(mse),
                'mae': float(mae),
                'original': family_original,
                'reconstructed': family_recon,
                'dimensions': len(indices)
            }

            logger.info(f"Family '{family_name}': MSE={mse:.6f}, MAE={mae:.6f}, dims={len(indices)}")

            # Visualize
            self._plot_family_reconstruction(
                family_original, family_recon, family_name, mse
            )

        return family_results

    def analyze_per_dimension_errors(self, features: torch.Tensor) -> Dict:
        """Analyze errors per dimension across all families."""
        logger.info("Analyzing per-dimension errors...")

        with torch.no_grad():
            outputs = self.vqvae(features)
            reconstructed = outputs['reconstruction']['features']

        # Convert to numpy
        features_np = features.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()

        # Compute per-dimension MSE
        dimension_mse = np.mean((features_np - reconstructed_np) ** 2, axis=0)

        # Per-family dimension errors
        family_dim_errors = {}
        for family_name, indices in self.group_indices.items():
            family_errors = dimension_mse[indices]
            family_dim_errors[family_name] = {
                'errors': family_errors.tolist(),
                'mean': float(np.mean(family_errors)),
                'std': float(np.std(family_errors)),
                'max': float(np.max(family_errors)),
                'high_error_dims': [
                    (int(idx), float(family_errors[idx]))
                    for idx in np.argsort(family_errors)[-10:][::-1]
                ]
            }

        # Visualizations
        self._plot_dimension_error_heatmap(dimension_mse, self.group_indices)
        self._plot_high_error_dimensions(family_dim_errors)

        return {
            'global_dimension_mse': dimension_mse.tolist(),
            'per_family': family_dim_errors
        }

    def analyze_codebook_utilization(self, features: torch.Tensor) -> Dict:
        """Analyze codebook usage patterns."""
        logger.info("Analyzing codebook utilization...")

        # Collect token assignments
        usage_counts = [
            np.zeros(q.num_embeddings, dtype=np.int64)
            for q in self.vqvae.quantizers
        ]

        with torch.no_grad():
            outputs = self.vqvae(features)
            tokens = outputs['tokens']  # [B, N×L]

            # Count token occurrences
            for quantizer_idx in range(tokens.shape[1]):
                token_ids = tokens[:, quantizer_idx].cpu().numpy()
                for token_id in token_ids:
                    usage_counts[quantizer_idx][token_id] += 1

        # Compute metrics per quantizer
        quantizer_metrics = []
        for q_idx, (counts, quantizer) in enumerate(zip(usage_counts, self.vqvae.quantizers)):
            total_used = np.sum(counts > 0)
            utilization = total_used / len(counts)

            # Compute entropy
            probs = counts / (np.sum(counts) + 1e-10)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            perplexity = np.exp(entropy)

            # Identify dead codes
            dead_codes = np.where(counts < 5)[0].tolist()

            quantizer_metrics.append({
                'quantizer_idx': q_idx,
                'num_tokens': len(counts),
                'utilization': float(utilization),
                'perplexity': float(perplexity),
                'entropy': float(entropy),
                'dead_codes': dead_codes,
                'dead_codes_count': len(dead_codes),
                'usage_counts': counts.tolist(),
                'avg_usage': float(np.mean(counts[counts > 0])) if np.any(counts > 0) else 0.0,
            })

            logger.info(
                f"Quantizer {q_idx}: utilization={utilization:.2%}, "
                f"perplexity={perplexity:.1f}, dead_codes={len(dead_codes)}"
            )

        # Global statistics
        global_stats = {
            'avg_utilization': float(np.mean([m['utilization'] for m in quantizer_metrics])),
            'avg_perplexity': float(np.mean([m['perplexity'] for m in quantizer_metrics])),
            'total_dead_codes': sum(m['dead_codes_count'] for m in quantizer_metrics),
        }

        # Visualizations
        self._plot_codebook_utilization(usage_counts, quantizer_metrics)

        return {
            'per_quantizer': quantizer_metrics,
            'global_stats': global_stats
        }

    def visualize_latent_space(self, features: torch.Tensor) -> Dict:
        """Visualize latent space with t-SNE."""
        logger.info("Visualizing latent space with t-SNE...")

        # Limit samples for t-SNE (computationally expensive)
        max_tsne_samples = min(5000, len(features))
        features_tsne = features[:max_tsne_samples]

        with torch.no_grad():
            # Get latents from encoder
            latents = self.vqvae.encode(features_tsne)  # List of [B, D] per level

        # Concatenate all latents
        latents_concat = torch.cat(latents, dim=1).cpu().numpy()
        logger.info(f"Latent shape: {latents_concat.shape}")

        # Compute t-SNE
        logger.info("Computing t-SNE embeddings...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
        embeddings = tsne.fit_transform(latents_concat)

        # Get token assignments for coloring
        with torch.no_grad():
            outputs = self.vqvae(features_tsne)
            tokens = outputs['tokens'].cpu().numpy()

        # Visualize by code assignment (first quantizer)
        self._plot_tsne_by_code(embeddings, tokens[:, 0], "Level 0")

        # Compute cluster quality
        if len(np.unique(tokens[:, 0])) > 1:
            silhouette = float(silhouette_score(latents_concat, tokens[:, 0]))
            davies_bouldin = float(davies_bouldin_score(latents_concat, tokens[:, 0]))
        else:
            silhouette = 0.0
            davies_bouldin = 0.0

        return {
            'tsne_embeddings': embeddings.tolist(),
            'token_assignments': tokens.tolist(),
            'cluster_quality': {
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
            }
        }

    def generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations based on diagnostic results."""
        recommendations = []

        # 1. Check per-family reconstruction errors
        family_mses = {
            name: data['mse']
            for name, data in results['per_family'].items()
        }
        min_mse = min(family_mses.values())
        max_mse = max(family_mses.values())

        if max_mse > 3 * min_mse:
            worst_family = max(family_mses, key=family_mses.get)
            recommendations.append(
                f"⚠️  '{worst_family}' family has {max_mse/min_mse:.1f}x higher reconstruction "
                f"error than others (MSE={max_mse:.6f}). Consider increasing compression ratio "
                f"to allocate more latent capacity (e.g., [0.25, 0.75, 2.0])."
            )

        # 2. Check codebook utilization
        avg_util = results['codebook']['global_stats']['avg_utilization']
        if avg_util < 0.5:
            recommendations.append(
                f"⚠️  Low codebook utilization ({avg_util:.1%}). Consider reducing codebook sizes "
                f"or increasing beta (commitment loss weight) to encourage better code usage."
            )
        elif avg_util > 0.9:
            recommendations.append(
                f"✓ High codebook utilization ({avg_util:.1%}). Codebook sizes are appropriate."
            )

        # 3. Check dead codes
        total_dead = results['codebook']['global_stats']['total_dead_codes']
        if total_dead > 0:
            recommendations.append(
                f"⚠️  {total_dead} dead codes detected across quantizers. "
                f"Consider enabling dead code reset (dead_code_reset_interval=1000, threshold=10.0)."
            )

        # 4. Check high-error dimensions
        for family, dim_data in results['per_dimension']['per_family'].items():
            if dim_data['max'] > 0.1:
                high_error_dims = dim_data['high_error_dims'][:3]
                dim_str = ', '.join([f"dim{idx}" for idx, _ in high_error_dims])
                recommendations.append(
                    f"⚠️  High error in {family} dimensions: {dim_str}. "
                    f"Investigate feature extraction for these dimensions."
                )

        # 5. General health check
        if not recommendations:
            recommendations.append("✓ VQ-VAE reconstruction quality looks healthy overall.")

        return recommendations

    def save_reports(self, results: Dict):
        """Save JSON and text summary reports."""
        # JSON report
        json_path = self.output_dir / 'summary_report.json'
        with open(json_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._prepare_for_json(results)
            json.dump(json_results, f, indent=2)
        logger.info(f"Saved JSON report: {json_path}")

        # Text summary report
        text_path = self.output_dir / 'summary_report.txt'
        with open(text_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VQ-VAE RECONSTRUCTION ERROR DIAGNOSTIC REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Per-family summary
            f.write("Per-Family Reconstruction Errors:\n")
            f.write("-" * 40 + "\n")
            for family, data in results['per_family'].items():
                f.write(f"{family:15s}: MSE={data['mse']:.6f}, dims={data['dimensions']}\n")

            # Codebook utilization
            f.write("\nCodebook Utilization:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Utilization: {results['codebook']['global_stats']['avg_utilization']:.1%}\n")
            f.write(f"Average Perplexity: {results['codebook']['global_stats']['avg_perplexity']:.1f}\n")
            f.write(f"Total Dead Codes: {results['codebook']['global_stats']['total_dead_codes']}\n")

            # Recommendations
            f.write("\nRecommendations:\n")
            f.write("-" * 40 + "\n")
            for i, rec in enumerate(results['recommendations'], 1):
                f.write(f"{i}. {rec}\n")

        logger.info(f"Saved text report: {text_path}")

    def _prepare_for_json(self, obj):
        """Recursively convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items() if k not in ['original', 'reconstructed']}
        elif isinstance(obj, (list, tuple)):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj

    # === Visualization Methods ===

    def _plot_family_reconstruction(
        self, original: np.ndarray, reconstructed: np.ndarray, family_name: str, mse: float
    ):
        """Plot original vs reconstructed features for a family (4-panel)."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Limit to first 100 samples for visualization
        n_vis = min(100, len(original))
        original_vis = original[:n_vis]
        reconstructed_vis = reconstructed[:n_vis]
        error = np.abs(original_vis - reconstructed_vis)

        # Original features
        im1 = axes[0, 0].imshow(original_vis, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[0, 0].set_title(f'{family_name} - Original')
        axes[0, 0].set_xlabel('Dimension')
        axes[0, 0].set_ylabel('Sample')
        plt.colorbar(im1, ax=axes[0, 0])

        # Reconstructed features
        im2 = axes[0, 1].imshow(reconstructed_vis, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[0, 1].set_title(f'{family_name} - Reconstructed')
        axes[0, 1].set_xlabel('Dimension')
        plt.colorbar(im2, ax=axes[0, 1])

        # Absolute error
        im3 = axes[1, 0].imshow(error, aspect='auto', cmap='hot', interpolation='nearest')
        axes[1, 0].set_title(f'{family_name} - Absolute Error')
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Sample')
        plt.colorbar(im3, ax=axes[1, 0])

        # Error distribution
        axes[1, 1].hist(error.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(error.mean(), color='r', linestyle='--',
                          label=f'Mean: {error.mean():.4f}')
        axes[1, 1].legend()

        plt.suptitle(f'{family_name} Reconstruction Analysis (MSE: {mse:.6f})', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / 'visualizations' / 'family_reconstruction' / f'{family_name}_recon.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved family reconstruction plot: {output_path}")

    def _plot_dimension_error_heatmap(self, dimension_mse: np.ndarray, group_indices: Dict):
        """Plot per-dimension error heatmap across families."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create matrix: dimensions × families
        family_names = list(group_indices.keys())
        error_matrix = []
        dim_labels = []

        for family in family_names:
            indices = group_indices[family]
            errors = dimension_mse[indices]
            error_matrix.append(errors)
            dim_labels.extend([f"{family}_{i}" for i in range(len(indices))])

        error_matrix = np.concatenate(error_matrix)

        # Plot heatmap
        im = ax.imshow(error_matrix.reshape(-1, 1), aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_ylabel('Dimension')
        ax.set_title('Per-Dimension MSE')
        ax.set_xticks([])

        plt.colorbar(im, ax=ax, label='MSE')
        plt.tight_layout()

        output_path = self.output_dir / 'visualizations' / 'dimension_errors' / 'error_heatmap.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved dimension error heatmap: {output_path}")

    def _plot_high_error_dimensions(self, family_dim_errors: Dict):
        """Plot bar chart of highest-error dimensions per family."""
        fig, axes = plt.subplots(len(family_dim_errors), 1, figsize=(12, 4 * len(family_dim_errors)))

        if len(family_dim_errors) == 1:
            axes = [axes]

        for ax, (family, data) in zip(axes, family_dim_errors.items()):
            high_error_dims = data['high_error_dims']
            dims = [f"dim{idx}" for idx, _ in high_error_dims]
            errors = [err for _, err in high_error_dims]

            ax.bar(range(len(dims)), errors, color='orangered')
            ax.set_xticks(range(len(dims)))
            ax.set_xticklabels(dims, rotation=45, ha='right')
            ax.set_ylabel('MSE')
            ax.set_title(f'{family} - Top 10 High-Error Dimensions')
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / 'visualizations' / 'dimension_errors' / 'high_error_dims.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved high-error dimensions plot: {output_path}")

    def _plot_codebook_utilization(self, usage_counts: List[np.ndarray], metrics: List[Dict]):
        """Plot codebook utilization bar chart and heatmap."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Left: Utilization bar chart
        utilizations = [m['utilization'] for m in metrics]
        quantizer_labels = [f"Q{m['quantizer_idx']}" for m in metrics]

        axes[0].bar(range(len(utilizations)), utilizations, color='steelblue')
        axes[0].set_xlabel('Quantizer Index')
        axes[0].set_ylabel('Utilization (%)')
        axes[0].set_title('Codebook Utilization per Quantizer')
        axes[0].set_xticks(range(len(quantizer_labels)))
        axes[0].set_xticklabels(quantizer_labels, rotation=45)
        axes[0].axhline(y=0.667, color='r', linestyle='--', label='Baseline (66.7%)')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Right: Usage heatmap (log scale)
        max_codes = max(len(counts) for counts in usage_counts)
        usage_matrix = np.zeros((len(usage_counts), max_codes))
        for i, counts in enumerate(usage_counts):
            usage_matrix[i, :len(counts)] = counts

        # Replace zeros with NaN for log scale
        usage_matrix_masked = np.where(usage_matrix > 0, usage_matrix, np.nan)

        im = axes[1].imshow(usage_matrix_masked, aspect='auto', cmap='viridis',
                           norm=LogNorm(vmin=1, vmax=usage_matrix.max()), interpolation='nearest')
        axes[1].set_xlabel('Code Index')
        axes[1].set_ylabel('Quantizer Index')
        axes[1].set_title('Code Usage Heatmap (log scale)')
        axes[1].set_yticks(range(len(quantizer_labels)))
        axes[1].set_yticklabels(quantizer_labels)
        plt.colorbar(im, ax=axes[1], label='Usage Count')

        plt.tight_layout()

        output_path = self.output_dir / 'visualizations' / 'codebook' / 'utilization.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved codebook utilization plot: {output_path}")

    def _plot_tsne_by_code(self, embeddings: np.ndarray, code_assignments: np.ndarray, title: str):
        """Plot t-SNE colored by code assignment."""
        fig, ax = plt.subplots(figsize=(12, 10))

        unique_codes = np.unique(code_assignments)
        n_codes = len(unique_codes)

        # Use tab20 colormap for up to 20 codes, otherwise jet
        if n_codes <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, n_codes))
        else:
            colors = plt.cm.jet(np.linspace(0, 1, n_codes))

        for i, code in enumerate(unique_codes):
            mask = code_assignments == code
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[colors[i]],
                label=f'Code {code}',
                alpha=0.6,
                s=20,
            )

        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title(f't-SNE Latent Space - {title}')

        # Only show legend if not too many codes
        if n_codes <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = self.output_dir / 'visualizations' / 'latent_space' / f'tsne_{title.replace(" ", "_")}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved t-SNE plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose VQ-VAE reconstruction errors",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to VQ-VAE checkpoint directory'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to HDF5 dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='diagnostics/vqvae_recon',
        help='Output directory for diagnostic results'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10000,
        help='Number of samples to analyze (default: 10000)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Computation device (default: cuda)'
    )

    args = parser.parse_args()

    # Run diagnostics
    diagnostics = VQVAEDiagnostics(
        checkpoint_path=Path(args.checkpoint),
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        device=args.device
    )

    results = diagnostics.run_full_analysis()

    # Print summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print("\nPer-Family Reconstruction Errors:")
    for family, data in results['per_family'].items():
        print(f"  {family:15s}: MSE={data['mse']:.6f}")

    print("\nCodebook Utilization:")
    print(f"  Average: {results['codebook']['global_stats']['avg_utilization']:.1%}")
    print(f"  Dead Codes: {results['codebook']['global_stats']['total_dead_codes']}")

    print("\nRecommendations:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. {rec}")

    print(f"\nFull results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
