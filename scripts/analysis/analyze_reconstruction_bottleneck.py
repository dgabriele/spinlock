"""Analyze what limits VQ-VAE reconstruction quality.

Investigates:
1. Per-feature reconstruction errors
2. Which features reconstruct worst
3. Whether errors are in VQ bottleneck or decoder
4. Systematic vs random error patterns
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path
import argparse
from collections import defaultdict


def load_checkpoint(checkpoint_path):
    """Load VQ-VAE checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return checkpoint


def analyze_reconstruction_errors(
    model,
    checkpoint,
    dataset_path,
    device='cuda',
    num_samples=5000,
):
    """Analyze reconstruction errors in detail.

    Args:
        model: VQ-VAE model
        checkpoint: Checkpoint dict containing encoder_state_dicts and other metadata
        dataset_path: Path to HDF5 dataset
        device: Device to use
        num_samples: Number of samples to analyze

    Returns:
        Dict with analysis results
    """
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()

    # Also set submodules to eval (important for BatchNorm, etc.)
    for module in model.modules():
        module.eval()

    # Load frozen encoders and normalization stats from checkpoint
    encoder_state_dicts = checkpoint.get('encoder_state_dicts', {})
    norm_stats = checkpoint.get('normalization_stats', {})

    # Load and encode features using the same process as training
    print("Loading and encoding features...")
    from spinlock.encoding.encoders import get_encoder
    import numpy as np

    all_features = []
    all_feature_names = []

    with h5py.File(dataset_path, 'r') as f:
        # Load INITIAL features first (they come first in concatenation)
        if 'features/initial/aggregated/features' in f:
            initial_features = np.array(f['features/initial/aggregated/features'][:num_samples])
            print(f"  Loaded initial manual features: {initial_features.shape}")

            # Normalize INITIAL features
            if 'initial' in norm_stats:
                mean, std = norm_stats['initial']  # Tuple of (mean, std)
                if isinstance(mean, torch.Tensor):
                    mean = mean.cpu().numpy()
                if isinstance(std, torch.Tensor):
                    std = std.cpu().numpy()
                if len(mean) > 0:  # Check if not empty
                    initial_features = (initial_features - mean) / (std + 1e-8)
                    print(f"    Normalized with mean={mean[:3]}, std={std[:3]}")

            all_features.append(initial_features)

        # Load TEMPORAL features and encode with frozen encoder
        if 'temporal' in encoder_state_dicts:
            temporal_raw = np.array(f['features/temporal/features'][:num_samples])
            print(f"  Loaded raw temporal features: {temporal_raw.shape}")

            # Create encoder and load frozen weights
            input_dim = temporal_raw.shape[2]  # Feature dimension
            temporal_encoder_info = encoder_state_dicts['temporal']

            # The encoder state dict contains metadata + actual weights
            if 'state_dict' in temporal_encoder_info:
                actual_state_dict = temporal_encoder_info['state_dict']
            else:
                actual_state_dict = temporal_encoder_info

            encoder = get_encoder('TemporalCNNEncoder',
                                input_dim=input_dim,
                                embedding_dim=128,
                                architecture='resnet1d_3')
            encoder.load_state_dict(actual_state_dict)
            encoder = encoder.to(device)
            encoder.eval()

            # Encode temporal features
            temporal_features = []
            batch_size = 256
            for i in range(0, len(temporal_raw), batch_size):
                batch = temporal_raw[i:i+batch_size]
                batch_tensor = torch.from_numpy(batch).float().to(device)
                with torch.no_grad():
                    encoded = encoder(batch_tensor)
                temporal_features.append(encoded.cpu().numpy())
            temporal_features = np.concatenate(temporal_features, axis=0)
            print(f"  Encoded temporal features: {temporal_features.shape}")
            print(f"    Temporal stats BEFORE norm: min={temporal_features.min():.3f}, max={temporal_features.max():.3f}, mean={temporal_features.mean():.3f}, std={temporal_features.std():.3f}")

            # Normalize TEMPORAL features (if stats exist)
            if 'temporal' in norm_stats:
                mean, std = norm_stats['temporal']  # Tuple of (mean, std)
                if isinstance(mean, torch.Tensor):
                    mean = mean.cpu().numpy()
                if isinstance(std, torch.Tensor):
                    std = std.cpu().numpy()
                if len(mean) > 0:  # Check if not empty (temporal encoder output isn't normalized)
                    temporal_features = (temporal_features - mean) / (std + 1e-8)
                    print(f"    Normalized with mean={mean[:3]}, std={std[:3]}")
                else:
                    print(f"    Temporal features not normalized (encoder output)")

            print(f"    Temporal stats AFTER norm: min={temporal_features.min():.3f}, max={temporal_features.max():.3f}, mean={temporal_features.mean():.3f}, std={temporal_features.std():.3f}")
            all_features.append(temporal_features)

        # Load initial features (hybrid mode with raw ICs)
        if 'inputs/fields' in f:
            raw_ics_data = np.array(f['inputs/fields'][:num_samples])
            # Average across realizations
            if len(raw_ics_data.shape) == 4:
                raw_ics = raw_ics_data.mean(axis=1, keepdims=True)  # [N, 1, H, W]
            elif len(raw_ics_data.shape) == 5:
                raw_ics = raw_ics_data.mean(axis=1)  # [N, C, H, W]
            print(f"  Loaded raw ICs for hybrid encoder: {raw_ics.shape}")
            raw_ics = torch.from_numpy(raw_ics).float().to(device)
        else:
            raw_ics = None

    # Concatenate all features
    features = np.concatenate(all_features, axis=1)
    print(f"Total concatenated features (normalized): {features.shape}")

    # Apply feature mask (feature cleaning)
    if 'feature_mask' in checkpoint:
        feature_mask = checkpoint['feature_mask']
        if isinstance(feature_mask, torch.Tensor):
            feature_mask = feature_mask.cpu().numpy()
        features = features[:, feature_mask]
        print(f"After feature cleaning (masked): {features.shape}")

    features_np = features  # Keep numpy version
    features = torch.from_numpy(features).float()

    # Run through model in batches (to avoid OOM)
    batch_size = 256
    all_reconstructions = []
    all_targets = []
    all_latent_vectors = []
    all_quantized_vectors = []
    all_encodings = []

    print(f"Running inference in batches of {batch_size}...")
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i+batch_size].to(device)
            batch_raw_ics = raw_ics[i:i+batch_size] if raw_ics is not None else None

            if batch_raw_ics is not None:
                outputs = model(batch_features, raw_ics=batch_raw_ics)
            else:
                outputs = model(batch_features)

            # Get reconstruction target
            if 'input_features' in outputs:
                target = outputs['input_features']
            else:
                target = batch_features.to(device)

            reconstruction = outputs['reconstruction']['features']

            all_reconstructions.append(reconstruction.cpu())
            all_targets.append(target.cpu())

            # Collect VQ outputs for analysis
            if 'latent_vectors' in outputs:
                all_latent_vectors.append([lv.cpu() for lv in outputs['latent_vectors']])
            if 'quantized_vectors' in outputs:
                all_quantized_vectors.append([qv.cpu() for qv in outputs['quantized_vectors']])
            if 'encodings' in outputs:
                all_encodings.append([enc.cpu() for enc in outputs['encodings']])

    # Concatenate all batches
    reconstruction = torch.cat(all_reconstructions, dim=0)
    target = torch.cat(all_targets, dim=0)

    # Concatenate VQ outputs
    latent_vectors = [torch.cat([batch[i] for batch in all_latent_vectors], dim=0)
                     for i in range(len(all_latent_vectors[0]))] if all_latent_vectors else []
    quantized_vectors = [torch.cat([batch[i] for batch in all_quantized_vectors], dim=0)
                        for i in range(len(all_quantized_vectors[0]))] if all_quantized_vectors else []
    encodings = [torch.cat([batch[i] for batch in all_encodings], dim=0)
                for i in range(len(all_encodings[0]))] if all_encodings else []

    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Reconstruction stats: min={reconstruction.min():.3f}, max={reconstruction.max():.3f}, mean={reconstruction.mean():.3f}, std={reconstruction.std():.3f}")
    print(f"Target stats: min={target.min():.3f}, max={target.max():.3f}, mean={target.mean():.3f}, std={target.std():.3f}")

    # 1. Overall reconstruction error
    mse_loss = ((reconstruction - target) ** 2).mean().item()
    mae_loss = (reconstruction - target).abs().mean().item()

    print(f"\n=== Overall Reconstruction ===")
    print(f"MSE: {mse_loss:.6f}")
    print(f"MAE: {mae_loss:.6f}")
    print(f"RMSE: {np.sqrt(mse_loss):.6f}")
    print(f"Quality (1-MSE): {1-mse_loss:.4f}")

    # 2. Per-feature errors
    per_feature_mse = ((reconstruction - target) ** 2).mean(dim=0).cpu().numpy()
    per_feature_mae = (reconstruction - target).abs().mean(dim=0).cpu().numpy()

    print(f"\n=== Per-Feature Analysis ===")
    print(f"Features: {len(per_feature_mse)}")
    print(f"MSE range: {per_feature_mse.min():.6f} to {per_feature_mse.max():.6f}")
    print(f"MSE mean: {per_feature_mse.mean():.6f}")
    print(f"MSE std: {per_feature_mse.std():.6f}")

    # Worst features
    worst_indices = np.argsort(per_feature_mse)[-10:][::-1]
    print(f"\nWorst 10 features (by MSE):")
    for i, idx in enumerate(worst_indices, 1):
        print(f"  {i}. Feature {idx}: MSE={per_feature_mse[idx]:.6f}, MAE={per_feature_mae[idx]:.6f}")

    # Best features
    best_indices = np.argsort(per_feature_mse)[:10]
    print(f"\nBest 10 features (by MSE):")
    for i, idx in enumerate(best_indices, 1):
        print(f"  {i}. Feature {idx}: MSE={per_feature_mse[idx]:.6f}, MAE={per_feature_mae[idx]:.6f}")

    # 3. VQ bottleneck analysis
    print(f"\n=== VQ Bottleneck Analysis ===")

    if latent_vectors and quantized_vectors:
        # Compare pre-quantization vs post-quantization latents
        total_vq_error = 0
        for pre_q, post_q in zip(latent_vectors, quantized_vectors):
            vq_error = ((pre_q - post_q) ** 2).mean().item()
            total_vq_error += vq_error

        avg_vq_error = total_vq_error / len(latent_vectors)
        print(f"Average VQ quantization error: {avg_vq_error:.6f}")
        print(f"VQ error / Total error: {avg_vq_error / mse_loss * 100:.1f}%")

        # 4. Decoder analysis - reconstruct from quantized codes
        print(f"\n=== Decoder Analysis ===")

        # The decoder takes quantized vectors and reconstructs
        # We can't easily separate this without model internals,
        # but we can estimate decoder error as: total_error - vq_error
        estimated_decoder_error = mse_loss - avg_vq_error
        print(f"Estimated decoder error: {estimated_decoder_error:.6f}")
        print(f"Decoder error / Total error: {estimated_decoder_error / mse_loss * 100:.1f}%")
    else:
        print("VQ outputs not available in model outputs")
        avg_vq_error = 0
        estimated_decoder_error = mse_loss

    # 5. Codebook utilization
    print(f"\n=== Codebook Utilization ===")
    if encodings:
        for level_idx, level_encodings in enumerate(encodings):
            # encodings: [batch, num_tokens] - one-hot encoded
            used_codes = (level_encodings.sum(dim=0) > 0).sum().item()
            total_codes = level_encodings.shape[1]
            utilization = used_codes / total_codes * 100
            print(f"Level {level_idx}: {used_codes}/{total_codes} codes used ({utilization:.1f}%)")
    else:
        print("Encodings not available in model outputs")

    # 6. Error distribution
    print(f"\n=== Error Distribution ===")
    errors = (reconstruction - target).cpu().numpy()
    print(f"Error mean: {errors.mean():.6f}")
    print(f"Error std: {errors.std():.6f}")
    print(f"Error min: {errors.min():.6f}")
    print(f"Error max: {errors.max():.6f}")

    # Check if errors are systematic (bias) or random
    per_sample_mean_error = errors.mean(axis=1)
    print(f"Per-sample mean error: {per_sample_mean_error.mean():.6f} ± {per_sample_mean_error.std():.6f}")

    return {
        'overall_mse': mse_loss,
        'overall_mae': mae_loss,
        'per_feature_mse': per_feature_mse,
        'per_feature_mae': per_feature_mae,
        'vq_error': avg_vq_error,
        'decoder_error_estimate': estimated_decoder_error,
        'worst_features': worst_indices,
        'best_features': best_indices,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to VQ-VAE checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset HDF5 file')
    parser.add_argument('--num-samples', type=int, default=5000,
                       help='Number of samples to analyze')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)

    # Reconstruct model
    # The checkpoint's model_config is for the ADJUSTED vqvae (already expanded)
    # We'll create a dummy wrapper and load the state dict directly
    from spinlock.encoding.categorical_vqvae import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig
    from spinlock.encoding.vqvae_with_initial import VQVAEWithInitial
    from spinlock.encoding.encoders.initial_hybrid import InitialHybridEncoder

    model_config_dict = checkpoint['model_config']
    config_dict = checkpoint['config']

    # Check if this is a hybrid model (has initial encoder)
    state_dict = checkpoint['model_state_dict']
    has_initial_encoder = any('initial_encoder' in k for k in state_dict.keys())

    if has_initial_encoder:
        print("Detected hybrid model with InitialHybridEncoder")

        # The model_config is for the ALREADY ADJUSTED inner vqvae
        # We need to create the wrapper differently
        # Strategy: Create inner vqvae directly, then manually add initial_encoder

        # Create inner vqvae with adjusted config
        vqvae_config = CategoricalVQVAEConfig(**model_config_dict)
        inner_vqvae = CategoricalHierarchicalVQVAE(vqvae_config)

        # Create initial encoder
        initial_config = config_dict['families']['initial']
        encoder_params = initial_config.get('encoder_params', {})
        initial_encoder = InitialHybridEncoder(
            manual_dim=encoder_params.get('manual_dim', 14),
            cnn_embedding_dim=encoder_params.get('cnn_embedding_dim', 28),
            encode_manual=encoder_params.get('encode_manual', False),
            in_channels=encoder_params.get('in_channels', 1),
        )
        initial_encoder.eval()  # Set to eval mode BEFORE loading weights
        print(f"  Created InitialHybridEncoder: manual_dim={encoder_params.get('manual_dim', 14)}, cnn_dim={encoder_params.get('cnn_embedding_dim', 28)}")

        # Create a simple wrapper
        class SimpleWrapper(nn.Module):
            def __init__(self, vqvae, initial_encoder):
                super().__init__()
                self.vqvae = vqvae
                self.initial_encoder = initial_encoder

            def forward(self, x, raw_ics=None):
                if raw_ics is not None:
                    # Expand initial features with CNN
                    # Assume initial features are at the beginning
                    manual_features = x[:, :self.initial_encoder.manual_dim]
                    other_features = x[:, self.initial_encoder.manual_dim:]

                    # Encode (note: signature is forward(manual_features, raw_ics))
                    with torch.no_grad():  # Ensure no gradient computation
                        expanded_initial = self.initial_encoder(manual_features, raw_ics)

                    # Debug: Check expanded_initial stats (first batch only)
                    if not hasattr(self, '_debug_printed'):
                        print(f"      CNN output stats: min={expanded_initial.min():.3f}, max={expanded_initial.max():.3f}, mean={expanded_initial.mean():.3f}, std={expanded_initial.std():.3f}")
                        self._debug_printed = True

                    # Concatenate
                    expanded_x = torch.cat([expanded_initial, other_features], dim=1)
                else:
                    expanded_x = x

                # Pass to vqvae
                outputs = self.vqvae(expanded_x)

                # Add input_features to outputs (for reconstruction target)
                if raw_ics is not None:
                    outputs['input_features'] = expanded_x

                return outputs

        model = SimpleWrapper(inner_vqvae, initial_encoder)
    else:
        print("Detected standard model (no hybrid encoder)")
        model_config = CategoricalVQVAEConfig(**model_config_dict)
        model = CategoricalHierarchicalVQVAE(model_config)

    # Remove _orig_mod prefix from torch.compile()
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("Removing _orig_mod prefix from torch.compile checkpoint")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    print(f"Loading model state dict...")
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"  Missing keys: {len(load_result.missing_keys)}")
        for key in load_result.missing_keys[:5]:
            print(f"    - {key}")
    if load_result.unexpected_keys:
        print(f"  Unexpected keys: {len(load_result.unexpected_keys)}")
        for key in load_result.unexpected_keys[:5]:
            print(f"    - {key}")

    print("Model loaded successfully")

    print(f"Analyzing dataset: {args.dataset}")
    print(f"Using {args.num_samples} samples")

    results = analyze_reconstruction_errors(
        model,
        checkpoint,
        args.dataset,
        device=args.device,
        num_samples=args.num_samples,
    )

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
