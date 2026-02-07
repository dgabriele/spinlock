"""Main validator for MNO-VQ-VAE distribution alignment."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
from pathlib import Path

from spinlock.mno.validation_utils import load_mno_checkpoint, load_vqvae_checkpoint
from spinlock.encoding.unified_feature_pipeline import UnifiedFeaturePipeline
from spinlock.mno.validation.metrics import ValidationMetrics
from spinlock.mno.validation.config import ValidationConfig


@dataclass
class ValidationResult:
    """Container for validation results."""
    mno_reconstruction_error: float
    cno_reconstruction_error: float
    reconstruction_ratio: float
    per_level_accuracy: Dict[str, float]
    token_distribution_kl: float
    pass_threshold: bool
    num_samples: int


class MNOVQVAEValidator:
    """
    Validates VQ-VAE tokenization quality on MNO-generated rollouts.

    Tests whether VQ-VAE (trained on CNO ground truth) can reliably tokenize
    MNO outputs with acceptable reconstruction quality, validating distribution
    alignment.

    Design Principles:
    - Reuses existing checkpoint loaders (DRY)
    - Composes feature pipeline (no reimplementation)
    - Modular metrics computation (testable)
    - Clear pass/fail criteria (actionable)

    Example:
        validator = MNOVQVAEValidator(
            mno_checkpoint="checkpoints/mno/50k_baseline/meta_operator_best.pt",
            vqvae_checkpoint="checkpoints/vqvae/50k_baseline/best_model.pt",
            device="cuda"
        )

        result = validator.validate(
            dataset_path="datasets/cno_50k_v3_1.h5",
            num_samples=100
        )

        if result.pass_threshold:
            print("✓ VQ-VAE can tokenize MNO outputs")
        else:
            print("✗ Distribution mismatch detected")
    """

    def __init__(
        self,
        mno_checkpoint: Path,
        vqvae_checkpoint: Path,
        config: Optional[ValidationConfig] = None,
        device: str = "cuda"
    ):
        """
        Initialize validator with trained checkpoints.

        Args:
            mno_checkpoint: Path to trained MNO checkpoint
            vqvae_checkpoint: Path to trained VQ-VAE checkpoint
            config: Validation configuration (uses defaults if None)
            device: Torch device for inference
        """
        self.device = torch.device(device)
        self.config = config or ValidationConfig()

        # Load models using existing utilities (DRY)
        print("Loading MNO checkpoint...")
        self.mno = load_mno_checkpoint(str(mno_checkpoint), device=device)

        print("Loading VQ-VAE checkpoint...")
        self.vqvae = load_vqvae_checkpoint(str(vqvae_checkpoint), device=device)

        # Setup feature pipeline (reuses VQ-VAE's frozen encoders and normalization)
        print("Setting up feature pipeline...")
        self.feature_pipeline = UnifiedFeaturePipeline.from_checkpoint(
            vqvae_checkpoint, device=device
        )

        # Get CNO baseline reconstruction error from VQ-VAE checkpoint
        vqvae_ckpt = torch.load(vqvae_checkpoint, map_location=device, weights_only=False)
        self.cno_reconstruction_error = self._extract_cno_baseline(vqvae_ckpt)

        print(f"CNO baseline reconstruction error: {self.cno_reconstruction_error:.6f}")

    def _extract_cno_baseline(self, vqvae_checkpoint: Dict) -> float:
        """Extract CNO reconstruction error from VQ-VAE training history."""
        if 'history' in vqvae_checkpoint:
            history = vqvae_checkpoint['history']
            # Use final validation reconstruction error
            if 'val_loss' in history and len(history['val_loss']) > 0:
                # Assumes val_loss is dominated by reconstruction error
                return history['val_loss'][-1]

        # Fallback: use reconstruction error from checkpoint metadata
        if 'best_loss' in vqvae_checkpoint:
            return vqvae_checkpoint['best_loss']

        # Conservative default if not found
        return 0.027  # From 50K baseline docs

    def validate(
        self,
        dataset_path: Path,
        num_samples: Optional[int] = None,
        batch_size: int = 8
    ) -> ValidationResult:
        """
        Run validation on dataset samples.

        Workflow:
        1. Load validation samples (ICs, params) from dataset
        2. Generate MNO rollouts (256 timesteps)
        3. Extract features from MNO rollouts
        4. Tokenize features with VQ-VAE
        5. Decode tokens back to features
        6. Measure reconstruction error
        7. Compare to CNO baseline

        Args:
            dataset_path: Path to HDF5 dataset with ICs and params
            num_samples: Number of samples to validate (None = use config default)
            batch_size: Batch size for inference

        Returns:
            ValidationResult with metrics and pass/fail status
        """
        num_samples = num_samples or self.config.num_samples
        print(f"\n{'='*60}")
        print(f"MNO-VQ-VAE Distribution Alignment Validation")
        print(f"{'='*60}")
        print(f"Samples: {num_samples}")
        print(f"Dataset: {dataset_path}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}\n")

        # Step 1: Load dataset samples
        print("Step 1: Loading validation samples...")
        ics, params = self._load_validation_samples(dataset_path, num_samples)

        # Step 2: Generate MNO rollouts
        print("Step 2: Generating MNO rollouts...")
        mno_rollouts = self._generate_mno_rollouts(ics, params, batch_size)

        # Step 3: Extract features from MNO rollouts
        print("Step 3: Extracting features from MNO rollouts...")
        mno_features = self._extract_features(mno_rollouts, ics, batch_size)

        # Step 4: Tokenize with VQ-VAE
        print("Step 4: Tokenizing features with VQ-VAE...")
        tokens, vq_output = self._tokenize_features(mno_features, batch_size)

        # Step 5: Decode tokens back to features
        print("Step 5: Decoding tokens back to features...")
        reconstructed_features = self._decode_tokens(tokens, batch_size)

        # Step 6: Compute validation metrics
        print("Step 6: Computing validation metrics...")
        metrics = ValidationMetrics.compute(
            features_original=mno_features,
            features_reconstructed=reconstructed_features,
            tokens=tokens,
            cno_baseline_error=self.cno_reconstruction_error,
            config=self.config
        )

        # Step 7: Create result
        result = ValidationResult(
            mno_reconstruction_error=metrics['reconstruction_mse'],
            cno_reconstruction_error=self.cno_reconstruction_error,
            reconstruction_ratio=metrics['reconstruction_ratio'],
            per_level_accuracy=metrics['per_level_token_accuracy'],
            token_distribution_kl=metrics.get('token_distribution_kl', 0.0),
            pass_threshold=metrics['reconstruction_ratio'] <= self.config.max_reconstruction_ratio,
            num_samples=num_samples
        )

        # Step 8: Print summary
        self._print_summary(result)

        return result

    def _load_validation_samples(
        self,
        dataset_path: Path,
        num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load ICs and params from validation split."""
        import h5py

        with h5py.File(dataset_path, 'r') as f:
            total_samples = f['inputs/fields'].shape[0]

            # Use last num_samples as validation set (simple split)
            val_start = max(0, total_samples - num_samples)
            val_end = total_samples

            fields = f['inputs/fields'][val_start:val_end]

            # Handle both formats:
            # New format: [N, R, C, H, W] with realizations
            # Old format: [N, C, H, W] without realizations
            if len(fields.shape) == 5:
                # New format with realizations - use first realization
                ics = torch.from_numpy(fields[:, 0, ...]).float()  # [N, C, H, W]
            else:
                # Old format without realizations
                ics = torch.from_numpy(fields).float()  # [N, C, H, W]

            params = torch.from_numpy(
                f['parameters/params'][val_start:val_end]
            ).float()

        print(f"  Loaded {len(ics)} samples (ICs: {ics.shape}, params: {params.shape})")
        return ics, params

    def _generate_mno_rollouts(
        self,
        ics: torch.Tensor,
        params: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """Generate MNO rollouts in batches."""
        rollouts = []

        num_batches = (len(ics) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, len(ics))

            batch_ics = ics[start:end].to(self.device)
            batch_params = params[start:end].to(self.device)

            with torch.no_grad():
                # Generate 256-step rollout
                batch_rollouts = self.mno.rollout(
                    batch_ics,
                    steps=self.config.rollout_steps,
                    return_all_steps=True,
                    params=batch_params
                )

            rollouts.append(batch_rollouts[:, 1:, ...].cpu())  # Remove IC from output
            print(f"  Generated batch {i+1}/{num_batches}")

        result = torch.cat(rollouts, dim=0)
        print(f"  MNO rollouts shape: {result.shape}")
        return result

    def _extract_features(
        self,
        rollouts: torch.Tensor,
        ics: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """Extract features from rollouts using feature pipeline."""
        features_list = []

        num_batches = (len(rollouts) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, len(rollouts))

            batch_rollouts = rollouts[start:end].to(self.device)
            batch_ics = ics[start:end].to(self.device)

            # Check if realization dimension exists in the data
            # If rollouts is [B, T, C, H, W], add realization dim: [B, 1, T, C, H, W]
            if len(batch_rollouts.shape) == 5:
                batch_rollouts = batch_rollouts.unsqueeze(1)
            if len(batch_ics.shape) == 3:
                batch_ics = batch_ics.unsqueeze(1)

            with torch.no_grad():
                # Extract and normalize features
                batch_features = self.feature_pipeline(
                    batch_rollouts,
                    batch_ics,
                    normalize=True
                )

            features_list.append(batch_features.cpu())
            print(f"  Extracted features batch {i+1}/{num_batches}")

        result = torch.cat(features_list, dim=0)
        print(f"  Features shape: {result.shape}")
        return result

    def _tokenize_features(
        self,
        features: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, Dict]:
        """Tokenize features with VQ-VAE."""
        tokens_list = []
        vq_outputs = []

        num_batches = (len(features) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, len(features))

            batch_features = features[start:end].to(self.device)

            with torch.no_grad():
                # Full forward pass to get tokens and VQ output
                output = self.vqvae(batch_features)
                tokens_list.append(output['tokens'].cpu())
                vq_outputs.append({
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in output.items()
                })

            print(f"  Tokenized batch {i+1}/{num_batches}")

        tokens = torch.cat(tokens_list, dim=0)
        print(f"  Tokens shape: {tokens.shape}")
        return tokens, vq_outputs[0]  # Return first batch VQ output for inspection

    def _decode_tokens(
        self,
        tokens: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """Decode tokens back to features."""
        reconstructed_list = []

        num_batches = (len(tokens) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, len(tokens))

            batch_tokens = tokens[start:end].to(self.device)

            with torch.no_grad():
                batch_reconstructed = self.vqvae.decode_from_tokens(batch_tokens)

            reconstructed_list.append(batch_reconstructed.cpu())
            print(f"  Decoded batch {i+1}/{num_batches}")

        result = torch.cat(reconstructed_list, dim=0)
        print(f"  Reconstructed features shape: {result.shape}")
        return result

    def _print_summary(self, result: ValidationResult):
        """Print validation summary."""
        print(f"\n{'='*60}")
        print("VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"\nReconstruction Error:")
        print(f"  MNO rollouts:     {result.mno_reconstruction_error:.6f}")
        print(f"  CNO baseline:     {result.cno_reconstruction_error:.6f}")
        print(f"  Ratio (MNO/CNO):  {result.reconstruction_ratio:.3f}x")

        if result.reconstruction_ratio <= 1.5:
            status = "✓ EXCELLENT"
        elif result.reconstruction_ratio <= 2.0:
            status = "✓ GOOD"
        elif result.reconstruction_ratio <= 3.0:
            status = "⚠ ACCEPTABLE"
        else:
            status = "✗ POOR"

        print(f"  Status:           {status}")

        print(f"\n{'='*60}")
        print(f"Pass Threshold: {result.pass_threshold}")
        print(f"{'='*60}\n")
