"""Main validator for MNO-VQ-VAE distribution alignment.

This validator tests whether VQ-VAE (trained on CNO ground truth) can reliably
tokenize MNO outputs with acceptable reconstruction quality.

V2 Integration:
- Uses VQTokenizer with built-in feature processing
- No external feature pipeline needed
- Type-safe checkpoint access via Pydantic schema
- Clean forward pass with integrated reconstruction
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
from pathlib import Path

from spinlock.mno.validation_utils import load_mno_checkpoint, load_vqvae_checkpoint
from spinlock.mno.validation.metrics import ValidationMetrics
from spinlock.mno.validation.config import ValidationConfig
from spinlock.tokens import load_checkpoint, TokenizerCheckpoint


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
    - Uses VQTokenizer with integrated feature processing (V2)
    - Modular metrics computation (testable)
    - Clear pass/fail criteria (actionable)

    V2 Architecture:
    - VQTokenizer.model.forward() handles feature encoding internally
    - Reconstruction happens in encoded feature space
    - No external feature pipeline needed
    - Type-safe checkpoint access via Pydantic

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
            print("[PASS] VQ-VAE can tokenize MNO outputs")
        else:
            print("[FAIL] Distribution mismatch detected")
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

        # Load checkpoint with Pydantic schema for type-safe access
        print("Loading checkpoint metadata...")
        self.checkpoint = load_checkpoint(Path(vqvae_checkpoint))

        # Extract CNO baseline reconstruction error from training history
        self.cno_reconstruction_error = self._extract_cno_baseline()

        print(f"CNO baseline reconstruction error: {self.cno_reconstruction_error:.6f}")

    def _extract_cno_baseline(self) -> float:
        """Extract CNO reconstruction MSE from VQ-VAE training history.

        IMPORTANT: Returns UNNORMALIZED reconstruction MSE to match MNO validation.

        The VQTokenizer loss may use normalize_reconstruction=True which divides
        MSE by feature dimension. We need to unnormalize it to get raw MSE for
        fair comparison with MNO validation metrics.

        Returns:
            CNO baseline reconstruction MSE (unnormalized, in encoded feature space)
        """
        # Try training history metrics (most reliable and detailed)
        if self.checkpoint.metadata and 'training_history' in self.checkpoint.metadata:
            history = self.checkpoint.metadata['training_history']

            # Check for val_metrics which contains per-component losses
            if 'val_metrics' in history and len(history['val_metrics']) > 0:
                final_metrics = history['val_metrics'][-1]
                if 'reconstruction' in final_metrics:
                    recon_loss = float(final_metrics['reconstruction'])

                    # Unnormalize if needed
                    if self.checkpoint.config.loss.normalize_reconstruction:
                        # Loss was divided by feature dimension, multiply back
                        total_dim = self._get_total_encoded_dim()
                        recon_mse = recon_loss * total_dim
                        print(f"  Normalized recon loss: {recon_loss:.6f}")
                        print(f"  Total encoded dim: {total_dim}")
                        print(f"  Unnormalized MSE: {recon_mse:.6f}")
                        return recon_mse
                    else:
                        return recon_loss

            # Fallback to val_losses (total loss)
            if 'val_losses' in history and len(history['val_losses']) > 0:
                print("  Warning: Using total val_loss (includes auxiliary losses)")
                return float(history['val_losses'][-1])

        # Fallback: use val_loss from checkpoint
        if self.checkpoint.val_loss is not None:
            print("  Warning: Using checkpoint val_loss")
            return float(self.checkpoint.val_loss)

        # Conservative default if not found
        print("  Warning: Could not extract CNO baseline from checkpoint, using default")
        return 0.027

    def _get_total_encoded_dim(self) -> int:
        """Calculate total encoded feature dimension from config."""
        total_dim = 0

        # Temporal encoder output dimension
        if self.checkpoint.config.encoder.temporal:
            total_dim += sum(self.checkpoint.config.encoder.temporal.level_dims)

        # Initial encoder output dimension
        if self.checkpoint.config.encoder.initial:
            # CNN embedding dimension
            total_dim += self.checkpoint.config.encoder.initial.cnn_embedding_dim
            # Manual features dimension (if not encoded, use raw manual_dim)
            total_dim += self.checkpoint.config.encoder.initial.manual_dim

        return total_dim  # From 50K baseline docs

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
        3. Process rollouts through VQTokenizer.model.forward()
           - Internally handles feature encoding
           - Returns reconstruction in encoded space
        4. Measure reconstruction error in encoded feature space
        5. Compare to CNO baseline

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

        # Step 3: Process rollouts through VQTokenizer (integrated feature extraction + reconstruction)
        print("Step 3: Processing rollouts through VQTokenizer...")
        original_features, reconstructed_features, tokens = self._process_with_tokenizer(
            mno_rollouts, ics, batch_size
        )

        # Step 4: Compute validation metrics
        print("Step 4: Computing validation metrics...")
        metrics = ValidationMetrics.compute(
            features_original=original_features,
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

    def _process_with_tokenizer(
        self,
        rollouts: torch.Tensor,
        ics: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Process MNO rollouts through VQTokenizer with integrated feature handling.

        V2 Architecture:
        - Extracts temporal features from rollouts
        - VQTokenizer.model.forward() handles encoding internally
        - Returns reconstruction in encoded feature space
        - No external feature pipeline needed

        Args:
            rollouts: MNO rollouts [N, T, C, H, W]
            ics: Initial conditions [N, C, H, W]
            batch_size: Batch size for processing

        Returns:
            Tuple of (original_encoded, reconstructed_encoded, tokens)
            - original_encoded: [N, D_encoded] in encoded feature space
            - reconstructed_encoded: [N, D_encoded] reconstructed features
            - tokens: Dict mapping "family_category_Ll" → token indices [N]
        """
        from spinlock.mno.feature_extraction import MNOFeatureExtractor

        # Initialize feature extractor for temporal features
        feature_extractor = MNOFeatureExtractor(device=str(self.device))

        original_encoded_list = []
        reconstructed_list = []
        tokens_dict = {}

        num_batches = (len(rollouts) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, len(rollouts))

            batch_rollouts = rollouts[start:end].to(self.device)
            batch_ics = ics[start:end].to(self.device)

            with torch.no_grad():
                # Extract temporal features from MNO rollouts
                # MNOFeatureExtractor returns {'temporal': [B, T, D], ...}
                extracted = feature_extractor.extract(batch_rollouts)
                temporal_features = extracted['temporal']  # [B, T, D]

                # Prepare inputs for VQTokenizer
                initial_raw = None
                initial_manual = None

                if 'initial' in self.vqvae.model.families:
                    # VQTokenizer expects initial conditions
                    initial_raw = batch_ics  # [B, C, H, W]

                    # Check if hybrid encoder (requires both manual AND raw, even if encode_manual=False)
                    from spinlock.encoding.encoders.initial_hybrid import InitialHybridEncoder
                    if isinstance(self.vqvae.model.initial_encoder, InitialHybridEncoder):
                        # InitialHybridEncoder always requires manual_features argument
                        # Get the expected manual dimension from config
                        manual_dim = self.checkpoint.config.encoder.initial.manual_dim

                        # If encode_manual=True, extract real manual features
                        # If encode_manual=False, provide zeros (Identity encoder will pass through)
                        encode_manual = self.checkpoint.config.encoder.initial.encode_manual
                        if encode_manual:
                            # Extract manual features from ICs using InitialManualExtractor
                            from spinlock.features.initial.manual_extractors import InitialManualExtractor
                            manual_extractor = InitialManualExtractor(device=self.device)
                            # Add realization dimension for feature extractor: [B, C, H, W] → [B, 1, C, H, W]
                            ics_with_realization = batch_ics.unsqueeze(1)
                            # Extract and remove realization dimension: [B, 1, D_manual] → [B, D_manual]
                            manual_features = manual_extractor.extract_all(ics_with_realization)[:, 0, :]
                            initial_manual = manual_features  # [B, D_manual]
                        else:
                            # Provide zeros - encoder uses Identity so output won't be used anyway
                            initial_manual = torch.zeros(
                                batch_ics.shape[0], manual_dim, device=self.device
                            )

                # Forward pass through VQTokenizer model
                # Returns dict with 'reconstructed', 'quantized', 'vq_loss', 'encodings', etc.
                output = self.vqvae.model.forward(
                    temporal_features=temporal_features,
                    initial_manual=initial_manual,
                    initial_raw=initial_raw,
                    temporal_mask=None,
                    temporal_lengths=None,
                )

                # Extract results
                original_encoded_list.append(output['original_encoded'].cpu())
                reconstructed_list.append(output['reconstructed'].cpu())

                # Extract token indices from encodings
                # output['encodings'] is dict of quantized vectors, need to get indices
                batch_tokens = self.vqvae.model.encode(
                    temporal_features=temporal_features,
                    initial_manual=initial_manual,
                    initial_raw=initial_raw,
                    temporal_mask=None,
                    temporal_lengths=None,
                )

                # Accumulate tokens
                for key, indices in batch_tokens.items():
                    if key not in tokens_dict:
                        tokens_dict[key] = []
                    tokens_dict[key].append(indices.cpu())

            print(f"  Processed batch {i+1}/{num_batches}")

        # Concatenate results
        original_encoded = torch.cat(original_encoded_list, dim=0)
        reconstructed = torch.cat(reconstructed_list, dim=0)

        # Concatenate tokens
        tokens = {key: torch.cat(indices_list, dim=0) for key, indices_list in tokens_dict.items()}

        print(f"  Original encoded shape: {original_encoded.shape}")
        print(f"  Reconstructed shape: {reconstructed.shape}")
        print(f"  Tokens: {len(tokens)} categories")

        return original_encoded, reconstructed, tokens

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
            status = "[PASS] EXCELLENT"
        elif result.reconstruction_ratio <= 2.0:
            status = "[PASS] GOOD"
        elif result.reconstruction_ratio <= 3.0:
            status = "[WARN] ACCEPTABLE"
        else:
            status = "[FAIL] POOR"

        print(f"  Status:           {status}")

        print(f"\n{'='*60}")
        print(f"Pass Threshold: {'PASSED' if result.pass_threshold else 'FAILED'}")
        print(f"{'='*60}\n")
