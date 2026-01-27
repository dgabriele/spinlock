"""
Core evaluation classes for MNO diagnostics.

Provides the main evaluation orchestrator that coordinates physics fidelity
and tokenization evaluation.
"""

import torch
import torch.nn as nn
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

from spinlock.noa.backbone import NOABackbone
from spinlock.noa.validation_utils import load_mno_checkpoint
from spinlock.noa.cno_replay import CNOReplayer
from spinlock.diagnostics.metrics import (
    compute_physics_metrics,
    compute_error_growth_stats,
    compute_tokenization_metrics,
    compute_distribution_metrics,
)
from spinlock.diagnostics.feature_alignment import (
    extract_features_from_rollouts,
    load_vqvae_for_tokenization,
    tokenize_features,
    get_token_distribution,
)


class MNODiagnosticEvaluator:
    """Main evaluation orchestrator for MNO diagnostics.

    Coordinates physics fidelity and tokenization evaluation across
    training and held-out splits.
    """

    def __init__(
        self,
        mno_checkpoint: Path,
        vqvae_checkpoint: Path,
        dataset_path: Path,
        substrate_config_path: Path,
        train_split_end: int = 10240,
        device: str = "cuda",
    ):
        """Initialize MNO diagnostic evaluator.

        Args:
            mno_checkpoint: Path to MNO checkpoint (.pt file)
            vqvae_checkpoint: Path to VQ-VAE checkpoint directory
            dataset_path: Path to CNO dataset HDF5 file
            substrate_config_path: Path to CNO generation config YAML
            train_split_end: Index where training split ends (default 10240)
            device: Computation device
        """
        self.mno_checkpoint = Path(mno_checkpoint)
        self.vqvae_checkpoint = Path(vqvae_checkpoint)
        self.dataset_path = Path(dataset_path)
        self.substrate_config_path = Path(substrate_config_path)
        self.train_split_end = train_split_end
        self.device = torch.device(device)

        # Models (loaded on demand)
        self.mno: Optional[NOABackbone] = None
        self.cno_replayer: Optional[CNOReplayer] = None
        self.vqvae: Optional[nn.Module] = None
        self.vqvae_metadata: Optional[Dict] = None

        # Dataset info
        self.dataset_size: Optional[int] = None
        self.num_channels: Optional[int] = None
        self.spatial_size: Optional[Tuple[int, int]] = None

    def _load_models(self):
        """Load all required models."""
        if self.mno is None:
            print("Loading MNO...")
            self.mno = load_mno_checkpoint(str(self.mno_checkpoint), device=str(self.device))

        if self.cno_replayer is None:
            print("Loading CNO replayer...")
            self.cno_replayer = CNOReplayer.from_config(
                str(self.substrate_config_path),
                device=str(self.device),
            )

        if self.vqvae is None:
            print("Loading VQ-VAE...")
            self.vqvae, self.vqvae_metadata = load_vqvae_for_tokenization(
                self.vqvae_checkpoint,
                device=str(self.device),
            )

    def _load_dataset_info(self):
        """Load dataset metadata."""
        if self.dataset_size is not None:
            return

        with h5py.File(self.dataset_path, 'r') as f:
            # Check dataset structure
            if 'inputs/fields' in f:
                self.dataset_size = f['inputs/fields'].shape[0]
                self.num_channels = f['inputs/fields'].shape[-3]
                self.spatial_size = (f['inputs/fields'].shape[-2], f['inputs/fields'].shape[-1])
            else:
                raise ValueError(f"Dataset does not have expected structure: {list(f.keys())}")

        print(f"Dataset info:")
        print(f"  Total samples: {self.dataset_size}")
        print(f"  Channels: {self.num_channels}")
        print(f"  Spatial size: {self.spatial_size}")

    def _get_split_indices(
        self,
        split: str,
        max_samples: Optional[int] = None,
    ) -> List[int]:
        """Get indices for a dataset split.

        Args:
            split: 'train', 'holdout', or 'both'
            max_samples: Maximum samples per split (None for all)

        Returns:
            List of sample indices
        """
        self._load_dataset_info()

        if split == "train":
            indices = list(range(0, min(self.train_split_end, self.dataset_size)))
        elif split == "holdout":
            indices = list(range(self.train_split_end, self.dataset_size))
        elif split == "both":
            indices = list(range(0, self.dataset_size))
        else:
            raise ValueError(f"Unknown split: {split}")

        # Sample if needed
        if max_samples is not None and len(indices) > max_samples:
            import random
            random.seed(42)  # Reproducible sampling
            indices = random.sample(indices, max_samples)
            indices.sort()  # Keep sorted for efficient HDF5 access

        return indices

    def evaluate_physics_fidelity(
        self,
        split: str = "both",
        max_samples: Optional[int] = None,
        batch_size: int = 8,
        num_timesteps: int = 256,
        save_rollouts_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Evaluate physics fidelity (MNO vs CNO rollout comparison).

        Args:
            split: 'train', 'holdout', or 'both'
            max_samples: Maximum samples per split
            batch_size: Batch size for evaluation
            num_timesteps: Number of timesteps to roll out
            save_rollouts_path: Optional path to save rollouts (HDF5)

        Returns:
            Dictionary with physics metrics per split
        """
        self._load_models()
        indices = self._get_split_indices(split, max_samples)

        print(f"\nEvaluating physics fidelity on {len(indices)} samples ({split} split)...")

        evaluator = PhysicsFidelityEvaluator(
            mno=self.mno,
            cno_replayer=self.cno_replayer,
            dataset_path=self.dataset_path,
            device=self.device,
        )

        results = evaluator.evaluate(
            indices=indices,
            batch_size=batch_size,
            num_timesteps=num_timesteps,
            save_rollouts_path=save_rollouts_path,
        )

        return results

    def evaluate_tokenization(
        self,
        split: str = "both",
        max_samples: Optional[int] = None,
        batch_size: int = 8,
        num_timesteps: int = 256,
        save_tokens_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Evaluate tokenization quality and distribution shift.

        Args:
            split: 'train', 'holdout', or 'both'
            max_samples: Maximum samples per split
            batch_size: Batch size for evaluation
            num_timesteps: Number of timesteps to roll out
            save_tokens_path: Optional path to save tokens (HDF5)

        Returns:
            Dictionary with tokenization metrics
        """
        self._load_models()
        indices = self._get_split_indices(split, max_samples)

        print(f"\nEvaluating tokenization on {len(indices)} samples ({split} split)...")

        evaluator = TokenizationEvaluator(
            mno=self.mno,
            cno_replayer=self.cno_replayer,
            vqvae=self.vqvae,
            vqvae_checkpoint=self.vqvae_checkpoint,
            dataset_path=self.dataset_path,
            device=self.device,
        )

        results = evaluator.evaluate(
            indices=indices,
            batch_size=batch_size,
            num_timesteps=num_timesteps,
            save_tokens_path=save_tokens_path,
        )

        return results


class PhysicsFidelityEvaluator:
    """Evaluates physics fidelity by comparing MNO vs CNO rollouts."""

    def __init__(
        self,
        mno: NOABackbone,
        cno_replayer: CNOReplayer,
        dataset_path: Path,
        device: torch.device,
    ):
        """Initialize physics fidelity evaluator.

        Args:
            mno: Loaded MNO model
            cno_replayer: CNO replayer
            dataset_path: Path to dataset
            device: Computation device
        """
        self.mno = mno
        self.cno_replayer = cno_replayer
        self.dataset_path = dataset_path
        self.device = device

    def evaluate(
        self,
        indices: List[int],
        batch_size: int = 8,
        num_timesteps: int = 256,
        save_rollouts_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Evaluate physics fidelity on given indices.

        Args:
            indices: Sample indices to evaluate
            batch_size: Batch size
            num_timesteps: Number of timesteps
            save_rollouts_path: Optional path to save rollouts

        Returns:
            Dictionary with metrics and per-sample results
        """
        num_samples = len(indices)
        num_batches = (num_samples + batch_size - 1) // batch_size

        # Accumulators
        all_per_timestep_mse = []
        all_per_timestep_rel_l2 = []
        all_per_timestep_nrmse = []
        all_sample_results = []

        # Optional rollout storage
        if save_rollouts_path:
            mno_rollouts_list = []
            cno_rollouts_list = []

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Physics evaluation"):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, num_samples)
                batch_indices = indices[batch_start:batch_end]

                # Load batch data
                ics, params = self._load_batch(batch_indices)

                # Generate MNO rollout (pass params for conditioning)
                mno_rollout = self._mno_rollout(ics, params, num_timesteps)

                # Generate CNO rollout (averaged over realizations)
                cno_rollout = self._cno_rollout(params, ics, num_timesteps)

                # Compute metrics
                metrics = compute_physics_metrics(mno_rollout, cno_rollout, per_timestep=True)

                # Accumulate
                all_per_timestep_mse.append(metrics['mse'].cpu())
                all_per_timestep_rel_l2.append(metrics['relative_l2'].cpu())
                all_per_timestep_nrmse.append(metrics['nrmse'].cpu())

                # Store per-sample metrics
                for i in range(len(batch_indices)):
                    all_sample_results.append({
                        'sample_idx': batch_indices[i],
                        'final_mse': metrics['mse'][-1].item(),
                        'mean_mse': metrics['mse'].mean().item(),
                        'final_rel_l2': metrics['relative_l2'][-1].item(),
                    })

                # Save rollouts if requested
                if save_rollouts_path:
                    mno_rollouts_list.append(mno_rollout.cpu())
                    cno_rollouts_list.append(cno_rollout.cpu())

                # Memory cleanup
                if batch_idx % 4 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Aggregate metrics
        mean_per_timestep_mse = torch.stack(all_per_timestep_mse).mean(dim=0)
        std_per_timestep_mse = torch.stack(all_per_timestep_mse).std(dim=0)
        mean_per_timestep_rel_l2 = torch.stack(all_per_timestep_rel_l2).mean(dim=0)
        mean_per_timestep_nrmse = torch.stack(all_per_timestep_nrmse).mean(dim=0)

        # Compute error growth statistics
        error_growth = compute_error_growth_stats(mean_per_timestep_mse)

        # Save rollouts if requested
        if save_rollouts_path:
            self._save_rollouts(
                save_rollouts_path,
                torch.cat(mno_rollouts_list, dim=0),
                torch.cat(cno_rollouts_list, dim=0),
                indices,
            )

        return {
            'mean_mse': mean_per_timestep_mse.mean().item(),
            'std_mse': std_per_timestep_mse.mean().item(),
            'final_mse': mean_per_timestep_mse[-1].item(),
            'relative_l2': mean_per_timestep_rel_l2.mean().item(),
            'nrmse': mean_per_timestep_nrmse.mean().item(),
            'per_timestep_mse': mean_per_timestep_mse.numpy().tolist(),
            'per_timestep_rel_l2': mean_per_timestep_rel_l2.numpy().tolist(),
            'per_timestep_nrmse': mean_per_timestep_nrmse.numpy().tolist(),
            'error_growth': error_growth,
            'per_sample_results': all_sample_results,
        }

    def _load_batch(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a batch of ICs and parameters from dataset."""
        with h5py.File(self.dataset_path, 'r') as f:
            ics = torch.from_numpy(f['inputs/fields'][indices]).float()
            # Extract first channel only if multi-channel (MNO expects 1 channel)
            if ics.shape[1] > 1:
                ics = ics[:, :1]  # [B, 1, H, W]

            # Try both parameter paths (different dataset versions)
            if 'inputs/sobol_parameters' in f:
                params = torch.from_numpy(f['inputs/sobol_parameters'][indices]).float()
            elif 'parameters/params' in f:
                params = torch.from_numpy(f['parameters/params'][indices]).float()
            else:
                raise KeyError("Could not find parameters in dataset (tried 'inputs/sobol_parameters' and 'parameters/params')")

        return ics, params

    def _mno_rollout(
        self,
        ics: torch.Tensor,
        params: torch.Tensor,
        num_timesteps: int
    ) -> torch.Tensor:
        """Generate MNO rollout from initial conditions.

        Args:
            ics: Initial conditions [B, C, H, W]
            params: Operator parameters [B, D]
            num_timesteps: Number of timesteps

        Returns:
            Rollout [B, T+1, C, H, W]
        """
        ics = ics.to(self.device)
        params = params.to(self.device)

        # Use MNO's rollout method which handles param conditioning
        rollout = self.mno.rollout(
            u0=ics,
            steps=num_timesteps,
            return_all_steps=True,
            params=params,
        )  # [B, T+1, C, H, W]

        return rollout

    def _cno_rollout(
        self,
        params: torch.Tensor,
        ics: torch.Tensor,
        num_timesteps: int,
        num_realizations: int = 3,
    ) -> torch.Tensor:
        """Generate CNO rollout averaged over realizations.

        Args:
            params: Parameter vectors [B, D]
            ics: Initial conditions [B, C, H, W]
            num_timesteps: Number of timesteps
            num_realizations: Number of stochastic realizations

        Returns:
            Averaged rollout [B, T+1, C, H, W]
        """
        B = ics.shape[0]
        all_rollouts = []

        for i in range(B):
            rollout = self.cno_replayer.rollout(
                params_vector=params[i].cpu().numpy(),
                ic=ics[i],
                timesteps=num_timesteps,
                num_realizations=num_realizations,
                return_all_steps=True,
            )  # [1, M, T+1, C, H, W]

            # Average over realizations
            averaged = rollout.mean(dim=1)  # [1, T+1, C, H, W]
            all_rollouts.append(averaged)

        return torch.cat(all_rollouts, dim=0)  # [B, T+1, C, H, W]

    def _save_rollouts(
        self,
        save_path: Path,
        mno_rollouts: torch.Tensor,
        cno_rollouts: torch.Tensor,
        indices: List[int],
    ):
        """Save rollouts to HDF5 file."""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(save_path, 'w') as f:
            f.create_dataset('mno_rollouts', data=mno_rollouts.numpy(), compression='gzip')
            f.create_dataset('cno_rollouts', data=cno_rollouts.numpy(), compression='gzip')
            f.create_dataset('sample_indices', data=np.array(indices))


class TokenizationEvaluator:
    """Evaluates tokenization quality and distribution shift."""

    def __init__(
        self,
        mno: NOABackbone,
        cno_replayer: CNOReplayer,
        vqvae: nn.Module,
        vqvae_checkpoint: Path,
        dataset_path: Path,
        device: torch.device,
    ):
        """Initialize tokenization evaluator.

        Args:
            mno: Loaded MNO model
            cno_replayer: CNO replayer
            vqvae: Loaded VQ-VAE model
            vqvae_checkpoint: Path to VQ-VAE checkpoint
            dataset_path: Path to dataset
            device: Computation device
        """
        self.mno = mno
        self.cno_replayer = cno_replayer
        self.vqvae = vqvae
        self.vqvae_checkpoint = vqvae_checkpoint
        self.dataset_path = dataset_path
        self.device = device

    def evaluate(
        self,
        indices: List[int],
        batch_size: int = 8,
        num_timesteps: int = 256,
        save_tokens_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Evaluate tokenization on given indices.

        Args:
            indices: Sample indices to evaluate
            batch_size: Batch size
            num_timesteps: Number of timesteps
            save_tokens_path: Optional path to save tokens

        Returns:
            Dictionary with tokenization metrics
        """
        # Generate MNO rollouts
        print("  Generating MNO rollouts...")
        mno_rollouts = self._generate_mno_rollouts(indices, batch_size, num_timesteps)

        # Generate CNO rollouts for baseline
        print("  Generating CNO rollouts...")
        cno_rollouts = self._generate_cno_rollouts(indices, batch_size, num_timesteps)

        # Extract features
        print("  Extracting features from rollouts...")
        mno_features = extract_features_from_rollouts(
            mno_rollouts,
            self.vqvae_checkpoint,
            device=str(self.device),
            batch_size=batch_size,
        )

        cno_features = extract_features_from_rollouts(
            cno_rollouts,
            self.vqvae_checkpoint,
            device=str(self.device),
            batch_size=batch_size,
        )

        # Tokenize
        print("  Tokenizing features...")
        mno_tokens_data = tokenize_features(
            mno_features,
            self.vqvae,
            device=str(self.device),
            batch_size=256,
            return_reconstructed=True,
        )

        cno_tokens_data = tokenize_features(
            cno_features,
            self.vqvae,
            device=str(self.device),
            batch_size=256,
            return_reconstructed=True,
        )

        # Compute metrics
        print("  Computing metrics...")

        # Get number of embeddings from VQ-VAE
        # Assuming all levels have same codebook size
        num_embeddings = self.vqvae.vq_layers[0].num_embeddings

        mno_metrics = compute_tokenization_metrics(
            features=mno_features,
            reconstructed=mno_tokens_data['reconstructed'],
            tokens=mno_tokens_data['tokens'],
            num_embeddings=num_embeddings,
        )

        cno_metrics = compute_tokenization_metrics(
            features=cno_features,
            reconstructed=cno_tokens_data['reconstructed'],
            tokens=cno_tokens_data['tokens'],
            num_embeddings=num_embeddings,
        )

        # Distribution comparison
        dist_metrics = compute_distribution_metrics(
            tokens_a=mno_tokens_data['tokens'].numpy(),
            tokens_b=cno_tokens_data['tokens'].numpy(),
            num_embeddings=num_embeddings,
            name_a="mno",
            name_b="cno",
        )

        # Save tokens if requested
        if save_tokens_path:
            self._save_tokens(
                save_tokens_path,
                mno_tokens_data,
                cno_tokens_data,
                mno_features,
                cno_features,
                indices,
            )

        return {
            'mno': mno_metrics,
            'cno': cno_metrics,
            'distribution_comparison': dist_metrics,
        }

    def _generate_mno_rollouts(
        self,
        indices: List[int],
        batch_size: int,
        num_timesteps: int,
    ) -> torch.Tensor:
        """Generate MNO rollouts for given indices."""
        num_samples = len(indices)
        num_batches = (num_samples + batch_size - 1) // batch_size
        all_rollouts = []

        with torch.no_grad():
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, num_samples)
                batch_indices = indices[batch_start:batch_end]

                # Load ICs and params
                with h5py.File(self.dataset_path, 'r') as f:
                    ics = torch.from_numpy(f['inputs/fields'][batch_indices]).float()
                    # Extract first channel only if multi-channel
                    if ics.shape[1] > 1:
                        ics = ics[:, :1]  # [B, 1, H, W]

                    # Try both parameter paths
                    if 'inputs/sobol_parameters' in f:
                        params = torch.from_numpy(f['inputs/sobol_parameters'][batch_indices]).float()
                    elif 'parameters/params' in f:
                        params = torch.from_numpy(f['parameters/params'][batch_indices]).float()
                    else:
                        raise KeyError("Could not find parameters in dataset")

                ics = ics.to(self.device)
                params = params.to(self.device)

                # Use MNO's rollout method which handles param conditioning
                rollout = self.mno.rollout(
                    u0=ics,
                    steps=num_timesteps,
                    return_all_steps=True,
                    params=params,
                )  # [B, T+1, C, H, W]

                all_rollouts.append(rollout.cpu())

        return torch.cat(all_rollouts, dim=0)

    def _generate_cno_rollouts(
        self,
        indices: List[int],
        batch_size: int,
        num_timesteps: int,
    ) -> torch.Tensor:
        """Generate CNO rollouts for given indices."""
        num_samples = len(indices)
        all_rollouts = []

        with h5py.File(self.dataset_path, 'r') as f:
            ics = torch.from_numpy(f['inputs/fields'][indices]).float()
            # Extract first channel only if multi-channel
            if ics.shape[1] > 1:
                ics = ics[:, :1]  # [N, 1, H, W]

            # Try both parameter paths (different dataset versions)
            if 'inputs/sobol_parameters' in f:
                params = torch.from_numpy(f['inputs/sobol_parameters'][indices]).float()
            elif 'parameters/params' in f:
                params = torch.from_numpy(f['parameters/params'][indices]).float()
            else:
                raise KeyError("Could not find parameters in dataset")

        for i in range(num_samples):
            rollout = self.cno_replayer.rollout(
                params_vector=params[i].cpu().numpy(),
                ic=ics[i],
                timesteps=num_timesteps,
                num_realizations=3,
                return_all_steps=True,
            )  # [1, M, T+1, C, H, W]

            # Average over realizations
            averaged = rollout.mean(dim=1)  # [1, T+1, C, H, W]
            all_rollouts.append(averaged)

        return torch.cat(all_rollouts, dim=0)

    def _save_tokens(
        self,
        save_path: Path,
        mno_tokens_data: Dict,
        cno_tokens_data: Dict,
        mno_features: torch.Tensor,
        cno_features: torch.Tensor,
        indices: List[int],
    ):
        """Save tokens and features to HDF5 file."""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(save_path, 'w') as f:
            # MNO data
            f.create_dataset('mno_tokens', data=mno_tokens_data['tokens'].numpy(), compression='gzip')
            f.create_dataset('mno_features', data=mno_features.numpy(), compression='gzip')
            f.create_dataset('mno_reconstructed', data=mno_tokens_data['reconstructed'].numpy(), compression='gzip')

            # CNO data
            f.create_dataset('cno_tokens', data=cno_tokens_data['tokens'].numpy(), compression='gzip')
            f.create_dataset('cno_features', data=cno_features.numpy(), compression='gzip')
            f.create_dataset('cno_reconstructed', data=cno_tokens_data['reconstructed'].numpy(), compression='gzip')

            # Metadata
            f.create_dataset('sample_indices', data=np.array(indices))
