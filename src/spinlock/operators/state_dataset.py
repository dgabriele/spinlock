"""Dataset for loading ICs and parameter vectors for CNO replay.

Used in Stage 1 meta-operator training to load initial conditions
and Sobol parameter vectors for CNO target trajectory generation.

Supports stratified subsampling to preserve Sobol sequence properties
when training on a subset of the full dataset.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import h5py


class NOAStateDataset(Dataset):
    """Dataset that loads ICs and parameter vectors for CNO replay.

    Loads from HDF5 datasets with shape [N, M, C, H, W]:
    - N: number of samples (operators)
    - M: number of realizations per sample
    - C: number of channels (read from metadata/num_channels)
    - H, W: spatial dimensions

    Provides:
    - ic: Initial condition [C, H, W]
    - params: Sobol parameter vector [d,] for CNO reconstruction
    - sample_idx: Original index in the dataset (for debugging/tracking)

    This dataset is used in Stage 1 training where we need to:
    1. Load an initial condition
    2. Use NOA to predict future states
    3. Use CNO to generate target trajectories from parameters
    4. Compare NOA predictions vs CNO targets

    Sampling Strategies:
    - "stratified": Sample uniformly across the Sobol sequence (preserves space-filling)
    - "random": Random sampling with fixed seed (reproducible but loses structure)
    - "sequential": Take first n samples (legacy behavior, NOT recommended for subsampling)

    The Sobol sequence used to generate the dataset has optimal space-filling
    properties across the FULL sequence. When subsampling, "stratified" preserves
    this property by taking evenly-spaced samples, while "sequential" breaks it
    by clustering samples at the start of the sequence.

    Example:
        >>> # Stratified subsampling (recommended for training)
        >>> dataset = NOAStateDataset(
        ...     "datasets/100k.h5",
        ...     max_samples=1000,
        ...     sampling_strategy="stratified"
        ... )
        >>> # Uses samples [0, 100, 200, ..., 99900] - uniform coverage
    """

    def __init__(
        self,
        dataset_path: str,
        max_samples: int | None = None,
        realization_idx: int = 0,
        sampling_strategy: str = "stratified",
        random_seed: int = 42,
    ):
        """Initialize dataset with stratified subsampling.

        Args:
            dataset_path: Path to HDF5 dataset
            max_samples: Maximum number of samples to load (None = all)
            realization_idx: Which realization to use for IC (0 to M-1)
            sampling_strategy: How to subsample when max_samples < total
                - "stratified": Uniform spacing across sequence (default, recommended)
                - "random": Random sampling with fixed seed (reproducible)
                - "sequential": First n samples (legacy, breaks Sobol properties)
            random_seed: Seed for random sampling strategy (default: 42)
        """
        self.dataset_path = Path(dataset_path)
        self.realization_idx = realization_idx
        self.sampling_strategy = sampling_strategy

        with h5py.File(self.dataset_path, "r") as f:
            total = f["inputs/fields"].shape[0]
            n = max_samples if max_samples is not None else total
            n = min(n, total)

            # Determine indices based on sampling strategy
            if n == total:
                # Using full dataset - no subsampling needed
                indices = np.arange(n)
            elif sampling_strategy == "stratified":
                # Stratified: uniformly spaced across the full Sobol sequence
                # WARNING: Breaks Sobol space-filling due to stride sampling
                # (Sobol sequences are prefix-optimal, stride breaks this property)
                stride = total // n
                indices = np.arange(0, total, stride)[:n]
                print(f"  Using stratified sampling: {n} samples with stride {stride} from {total} total")
                print(f"  ⚠️  WARNING: Stride sampling breaks Sobol prefix-optimality!")
                print(f"  ⚠️  Consider using sampling_strategy='sequential' for better coverage.")
            elif sampling_strategy == "random":
                # Random: reproducible random subset
                # Loses Sobol structure but avoids prefix clustering
                np.random.seed(random_seed)
                indices = np.random.choice(total, size=n, replace=False)
                indices = np.sort(indices)  # Sort for HDF5 cache efficiency
                print(f"  Using random sampling: {n} samples from {total} total (seed={random_seed})")
            elif sampling_strategy == "sequential":
                # Sequential: first n samples (prefix-optimal for Sobol sequences)
                # CORRECT: Preserves Sobol space-filling properties
                # (Sobol sequences fill space progressively - first N samples have optimal coverage)
                indices = np.arange(n)
                print(f"  Using sequential sampling: first {n} samples from {total} total")
                print(f"  ✓ Sequential sampling preserves Sobol prefix-optimality (recommended)")
            else:
                raise ValueError(
                    f"Unknown sampling_strategy: '{sampling_strategy}'. "
                    f"Must be one of: 'stratified', 'random', 'sequential'"
                )

            # Store original indices for debugging/tracking
            self.indices = indices

            # Load ICs using selected indices
            # Note: Fancy indexing with sorted indices is cache-friendly
            # New standard: Dataset has shape [N, M, C, H, W] with explicit channel dimension
            # Select one realization - result is [N', C, H, W]
            inputs = f["inputs/fields"][indices, realization_idx, :, :, :]

            # Load Sobol parameter vectors for CNO replay
            params = f["parameters/params"][indices]

            # Check how many samples are actually populated (non-zero)
            # This handles cases where HDF5 file is pre-allocated but only partially filled
            sample_norms = np.linalg.norm(inputs.reshape(inputs.shape[0], -1), axis=1)
            populated_mask = sample_norms > 1e-10  # Threshold for "non-zero"
            num_populated = populated_mask.sum()
            num_allocated = inputs.shape[0]

            # Trim to populated samples only
            if num_populated < num_allocated:
                inputs = inputs[populated_mask]
                params = params[populated_mask]
                indices = indices[populated_mask]
                print(f"  Dataset pre-allocated to {num_allocated} but only {num_populated} samples populated")
                print(f"  Trimmed to {num_populated} non-zero samples")

            self.ics = torch.from_numpy(inputs).float()  # [N', C, H, W] - keep as-is
            self.params = torch.from_numpy(params).float()
            self.indices = indices

        # Update n_samples to reflect actual populated samples
        self.n_samples = self.ics.shape[0]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.

        Returns:
            Dictionary with:
                'ic': Initial condition [C, H, W] where C is dataset channel count
                'params': Sobol parameter vector [d,]
                'sample_idx': Original index in the full dataset (for debugging/tracking)
        """
        return {
            "ic": self.ics[idx],
            "params": self.params[idx],
            "sample_idx": int(self.indices[idx]),  # Return original dataset index
        }
