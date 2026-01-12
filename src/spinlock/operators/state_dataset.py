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
                # Preserves space-filling properties of the Sobol sequence
                stride = total // n
                indices = np.arange(0, total, stride)[:n]
                print(f"  Using stratified sampling: {n} samples with stride {stride} from {total} total")
            elif sampling_strategy == "random":
                # Random: reproducible random subset
                # Loses Sobol structure but avoids prefix clustering
                np.random.seed(random_seed)
                indices = np.random.choice(total, size=n, replace=False)
                indices = np.sort(indices)  # Sort for HDF5 cache efficiency
                print(f"  Using random sampling: {n} samples from {total} total (seed={random_seed})")
            elif sampling_strategy == "sequential":
                # Sequential: first n samples (legacy behavior)
                # WARNING: Breaks Sobol space-filling properties for subsamples
                indices = np.arange(n)
                print(f"  Using sequential sampling: first {n} samples from {total} total")
                print(f"  ⚠️  WARNING: Sequential sampling breaks Sobol properties for subsamples!")
                print(f"  ⚠️  Consider using sampling_strategy='stratified' instead.")
            else:
                raise ValueError(
                    f"Unknown sampling_strategy: '{sampling_strategy}'. "
                    f"Must be one of: 'stratified', 'random', 'sequential'"
                )

            # Store original indices for debugging/tracking
            self.indices = indices

            # Load ICs using selected indices
            # Note: Fancy indexing with sorted indices is cache-friendly
            inputs = f["inputs/fields"][indices, realization_idx, :, :]
            self.ics = torch.from_numpy(inputs).float().unsqueeze(1)  # Add channel dim

            # Load Sobol parameter vectors for CNO replay
            self.params = torch.from_numpy(f["parameters/params"][indices]).float()

        self.n_samples = n

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.

        Returns:
            Dictionary with:
                'ic': Initial condition [C, H, W]
                'params': Sobol parameter vector [d,]
                'sample_idx': Original index in the full dataset (for debugging/tracking)
        """
        return {
            "ic": self.ics[idx],
            "params": self.params[idx],
            "sample_idx": int(self.indices[idx]),  # Return original dataset index
        }
