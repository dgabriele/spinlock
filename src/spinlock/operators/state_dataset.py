"""Dataset for loading ICs and parameter vectors for CNO replay.

Used in Stage 1 meta-operator training to load initial conditions
and Sobol parameter vectors for CNO target trajectory generation.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import h5py


class NOAStateDataset(Dataset):
    """Dataset that loads ICs and parameter vectors for CNO replay.

    Provides:
    - ic: Initial condition [C, H, W]
    - params: Sobol parameter vector [d,] for CNO reconstruction
    - sample_idx: Original index (for debugging/tracking)

    This dataset is used in Stage 1 training where we need to:
    1. Load an initial condition
    2. Use NOA to predict future states
    3. Use CNO to generate target trajectories from parameters
    4. Compare NOA predictions vs CNO targets
    """

    def __init__(
        self,
        dataset_path: str,
        max_samples: int | None = None,
        realization_idx: int = 0,
    ):
        """Initialize dataset.

        Args:
            dataset_path: Path to HDF5 dataset
            max_samples: Maximum number of samples to load (None = all)
            realization_idx: Which realization to use for IC (0 to M-1)
        """
        self.dataset_path = Path(dataset_path)
        self.realization_idx = realization_idx

        with h5py.File(self.dataset_path, "r") as f:
            total = f["inputs/fields"].shape[0]
            n = max_samples if max_samples is not None else total
            n = min(n, total)

            # Load ICs: [N, M, H, W] -> [N, 1, H, W] using specified realization
            inputs = f["inputs/fields"][:n, realization_idx, :, :]
            self.ics = torch.from_numpy(inputs).float().unsqueeze(1)  # Add channel dim

            # Load Sobol parameter vectors for CNO replay
            self.params = torch.from_numpy(f["parameters/params"][:n]).float()

        self.n_samples = n

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.

        Returns:
            Dictionary with:
                'ic': Initial condition [C, H, W]
                'params': Sobol parameter vector [d,]
                'sample_idx': Original index for debugging
        """
        return {
            "ic": self.ics[idx],
            "params": self.params[idx],
            "sample_idx": idx,
        }
