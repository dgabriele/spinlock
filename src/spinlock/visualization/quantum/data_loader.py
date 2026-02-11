"""QBM dataset loading and preprocessing for visualization."""
import h5py
import torch
from typing import Tuple, Optional, Literal
from pathlib import Path


class QBMDatasetLoader:
    """Load QBM datasets with complex wavefunction handling.

    Converts [N, M, 2, H, W] format (real/imag) to complex tensors
    or extracts probability densities |ψ|² for rendering.

    Context manager for safe HDF5 file handling.
    """

    def __init__(self, dataset_path: Path):
        """Initialize loader with dataset path.

        Args:
            dataset_path: Path to HDF5 dataset file
        """
        self.dataset_path = Path(dataset_path)
        self._h5_file: Optional[h5py.File] = None

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    def __enter__(self):
        """Open HDF5 file for reading."""
        self._h5_file = h5py.File(self.dataset_path, 'r')
        return self

    def __exit__(self, *args):
        """Close HDF5 file."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def get_dataset_info(self) -> dict:
        """Get dataset shape and metadata.

        Returns:
            Dictionary with dataset dimensions and structure
        """
        if self._h5_file is None:
            raise RuntimeError("Dataset not opened. Use context manager.")

        outputs_shape = self._h5_file['outputs'].shape
        temporal_shape = self._h5_file['features/temporal'].shape

        return {
            'num_samples': outputs_shape[0],
            'num_realizations': outputs_shape[1],
            'num_timesteps': outputs_shape[2],
            'num_channels': outputs_shape[3],
            'spatial_dims': outputs_shape[4:],
            'temporal_features': temporal_shape[2] if len(temporal_shape) > 2 else 0,
        }

    def load_rollouts(
        self,
        indices: torch.Tensor,
        output_format: Literal["probability", "complex", "real_imag"] = "probability"
    ) -> torch.Tensor:
        """Load QBM rollouts in specified format.

        Args:
            indices: Sample indices to load [N]
            output_format:
                - "probability": |ψ|² [N, M, T, H, W]
                - "complex": Complex tensor [N, M, T, H, W] (dtype=complex64)
                - "real_imag": Separate channels [N, M, T, 2, H, W]

        Returns:
            Rollouts in requested format
        """
        if self._h5_file is None:
            raise RuntimeError("Dataset not opened. Use context manager.")

        # Convert indices to numpy for HDF5 indexing
        indices_np = indices.cpu().numpy()

        # Load from HDF5
        raw_data = self._h5_file['outputs'][indices_np]  # [N, M, T, 2, H, W]
        tensor = torch.from_numpy(raw_data).float()

        if output_format == "probability":
            # |ψ|² = real² + imag²
            real, imag = tensor[..., 0, :, :], tensor[..., 1, :, :]
            return real**2 + imag**2  # [N, M, T, H, W]

        elif output_format == "complex":
            # Convert to complex tensor
            real, imag = tensor[..., 0, :, :], tensor[..., 1, :, :]
            return torch.complex(real, imag)  # [N, M, T, H, W] complex64

        elif output_format == "real_imag":
            return tensor  # [N, M, T, 2, H, W]

        else:
            raise ValueError(f"Unknown format: {output_format}")

    def load_quantum_features(self, indices: torch.Tensor) -> torch.Tensor:
        """Load quantum-specific features (purity, entropy, coherence).

        Args:
            indices: Sample indices [N]

        Returns:
            Quantum features [N, T, ~10] (last quantum dims of temporal)
        """
        if self._h5_file is None:
            raise RuntimeError("Dataset not opened. Use context manager.")

        # Convert indices to numpy
        indices_np = indices.cpu().numpy()

        # Load full temporal features
        temporal = self._h5_file['features/temporal'][indices_np]  # [N, T, 188]

        # Quantum features start at index 178 (last 10 features)
        quantum_start_idx = 178
        return torch.from_numpy(temporal[:, :, quantum_start_idx:]).float()

    def load_parameters(self, indices: torch.Tensor) -> dict:
        """Load QBM parameters (gamma, nbar, alpha_init).

        Args:
            indices: Sample indices [N]

        Returns:
            Dictionary with gamma, nbar, alpha_init tensors
        """
        if self._h5_file is None:
            raise RuntimeError("Dataset not opened. Use context manager.")

        # Convert indices to numpy
        indices_np = indices.cpu().numpy()

        return {
            'gamma': torch.from_numpy(self._h5_file['parameters/gamma'][indices_np]).float(),
            'nbar': torch.from_numpy(self._h5_file['parameters/nbar'][indices_np]).float(),
            'alpha_init': torch.from_numpy(self._h5_file['parameters/alpha_init'][indices_np]).float(),
        }

    def load_inputs(self, indices: torch.Tensor, output_format: str = "probability") -> torch.Tensor:
        """Load initial conditions.

        Args:
            indices: Sample indices [N]
            output_format: Same as load_rollouts

        Returns:
            Initial conditions in requested format
        """
        if self._h5_file is None:
            raise RuntimeError("Dataset not opened. Use context manager.")

        indices_np = indices.cpu().numpy()
        raw_data = self._h5_file['inputs'][indices_np]  # [N, M, 2, H, W]
        tensor = torch.from_numpy(raw_data).float()

        if output_format == "probability":
            real, imag = tensor[..., 0, :, :], tensor[..., 1, :, :]
            return real**2 + imag**2
        elif output_format == "complex":
            real, imag = tensor[..., 0, :, :], tensor[..., 1, :, :]
            return torch.complex(real, imag)
        else:
            return tensor
