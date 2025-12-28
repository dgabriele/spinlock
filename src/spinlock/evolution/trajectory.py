"""
HDF5 storage for temporal trajectories.

Provides efficient storage for evolved operator trajectories with metrics.
Follows the existing HDF5DatasetWriter pattern for consistency.

Schema:
    /trajectories/         [N_ops, M_real, T_steps, C, H, W]
    /metrics/
        energy             [N_ops, M_real, T_steps]
        entropy            [N_ops, M_real, T_steps]
        autocorrelation    [N_ops, M_real, T_steps]
        variance           [N_ops, M_real, T_steps]
    /metadata/
        policy             (JSON string)
        num_timesteps
        num_operators
        num_realizations
"""

import h5py
import torch
import json
from pathlib import Path
from typing import List, Dict, Any
from .metrics import TrajectoryMetrics


class TrajectoryWriter:
    """
    HDF5 writer for temporal trajectories with metrics.

    Context manager for safe file handling. Follows the existing
    HDF5DatasetWriter pattern.

    Example:
        ```python
        with TrajectoryWriter(
            output_path=Path("trajectories.h5"),
            num_operators=10,
            num_realizations=10,
            num_timesteps=100,
            grid_size=64,
            num_channels=3
        ) as writer:

            for op_idx in range(10):
                for real_idx in range(10):
                    trajectory, metrics = evolve_operator(...)
                    writer.write_trajectory(
                        op_idx,
                        real_idx,
                        trajectory,
                        metrics
                    )

            writer.write_metadata({
                "policy": "convex",
                "alpha": 0.7
            })
        ```
    """

    def __init__(
        self,
        output_path: Path,
        num_operators: int,
        num_realizations: int,
        num_timesteps: int,
        grid_size: int,
        num_channels: int,
        compression: str = "gzip",
        compression_level: int = 4
    ):
        """
        Initialize trajectory writer.

        Args:
            output_path: Path to output HDF5 file
            num_operators: Number of operators (N)
            num_realizations: Number of realizations per operator (M)
            num_timesteps: Number of timesteps (T)
            grid_size: Spatial grid dimension (H=W)
            num_channels: Number of channels (C)
            compression: Compression algorithm ("gzip", "lzf", "none")
            compression_level: Compression level (0-9 for gzip)
        """
        self.output_path = output_path
        self.num_operators = num_operators
        self.num_realizations = num_realizations
        self.num_timesteps = num_timesteps
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.compression = compression if compression != "none" else None
        self.compression_opts = compression_level if compression == "gzip" else None

        self.file = None

    def __enter__(self):
        """Open HDF5 file and create datasets."""
        self.file = h5py.File(self.output_path, "w")

        # Create trajectories dataset [N, M, T, C, H, W]
        self.file.create_dataset(
            "trajectories",
            shape=(
                self.num_operators,
                self.num_realizations,
                self.num_timesteps,
                self.num_channels,
                self.grid_size,
                self.grid_size
            ),
            dtype='float32',
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=(1, 1, self.num_timesteps, self.num_channels, self.grid_size, self.grid_size)
        )

        # Create metrics group and datasets [N, M, T]
        metrics_group = self.file.create_group("metrics")
        metric_names = ["energy", "entropy", "autocorrelation", "variance", "mean_magnitude"]

        for metric_name in metric_names:
            metrics_group.create_dataset(
                metric_name,
                shape=(self.num_operators, self.num_realizations, self.num_timesteps),
                dtype='float32',
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=(1, 1, self.num_timesteps)
            )

        # Create metadata group
        self.file.create_group("metadata")

        return self

    def write_trajectory(
        self,
        operator_idx: int,
        realization_idx: int,
        trajectory: torch.Tensor,  # [T, C, H, W]
        metrics: List[TrajectoryMetrics]
    ):
        """
        Write single trajectory and associated metrics.

        Args:
            operator_idx: Operator index (0 to N-1)
            realization_idx: Realization index (0 to M-1)
            trajectory: Trajectory tensor [T, C, H, W]
            metrics: List of TrajectoryMetrics (length T)

        Raises:
            ValueError: If indices are out of bounds or shapes mismatch
        """
        if operator_idx >= self.num_operators:
            raise ValueError(f"Operator index {operator_idx} >= {self.num_operators}")

        if realization_idx >= self.num_realizations:
            raise ValueError(f"Realization index {realization_idx} >= {self.num_realizations}")

        if trajectory.shape[0] != self.num_timesteps:
            raise ValueError(
                f"Trajectory timesteps {trajectory.shape[0]} != {self.num_timesteps}"
            )

        if len(metrics) != self.num_timesteps:
            raise ValueError(
                f"Metrics length {len(metrics)} != {self.num_timesteps}"
            )

        # Write trajectory
        traj_np = trajectory.cpu().numpy()
        self.file["trajectories"][operator_idx, realization_idx] = traj_np

        # Write metrics
        for t, m in enumerate(metrics):
            self.file["metrics/energy"][operator_idx, realization_idx, t] = m.energy
            self.file["metrics/entropy"][operator_idx, realization_idx, t] = m.entropy
            self.file["metrics/autocorrelation"][operator_idx, realization_idx, t] = m.autocorrelation
            self.file["metrics/variance"][operator_idx, realization_idx, t] = m.variance
            self.file["metrics/mean_magnitude"][operator_idx, realization_idx, t] = m.mean_magnitude

    def write_metadata(self, metadata: Dict[str, Any]):
        """
        Write evolution metadata.

        Args:
            metadata: Dictionary of metadata to store

        Example:
            ```python
            writer.write_metadata({
                "policy": "convex",
                "alpha": 0.7,
                "num_timesteps": 100,
                "device": "cuda"
            })
            ```
        """
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                # Serialize complex types as JSON
                self.file["metadata"].attrs[key] = json.dumps(value)
            elif isinstance(value, Path):
                # Convert paths to strings
                self.file["metadata"].attrs[key] = str(value)
            else:
                # Store simple types directly
                self.file["metadata"].attrs[key] = value

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close HDF5 file."""
        if self.file:
            self.file.close()


class TrajectoryReader:
    """
    HDF5 reader for temporal trajectories.

    Context manager for safe file reading. Provides convenient access
    to stored trajectories and metrics.

    Example:
        ```python
        with TrajectoryReader(Path("trajectories.h5")) as reader:
            metadata = reader.get_metadata()
            trajectory = reader.get_trajectory(op_idx=0, real_idx=0)
            metrics = reader.get_metrics(op_idx=0, real_idx=0)
        ```
    """

    def __init__(self, file_path: Path):
        """
        Initialize trajectory reader.

        Args:
            file_path: Path to HDF5 trajectory file
        """
        self.file_path = file_path
        self.file = None

    def __enter__(self):
        """Open HDF5 file."""
        self.file = h5py.File(self.file_path, "r")
        return self

    def get_trajectory(
        self,
        operator_idx: int,
        realization_idx: int
    ) -> torch.Tensor:
        """
        Get trajectory for specific operator and realization.

        Args:
            operator_idx: Operator index
            realization_idx: Realization index

        Returns:
            Trajectory tensor [T, C, H, W]
        """
        traj_np = self.file["trajectories"][operator_idx, realization_idx]
        return torch.from_numpy(traj_np)

    def get_metrics(
        self,
        operator_idx: int,
        realization_idx: int
    ) -> List[TrajectoryMetrics]:
        """
        Get metrics for specific operator and realization.

        Args:
            operator_idx: Operator index
            realization_idx: Realization index

        Returns:
            List of TrajectoryMetrics (length T)
        """
        T = self.file["trajectories"].shape[2]

        metrics = []
        for t in range(T):
            m = TrajectoryMetrics(
                energy=float(self.file["metrics/energy"][operator_idx, realization_idx, t]),
                entropy=float(self.file["metrics/entropy"][operator_idx, realization_idx, t]),
                autocorrelation=float(self.file["metrics/autocorrelation"][operator_idx, realization_idx, t]),
                variance=float(self.file["metrics/variance"][operator_idx, realization_idx, t]),
                mean_magnitude=float(self.file["metrics/mean_magnitude"][operator_idx, realization_idx, t])
            )
            metrics.append(m)

        return metrics

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get evolution metadata.

        Returns:
            Dictionary of metadata
        """
        metadata = {}
        for key, value in self.file["metadata"].attrs.items():
            # Try to parse JSON strings
            if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                try:
                    metadata[key] = json.loads(value)
                except json.JSONDecodeError:
                    metadata[key] = value
            else:
                metadata[key] = value

        return metadata

    @property
    def shape_info(self) -> Dict[str, int]:
        """Get dataset shape information."""
        traj_shape = self.file["trajectories"].shape
        return {
            "num_operators": traj_shape[0],
            "num_realizations": traj_shape[1],
            "num_timesteps": traj_shape[2],
            "num_channels": traj_shape[3],
            "grid_size": traj_shape[4]
        }

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close HDF5 file."""
        if self.file:
            self.file.close()
