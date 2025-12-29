"""
HDF5 storage backend for neural operator datasets.

Provides efficient storage with:
- Chunked writes for streaming
- Compression (gzip, lzf)
- Metadata tracking
- Extensible schema

Design principles:
- Streaming writes: Handle datasets larger than RAM
- Compression: Reduce storage footprint
- Metadata: Full reproducibility
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, cast
from datetime import datetime
import json


class HDF5DatasetWriter:
    """
    Efficient HDF5 writer for neural operator datasets.

    Schema:
        /metadata/
            - config (JSON string)
            - creation_date
            - version
            - sampling_metrics (JSON string)
            - ic_types [N] - Initial condition type per sample (optional)
            - evolution_policies [N] - Evolution policy per sample (optional)
            - grid_sizes [N] - Grid size per sample (optional)
            - noise_regimes [N] - Noise regime classification (optional)
        /parameters/
            - params [N, P] - Parameter sets
        /inputs/
            - fields [N, C_in, H, W] - Input fields
        /outputs/
            - fields [N, M, C_out, H, W] - Output fields (M realizations)

    Example:
        ```python
        with HDF5DatasetWriter(
            output_path=Path("dataset.h5"),
            grid_size=64,
            input_channels=3,
            output_channels=3,
            num_realizations=10,
            num_parameter_sets=1000,
            track_ic_metadata=True  # Enable discovery metadata
        ) as writer:

            for batch_params, batch_inputs, batch_outputs in data_generator:
                writer.write_batch(
                    batch_params, batch_inputs, batch_outputs,
                    ic_types=["multiscale_grf", "localized", ...],
                    evolution_policies=["autoregressive", ...],
                    grid_sizes=[64, 64, ...],
                    noise_regimes=["low", "medium", ...]
                )

            writer.write_metadata({"config": config_dict, "metrics": metrics})
        ```
    """

    def __init__(
        self,
        output_path: Path,
        grid_size: int,
        input_channels: int,
        output_channels: int,
        num_realizations: int,
        num_parameter_sets: int,
        compression: str = "gzip",
        compression_opts: int = 4,
        chunk_size: Optional[int] = None,
        track_ic_metadata: bool = True,
        store_trajectories: bool = True,
        num_timesteps: int = 1,
    ):
        """
        Initialize HDF5 dataset writer.

        Args:
            output_path: Path to output HDF5 file
            grid_size: Spatial grid dimension
            input_channels: Number of input channels
            output_channels: Number of output channels
            num_realizations: Number of stochastic realizations per parameter set
            num_parameter_sets: Total number of parameter sets
            compression: Compression algorithm ("gzip", "lzf", "none")
            compression_opts: Compression level (0-9 for gzip)
            chunk_size: Chunk size for HDF5 (defaults to 100)
            track_ic_metadata: Enable IC-behavior metadata tracking for discovery
            store_trajectories: Whether to store raw trajectories (False = feature-only mode)
            num_timesteps: Number of timesteps (1 for snapshot mode, >1 for temporal mode)
        """
        self.output_path = output_path
        self.grid_size = grid_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_realizations = num_realizations
        self.num_parameter_sets = num_parameter_sets
        self.compression = compression if compression != "none" else None
        self.compression_opts = compression_opts if compression == "gzip" else None
        self.track_ic_metadata = track_ic_metadata
        self.store_trajectories = store_trajectories
        self.num_timesteps = num_timesteps

        # Optimal chunk size
        self.chunk_size = chunk_size or min(100, num_parameter_sets)

        self.file: Optional[h5py.File] = None
        self.current_idx = 0

        # Write buffering to reduce HDF5 I/O overhead
        self._write_buffer: Dict[str, List[np.ndarray]] = {
            "parameters": [],
            "inputs": [],
            "outputs": [],
            "ic_types": [],
            "evolution_policies": [],
            "grid_sizes": [],
            "noise_regimes": [],
        }
        self._buffer_size = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self) -> None:
        """Create HDF5 file and initialize datasets."""
        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.file = h5py.File(self.output_path, "w")

        # Metadata group
        meta = self.file.create_group("metadata")
        meta.attrs["creation_date"] = datetime.now().isoformat()
        meta.attrs["version"] = "1.0"
        meta.attrs["grid_size"] = self.grid_size
        meta.attrs["num_realizations"] = self.num_realizations
        meta.attrs["num_parameter_sets"] = self.num_parameter_sets
        meta.attrs["track_ic_metadata"] = self.track_ic_metadata

        # Discovery-focused metadata datasets (per-sample)
        if self.track_ic_metadata:
            meta.create_dataset(
                "ic_types",
                shape=(self.num_parameter_sets,),
                dtype=h5py.string_dtype(encoding='utf-8'),
                chunks=(self.chunk_size,),
            )
            meta.create_dataset(
                "evolution_policies",
                shape=(self.num_parameter_sets,),
                dtype=h5py.string_dtype(encoding='utf-8'),
                chunks=(self.chunk_size,),
            )
            meta.create_dataset(
                "grid_sizes",
                shape=(self.num_parameter_sets,),
                dtype=np.int32,
                chunks=(self.chunk_size,),
            )
            meta.create_dataset(
                "noise_regimes",
                shape=(self.num_parameter_sets,),
                dtype=h5py.string_dtype(encoding='utf-8'),
                chunks=(self.chunk_size,),
            )

        # Parameters group (created dynamically when we know dimensions)
        self.file.create_group("parameters")

        # Inputs group
        inputs_group = self.file.create_group("inputs")
        inputs_group.create_dataset(
            "fields",
            shape=(self.num_parameter_sets, self.input_channels, self.grid_size, self.grid_size),
            dtype=np.float32,
            chunks=(self.chunk_size, self.input_channels, self.grid_size, self.grid_size),
            compression=self.compression,
            compression_opts=self.compression_opts,
        )

        # Outputs group (only if storing trajectories)
        if self.store_trajectories:
            outputs_group = self.file.create_group("outputs")

            # Temporal mode (T > 1): shape [N, M, T, C, H, W]
            if self.num_timesteps > 1:
                outputs_group.create_dataset(
                    "fields",
                    shape=(
                        self.num_parameter_sets,
                        self.num_realizations,
                        self.num_timesteps,
                        self.output_channels,
                        self.grid_size,
                        self.grid_size,
                    ),
                    dtype=np.float32,
                    chunks=(self.chunk_size, 1, 1, self.output_channels, self.grid_size, self.grid_size),
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )
            # Snapshot mode (T = 1): shape [N, M, C, H, W]
            else:
                outputs_group.create_dataset(
                    "fields",
                    shape=(
                        self.num_parameter_sets,
                        self.num_realizations,
                        self.output_channels,
                        self.grid_size,
                        self.grid_size,
                    ),
                    dtype=np.float32,
                    chunks=(self.chunk_size, 1, self.output_channels, self.grid_size, self.grid_size),
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )
        else:
            # Feature-only mode: create placeholder group with metadata
            outputs_group = self.file.create_group("outputs")
            outputs_group.attrs["feature_only_mode"] = True
            outputs_group.attrs["note"] = (
                "Trajectories not stored. Feature-only mode enabled to save storage. "
                "Features extracted during generation and stored in /features/ group."
            )

    def write_batch(
        self,
        parameters: np.ndarray,
        inputs: Union[torch.Tensor, np.ndarray],
        outputs: Union[torch.Tensor, np.ndarray],
        ic_types: Optional[List[str]] = None,
        evolution_policies: Optional[List[str]] = None,
        grid_sizes: Optional[List[int]] = None,
        noise_regimes: Optional[List[str]] = None,
    ) -> None:
        """
        Write a batch of data to HDF5 with buffering for performance.

        Accumulates small batches in memory and flushes full chunks to HDF5,
        reducing lock contention and improving compression efficiency.

        Args:
            parameters: [B, P] parameter values
            inputs: [B, C_in, H, W] input fields
            outputs: [B, M, C_out, H, W] or [B, M, T, C_out, H, W] output fields
            ic_types: List of IC type strings (optional, length B)
            evolution_policies: List of evolution policy strings (optional, length B)
            grid_sizes: List of grid sizes (optional, length B)
            noise_regimes: List of noise regime strings (optional, length B)

        Example:
            ```python
            writer.write_batch(
                parameters=param_batch,  # (32, 13)
                inputs=input_batch,      # (32, 3, 64, 64)
                outputs=output_batch,    # (32, 10, 3, 64, 64)
                ic_types=["multiscale_grf", "localized", ...],
                evolution_policies=["autoregressive", ...],
                grid_sizes=[64, 64, ...],
                noise_regimes=["low", "medium", ...]
            )
            ```
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened. Use context manager or call open()")

        batch_size = parameters.shape[0]

        # Convert tensors to numpy
        inputs_np = inputs.cpu().numpy() if isinstance(inputs, torch.Tensor) else inputs
        outputs_np = outputs.cpu().numpy() if isinstance(outputs, torch.Tensor) else outputs

        # Create parameters dataset on first write
        param_group = cast(h5py.Group, self.file["parameters"])
        if "params" not in param_group:
            param_dim = parameters.shape[1]
            param_group.create_dataset(
                "params",
                shape=(self.num_parameter_sets, param_dim),
                dtype=np.float32,
                chunks=(self.chunk_size, param_dim),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

        # Accumulate in buffer
        self._write_buffer["parameters"].append(parameters)
        self._write_buffer["inputs"].append(inputs_np)
        self._write_buffer["outputs"].append(outputs_np)

        if ic_types is not None:
            if len(ic_types) != batch_size:
                raise ValueError(f"ic_types length {len(ic_types)} != batch_size {batch_size}")
            self._write_buffer["ic_types"].extend(ic_types)

        if evolution_policies is not None:
            if len(evolution_policies) != batch_size:
                raise ValueError(f"evolution_policies length {len(evolution_policies)} != batch_size {batch_size}")
            self._write_buffer["evolution_policies"].extend(evolution_policies)

        if grid_sizes is not None:
            if len(grid_sizes) != batch_size:
                raise ValueError(f"grid_sizes length {len(grid_sizes)} != batch_size {batch_size}")
            self._write_buffer["grid_sizes"].extend(grid_sizes)

        if noise_regimes is not None:
            if len(noise_regimes) != batch_size:
                raise ValueError(f"noise_regimes length {len(noise_regimes)} != batch_size {batch_size}")
            self._write_buffer["noise_regimes"].extend(noise_regimes)

        self._buffer_size += batch_size

        # Flush when buffer reaches chunk_size
        if self._buffer_size >= self.chunk_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush accumulated buffer to HDF5."""
        if self._buffer_size == 0:
            return

        # Concatenate buffered arrays
        parameters = np.concatenate(self._write_buffer["parameters"], axis=0)
        inputs_np = np.concatenate(self._write_buffer["inputs"], axis=0)
        outputs_np = np.concatenate(self._write_buffer["outputs"], axis=0)

        actual_batch_size = parameters.shape[0]
        end_idx = self.current_idx + actual_batch_size

        if end_idx > self.num_parameter_sets:
            raise ValueError(
                f"Writing {actual_batch_size} samples would exceed dataset size "
                f"({end_idx} > {self.num_parameter_sets})"
            )

        # Write to HDF5
        cast(h5py.Dataset, self.file["parameters/params"])[self.current_idx : end_idx] = parameters
        cast(h5py.Dataset, self.file["inputs/fields"])[self.current_idx : end_idx] = inputs_np

        # Only write trajectories if storing them (not feature-only mode)
        if self.store_trajectories:
            cast(h5py.Dataset, self.file["outputs/fields"])[self.current_idx : end_idx] = outputs_np

        # Write discovery metadata if tracking enabled
        if self.track_ic_metadata:
            meta_group = cast(h5py.Group, self.file["metadata"])

            if self._write_buffer["ic_types"]:
                ic_types = np.array(self._write_buffer["ic_types"], dtype=h5py.string_dtype())
                cast(h5py.Dataset, meta_group["ic_types"])[self.current_idx : end_idx] = ic_types

            if self._write_buffer["evolution_policies"]:
                evolution_policies = np.array(self._write_buffer["evolution_policies"], dtype=h5py.string_dtype())
                cast(h5py.Dataset, meta_group["evolution_policies"])[self.current_idx : end_idx] = evolution_policies

            if self._write_buffer["grid_sizes"]:
                grid_sizes = np.array(self._write_buffer["grid_sizes"], dtype=np.int32)
                cast(h5py.Dataset, meta_group["grid_sizes"])[self.current_idx : end_idx] = grid_sizes

            if self._write_buffer["noise_regimes"]:
                noise_regimes = np.array(self._write_buffer["noise_regimes"], dtype=h5py.string_dtype())
                cast(h5py.Dataset, meta_group["noise_regimes"])[self.current_idx : end_idx] = noise_regimes

        self.current_idx = end_idx

        # Clear buffer
        for key in self._write_buffer:
            self._write_buffer[key].clear()
        self._buffer_size = 0

        # Flush to disk periodically
        if self.current_idx % (self.chunk_size * 10) == 0:
            self.file.flush()

    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Write metadata to HDF5 file.

        Args:
            metadata: Dictionary of metadata to store

        Example:
            ```python
            writer.write_metadata({
                "config": config.model_dump(),
                "sampling_metrics": metrics_dict,
                "experiment_name": "test_run"
            })
            ```
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        meta_group = self.file["metadata"]

        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                # Store complex types as JSON
                meta_group.attrs[key] = json.dumps(value, indent=2)
            elif isinstance(value, (str, int, float, bool)):
                meta_group.attrs[key] = value
            else:
                # Try to convert to JSON
                try:
                    meta_group.attrs[key] = json.dumps(value, indent=2)
                except TypeError:
                    print(f"Warning: Could not serialize metadata key '{key}'")

    def close(self) -> None:
        """Close HDF5 file, flushing any remaining buffered data."""
        if self.file:
            # Flush any remaining buffered data
            self._flush_buffer()

            self.file.close()
            self.file = None


class HDF5DatasetReader:
    """
    Reader for neural operator HDF5 datasets.

    Example:
        ```python
        with HDF5DatasetReader(Path("dataset.h5")) as reader:
            metadata = reader.get_metadata()
            params = reader.get_parameters()
            inputs = reader.get_inputs(0)  # Get first sample
            outputs = reader.get_outputs(slice(0, 10))  # Get first 10 samples
        ```
    """

    def __init__(self, dataset_path: Path):
        """
        Initialize HDF5 dataset reader.

        Args:
            dataset_path: Path to HDF5 dataset
        """
        self.dataset_path = dataset_path
        self.file: Optional[h5py.File] = None

    def __enter__(self):
        self.file = h5py.File(self.dataset_path, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        metadata = dict(self.file["metadata"].attrs)

        # Parse JSON strings
        for key, value in metadata.items():
            if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                try:
                    metadata[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass

        return metadata

    def get_parameters(self, idx: Optional[Union[int, slice]] = None) -> np.ndarray:
        """
        Get parameter sets.

        Args:
            idx: Index or slice (None = all)

        Returns:
            Parameter array
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        dataset = cast(h5py.Dataset, self.file["parameters/params"])
        if idx is None:
            return dataset[:]  # type: ignore
        return dataset[idx]  # type: ignore

    def get_inputs(self, idx: Optional[Union[int, slice]] = None) -> np.ndarray:
        """
        Get input fields.

        Args:
            idx: Index or slice (None = all)

        Returns:
            Input field array
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        dataset = cast(h5py.Dataset, self.file["inputs/fields"])
        if idx is None:
            return dataset[:]  # type: ignore
        return dataset[idx]  # type: ignore

    def get_outputs(self, idx: Optional[Union[int, slice]] = None) -> np.ndarray:
        """
        Get output fields.

        Args:
            idx: Index or slice (None = all)

        Returns:
            Output field array
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        dataset = cast(h5py.Dataset, self.file["outputs/fields"])
        if idx is None:
            return dataset[:]  # type: ignore
        return dataset[idx]  # type: ignore

    @property
    def num_samples(self) -> int:
        """Total number of samples in dataset."""
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")
        dataset = cast(h5py.Dataset, self.file["parameters/params"])
        return dataset.shape[0]

    @property
    def shape_info(self) -> Dict[str, tuple]:
        """Get shapes of all datasets."""
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        return {
            "parameters": cast(h5py.Dataset, self.file["parameters/params"]).shape,
            "inputs": cast(h5py.Dataset, self.file["inputs/fields"]).shape,
            "outputs": cast(h5py.Dataset, self.file["outputs/fields"]).shape,
        }

    def get_ic_types(self, idx: Optional[Union[int, slice]] = None) -> Optional[np.ndarray]:
        """
        Get IC types metadata.

        Args:
            idx: Index or slice (None = all)

        Returns:
            IC types array or None if not tracked
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        meta_group = cast(h5py.Group, self.file["metadata"])
        if "ic_types" not in meta_group:
            return None

        dataset = cast(h5py.Dataset, meta_group["ic_types"])
        if idx is None:
            return dataset[:]  # type: ignore
        return dataset[idx]  # type: ignore

    def get_evolution_policies(self, idx: Optional[Union[int, slice]] = None) -> Optional[np.ndarray]:
        """
        Get evolution policies metadata.

        Args:
            idx: Index or slice (None = all)

        Returns:
            Evolution policies array or None if not tracked
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        meta_group = cast(h5py.Group, self.file["metadata"])
        if "evolution_policies" not in meta_group:
            return None

        dataset = cast(h5py.Dataset, meta_group["evolution_policies"])
        if idx is None:
            return dataset[:]  # type: ignore
        return dataset[idx]  # type: ignore

    def get_grid_sizes(self, idx: Optional[Union[int, slice]] = None) -> Optional[np.ndarray]:
        """
        Get grid sizes metadata.

        Args:
            idx: Index or slice (None = all)

        Returns:
            Grid sizes array or None if not tracked
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        meta_group = cast(h5py.Group, self.file["metadata"])
        if "grid_sizes" not in meta_group:
            return None

        dataset = cast(h5py.Dataset, meta_group["grid_sizes"])
        if idx is None:
            return dataset[:]  # type: ignore
        return dataset[idx]  # type: ignore

    def get_noise_regimes(self, idx: Optional[Union[int, slice]] = None) -> Optional[np.ndarray]:
        """
        Get noise regimes metadata.

        Args:
            idx: Index or slice (None = all)

        Returns:
            Noise regimes array or None if not tracked
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        meta_group = cast(h5py.Group, self.file["metadata"])
        if "noise_regimes" not in meta_group:
            return None

        dataset = cast(h5py.Dataset, meta_group["noise_regimes"])
        if idx is None:
            return dataset[:]  # type: ignore
        return dataset[idx]  # type: ignore
