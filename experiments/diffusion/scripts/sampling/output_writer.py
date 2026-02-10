"""Output writer for sampled data."""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import h5py
import numpy as np
import torch


class SampleWriter:
    """Write samples to disk in HDF5 format.

    Args:
        output_dir: Directory to save outputs
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_samples(
        self,
        tokens: List[Dict[str, torch.Tensor]],
        theta: Optional[np.ndarray],
        u0: Optional[np.ndarray],
        trajectories: Optional[np.ndarray],
        metadata: Dict[str, Any]
    ):
        """Write all sample data to disk.

        Args:
            tokens: List of token dicts
            theta: Parameters [N, 14]
            u0: Initial conditions [N, C, H, W]
            trajectories: Rollouts [N, M, T+1, C, H, W]
            metadata: Generation metadata
        """
        hdf5_path = self.output_dir / "samples.h5"
        json_path = self.output_dir / "metadata.json"

        self._write_hdf5(hdf5_path, tokens, theta, u0, trajectories, metadata)
        self._write_json(json_path, metadata)

    def _write_hdf5(
        self,
        path: Path,
        tokens: List[Dict[str, torch.Tensor]],
        theta: Optional[np.ndarray],
        u0: Optional[np.ndarray],
        trajectories: Optional[np.ndarray],
        metadata: Dict[str, Any]
    ):
        """Write HDF5 file."""
        with h5py.File(path, 'w') as f:
            # Write tokens
            self._write_tokens(f, tokens)

            # Write decoded features
            if theta is not None or u0 is not None:
                decoded_group = f.create_group('decoded')
                if theta is not None:
                    decoded_group.create_dataset('theta', data=theta)
                if u0 is not None:
                    decoded_group.create_dataset('u0', data=u0)

            # Write trajectories
            if trajectories is not None:
                f.create_dataset('trajectories', data=trajectories)

            # Write metadata as attributes
            self._write_metadata_attrs(f, metadata)

    def _write_tokens(self, f: h5py.File, tokens: List[Dict[str, torch.Tensor]]):
        """Write token indices."""
        if not tokens:
            return

        tokens_group = f.create_group('tokens')

        # Stack tokens by key
        keys = tokens[0].keys()
        for key in keys:
            token_array = np.stack([
                t[key].numpy() if isinstance(t[key], torch.Tensor) else t[key]
                for t in tokens
            ])
            tokens_group.create_dataset(key, data=token_array)

    def _write_metadata_attrs(self, f: h5py.File, metadata: Dict[str, Any]):
        """Write metadata as HDF5 attributes."""
        for key, value in metadata.items():
            if isinstance(value, (int, float, str, bool)):
                f.attrs[key] = value
            elif isinstance(value, dict):
                f.attrs[key] = json.dumps(value)

    def _write_json(self, path: Path, metadata: Dict[str, Any]):
        """Write metadata JSON."""
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
