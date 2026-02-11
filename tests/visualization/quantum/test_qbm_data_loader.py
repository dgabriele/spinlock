"""Tests for QBM data loader."""
import pytest
import h5py
import torch
import numpy as np
from pathlib import Path
from spinlock.visualization.quantum.data_loader import QBMDatasetLoader


@pytest.fixture
def mock_qbm_dataset(tmp_path):
    """Create a minimal mock QBM dataset."""
    dataset_path = tmp_path / "test_qbm.h5"

    with h5py.File(dataset_path, 'w') as f:
        # Create minimal QBM structure matching real datasets
        N, M, T, H, W = 10, 3, 20, 64, 64

        # Outputs: [N, M, T, 2, H, W] - complex wavefunctions
        f.create_dataset('outputs', data=np.random.randn(N, M, T, 2, H, W).astype(np.float32))

        # Inputs: [N, M, 2, H, W] - initial conditions
        f.create_dataset('inputs', data=np.random.randn(N, M, 2, H, W).astype(np.float32))

        # Temporal features: [N, T, 188] - includes quantum features at end
        f.create_dataset('features/temporal', data=np.random.randn(N, T, 188).astype(np.float32))

        # Parameters
        f.create_dataset('parameters/gamma', data=np.random.rand(N).astype(np.float32))
        f.create_dataset('parameters/nbar', data=np.random.rand(N).astype(np.float32))
        f.create_dataset('parameters/alpha_init', data=np.random.rand(N).astype(np.float32))

    return dataset_path


def test_loader_context_manager(mock_qbm_dataset):
    """Test loader context manager."""
    with QBMDatasetLoader(mock_qbm_dataset) as loader:
        assert loader._h5_file is not None

    # File should be closed after context
    # (can't easily test this without accessing private attributes)


def test_get_dataset_info(mock_qbm_dataset):
    """Test dataset info extraction."""
    with QBMDatasetLoader(mock_qbm_dataset) as loader:
        info = loader.get_dataset_info()

        assert info['num_samples'] == 10
        assert info['num_realizations'] == 3
        assert info['num_timesteps'] == 20
        assert info['num_channels'] == 2
        assert info['spatial_dims'] == (64, 64)
        assert info['temporal_features'] == 188


def test_load_rollouts_probability_format(mock_qbm_dataset):
    """Test loading rollouts as probability density."""
    with QBMDatasetLoader(mock_qbm_dataset) as loader:
        indices = torch.arange(5)
        rollouts = loader.load_rollouts(indices, output_format="probability")

        # Should be [N, M, T, H, W]
        assert rollouts.shape == (5, 3, 20, 64, 64)
        assert rollouts.dtype == torch.float32

        # All values should be non-negative (probability density)
        assert (rollouts >= 0).all()


def test_load_rollouts_complex_format(mock_qbm_dataset):
    """Test loading rollouts as complex tensor."""
    with QBMDatasetLoader(mock_qbm_dataset) as loader:
        indices = torch.arange(5)
        rollouts = loader.load_rollouts(indices, output_format="complex")

        # Should be [N, M, T, H, W] with complex dtype
        assert rollouts.shape == (5, 3, 20, 64, 64)
        assert rollouts.dtype == torch.complex64


def test_load_rollouts_real_imag_format(mock_qbm_dataset):
    """Test loading rollouts as separate real/imag channels."""
    with QBMDatasetLoader(mock_qbm_dataset) as loader:
        indices = torch.arange(5)
        rollouts = loader.load_rollouts(indices, output_format="real_imag")

        # Should be [N, M, T, 2, H, W]
        assert rollouts.shape == (5, 3, 20, 2, 64, 64)
        assert rollouts.dtype == torch.float32


def test_load_quantum_features(mock_qbm_dataset):
    """Test loading quantum features."""
    with QBMDatasetLoader(mock_qbm_dataset) as loader:
        indices = torch.arange(5)
        quantum_feats = loader.load_quantum_features(indices)

        # Should be [N, T, 10] - last 10 features from temporal (188 - 178 = 10)
        assert quantum_feats.shape == (5, 20, 10)
        assert quantum_feats.dtype == torch.float32


def test_load_parameters(mock_qbm_dataset):
    """Test loading QBM parameters."""
    with QBMDatasetLoader(mock_qbm_dataset) as loader:
        indices = torch.arange(5)
        params = loader.load_parameters(indices)

        assert 'gamma' in params
        assert 'nbar' in params
        assert 'alpha_init' in params

        assert params['gamma'].shape == (5,)
        assert params['nbar'].shape == (5,)
        assert params['alpha_init'].shape == (5,)


def test_load_inputs(mock_qbm_dataset):
    """Test loading initial conditions."""
    with QBMDatasetLoader(mock_qbm_dataset) as loader:
        indices = torch.arange(5)
        inputs = loader.load_inputs(indices, output_format="probability")

        # Should be [N, M, H, W]
        assert inputs.shape == (5, 3, 64, 64)
        assert (inputs >= 0).all()


def test_invalid_output_format(mock_qbm_dataset):
    """Test that invalid format raises error."""
    with QBMDatasetLoader(mock_qbm_dataset) as loader:
        indices = torch.arange(5)

        with pytest.raises(ValueError, match="Unknown format"):
            loader.load_rollouts(indices, output_format="invalid")


def test_file_not_found():
    """Test that missing file raises error."""
    with pytest.raises(FileNotFoundError):
        QBMDatasetLoader(Path("/nonexistent/path.h5"))


def test_consistency_probability_from_real_imag(mock_qbm_dataset):
    """Test that probability extraction is consistent."""
    with QBMDatasetLoader(mock_qbm_dataset) as loader:
        indices = torch.arange(5)

        # Load as probability
        prob_direct = loader.load_rollouts(indices, output_format="probability")

        # Load as real/imag and compute probability manually
        real_imag = loader.load_rollouts(indices, output_format="real_imag")
        prob_manual = real_imag[..., 0, :, :]**2 + real_imag[..., 1, :, :]**2

        # Should be identical
        assert torch.allclose(prob_direct, prob_manual, atol=1e-6)
