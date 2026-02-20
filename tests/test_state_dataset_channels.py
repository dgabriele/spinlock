"""Unit tests for SpinlockDataset — multi-channel loading, sampling strategies."""

import pytest
import h5py
import torch
import tempfile
import numpy as np
from pathlib import Path
from spinlock.data import SpinlockDataset


def _make_hdf5(path, n=100, m=3, c=3, h=64, w=64, p=14):
    """Create a valid Spinlock HDF5 file with required metadata attributes."""
    with h5py.File(path, 'w') as f:
        ds = f.create_dataset('inputs/fields', data=np.random.randn(n, m, c, h, w))
        ds.attrs['format'] = 'NMCHW'
        ds.attrs['num_channels'] = c
        ds.attrs['num_realizations'] = m
        f.create_dataset('parameters/params', data=np.random.randn(n, p))


def test_single_channel_dataset():
    """Test single-channel format [N, M, C=1, H, W] loads correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_single.h5"
        _make_hdf5(dataset_path, c=1)

        dataset = SpinlockDataset(dataset_path, max_samples=10)
        sample = dataset[0]
        assert sample['ic'].shape == (1, 64, 64), f"Expected (1, 64, 64), got {sample['ic'].shape}"
        assert sample['params'].shape == (14,), f"Expected (14,), got {sample['params'].shape}"
        assert 'sample_idx' in sample, "sample_idx missing from sample"


def test_multi_channel_dataset():
    """Test multi-channel format [N, M, C=3, H, W] loads correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_multi.h5"
        _make_hdf5(dataset_path, c=3)

        dataset = SpinlockDataset(dataset_path, max_samples=10)
        sample = dataset[0]
        assert sample['ic'].shape == (3, 64, 64), f"Expected (3, 64, 64), got {sample['ic'].shape}"
        assert sample['params'].shape == (14,), f"Expected (14,), got {sample['params'].shape}"
        assert 'sample_idx' in sample, "sample_idx missing from sample"


def test_dataset_different_realizations():
    """Test that different realizations produce different ICs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_realizations.h5"
        _make_hdf5(dataset_path, n=10, m=5, c=3)

        dataset_real0 = SpinlockDataset(dataset_path, max_samples=10, realization_idx=0)
        dataset_real1 = SpinlockDataset(dataset_path, max_samples=10, realization_idx=1)

        sample_real0 = dataset_real0[0]
        sample_real1 = dataset_real1[0]

        assert torch.allclose(sample_real0['params'], sample_real1['params']), \
            "Parameters should be identical across realizations"
        assert not torch.allclose(sample_real0['ic'], sample_real1['ic']), \
            "ICs should differ for different realizations"


def test_100k_baseline_dataset():
    """Test 50k baseline dataset loads correctly."""
    dataset_path = Path("datasets/50k_baseline.h5")
    if not dataset_path.exists():
        pytest.skip("Dataset not found")

    dataset = SpinlockDataset(dataset_path, max_samples=10)
    sample = dataset[0]
    assert sample['ic'].shape == (3, 64, 64), f"Expected (3, 64, 64), got {sample['ic'].shape}"
    assert sample['params'].shape[0] == 14, f"Expected 14 params, got {sample['params'].shape[0]}"
    assert 'sample_idx' in sample, "sample_idx missing from sample"


def test_stratified_sampling():
    """Test stratified sampling produces uniformly spaced indices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_stratified.h5"
        _make_hdf5(dataset_path, n=1000)

        dataset = SpinlockDataset(dataset_path, max_samples=10, sampling_strategy="stratified")

        indices = dataset.indices
        expected_stride = 1000 // 10
        expected_indices = np.arange(0, 1000, expected_stride)[:10]
        np.testing.assert_array_equal(indices, expected_indices)


def test_sequential_sampling():
    """Test sequential sampling takes first n samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_sequential.h5"
        _make_hdf5(dataset_path, n=1000)

        dataset = SpinlockDataset(dataset_path, max_samples=10, sampling_strategy="sequential")

        indices = dataset.indices
        np.testing.assert_array_equal(indices, np.arange(10))


def test_random_sampling():
    """Test random sampling is reproducible with fixed seed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_random.h5"
        _make_hdf5(dataset_path, n=1000)

        dataset1 = SpinlockDataset(dataset_path, max_samples=10, sampling_strategy="random", random_seed=42)
        dataset2 = SpinlockDataset(dataset_path, max_samples=10, sampling_strategy="random", random_seed=42)
        np.testing.assert_array_equal(dataset1.indices, dataset2.indices)

        dataset3 = SpinlockDataset(dataset_path, max_samples=10, sampling_strategy="random", random_seed=123)
        assert not np.array_equal(dataset1.indices, dataset3.indices)


def test_introspection_only():
    """Test SpinlockDataset introspection mode (no data loading)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_introspect.h5"
        _make_hdf5(dataset_path, n=100, c=3, p=14)

        ds = SpinlockDataset(dataset_path)  # no max_samples → introspection only
        assert ds.num_channels == 3
        assert ds.theta_param_dim == 14
        assert not ds._dataset_mode
