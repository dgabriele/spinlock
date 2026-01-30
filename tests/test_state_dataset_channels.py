"""Unit tests for multi-channel dataset loading."""

import pytest
import h5py
import torch
import tempfile
import numpy as np
from pathlib import Path
from spinlock.operators.state_dataset import NOAStateDataset


def test_single_channel_dataset():
    """Test single-channel format [N, M, C=1, H, W] loads correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_single.h5"
        with h5py.File(dataset_path, 'w') as f:
            # New standard format with explicit C=1 dimension
            f.create_dataset('inputs/fields', data=np.random.randn(100, 3, 1, 64, 64))
            f.create_dataset('parameters/params', data=np.random.randn(100, 14))
            f.create_group('metadata')
            f['metadata'].attrs['num_channels'] = 1

        dataset = NOAStateDataset(dataset_path, max_samples=10)
        sample = dataset[0]
        assert sample['ic'].shape == (1, 64, 64), f"Expected (1, 64, 64), got {sample['ic'].shape}"
        assert sample['params'].shape == (14,), f"Expected (14,), got {sample['params'].shape}"
        assert 'sample_idx' in sample, "sample_idx missing from sample"


def test_multi_channel_dataset():
    """Test multi-channel format [N, M, C=3, H, W] loads correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_multi.h5"
        with h5py.File(dataset_path, 'w') as f:
            # Multi-channel format with C=3 dimension
            f.create_dataset('inputs/fields', data=np.random.randn(100, 3, 3, 64, 64))
            f.create_dataset('parameters/params', data=np.random.randn(100, 14))
            f.create_group('metadata')
            f['metadata'].attrs['num_channels'] = 3

        dataset = NOAStateDataset(dataset_path, max_samples=10)
        sample = dataset[0]
        assert sample['ic'].shape == (3, 64, 64), f"Expected (3, 64, 64), got {sample['ic'].shape}"
        assert sample['params'].shape == (14,), f"Expected (14,), got {sample['params'].shape}"
        assert 'sample_idx' in sample, "sample_idx missing from sample"


def test_dataset_different_realizations():
    """Test that different realizations produce different ICs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_realizations.h5"
        with h5py.File(dataset_path, 'w') as f:
            # Create dataset with distinct realizations
            data = np.random.randn(10, 5, 3, 64, 64)
            f.create_dataset('inputs/fields', data=data)
            f.create_dataset('parameters/params', data=np.random.randn(10, 14))
            f.create_group('metadata')
            f['metadata'].attrs['num_channels'] = 3

        # Load with different realizations
        dataset_real0 = NOAStateDataset(dataset_path, max_samples=10, realization_idx=0)
        dataset_real1 = NOAStateDataset(dataset_path, max_samples=10, realization_idx=1)

        sample_real0 = dataset_real0[0]
        sample_real1 = dataset_real1[0]

        # Same params but different ICs
        assert torch.allclose(sample_real0['params'], sample_real1['params']), "Parameters should be identical"
        assert not torch.allclose(sample_real0['ic'], sample_real1['ic']), "ICs should differ for different realizations"


def test_100k_baseline_dataset():
    """Test 100k baseline dataset loads correctly."""
    dataset_path = Path("datasets/100k_baseline_dev.h5")
    if not dataset_path.exists():
        pytest.skip("Dataset not found")

    dataset = NOAStateDataset(dataset_path, max_samples=10)
    sample = dataset[0]
    assert sample['ic'].shape == (3, 64, 64), f"Expected (3, 64, 64), got {sample['ic'].shape}"
    assert sample['params'].shape[0] == 14, f"Expected 14 params, got {sample['params'].shape[0]}"
    assert 'sample_idx' in sample, "sample_idx missing from sample"


def test_stratified_sampling():
    """Test stratified sampling produces uniformly spaced indices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_stratified.h5"
        with h5py.File(dataset_path, 'w') as f:
            f.create_dataset('inputs/fields', data=np.random.randn(1000, 3, 3, 64, 64))
            f.create_dataset('parameters/params', data=np.random.randn(1000, 14))
            f.create_group('metadata')
            f['metadata'].attrs['num_channels'] = 3

        dataset = NOAStateDataset(
            dataset_path,
            max_samples=10,
            sampling_strategy="stratified"
        )

        # Check indices are uniformly spaced
        indices = dataset.indices
        expected_stride = 1000 // 10
        expected_indices = np.arange(0, 1000, expected_stride)[:10]
        np.testing.assert_array_equal(indices, expected_indices)


def test_sequential_sampling():
    """Test sequential sampling takes first n samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_sequential.h5"
        with h5py.File(dataset_path, 'w') as f:
            f.create_dataset('inputs/fields', data=np.random.randn(1000, 3, 3, 64, 64))
            f.create_dataset('parameters/params', data=np.random.randn(1000, 14))
            f.create_group('metadata')
            f['metadata'].attrs['num_channels'] = 3

        dataset = NOAStateDataset(
            dataset_path,
            max_samples=10,
            sampling_strategy="sequential"
        )

        # Check indices are sequential
        indices = dataset.indices
        expected_indices = np.arange(10)
        np.testing.assert_array_equal(indices, expected_indices)


def test_random_sampling():
    """Test random sampling is reproducible with fixed seed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_random.h5"
        with h5py.File(dataset_path, 'w') as f:
            f.create_dataset('inputs/fields', data=np.random.randn(1000, 3, 3, 64, 64))
            f.create_dataset('parameters/params', data=np.random.randn(1000, 14))
            f.create_group('metadata')
            f['metadata'].attrs['num_channels'] = 3

        # Create two datasets with same seed
        dataset1 = NOAStateDataset(
            dataset_path,
            max_samples=10,
            sampling_strategy="random",
            random_seed=42
        )
        dataset2 = NOAStateDataset(
            dataset_path,
            max_samples=10,
            sampling_strategy="random",
            random_seed=42
        )

        # Indices should be identical
        np.testing.assert_array_equal(dataset1.indices, dataset2.indices)

        # Create dataset with different seed
        dataset3 = NOAStateDataset(
            dataset_path,
            max_samples=10,
            sampling_strategy="random",
            random_seed=123
        )

        # Indices should differ
        assert not np.array_equal(dataset1.indices, dataset3.indices)
