"""Integration tests for visualize-qbm CLI command."""
import pytest
import h5py
import torch
import numpy as np
from pathlib import Path
from argparse import Namespace
from spinlock.cli.visualize_qbm import VisualizeQBMCommand


@pytest.fixture
def mock_qbm_dataset(tmp_path):
    """Create a minimal mock QBM dataset for testing."""
    dataset_path = tmp_path / "test_qbm.h5"

    # Create minimal dataset with small dimensions for speed
    N, M, T, H, W = 10, 3, 20, 32, 32

    with h5py.File(dataset_path, 'w') as f:
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


def test_visualize_qbm_probability_mode(mock_qbm_dataset, tmp_path):
    """Test probability density rendering mode."""
    output = tmp_path / "output.mp4"

    args = Namespace(
        dataset=mock_qbm_dataset,
        output=output,
        n_rollouts=2,
        sampling_method='random',
        renderer='probability',
        overlay_observable='purity',
        colormap='viridis',
        wigner_colormap='RdBu_r',
        fps=10,
        stride=2,
        device='cpu',
        seed=42,
        verbose=False,
    )

    command = VisualizeQBMCommand()
    exit_code = command.execute(args)

    assert exit_code == 0
    assert output.exists()
    assert output.stat().st_size > 0


def test_visualize_qbm_wigner_mode(mock_qbm_dataset, tmp_path):
    """Test Wigner phase-space rendering mode."""
    output = tmp_path / "wigner.mp4"

    args = Namespace(
        dataset=mock_qbm_dataset,
        output=output,
        n_rollouts=2,
        sampling_method='random',
        renderer='wigner',
        overlay_observable='entropy',
        colormap='viridis',
        wigner_colormap='RdBu_r',
        fps=10,
        stride=2,
        device='cpu',
        seed=42,
        verbose=False,
    )

    command = VisualizeQBMCommand()
    exit_code = command.execute(args)

    assert exit_code == 0
    assert output.exists()


def test_visualize_qbm_both_mode(mock_qbm_dataset, tmp_path):
    """Test side-by-side rendering mode."""
    output = tmp_path / "both.mp4"

    args = Namespace(
        dataset=mock_qbm_dataset,
        output=output,
        n_rollouts=2,
        sampling_method='random',
        renderer='both',
        overlay_observable='coherence_mean',
        colormap='viridis',
        wigner_colormap='RdBu_r',
        fps=10,
        stride=2,
        device='cpu',
        seed=42,
        verbose=False,
    )

    command = VisualizeQBMCommand()
    exit_code = command.execute(args)

    assert exit_code == 0
    assert output.exists()


def test_visualize_qbm_no_overlay(mock_qbm_dataset, tmp_path):
    """Test rendering without observable overlay."""
    output = tmp_path / "no_overlay.mp4"

    args = Namespace(
        dataset=mock_qbm_dataset,
        output=output,
        n_rollouts=2,
        sampling_method='random',
        renderer='probability',
        overlay_observable='none',
        colormap='viridis',
        wigner_colormap='RdBu_r',
        fps=10,
        stride=2,
        device='cpu',
        seed=42,
        verbose=False,
    )

    command = VisualizeQBMCommand()
    exit_code = command.execute(args)

    assert exit_code == 0
    assert output.exists()


def test_visualize_qbm_sobol_sampling(mock_qbm_dataset, tmp_path):
    """Test Sobol quasi-random sampling."""
    output = tmp_path / "sobol.mp4"

    args = Namespace(
        dataset=mock_qbm_dataset,
        output=output,
        n_rollouts=3,
        sampling_method='sobol',
        renderer='probability',
        overlay_observable='none',
        colormap='viridis',
        wigner_colormap='RdBu_r',
        fps=10,
        stride=2,
        device='cpu',
        seed=42,
        verbose=False,
    )

    command = VisualizeQBMCommand()
    exit_code = command.execute(args)

    assert exit_code == 0
    assert output.exists()


def test_visualize_qbm_diverse_sampling(mock_qbm_dataset, tmp_path):
    """Test diversity-based sampling."""
    output = tmp_path / "diverse.mp4"

    args = Namespace(
        dataset=mock_qbm_dataset,
        output=output,
        n_rollouts=3,
        sampling_method='diverse',
        renderer='probability',
        overlay_observable='none',
        colormap='viridis',
        wigner_colormap='RdBu_r',
        fps=10,
        stride=2,
        device='cpu',
        seed=42,
        verbose=False,
    )

    command = VisualizeQBMCommand()
    exit_code = command.execute(args)

    assert exit_code == 0
    assert output.exists()


def test_visualize_qbm_verbose(mock_qbm_dataset, tmp_path, capsys):
    """Test verbose output."""
    output = tmp_path / "verbose.mp4"

    args = Namespace(
        dataset=mock_qbm_dataset,
        output=output,
        n_rollouts=2,
        sampling_method='random',
        renderer='probability',
        overlay_observable='purity',
        colormap='viridis',
        wigner_colormap='RdBu_r',
        fps=10,
        stride=2,
        device='cpu',
        seed=42,
        verbose=True,
    )

    command = VisualizeQBMCommand()
    exit_code = command.execute(args)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Loading QBM dataset" in captured.out
    assert "Initializing renderers" in captured.out
    assert "Rendering frames" in captured.out
    assert "Video encoding complete" in captured.out


def test_visualize_qbm_missing_dataset(tmp_path):
    """Test error handling for missing dataset."""
    output = tmp_path / "output.mp4"

    args = Namespace(
        dataset=Path("/nonexistent/dataset.h5"),
        output=output,
        n_rollouts=2,
        sampling_method='random',
        renderer='probability',
        overlay_observable='none',
        colormap='viridis',
        wigner_colormap='RdBu_r',
        fps=10,
        stride=2,
        device='cpu',
        seed=42,
        verbose=False,
    )

    command = VisualizeQBMCommand()
    exit_code = command.execute(args)

    # Should return non-zero exit code
    assert exit_code != 0
