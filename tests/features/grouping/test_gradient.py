"""Tests for GradientRefiner."""

import pytest
import numpy as np
import torch

from spinlock.features.grouping.gradient import GradientRefiner, DifferentiableAssigner
from spinlock.features.grouping.models import GradientParams


@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    np.random.seed(42)
    # Create 100 samples with 10 features
    N, D = 100, 10
    features = np.random.randn(N, D).astype(np.float32)
    return features


@pytest.fixture
def feature_names():
    """Create feature names."""
    return [f"feat_{i}" for i in range(10)]


def test_gradient_refiner_init():
    """Test GradientRefiner initialization."""
    config = GradientParams()
    refiner = GradientRefiner(config)

    assert refiner.config == config
    assert refiner.custom_loss_fn is None
    assert isinstance(refiner.device, torch.device)


def test_set_custom_loss():
    """Test setting custom loss function."""
    config = GradientParams()
    refiner = GradientRefiner(config)

    def custom_loss(features, assignment_probs):
        return torch.tensor(0.0)

    refiner.set_custom_loss(custom_loss)
    assert refiner.custom_loss_fn is not None


def test_get_temperature():
    """Test temperature annealing."""
    config = GradientParams(
        num_epochs=100,
        temperature_start=1.0,
        temperature_end=0.5,
    )
    refiner = GradientRefiner(config)

    # At epoch 0, should be start temp
    temp = refiner._get_temperature(0)
    assert temp == 1.0

    # At epoch 100, should be end temp
    temp = refiner._get_temperature(100)
    assert temp == 0.5

    # At epoch 50, should be halfway
    temp = refiner._get_temperature(50)
    assert 0.7 < temp < 0.8


def test_orthogonality_loss():
    """Test orthogonality loss computation."""
    config = GradientParams()
    refiner = GradientRefiner(config)

    # Create features and assignments
    features = torch.randn(100, 10)
    assignment_probs = torch.randn(10, 3).softmax(dim=1)

    loss = refiner._orthogonality_loss(features, assignment_probs)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert loss >= 0  # Should be non-negative


def test_informativeness_loss():
    """Test informativeness loss computation."""
    config = GradientParams()
    refiner = GradientRefiner(config)

    # Create features and assignments
    features = torch.randn(100, 10)
    assignment_probs = torch.randn(10, 3).softmax(dim=1)

    loss = refiner._informativeness_loss(features, assignment_probs)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    # Should be negative (negative variance)
    assert loss <= 0


def test_refine_basic(sample_features, feature_names):
    """Test basic gradient refinement."""
    config = GradientParams(
        num_epochs=10,  # Short for testing
        device="cpu",  # Use CPU for testing
    )
    refiner = GradientRefiner(config)

    # Refine with 3 groups
    groups = refiner.refine(sample_features, feature_names, num_groups=3)

    # Should have 3 groups
    assert len(groups) <= 3  # May have fewer if some are empty

    # All features should be assigned
    all_indices = []
    for indices in groups.values():
        all_indices.extend(indices)
    assert len(set(all_indices)) == 10


def test_refine_with_init(sample_features, feature_names):
    """Test gradient refinement with initialization."""
    config = GradientParams(
        num_epochs=10,
        device="cpu",
    )
    refiner = GradientRefiner(config)

    # Initial groups
    init_groups = {
        "group_1": [0, 1, 2],
        "group_2": [3, 4, 5],
        "group_3": [6, 7, 8, 9],
    }

    groups = refiner.refine(
        sample_features,
        feature_names,
        num_groups=3,
        init_groups=init_groups,
    )

    # Should have 3 groups
    assert len(groups) <= 3

    # All features should be assigned
    all_indices = []
    for indices in groups.values():
        all_indices.extend(indices)
    assert len(set(all_indices)) == 10


def test_differentiable_assigner():
    """Test DifferentiableAssigner module."""
    assigner = DifferentiableAssigner(num_features=10, num_groups=3)

    assert assigner.num_features == 10
    assert assigner.num_groups == 3
    assert assigner.logits.shape == (10, 3)

    # Test forward pass
    probs = assigner(temperature=1.0)
    assert probs.shape == (10, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(10), atol=1e-5)

    # Test hard assignments
    hard = assigner.get_hard_assignments()
    assert hard.shape == (10,)
    assert hard.min() >= 0
    assert hard.max() < 3


def test_differentiable_assigner_init_from_groups():
    """Test DifferentiableAssigner initialization from groups."""
    init_groups = {
        "group_1": [0, 1, 2],
        "group_2": [3, 4, 5],
        "group_3": [6, 7, 8, 9],
    }

    assigner = DifferentiableAssigner(
        num_features=10,
        num_groups=3,
        init_groups=init_groups,
    )

    # Get hard assignments
    hard = assigner.get_hard_assignments()

    # Check initialization worked
    assert torch.all(hard[:3] == 0)  # First 3 features in group 0
    assert torch.all(hard[3:6] == 1)  # Next 3 features in group 1
    assert torch.all(hard[6:] == 2)  # Last 4 features in group 2
