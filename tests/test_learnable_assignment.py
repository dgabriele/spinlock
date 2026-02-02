"""Tests for learnable category assignment modules."""

import pytest
import torch
import numpy as np
from spinlock.encoding.learnable_assignment import (
    SoftAssignmentMatrix,
    PerFamilyAssignmentMatrix,
    initialize_from_clustering
)
from spinlock.encoding.soft_routing import SoftGroupedFeatureExtractor
from spinlock.encoding.training.assignment_losses import (
    soft_orthogonality_loss,
    soft_balance_loss,
    family_constraint_loss
)
from spinlock.encoding.training.annealing import TemperatureScheduler


class TestSoftAssignmentMatrix:
    """Test SoftAssignmentMatrix module."""

    def test_forward(self):
        """Test forward pass returns valid probabilities."""
        num_features = 10
        num_categories = 3

        matrix = SoftAssignmentMatrix(num_features, num_categories)
        soft_assign = matrix(temperature=1.0)

        # Check shape
        assert soft_assign.shape == (num_features, num_categories)

        # Check probabilities sum to 1
        assert torch.allclose(soft_assign.sum(dim=1), torch.ones(num_features))

        # Check all values in [0, 1]
        assert (soft_assign >= 0).all()
        assert (soft_assign <= 1).all()

    def test_temperature_effect(self):
        """Test that temperature affects sharpness."""
        num_features = 5
        num_categories = 3

        matrix = SoftAssignmentMatrix(num_features, num_categories)

        # High temperature → more uniform
        soft_high = matrix(temperature=10.0)
        entropy_high = -(soft_high * torch.log(soft_high + 1e-10)).sum()

        # Low temperature → more peaked
        soft_low = matrix(temperature=0.1)
        entropy_low = -(soft_low * torch.log(soft_low + 1e-10)).sum()

        # High temp should have higher entropy
        assert entropy_high > entropy_low

    def test_get_hard_assignments(self):
        """Test hard assignment extraction."""
        num_features = 10
        num_categories = 3

        matrix = SoftAssignmentMatrix(num_features, num_categories)
        hard_assign = matrix.get_hard_assignments()

        # Check shape and values
        assert hard_assign.shape == (num_features,)
        assert (hard_assign >= 0).all()
        assert (hard_assign < num_categories).all()

    def test_gradient_flow(self):
        """Test gradients flow through forward pass."""
        num_features = 5
        num_categories = 3

        matrix = SoftAssignmentMatrix(num_features, num_categories)
        soft_assign = matrix(temperature=1.0)

        # Compute simple loss
        loss = soft_assign.sum()
        loss.backward()

        # Check gradients exist
        assert matrix.logits.grad is not None
        assert not torch.isnan(matrix.logits.grad).any()


class TestPerFamilyAssignmentMatrix:
    """Test PerFamilyAssignmentMatrix module."""

    def test_block_diagonal_structure(self):
        """Test block-diagonal constraint enforcement."""
        feature_families = {
            "family_a": [0, 1, 2],
            "family_b": [3, 4, 5, 6]
        }
        categories_per_family = {
            "family_a": 2,
            "family_b": 3
        }

        matrix = PerFamilyAssignmentMatrix(
            feature_families, categories_per_family
        )
        soft_assign = matrix(temperature=1.0)

        # Check total shape
        assert soft_assign.shape == (7, 5)  # 7 features, 5 total categories

        # Check block-diagonal structure (cross-family should be zero)
        # Family A features [0,1,2] should only assign to categories [0,1]
        assert torch.allclose(soft_assign[0:3, 2:5], torch.zeros(3, 3))

        # Family B features [3,4,5,6] should only assign to categories [2,3,4]
        assert torch.allclose(soft_assign[3:7, 0:2], torch.zeros(4, 2))

    def test_no_cross_family_leakage(self):
        """Test that features never assign to wrong family categories."""
        feature_families = {
            "family_a": [0, 1],
            "family_b": [2, 3]
        }
        categories_per_family = {
            "family_a": 2,
            "family_b": 2
        }

        matrix = PerFamilyAssignmentMatrix(
            feature_families, categories_per_family
        )
        soft_assign = matrix(temperature=0.1)  # Low temp for sharp assignments

        # Family A features should have zero prob for family B categories
        family_a_cross = soft_assign[0:2, 2:4]
        assert torch.allclose(family_a_cross, torch.zeros(2, 2))

        # Family B features should have zero prob for family A categories
        family_b_cross = soft_assign[2:4, 0:2]
        assert torch.allclose(family_b_cross, torch.zeros(2, 2))


class TestSoftRouting:
    """Test SoftGroupedFeatureExtractor."""

    def test_forward_shape(self):
        """Test forward pass produces correct shapes."""
        batch_size = 8
        input_dim = 20
        num_categories = 4
        embedding_dim = 16

        router = SoftGroupedFeatureExtractor(
            input_dim, num_categories, embedding_dim
        )

        x = torch.randn(batch_size, input_dim)
        soft_assign = torch.softmax(torch.randn(input_dim, num_categories), dim=1)

        category_embs = router(x, soft_assign)

        # Check output shape
        assert category_embs.shape == (batch_size, num_categories, embedding_dim)

    def test_gradient_flow(self):
        """Test gradients flow through soft routing."""
        router = SoftGroupedFeatureExtractor(10, 3, 8)

        x = torch.randn(4, 10)
        soft_assign = torch.softmax(torch.randn(10, 3), dim=1)
        soft_assign.requires_grad = True

        category_embs = router(x, soft_assign)
        loss = category_embs.sum()
        loss.backward()

        # Check gradients
        assert soft_assign.grad is not None
        assert not torch.isnan(soft_assign.grad).any()


class TestAssignmentLosses:
    """Test assignment loss functions."""

    def test_soft_balance_loss(self):
        """Test balance loss computation."""
        # Balanced assignments
        balanced = torch.tensor([
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        balance_balanced = soft_balance_loss(balanced)

        # Unbalanced assignments (all to category 0)
        unbalanced = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0]
        ])
        balance_unbalanced = soft_balance_loss(unbalanced)

        # Balanced should have lower loss
        assert balance_balanced < balance_unbalanced

    def test_family_constraint_loss(self):
        """Test family constraint loss."""
        feature_families = {
            "family_a": [0, 1],
            "family_b": [2, 3]
        }
        category_offsets = {
            "family_a": 0,
            "family_b": 2
        }
        categories_per_family = {
            "family_a": 2,
            "family_b": 2
        }

        # Perfect assignments (no cross-family)
        perfect = torch.tensor([
            [0.5, 0.5, 0.0, 0.0],  # Family A
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],  # Family B
            [0.0, 0.0, 0.5, 0.5]
        ])
        loss_perfect = family_constraint_loss(
            perfect, feature_families, category_offsets, categories_per_family
        )

        # Violated assignments (cross-family leakage)
        violated = torch.tensor([
            [0.3, 0.3, 0.2, 0.2],  # Family A leaking to B
            [0.3, 0.3, 0.2, 0.2],
            [0.2, 0.2, 0.3, 0.3],  # Family B leaking to A
            [0.2, 0.2, 0.3, 0.3]
        ])
        loss_violated = family_constraint_loss(
            violated, feature_families, category_offsets, categories_per_family
        )

        # Perfect should have lower loss
        assert loss_perfect < loss_violated


class TestTemperatureScheduler:
    """Test TemperatureScheduler."""

    def test_linear_schedule(self):
        """Test linear temperature annealing."""
        scheduler = TemperatureScheduler(
            start=1.0, end=0.1, total_epochs=100, schedule="linear"
        )

        # Check endpoints
        assert scheduler(0) == 1.0
        assert scheduler(100) == 0.1

        # Check midpoint
        mid_temp = scheduler(50)
        assert 0.4 < mid_temp < 0.6

    def test_exponential_schedule(self):
        """Test exponential temperature annealing."""
        scheduler = TemperatureScheduler(
            start=1.0, end=0.1, total_epochs=100, schedule="exponential"
        )

        # Check endpoints
        assert scheduler(0) == 1.0
        assert abs(scheduler(100) - 0.1) < 1e-6

        # Exponential should decay faster early on
        temp_25 = scheduler(25)
        temp_75 = scheduler(75)
        # More change in first quarter than last quarter
        assert (1.0 - temp_25) > (temp_75 - 0.1)

    def test_cosine_schedule(self):
        """Test cosine temperature annealing."""
        scheduler = TemperatureScheduler(
            start=1.0, end=0.1, total_epochs=100, schedule="cosine"
        )

        # Check endpoints
        assert abs(scheduler(0) - 1.0) < 1e-6
        assert abs(scheduler(100) - 0.1) < 1e-6

        # Cosine should be smooth
        temps = [scheduler(i) for i in range(0, 101, 10)]
        # Check monotonic decrease
        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
