"""Gradient-based grouping refinement via Gumbel-Softmax."""

from typing import Dict, List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import GradientParams


class GradientRefiner:
    """
    Gradient-based feature grouping via Gumbel-Softmax.

    Core objectives (standalone):
    1. Orthogonality: Minimize inter-group correlation (off-diagonal elements)
    2. Informativeness: Maximize per-group feature variance

    Interface for custom loss injection:
    3. Optional loss callback for downstream tasks (e.g., VQ-VAE reconstruction)
    """

    def __init__(self, config: GradientParams):
        self.config = config
        self.device = self._get_device()
        self.custom_loss_fn: Optional[Callable] = None  # Set via set_custom_loss()

    def set_custom_loss(self, loss_fn: Callable):
        """
        Set custom loss function for downstream optimization.

        Args:
            loss_fn: Callable(features, assignment_probs) -> torch.Tensor
                     Returns scalar loss to minimize
        """
        self.custom_loss_fn = loss_fn

    def _get_device(self) -> torch.device:
        """Determine compute device."""
        if self.config.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return torch.device("cuda")
        elif self.config.device == "cpu":
            return torch.device("cpu")
        else:  # auto
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def refine(
        self,
        features: np.ndarray,
        feature_names: List[str],
        num_groups: int,
        init_groups: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, List[int]]:
        """
        Refine grouping via gradient optimization.

        Args:
            features: Normalized features [N, D]
            feature_names: Feature names
            num_groups: Number of groups
            init_groups: Optional initialization from clustering

        Returns:
            Dict mapping group names to feature indices
        """
        # Convert to torch
        features_torch = torch.from_numpy(features).float().to(self.device)

        # Initialize assignment module
        assigner = DifferentiableAssigner(
            num_features=features.shape[1],
            num_groups=num_groups,
            init_groups=init_groups,
        ).to(self.device)

        # Optimize
        optimizer = torch.optim.Adam(assigner.parameters(), lr=self.config.learning_rate)

        for epoch in range(self.config.num_epochs):
            # Temperature annealing
            temperature = self._get_temperature(epoch)

            # Forward pass
            assignment_probs = assigner(temperature)  # [D, K]

            # Compute core losses (always)
            ortho_loss = self._orthogonality_loss(features_torch, assignment_probs)
            info_loss = self._informativeness_loss(features_torch, assignment_probs)

            total_loss = (
                self.config.orthogonality_weight * ortho_loss +
                self.config.informativeness_weight * info_loss
            )

            # Add custom loss if provided (e.g., VQ-VAE reconstruction)
            if self.custom_loss_fn is not None:
                custom_loss = self.custom_loss_fn(features_torch, assignment_probs)
                total_loss = total_loss + self.config.custom_loss_weight * custom_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Early stopping
            if ortho_loss.item() < self.config.orthogonality_target:
                break

        # Extract hard assignments
        with torch.no_grad():
            hard_assignments = assigner.get_hard_assignments()  # [D]

        # Convert to dict
        groups = {}
        for group_id in range(num_groups):
            indices = (hard_assignments == group_id).nonzero(as_tuple=True)[0].cpu().numpy().tolist()
            if indices:
                groups[f"group_{group_id + 1}"] = indices

        return groups

    def _get_temperature(self, epoch: int) -> float:
        """Compute annealing temperature."""
        progress = epoch / self.config.num_epochs
        return max(
            self.config.temperature_end,
            self.config.temperature_start - progress * (
                self.config.temperature_start - self.config.temperature_end
            )
        )

    def _orthogonality_loss(
        self,
        features: torch.Tensor,
        assignment_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute orthogonality loss (minimize inter-group correlation).

        Args:
            features: [N, D]
            assignment_probs: [D, K]

        Returns:
            Scalar loss
        """
        # Group features: [N, K]
        group_features = torch.matmul(features, assignment_probs)

        # Normalize
        group_features_norm = F.normalize(group_features, p=2, dim=0)

        # Correlation matrix: [K, K]
        corr_matrix = torch.matmul(group_features_norm.T, group_features_norm)

        # Off-diagonal penalty
        mask = ~torch.eye(corr_matrix.size(0), dtype=torch.bool, device=features.device)
        off_diagonal = torch.abs(corr_matrix[mask])

        return off_diagonal.mean()

    def _informativeness_loss(
        self,
        features: torch.Tensor,
        assignment_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute informativeness loss (maximize per-group feature variance).

        Encourages groups to capture meaningful variation in the data.

        Args:
            features: [N, D]
            assignment_probs: [D, K]

        Returns:
            Scalar loss (negative variance, to maximize via minimization)
        """
        # Encode to groups: [N, K]
        group_features = torch.matmul(features, assignment_probs)

        # Compute variance per group
        group_variances = torch.var(group_features, dim=0)

        # Return negative mean variance (minimize to maximize variance)
        return -group_variances.mean()


class DifferentiableAssigner(nn.Module):
    """Learnable assignment matrix with Gumbel-Softmax."""

    def __init__(
        self,
        num_features: int,
        num_groups: int,
        init_groups: Optional[Dict[str, List[int]]] = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups

        # Learnable logits [D, K]
        self.logits = nn.Parameter(torch.randn(num_features, num_groups) * 0.1)

        # Initialize from clustering if provided
        if init_groups is not None:
            self._init_from_groups(init_groups)

    def _init_from_groups(self, groups: Dict[str, List[int]]):
        """Initialize logits from hard assignments."""
        with torch.no_grad():
            # Reset logits
            self.logits.zero_()

            # Set high logit for assigned group
            for group_id, (name, indices) in enumerate(groups.items()):
                self.logits[indices, group_id] = 5.0  # Strong initialization

    def forward(self, temperature: float = 1.0) -> torch.Tensor:
        """
        Forward pass with Gumbel-Softmax sampling.

        Args:
            temperature: Gumbel-Softmax temperature

        Returns:
            Assignment probabilities [D, K]
        """
        return F.gumbel_softmax(self.logits, tau=temperature, hard=False)

    def get_hard_assignments(self) -> torch.Tensor:
        """Get hard assignments via argmax."""
        return torch.argmax(self.logits, dim=1)
