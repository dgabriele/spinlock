"""Feature importance weighting for VQ-VAE reconstruction loss.

Computes per-feature weights based on variance and information content,
allowing the model to focus reconstruction effort on high-value features.
"""

import numpy as np
import torch
from typing import Optional, Literal


def compute_feature_weights(
    train_features: np.ndarray,
    strategy: Literal["variance", "information", "variance_information", "uniform"] = "variance_information",
    normalize: bool = True,
) -> np.ndarray:
    """Compute importance weights from training data.

    Args:
        train_features: Training features [N, D]
        strategy: Weighting strategy
            - "variance": Weight by feature variance
            - "information": Weight by feature entropy
            - "variance_information": Geometric mean of variance and entropy (default)
            - "uniform": Equal weights (baseline)
        normalize: If True, normalize weights to sum to D (preserves overall loss scale)

    Returns:
        weights: Per-feature importance weights [D]
    """
    N, D = train_features.shape

    if strategy == "uniform":
        weights = np.ones(D)

    elif strategy == "variance":
        # Weight by variance (features with more variation are more important)
        variances = np.var(train_features, axis=0)
        # Normalize by mean variance to make weights relative
        weights = variances / (np.mean(variances) + 1e-8)

    elif strategy == "information":
        # Weight by entropy (features with more information content)
        entropies = []
        for d in range(D):
            # Discretize into bins for entropy calculation
            hist, _ = np.histogram(train_features[:, d], bins=50, density=True)
            # Normalize to probabilities
            p = hist / (hist.sum() + 1e-8)
            # Remove zeros
            p = p[p > 0]
            # Compute entropy: -sum(p * log(p))
            entropy = -np.sum(p * np.log(p + 1e-8))
            entropies.append(entropy)

        entropies = np.array(entropies)
        # Normalize by mean entropy
        weights = entropies / (np.mean(entropies) + 1e-8)

    elif strategy == "variance_information":
        # Geometric mean of variance and information weights
        # This balances both criteria

        # Variance component
        variances = np.var(train_features, axis=0)
        var_weights = variances / (np.mean(variances) + 1e-8)

        # Entropy component
        entropies = []
        for d in range(D):
            hist, _ = np.histogram(train_features[:, d], bins=50, density=True)
            p = hist / (hist.sum() + 1e-8)
            p = p[p > 0]
            entropy = -np.sum(p * np.log(p + 1e-8))
            entropies.append(entropy)
        entropies = np.array(entropies)
        ent_weights = entropies / (np.mean(entropies) + 1e-8)

        # Geometric mean: sqrt(var_weights * ent_weights)
        weights = np.sqrt(var_weights * ent_weights)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Normalize to preserve overall loss scale
    # Sum of weights = D (number of features)
    if normalize:
        weights = weights * D / (weights.sum() + 1e-8)

    return weights


def apply_feature_weights(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply feature weights to reconstruction loss.

    Args:
        reconstruction: Reconstructed features [batch, D]
        target: Target features [batch, D]
        weights: Feature weights [D] or None for uniform

    Returns:
        weighted_loss: Scalar weighted MSE loss
    """
    # Compute squared errors
    squared_errors = (reconstruction - target) ** 2  # [batch, D]

    if weights is not None:
        # Apply per-feature weights
        # weights: [D] -> [1, D] for broadcasting
        weighted_errors = squared_errors * weights.unsqueeze(0)  # [batch, D]
        return weighted_errors.mean()
    else:
        # Uniform weighting (standard MSE)
        return squared_errors.mean()


class LearnedFeatureWeights(torch.nn.Module):
    """Learnable per-feature attention weights.

    Adds trainable weights that are optimized during training to automatically
    discover which features are most important for reconstruction.

    Args:
        num_features: Number of input features
        init_strategy: Initialization strategy
            - "uniform": All weights = 1.0
            - "data": Initialize from data statistics (requires init_weights)
        l1_penalty: L1 regularization strength to encourage sparsity
    """

    def __init__(
        self,
        num_features: int,
        init_strategy: Literal["uniform", "data"] = "uniform",
        init_weights: Optional[np.ndarray] = None,
        l1_penalty: float = 0.01,
    ):
        super().__init__()

        # Initialize weights
        if init_strategy == "uniform":
            initial = torch.ones(num_features)
        elif init_strategy == "data":
            if init_weights is None:
                raise ValueError("init_weights required for 'data' initialization")
            initial = torch.from_numpy(init_weights).float()
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

        # Trainable weights (will be passed through sigmoid)
        self.raw_weights = torch.nn.Parameter(
            torch.log(initial + 1e-8)  # Log space for numerical stability
        )

        self.l1_penalty = l1_penalty

    def forward(self) -> torch.Tensor:
        """Get current feature weights (sigmoid activation).

        Returns:
            weights: Feature weights [D], positive values
        """
        return torch.sigmoid(self.raw_weights)

    def get_l1_loss(self) -> torch.Tensor:
        """Get L1 regularization loss to encourage sparsity.

        Returns:
            l1_loss: Scalar L1 penalty on weights
        """
        weights = self.forward()
        return self.l1_penalty * weights.abs().mean()

    def get_active_features(self, threshold: float = 0.1) -> torch.Tensor:
        """Get indices of features with weight > threshold.

        Args:
            threshold: Minimum weight to consider active

        Returns:
            active_indices: Boolean mask [D]
        """
        weights = self.forward()
        return weights > threshold
