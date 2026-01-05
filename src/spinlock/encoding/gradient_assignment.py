"""Gradient-based category assignment using Gumbel-Softmax.

Implements differentiable feature-to-category assignment optimization via:
- Gumbel-Softmax reparameterization for differentiable sampling
- Orthogonality loss (minimize inter-category correlation)
- Informativeness loss (maximize reconstruction quality)

Enhanced from unisim.system.models.gradient_assignment:
- GPU-first: Default to CUDA, fallback to CPU if unavailable
- Smart subsampling: 10K + 10% of excess for large datasets

References:
- Gumbel-Softmax: https://arxiv.org/abs/1611.01144
- Concrete Distribution: https://arxiv.org/abs/1611.00712
"""

from typing import List, Dict, Optional
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_subsample_size(n_samples: int, excess_fraction: float = 0.1) -> int:
    """Compute smart subsample size for gradient optimization.

    If dataset <= 10K: use full dataset
    Otherwise: 10K + excess_fraction * (N - 10K)

    This balances:
    - Small datasets: Use all data for best optimization
    - Large datasets: Subsample to fit in GPU memory while retaining diversity

    Args:
        n_samples: Total number of samples
        excess_fraction: Fraction of excess samples to include (default: 0.1 = 10%)

    Returns:
        Subsample size

    Examples:
        - 5K samples -> 5K (full)
        - 10K samples -> 10K (full)
        - 100K samples -> 10K + 0.1*(100K-10K) = 19K
        - 1M samples -> 10K + 0.1*(1M-10K) = 109K
    """
    BASE_SIZE = 10_000
    if n_samples <= BASE_SIZE:
        return n_samples
    excess = n_samples - BASE_SIZE
    return BASE_SIZE + int(excess * excess_fraction)


class DifferentiableCategoryAssigner(nn.Module):
    """Learnable feature-to-category assignment using Gumbel-Softmax.

    Maintains soft assignment probabilities [N_features, N_categories] that are
    optimized via gradient descent. Gumbel-Softmax provides differentiable sampling
    during training, allowing end-to-end optimization.
    """

    def __init__(
        self,
        num_features: int,
        num_categories: int,
        temperature: float = 1.0,
        init_logits: Optional[torch.Tensor] = None,
    ):
        """Initialize differentiable category assigner.

        Args:
            num_features: Number of input features
            num_categories: Number of target categories
            temperature: Gumbel-Softmax temperature (lower = more discrete)
                        tau=1.0: soft assignments
                        tau=0.5: nearly hard assignments
                        Annealed during training: 1.0 -> 0.5
            init_logits: Optional initialization from clustering [num_features, num_categories]
                        Strong initialization (logits=5.0) helps convergence
        """
        super().__init__()

        self.num_features = num_features
        self.num_categories = num_categories
        self.temperature = temperature

        # Learnable assignment logits [num_features, num_categories]
        if init_logits is not None:
            self.logits = nn.Parameter(init_logits.clone())
        else:
            # Random initialization with small values
            self.logits = nn.Parameter(
                torch.randn(num_features, num_categories) * 0.1
            )

    def forward(self, training: bool = True) -> torch.Tensor:
        """Compute soft or hard category assignments.

        Args:
            training: If True, use Gumbel-Softmax (differentiable)
                     If False, use softmax (deterministic)

        Returns:
            assignment_probs: [num_features, num_categories] soft assignments
                            Each row sums to 1.0
        """
        if training:
            # Gumbel-Softmax: differentiable categorical sampling
            assignment = F.gumbel_softmax(
                self.logits, tau=self.temperature, hard=False, dim=1
            )
        else:
            # Standard softmax (no Gumbel noise)
            assignment = F.softmax(self.logits, dim=1)

        return assignment

    def get_hard_assignments(self) -> Dict[str, List[int]]:
        """Convert soft assignments to hard category indices via argmax.

        Returns:
            Dict mapping category_name -> list of feature indices
            Example: {'cat_0': [0, 2, 5], 'cat_1': [1, 3, 4]}
        """
        with torch.no_grad():
            # Argmax along category dimension
            labels = self.logits.argmax(dim=1).cpu().numpy()

        # Build category assignments
        assignments = {}
        for cat_id in range(self.num_categories):
            indices = np.where(labels == cat_id)[0].tolist()
            if len(indices) > 0:
                assignments[f"cat_{cat_id}"] = indices

        return assignments


def optimize_category_assignment(
    features: np.ndarray,
    num_categories: int,
    init_assignments: Optional[Dict[str, List[int]]] = None,
    num_epochs: int = 500,
    learning_rate: float = 0.01,
    orthogonality_weight: float = 1.0,
    informativeness_weight: float = 1.0,
    orthogonality_target: float = 0.3,
    min_features_per_category: int = 3,
    random_seed: int = 42,
    subsample_excess_fraction: float = 0.1,
    device: str = "cuda",
) -> Dict[str, List[int]]:
    """Optimize category assignments using gradient descent.

    Two-objective optimization:
    1. Orthogonality: Minimize inter-category correlation (target <0.3)
    2. Informativeness: Maximize per-category reconstruction quality (minimize MSE)

    Uses Adam optimizer with temperature annealing (1.0 -> 0.5) for discrete assignments.

    Args:
        features: [N_samples, N_features] feature data
        num_categories: Number of target categories
        init_assignments: Optional initialization from clustering (strongly recommended)
        num_epochs: Training epochs (default 500)
        learning_rate: Adam learning rate (default 0.01)
        orthogonality_weight: Weight for orthogonality loss (default 1.0)
        informativeness_weight: Weight for informativeness loss (default 1.0)
        orthogonality_target: Target max correlation (used for early stopping)
        min_features_per_category: Min features per category (validation only)
        random_seed: Random seed for reproducibility
        subsample_excess_fraction: Fraction of excess samples for datasets > 10K
        device: Computation device ('cuda' or 'cpu'), defaults to 'cuda'

    Returns:
        Optimized category assignments
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Device selection: GPU-first
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    N_samples, N_features = features.shape

    # Smart subsampling for gradient optimization
    subsample_size = compute_subsample_size(N_samples, subsample_excess_fraction)
    if subsample_size < N_samples:
        print(f"  Subsampling for gradient optimization: {subsample_size:,} / {N_samples:,} samples")
        indices = np.random.choice(N_samples, subsample_size, replace=False)
        features = features[indices]
        N_samples = subsample_size

    # Initialize assignment module
    init_logits = None
    if init_assignments is not None:
        # Convert hard assignments to soft logits (strong initialization)
        init_logits = torch.zeros(N_features, num_categories)
        for cat_idx, (cat_name, indices) in enumerate(sorted(init_assignments.items())):
            init_logits[indices, cat_idx] = 5.0  # Strong confidence

    assigner = DifferentiableCategoryAssigner(
        num_features=N_features,
        num_categories=num_categories,
        temperature=1.0,
        init_logits=init_logits,
    ).to(device)

    optimizer = torch.optim.Adam(assigner.parameters(), lr=learning_rate)
    features_tensor = torch.from_numpy(features).float().to(device)

    print(f"\n  Gradient-based refinement:")
    print(f"    Device: {device}")
    print(f"    Epochs: {num_epochs}")
    print(f"    Learning rate: {learning_rate}")
    print(f"    Target orthogonality: <{orthogonality_target}")

    # Training loop
    best_loss = float("inf")
    best_state_dict = None
    early_stopped = False

    pbar = tqdm(
        range(num_epochs),
        desc="  Gradient refinement",
        unit="epoch",
        ncols=100,
        leave=True,
    )

    for epoch in pbar:
        optimizer.zero_grad()

        # Get soft assignments [N_features, N_categories]
        assignment_probs = assigner(training=True)

        # Compute orthogonality loss (minimize correlation between category embeddings)
        orthogonality_loss = compute_soft_orthogonality_loss(
            features_tensor, assignment_probs
        )

        # Compute informativeness loss (maximize per-category reconstruction)
        informativeness_loss = compute_soft_informativeness_loss(
            features_tensor, assignment_probs
        )

        # Total loss
        loss = (
            orthogonality_weight * orthogonality_loss
            + informativeness_weight * informativeness_loss
        )

        # Backprop + optimize
        loss.backward()
        optimizer.step()

        # Anneal temperature (make assignments more discrete over time)
        # Linear annealing: 1.0 -> 0.5
        assigner.temperature = max(0.5, 1.0 - (epoch / num_epochs) * 0.5)

        # Track best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state_dict = {k: v.clone() for k, v in assigner.state_dict().items()}

        # Update progress bar
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ortho=f"{orthogonality_loss.item():.4f}",
            temp=f"{assigner.temperature:.2f}",
        )

        # Early stopping if orthogonality target met
        if orthogonality_loss.item() < orthogonality_target:
            pbar.close()
            print(f"  ✓ Orthogonality target met: {orthogonality_loss.item():.4f} < {orthogonality_target}")
            print(f"    Early stopping at epoch {epoch}")
            early_stopped = True
            break

    if not early_stopped:
        pbar.close()

    # Restore best model
    if best_state_dict is not None:
        assigner.load_state_dict(best_state_dict)
        print(f"  Best loss: {best_loss:.4f}")

    # Extract hard assignments
    final_assignments = assigner.get_hard_assignments()

    # Validate assignments
    empty_categories = [
        cat for cat, indices in final_assignments.items() if len(indices) == 0
    ]
    small_categories = [
        (cat, len(indices))
        for cat, indices in final_assignments.items()
        if 0 < len(indices) < min_features_per_category
    ]

    if empty_categories:
        print(f"  ⚠ Empty categories: {empty_categories}")
    if small_categories:
        print(f"  ⚠ Small categories (<{min_features_per_category} features): {small_categories}")

    print(f"  Final: {len(final_assignments)} non-empty categories")

    return final_assignments


def compute_soft_orthogonality_loss(
    features: torch.Tensor, assignment_probs: torch.Tensor
) -> torch.Tensor:
    """Compute orthogonality loss for soft category assignments.

    Measures how independent categories are by computing correlation between
    category-aggregated feature embeddings.

    Loss = mean absolute off-diagonal correlation

    Args:
        features: [N_samples, N_features] feature data
        assignment_probs: [N_features, N_categories] soft assignments (rows sum to 1)

    Returns:
        Orthogonality loss (lower = more orthogonal categories)
        Range: [0, 1], target: <0.3
    """
    # Weighted feature embeddings per category
    # category_features[i, j] = sum_k (assignment_probs[k, j] * features[i, k])
    # Shape: [N_samples, N_categories]
    category_features = torch.matmul(features, assignment_probs)

    # Normalize each category embedding (L2 norm across samples)
    category_features_norm = F.normalize(category_features, p=2, dim=0)

    # Compute correlation matrix [N_categories, N_categories]
    # corr[i, j] = dot(category_i_normalized, category_j_normalized)
    corr_matrix = torch.matmul(category_features_norm.T, category_features_norm)

    # Off-diagonal correlations (penalize high correlation between different categories)
    N_categories = assignment_probs.size(1)
    mask = ~torch.eye(N_categories, dtype=torch.bool, device=features.device)
    off_diagonal = torch.abs(corr_matrix[mask])

    # Loss = mean absolute off-diagonal correlation
    loss = off_diagonal.mean()

    return loss


def compute_soft_informativeness_loss(
    features: torch.Tensor, assignment_probs: torch.Tensor
) -> torch.Tensor:
    """Compute informativeness loss for soft category assignments.

    Measures reconstruction quality when using only category-aggregated features.

    Process:
    1. Aggregate features per category (weighted by assignment probs)
    2. Reconstruct full feature space from category aggregates
    3. Compute MSE between reconstruction and original

    Args:
        features: [N_samples, N_features] feature data
        assignment_probs: [N_features, N_categories] soft assignments

    Returns:
        Informativeness loss (lower = more informative categories)
        MSE reconstruction error
    """
    # For each category, aggregate features using soft assignments
    # category_reconstructions[i, j] = sum_k (assignment_probs[k, j] * features[i, k])
    # Shape: [N_samples, N_categories]
    category_reconstructions = torch.matmul(features, assignment_probs)

    # Project back to full feature space
    # full_reconstruction[i, k] = sum_j (assignment_probs[k, j] * category_reconstructions[i, j])
    # Shape: [N_samples, N_features]
    reconstructions = torch.matmul(category_reconstructions, assignment_probs.T)

    # MSE loss
    loss = F.mse_loss(reconstructions, features)

    return loss


def validate_gradient_assignments(
    features: np.ndarray,
    assignments: Dict[str, List[int]],
    min_features_per_category: int = 3,
) -> Dict[str, float]:
    """Validate optimized assignments on original feature data.

    Computes final orthogonality and informativeness metrics.

    Args:
        features: [N_samples, N_features] feature data
        assignments: Optimized category assignments
        min_features_per_category: Minimum features per category

    Returns:
        Validation metrics dict with keys:
            - max_correlation: Max inter-category correlation
            - mean_correlation: Mean inter-category correlation
            - reconstruction_mse: Reconstruction MSE
            - num_categories: Number of non-empty categories
            - num_small_categories: Categories with <min features
    """
    from scipy.stats import pearsonr

    category_names = list(assignments.keys())
    N_categories = len(category_names)

    # Compute category centroids
    centroids = []
    for cat_name in category_names:
        indices = assignments[cat_name]
        if len(indices) == 0:
            continue
        centroid = features[:, indices].mean(axis=1)
        centroids.append(centroid)

    # Inter-category correlation
    if len(centroids) < 2:
        max_corr = 0.0
        mean_corr = 0.0
    else:
        correlations = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                corr, _ = pearsonr(centroids[i], centroids[j])
                correlations.append(abs(corr))
        max_corr = max(correlations) if correlations else 0.0
        mean_corr = float(np.mean(correlations)) if correlations else 0.0

    # Reconstruction MSE
    reconstructions = np.zeros_like(features)
    for cat_name in category_names:
        indices = assignments[cat_name]
        if len(indices) == 0:
            continue
        centroid = features[:, indices].mean(axis=1, keepdims=True)
        reconstructions[:, indices] = np.tile(centroid, (1, len(indices)))

    reconstruction_mse = float(np.mean((features - reconstructions) ** 2))

    # Count small categories
    num_small = sum(
        1
        for indices in assignments.values()
        if 0 < len(indices) < min_features_per_category
    )

    metrics = {
        "max_correlation": float(max_corr),
        "mean_correlation": mean_corr,
        "reconstruction_mse": reconstruction_mse,
        "num_categories": N_categories,
        "num_small_categories": num_small,
    }

    return metrics
