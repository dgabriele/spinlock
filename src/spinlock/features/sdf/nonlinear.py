"""
Nonlinear dynamics feature extraction.

Extracts trajectory-level features capturing nonlinear structure and complexity:
- Recurrence Quantification Analysis (RQA): recurrence rate, determinism, laminarity
- Correlation dimension: attractor complexity via Grassberger-Procaccia algorithm

These features are computationally expensive (O(T²)) and use temporal subsampling
for efficiency. They are opt-in via configuration.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from spinlock.features.sdf.config import SDFNonlinearConfig


class NonlinearFeatureExtractor:
    """
    Extract nonlinear dynamics features from trajectories.

    Operates on full trajectories [N, M, T, C, H, W] and computes
    trajectory-level summaries using phase space analysis.

    Example:
        >>> extractor = NonlinearFeatureExtractor(device='cuda')
        >>> trajectories = torch.randn(32, 10, 500, 3, 128, 128, device='cuda')
        >>> features = extractor.extract(trajectories)  # [N, M, num_features]
    """

    def __init__(self, device: torch.device = torch.device('cuda')):
        """
        Initialize nonlinear feature extractor.

        Args:
            device: Computation device (cuda or cpu)
        """
        self.device = device

    def extract(
        self,
        trajectories: torch.Tensor,  # [N, M, T, C, H, W]
        config: Optional['SDFNonlinearConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract nonlinear dynamics features.

        Args:
            trajectories: [N, M, T, C, H, W] operator rollouts
            config: Feature configuration

        Returns:
            Dictionary mapping feature names to tensors [N, M, C]
        """
        N, M, T, C, H, W = trajectories.shape

        # Compute spatial mean at each timestep → [N, M, T, C]
        time_series = trajectories.mean(dim=(-2, -1))

        features = {}

        # Default config if not provided
        if config is None:
            from spinlock.features.sdf.config import SDFNonlinearConfig
            config = SDFNonlinearConfig()

        include_all = True  # Extract all by default

        # Recurrence Quantification Analysis
        if include_all or config.include_recurrence:
            rqa_features = self._compute_rqa_metrics(
                time_series,
                epsilon=config.rqa_epsilon,
                embedding_dim=config.rqa_embedding_dim,
                tau=config.rqa_tau,
                subsample_factor=config.rqa_subsample_factor
            )
            features.update(rqa_features)

        # Correlation dimension
        if include_all or config.include_correlation_dim:
            corr_dim = self._compute_correlation_dimension(
                time_series,
                embedding_dim=config.corr_dim_embedding_dim,
                tau=config.corr_dim_tau,
                subsample_factor=config.corr_dim_subsample_factor
            )
            features['correlation_dimension'] = corr_dim

        # Phase 2 extension: Permutation entropy
        if include_all or config.include_permutation_entropy:
            perm_entropy = self._compute_permutation_entropy(
                time_series,
                embedding_dim=config.perm_entropy_embedding_dim,
                tau=config.perm_entropy_tau,
                subsample_factor=config.perm_entropy_subsample_factor
            )
            features['permutation_entropy'] = perm_entropy

        return features

    # =========================================================================
    # Recurrence Quantification Analysis (RQA)
    # =========================================================================

    def _compute_rqa_metrics(
        self,
        time_series: torch.Tensor,  # [N, M, T, C]
        epsilon: float = 0.1,
        embedding_dim: int = 3,
        tau: int = 1,
        subsample_factor: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Recurrence Quantification Analysis metrics.

        Builds recurrence plot via phase space embedding and analyzes
        diagonal/vertical line structures to detect hidden periodicities
        and deterministic chaos.

        Args:
            time_series: [N, M, T, C] temporal trajectories
            epsilon: Recurrence threshold (fraction of diameter)
            embedding_dim: Phase space embedding dimension
            tau: Time delay for embedding
            subsample_factor: Temporal subsampling to reduce O(T²) cost

        Returns:
            Dictionary with RQA features [N, M, C]:
                - recurrence_rate: % of recurrent points
                - determinism: % of recurrence points in diagonal structures
                - laminarity: % of recurrence points in vertical structures
                - entropy_diag_length: Shannon entropy of diagonal line lengths
        """
        N, M, T, C = time_series.shape

        # Temporal subsampling to reduce O(T²) cost
        T_sub = T // subsample_factor
        time_series_sub = time_series[:, :, ::subsample_factor, :]  # [N, M, T_sub, C]

        # Phase space embedding using time delay
        # For each channel independently
        result = {}

        for c in range(C):
            ts_c = time_series_sub[:, :, :, c]  # [N, M, T_sub]

            # Embed trajectory: create delayed coordinates
            # embedded shape: [N, M, T_sub - (embedding_dim-1)*tau, embedding_dim]
            embedded = []
            T_embedded = T_sub - (embedding_dim - 1) * tau
            if T_embedded < 2:
                # Not enough timesteps for embedding
                result[f'recurrence_rate_c{c}'] = torch.zeros(N, M, device=self.device)
                result[f'determinism_c{c}'] = torch.zeros(N, M, device=self.device)
                result[f'laminarity_c{c}'] = torch.zeros(N, M, device=self.device)
                result[f'entropy_diag_length_c{c}'] = torch.zeros(N, M, device=self.device)
                continue

            for i in range(embedding_dim):
                start_idx = i * tau
                end_idx = start_idx + T_embedded
                embedded.append(ts_c[:, :, start_idx:end_idx].unsqueeze(-1))

            embedded = torch.cat(embedded, dim=-1)  # [N, M, T_embedded, embedding_dim]

            # Compute pairwise distances in phase space
            # embedded_i: [N, M, T_embedded, 1, embedding_dim]
            # embedded_j: [N, M, 1, T_embedded, embedding_dim]
            embedded_i = embedded.unsqueeze(3)
            embedded_j = embedded.unsqueeze(2)

            # Euclidean distance: [N, M, T_embedded, T_embedded]
            distances = torch.norm(embedded_i - embedded_j, dim=-1)

            # Normalize epsilon by trajectory diameter (max distance)
            diameter = distances.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            threshold = epsilon * diameter

            # Recurrence matrix: [N, M, T_embedded, T_embedded]
            recurrence_matrix = (distances < threshold).float()

            # RQA metrics (per realization)
            recurrence_rate_vals = []
            determinism_vals = []
            laminarity_vals = []
            entropy_vals = []

            for n in range(N):
                for m in range(M):
                    R = recurrence_matrix[n, m].cpu().numpy()

                    # Recurrence rate: % of recurrent points
                    rr = R.sum() / (T_embedded ** 2)

                    # Determinism: % of recurrent points forming diagonal lines (length >= 2)
                    diag_points = self._count_diagonal_points(R, min_length=2)
                    det = diag_points / max(R.sum(), 1e-8)

                    # Laminarity: % of recurrent points forming vertical lines (length >= 2)
                    vert_points = self._count_vertical_points(R, min_length=2)
                    lam = vert_points / max(R.sum(), 1e-8)

                    # Entropy of diagonal line lengths
                    diag_lengths = self._extract_diagonal_lengths(R, min_length=2)
                    if len(diag_lengths) > 0:
                        hist, _ = np.histogram(diag_lengths, bins=20)
                        hist = hist / hist.sum()
                        hist = hist[hist > 0]  # Remove zeros
                        ent = -np.sum(hist * np.log(hist))
                    else:
                        ent = 0.0

                    recurrence_rate_vals.append(rr)
                    determinism_vals.append(det)
                    laminarity_vals.append(lam)
                    entropy_vals.append(ent)

            # Convert to tensors [N, M]
            recurrence_rate_tensor = torch.tensor(recurrence_rate_vals, device=self.device).reshape(N, M)
            determinism_tensor = torch.tensor(determinism_vals, device=self.device).reshape(N, M)
            laminarity_tensor = torch.tensor(laminarity_vals, device=self.device).reshape(N, M)
            entropy_tensor = torch.tensor(entropy_vals, device=self.device).reshape(N, M)

            # Store per-channel
            result[f'recurrence_rate_c{c}'] = recurrence_rate_tensor
            result[f'determinism_c{c}'] = determinism_tensor
            result[f'laminarity_c{c}'] = laminarity_tensor
            result[f'entropy_diag_length_c{c}'] = entropy_tensor

        # Aggregate across channels (mean)
        if C > 1:
            result['recurrence_rate'] = torch.stack([result[f'recurrence_rate_c{c}'] for c in range(C)], dim=-1).mean(dim=-1)
            result['determinism'] = torch.stack([result[f'determinism_c{c}'] for c in range(C)], dim=-1).mean(dim=-1)
            result['laminarity'] = torch.stack([result[f'laminarity_c{c}'] for c in range(C)], dim=-1).mean(dim=-1)
            result['entropy_diag_length'] = torch.stack([result[f'entropy_diag_length_c{c}'] for c in range(C)], dim=-1).mean(dim=-1)

        return result

    def _count_diagonal_points(self, R: np.ndarray, min_length: int = 2) -> int:
        """Count recurrence points forming diagonal lines of length >= min_length."""
        T = R.shape[0]
        count = 0

        for diag_offset in range(-T + 1, T):
            diag = np.diagonal(R, offset=diag_offset)
            count += self._count_consecutive_points(diag, min_length)

        return count

    def _count_vertical_points(self, R: np.ndarray, min_length: int = 2) -> int:
        """Count recurrence points forming vertical lines of length >= min_length."""
        T = R.shape[0]
        count = 0

        for col in range(T):
            vertical_line = R[:, col]
            count += self._count_consecutive_points(vertical_line, min_length)

        return count

    def _count_consecutive_points(self, line: np.ndarray, min_length: int) -> int:
        """Count points in consecutive sequences of length >= min_length."""
        count = 0
        current_length = 0

        for val in line:
            if val > 0:
                current_length += 1
            else:
                if current_length >= min_length:
                    count += current_length
                current_length = 0

        # Final sequence
        if current_length >= min_length:
            count += current_length

        return count

    def _extract_diagonal_lengths(self, R: np.ndarray, min_length: int = 2) -> list:
        """Extract lengths of diagonal line structures."""
        T = R.shape[0]
        lengths = []

        for diag_offset in range(-T + 1, T):
            diag = np.diagonal(R, offset=diag_offset)
            current_length = 0

            for val in diag:
                if val > 0:
                    current_length += 1
                else:
                    if current_length >= min_length:
                        lengths.append(current_length)
                    current_length = 0

            # Final sequence
            if current_length >= min_length:
                lengths.append(current_length)

        return lengths

    # =========================================================================
    # Correlation Dimension
    # =========================================================================

    def _compute_correlation_dimension(
        self,
        time_series: torch.Tensor,  # [N, M, T, C]
        embedding_dim: int = 5,
        tau: int = 1,
        subsample_factor: int = 10
    ) -> torch.Tensor:
        """
        Estimate correlation dimension D2 via Grassberger-Procaccia algorithm.

        Measures attractor complexity by analyzing how correlation integral
        C(r) scales with radius r in phase space. D2 ≈ d(log C(r)) / d(log r).

        Low D2 → simple attractor (limit cycle, fixed point)
        High D2 → complex/chaotic attractor

        Args:
            time_series: [N, M, T, C] temporal trajectories
            embedding_dim: Phase space embedding dimension
            tau: Time delay for embedding
            subsample_factor: Temporal subsampling to reduce O(T²) cost

        Returns:
            Tensor [N, M, C] with correlation dimension estimates
        """
        N, M, T, C = time_series.shape

        # Temporal subsampling
        T_sub = T // subsample_factor
        time_series_sub = time_series[:, :, ::subsample_factor, :]  # [N, M, T_sub, C]

        result = torch.zeros(N, M, C, device=self.device)

        for c in range(C):
            ts_c = time_series_sub[:, :, :, c]  # [N, M, T_sub]

            # Phase space embedding
            embedded = []
            T_embedded = T_sub - (embedding_dim - 1) * tau
            if T_embedded < 10:
                # Not enough points for reliable estimate
                continue

            for i in range(embedding_dim):
                start_idx = i * tau
                end_idx = start_idx + T_embedded
                embedded.append(ts_c[:, :, start_idx:end_idx].unsqueeze(-1))

            embedded = torch.cat(embedded, dim=-1)  # [N, M, T_embedded, embedding_dim]

            # Compute pairwise distances
            embedded_i = embedded.unsqueeze(3)
            embedded_j = embedded.unsqueeze(2)
            distances = torch.norm(embedded_i - embedded_j, dim=-1)  # [N, M, T_embedded, T_embedded]

            # Use log-spaced radii (10 points between 1st and 99th percentile of distances)
            for n in range(N):
                for m in range(M):
                    dists_flat = distances[n, m].flatten()
                    r_min = torch.quantile(dists_flat, 0.01).item()
                    r_max = torch.quantile(dists_flat, 0.99).item()

                    if r_max <= r_min:
                        continue

                    radii = torch.logspace(
                        np.log10(r_min), np.log10(r_max), steps=10, device=self.device
                    )

                    # Compute correlation integral C(r) for each radius
                    C_r = []
                    for r in radii:
                        count = (distances[n, m] < r).float().sum().item()
                        C_r.append(count / (T_embedded ** 2))

                    C_r = np.array(C_r)
                    log_r = np.log(radii.cpu().numpy())
                    log_C = np.log(C_r + 1e-10)

                    # Fit slope in log-log plot (correlation dimension)
                    # D2 ≈ slope of log(C) vs log(r)
                    valid_idx = np.isfinite(log_C) & (C_r > 1e-6)
                    if valid_idx.sum() >= 3:
                        slope, _ = np.polyfit(log_r[valid_idx], log_C[valid_idx], deg=1)
                        result[n, m, c] = slope
                    else:
                        result[n, m, c] = 0.0

        return result

    # =========================================================================
    # Permutation Entropy (Phase 2)
    # =========================================================================

    def _compute_permutation_entropy(
        self,
        time_series: torch.Tensor,  # [N, M, T, C]
        embedding_dim: int = 3,
        tau: int = 1,
        subsample_factor: int = 10
    ) -> torch.Tensor:
        """
        Compute permutation entropy (ordinal pattern complexity).

        Measures complexity of temporal ordinal patterns. Higher values indicate
        more unpredictable ordering (randomness), lower values indicate regularity.

        Args:
            time_series: [N, M, T, C] temporal trajectories
            embedding_dim: Order of permutation patterns (default: 3)
            tau: Time delay for embedding (default: 1)
            subsample_factor: Temporal subsampling to reduce cost

        Returns:
            Tensor [N, M, C] with permutation entropy values
        """
        N, M, T, C = time_series.shape

        # Temporal subsampling
        T_sub = T // subsample_factor
        time_series_sub = time_series[:, :, ::subsample_factor, :]  # [N, M, T_sub, C]

        result = torch.zeros(N, M, C, device=self.device)

        # Number of possible ordinal patterns
        from math import factorial
        num_patterns = factorial(embedding_dim)

        for c in range(C):
            ts_c = time_series_sub[:, :, :, c]  # [N, M, T_sub]

            # Build delay embedding
            T_embedded = T_sub - (embedding_dim - 1) * tau
            if T_embedded < 2 * num_patterns:
                # Not enough points for reliable entropy estimate
                continue

            # Extract delay vectors
            embedded = []
            for i in range(embedding_dim):
                start_idx = i * tau
                end_idx = start_idx + T_embedded
                embedded.append(ts_c[:, :, start_idx:end_idx].unsqueeze(-1))

            embedded = torch.cat(embedded, dim=-1)  # [N, M, T_embedded, embedding_dim]

            # Convert to ordinal patterns (rank ordering)
            # For each embedded vector, get permutation that sorts it
            for n in range(N):
                for m in range(M):
                    vectors = embedded[n, m].cpu().numpy()  # [T_embedded, embedding_dim]

                    # Get ordinal patterns (argsort gives ranks)
                    patterns = []
                    for vec in vectors:
                        # argsort gives indices that would sort the array
                        # This represents the ordinal pattern
                        pattern = tuple(np.argsort(vec))
                        patterns.append(pattern)

                    # Count pattern frequencies
                    from collections import Counter
                    pattern_counts = Counter(patterns)
                    probabilities = np.array(list(pattern_counts.values())) / len(patterns)

                    # Compute Shannon entropy of pattern distribution
                    # H = -sum(p * log(p))
                    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))

                    # Normalize by max entropy (log(num_patterns))
                    max_entropy = np.log(num_patterns)
                    normalized_entropy = entropy / (max_entropy + 1e-10)

                    result[n, m, c] = normalized_entropy

        return result

    def aggregate_realizations(
        self,
        features: Dict[str, torch.Tensor],
        methods: list = ['mean', 'std', 'cv']
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate features across realizations (M dimension).

        Args:
            features: Dictionary mapping feature names to tensors [N, M, ...]
            methods: Aggregation methods to apply

        Returns:
            Dictionary with aggregated features [N, ...]
        """
        aggregated = {}

        for name, values in features.items():
            # values shape: [N, M, ...] or [N, M]
            for method in methods:
                if method == 'mean':
                    aggregated[f'{name}_mean'] = values.mean(dim=1)
                elif method == 'std':
                    aggregated[f'{name}_std'] = values.std(dim=1)
                elif method == 'cv':
                    # Coefficient of variation: std / mean
                    mean = values.mean(dim=1)
                    std = values.std(dim=1)
                    aggregated[f'{name}_cv'] = std / (mean.abs() + 1e-8)
                elif method == 'min':
                    aggregated[f'{name}_min'] = values.min(dim=1)[0]
                elif method == 'max':
                    aggregated[f'{name}_max'] = values.max(dim=1)[0]

        return aggregated
