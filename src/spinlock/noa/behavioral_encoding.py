"""Behavioral encoding and signature extraction from token sequences.

Extracts compressed behavioral features from VQ-VAE token sequences:
- Token entropy (behavioral complexity)
- Level transitions (hierarchical progression patterns)
- Regime stability (run-length encoding)
- Trajectory characteristics (convergence, length, etc.)

These signatures enable:
1. Memory indexing (find behaviorally similar episodes)
2. Curiosity computation (novelty detection)
3. Regime classification (cluster by behavioral patterns)
"""

from typing import Dict, Any, List, Tuple
import torch
from torch import Tensor
import numpy as np
from scipy.stats import entropy as scipy_entropy


class BehavioralSignature:
    """Compressed representation of episode behavior from token sequences.

    A behavioral signature summarizes an episode's token sequence into
    a fixed-dimensional feature vector suitable for:
    - K-NN similarity search (episodic memory retrieval)
    - Clustering (regime discovery)
    - Curiosity signals (novelty quantification)
    """

    @staticmethod
    def from_tokens(
        token_sequence: Tensor,
        num_categories: int,
        num_levels: int,
    ) -> Dict[str, Any]:
        """Extract behavioral features from token sequence.

        Args:
            token_sequence: Token sequence [T, num_categories * num_levels]
                           Each row is concatenated tokens from all categories/levels
            num_categories: Number of VQ-VAE categories
            num_levels: Number of hierarchy levels per category

        Returns:
            Dictionary of behavioral features:
            - token_entropy: Shannon entropy of token distribution (complexity)
            - level_transition_counts: Transitions between hierarchy levels
            - regime_stability: Run-length statistics (how stable are regimes)
            - trajectory_length: Number of timesteps
            - final_tokens: Final token state (for simple comparison)
            - token_diversity: Fraction of unique tokens used
        """
        T = token_sequence.shape[0]
        num_tokens_per_step = num_categories * num_levels

        # Reshape to [T, num_categories, num_levels] for level analysis
        # token_sequence is [T, num_categories * num_levels]
        tokens_reshaped = token_sequence.view(T, num_categories, num_levels)

        # 1. Token entropy (behavioral complexity)
        # Flatten all tokens and compute distribution
        all_tokens = token_sequence.flatten().cpu().numpy()
        unique_tokens, counts = np.unique(all_tokens, return_counts=True)
        token_probs = counts / counts.sum()
        token_entropy = scipy_entropy(token_probs)

        # 2. Level transitions (hierarchical dynamics)
        # Count transitions between levels within each category
        level_transitions = BehavioralSignature._compute_level_transitions(
            tokens_reshaped, num_categories, num_levels
        )

        # 3. Regime stability (run-length encoding)
        # How long does system stay in same token state?
        stability_stats = BehavioralSignature._compute_stability(token_sequence)

        # 4. Trajectory length
        trajectory_length = T

        # 5. Final tokens (for simple matching)
        final_tokens = token_sequence[-1].cpu().numpy()

        # 6. Token diversity (exploration measure)
        num_unique = len(unique_tokens)
        total_possible = token_sequence.max().item() + 1  # Vocab size
        token_diversity = num_unique / total_possible if total_possible > 0 else 0.0

        return {
            "token_entropy": float(token_entropy),
            "level_transition_counts": level_transitions,
            "regime_stability": stability_stats,
            "trajectory_length": trajectory_length,
            "final_tokens": final_tokens,
            "token_diversity": float(token_diversity),
            "num_unique_tokens": int(num_unique),
        }

    @staticmethod
    def _compute_level_transitions(
        tokens: Tensor, num_categories: int, num_levels: int
    ) -> Dict[str, List[int]]:
        """Compute transitions between hierarchy levels.

        Args:
            tokens: Reshaped tokens [T, num_categories, num_levels]
            num_categories: Number of categories
            num_levels: Number of levels

        Returns:
            Dictionary with transition counts per category
        """
        T = tokens.shape[0]
        transitions = {f"category_{i}": [0] * num_levels for i in range(num_categories)}

        for cat_idx in range(num_categories):
            cat_tokens = tokens[:, cat_idx, :]  # [T, num_levels]

            for t in range(1, T):
                for level in range(num_levels):
                    if cat_tokens[t, level] != cat_tokens[t - 1, level]:
                        transitions[f"category_{cat_idx}"][level] += 1

        return transitions

    @staticmethod
    def _compute_stability(token_sequence: Tensor) -> Dict[str, float]:
        """Compute regime stability via run-length encoding.

        Args:
            token_sequence: Token sequence [T, D]

        Returns:
            Dictionary with stability statistics
        """
        T = token_sequence.shape[0]

        # Convert to numpy for run-length encoding
        tokens_np = token_sequence.cpu().numpy()

        # Compute run lengths (consecutive identical token states)
        run_lengths = []
        current_run = 1

        for t in range(1, T):
            if np.array_equal(tokens_np[t], tokens_np[t - 1]):
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1

        # Add final run
        run_lengths.append(current_run)

        # Statistics
        run_lengths = np.array(run_lengths)
        mean_run = float(run_lengths.mean()) if len(run_lengths) > 0 else 0.0
        max_run = int(run_lengths.max()) if len(run_lengths) > 0 else 0
        num_transitions = len(run_lengths) - 1

        return {
            "mean_run_length": mean_run,
            "max_run_length": max_run,
            "num_regime_transitions": num_transitions,
            "stability_ratio": mean_run / T if T > 0 else 0.0,  # Fraction of time stable
        }

    @staticmethod
    def similarity(sig1: Dict[str, Any], sig2: Dict[str, Any]) -> float:
        """Compute behavioral similarity between two signatures.

        Uses weighted combination of:
        - Token entropy difference (complexity similarity)
        - Final token cosine similarity (end-state matching)
        - Trajectory length ratio (temporal scale matching)

        Args:
            sig1, sig2: Behavioral signatures

        Returns:
            Similarity score in [0, 1] (1 = identical)
        """
        # 1. Entropy similarity (complexity)
        entropy_diff = abs(sig1["token_entropy"] - sig2["token_entropy"])
        max_entropy = max(sig1["token_entropy"], sig2["token_entropy"], 1.0)
        entropy_sim = 1.0 - (entropy_diff / max_entropy)

        # 2. Final token cosine similarity
        final1 = torch.tensor(sig1["final_tokens"], dtype=torch.float32)
        final2 = torch.tensor(sig2["final_tokens"], dtype=torch.float32)
        final_sim = torch.nn.functional.cosine_similarity(
            final1.unsqueeze(0), final2.unsqueeze(0)
        ).item()
        final_sim = (final_sim + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]

        # 3. Trajectory length similarity
        len1 = sig1["trajectory_length"]
        len2 = sig2["trajectory_length"]
        length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

        # Weighted combination
        # Weight final state higher (most distinctive)
        similarity = 0.5 * final_sim + 0.3 * entropy_sim + 0.2 * length_ratio

        return float(similarity)

    @staticmethod
    def to_vector(signature: Dict[str, Any], include_final_tokens: bool = True) -> np.ndarray:
        """Convert signature to fixed-dimensional vector for indexing.

        Args:
            signature: Behavioral signature
            include_final_tokens: Whether to include final_tokens in vector
                                 (increases dimensionality but adds specificity)

        Returns:
            Feature vector (numpy array)
        """
        features = [
            signature["token_entropy"],
            signature["token_diversity"],
            signature["trajectory_length"],
            signature["regime_stability"]["mean_run_length"],
            signature["regime_stability"]["max_run_length"],
            signature["regime_stability"]["num_regime_transitions"],
            signature["regime_stability"]["stability_ratio"],
        ]

        # Add transition counts (flatten)
        for cat_transitions in signature["level_transition_counts"].values():
            features.extend(cat_transitions)

        # Optionally add final tokens
        if include_final_tokens:
            features.extend(signature["final_tokens"].tolist())

        return np.array(features, dtype=np.float32)


class BehavioralEncoder:
    """Encode episodes into behavioral signatures for memory/analysis.

    Provides batch processing and caching for efficiency.
    """

    def __init__(self, num_categories: int, num_levels: int):
        """Initialize behavioral encoder.

        Args:
            num_categories: Number of VQ-VAE categories
            num_levels: Number of hierarchy levels per category
        """
        self.num_categories = num_categories
        self.num_levels = num_levels

    def encode_episode(self, episode) -> Dict[str, Any]:
        """Encode single episode into behavioral signature.

        Args:
            episode: Episode object with token_sequence attribute

        Returns:
            Behavioral signature dictionary
        """
        return BehavioralSignature.from_tokens(
            episode.token_sequence,
            self.num_categories,
            self.num_levels,
        )

    def encode_batch(self, episodes: List) -> List[Dict[str, Any]]:
        """Encode batch of episodes.

        Args:
            episodes: List of Episode objects

        Returns:
            List of behavioral signatures
        """
        return [self.encode_episode(ep) for ep in episodes]

    def similarity_matrix(self, signatures: List[Dict[str, Any]]) -> np.ndarray:
        """Compute pairwise similarity matrix for signatures.

        Args:
            signatures: List of behavioral signatures

        Returns:
            Similarity matrix [N, N] where N = len(signatures)
        """
        N = len(signatures)
        sim_matrix = np.zeros((N, N), dtype=np.float32)

        for i in range(N):
            for j in range(i, N):
                sim = BehavioralSignature.similarity(signatures[i], signatures[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        return sim_matrix


class RegimeClassifier:
    """Classify episodes into behavioral regimes via clustering.

    Uses behavioral signatures to discover natural groupings of
    episode behaviors (e.g., chaotic, periodic, equilibrium).
    """

    def __init__(self, encoder: BehavioralEncoder):
        """Initialize regime classifier.

        Args:
            encoder: Behavioral encoder
        """
        self.encoder = encoder
        self.cluster_labels = None
        self.cluster_centers = None

    def fit(
        self,
        episodes: List,
        n_clusters: int = 10,
        method: str = "kmeans",
    ) -> np.ndarray:
        """Cluster episodes into behavioral regimes.

        Args:
            episodes: List of Episode objects
            n_clusters: Number of regimes to discover
            method: Clustering method ("kmeans", "hierarchical", "dbscan")

        Returns:
            Cluster labels [N] where N = len(episodes)
        """
        from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

        # Encode episodes
        signatures = self.encoder.encode_batch(episodes)

        # Convert to vectors
        vectors = np.stack(
            [BehavioralSignature.to_vector(sig, include_final_tokens=False) for sig in signatures]
        )

        # Cluster
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == "hierarchical":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        labels = clusterer.fit_predict(vectors)

        self.cluster_labels = labels
        if hasattr(clusterer, "cluster_centers_"):
            self.cluster_centers = clusterer.cluster_centers_

        return labels

    def predict(self, episode) -> int:
        """Predict regime for new episode.

        Args:
            episode: Episode object

        Returns:
            Cluster label (regime ID)
        """
        if self.cluster_centers is None:
            raise ValueError("Must call fit() before predict()")

        # Encode episode
        signature = self.encoder.encode_episode(episode)
        vector = BehavioralSignature.to_vector(signature, include_final_tokens=False)

        # Find nearest cluster center
        distances = np.linalg.norm(self.cluster_centers - vector, axis=1)
        return int(np.argmin(distances))
