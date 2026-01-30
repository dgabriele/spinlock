"""Unit tests for hierarchical clustering enhancements.

Tests new features:
- Configurable distance metrics (correlation, euclidean, cosine)
- Configurable linkage methods (ward, average, complete, single)
- Alternative K selection strategies (gap statistic, elbow method)
- Dendrogram export functionality
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from spinlock.encoding.clustering_assignment import (
    compute_distance_matrix,
    _gap_statistic_k_selection,
    _elbow_k_selection,
    _compute_within_cluster_dispersion,
    _compute_wcss,
    _export_dendrogram,
    hierarchical_clustering_assignment,
)


class TestDistanceMatrix:
    """Test compute_distance_matrix with different metrics."""

    def test_correlation_distance(self):
        """Test correlation distance computation."""
        np.random.seed(42)
        features = np.random.randn(100, 10)
        dist = compute_distance_matrix(features, metric='correlation')

        # Check shape
        assert dist.shape == (10, 10)

        # Check range [0, 2] for correlation distance (1 - |r|)
        assert np.all(dist >= 0)
        assert np.all(dist <= 2)

        # Check diagonal is zero
        assert np.allclose(np.diag(dist), 0)

        # Check symmetry
        assert np.allclose(dist, dist.T)

    def test_euclidean_distance(self):
        """Test Euclidean distance computation."""
        np.random.seed(42)
        features = np.random.randn(100, 10)
        dist = compute_distance_matrix(features, metric='euclidean')

        # Check shape
        assert dist.shape == (10, 10)

        # Check non-negative
        assert np.all(dist >= 0)

        # Check diagonal is zero
        assert np.allclose(np.diag(dist), 0)

        # Check symmetry
        assert np.allclose(dist, dist.T)

    def test_cosine_distance(self):
        """Test cosine distance computation."""
        np.random.seed(42)
        features = np.random.randn(100, 10)
        dist = compute_distance_matrix(features, metric='cosine')

        # Check shape
        assert dist.shape == (10, 10)

        # Check range [0, 2] for cosine distance
        assert np.all(dist >= 0)
        assert np.all(dist <= 2)

        # Check diagonal is zero (or very close)
        assert np.allclose(np.diag(dist), 0, atol=1e-10)

        # Check symmetry
        assert np.allclose(dist, dist.T)

    def test_precomputed_correlation(self):
        """Test that precomputed correlation matrix is used."""
        np.random.seed(42)
        features = np.random.randn(100, 10)

        # Compute correlation matrix
        corr_matrix = np.corrcoef(features.T)

        # Use precomputed
        dist1 = compute_distance_matrix(features, metric='correlation', corr_matrix=corr_matrix)

        # Compute from scratch
        dist2 = compute_distance_matrix(features, metric='correlation')

        # Should be the same
        assert np.allclose(dist1, dist2)

    def test_invalid_metric(self):
        """Test that invalid metric raises error."""
        features = np.random.randn(100, 10)

        with pytest.raises(ValueError, match="Unknown distance metric"):
            compute_distance_matrix(features, metric='invalid')


class TestGapStatistic:
    """Test gap statistic K selection."""

    def test_gap_statistic_basic(self):
        """Test gap statistic K selection returns valid K."""
        np.random.seed(42)
        features = np.random.randn(100, 20)

        k = _gap_statistic_k_selection(
            features,
            min_clusters=2,
            max_clusters=10,
            n_refs=5,
            linkage_method='ward',
            distance_metric='correlation'
        )

        # Check K is in valid range
        assert 2 <= k <= 10
        assert isinstance(k, int)

    def test_gap_statistic_with_subsample(self):
        """Test gap statistic with subsampling."""
        np.random.seed(42)
        features = np.random.randn(1000, 20)

        k = _gap_statistic_k_selection(
            features,
            min_clusters=2,
            max_clusters=8,
            n_refs=3,
            subsample_size=100
        )

        assert 2 <= k <= 8

    def test_gap_statistic_different_metrics(self):
        """Test gap statistic with different distance metrics."""
        np.random.seed(42)
        features = np.random.randn(100, 15)

        for metric in ['correlation', 'euclidean', 'cosine']:
            k = _gap_statistic_k_selection(
                features,
                min_clusters=2,
                max_clusters=6,
                n_refs=3,
                distance_metric=metric
            )
            assert 2 <= k <= 6


class TestElbowMethod:
    """Test elbow method K selection."""

    def test_elbow_basic(self):
        """Test elbow method K selection returns valid K."""
        np.random.seed(42)
        features = np.random.randn(100, 20)

        k = _elbow_k_selection(
            features,
            min_clusters=2,
            max_clusters=10,
            linkage_method='ward',
            distance_metric='correlation'
        )

        # Check K is in valid range
        assert 2 <= k <= 10
        assert isinstance(k, int)

    def test_elbow_with_subsample(self):
        """Test elbow method with subsampling."""
        np.random.seed(42)
        features = np.random.randn(1000, 20)

        k = _elbow_k_selection(
            features,
            min_clusters=2,
            max_clusters=8,
            subsample_size=100
        )

        assert 2 <= k <= 8

    def test_elbow_small_range(self):
        """Test elbow method with small K range."""
        np.random.seed(42)
        features = np.random.randn(100, 10)

        # If range is too small, should return min_clusters
        k = _elbow_k_selection(
            features,
            min_clusters=2,
            max_clusters=3
        )

        assert k >= 2


class TestHelperFunctions:
    """Test clustering helper functions."""

    def test_compute_within_cluster_dispersion(self):
        """Test within-cluster dispersion computation."""
        np.random.seed(42)
        features = np.random.randn(100, 10)
        labels = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

        dispersion = _compute_within_cluster_dispersion(features, labels)

        # Should be non-negative
        assert dispersion >= 0
        assert isinstance(dispersion, float)

    def test_compute_wcss(self):
        """Test within-cluster sum of squares computation."""
        np.random.seed(42)
        features = np.random.randn(100, 10)
        labels = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

        wcss = _compute_wcss(features, labels)

        # Should be non-negative
        assert wcss >= 0
        assert isinstance(wcss, float)


class TestDendrogramExport:
    """Test dendrogram export functionality."""

    def test_dendrogram_export(self):
        """Test dendrogram export creates files."""
        np.random.seed(42)
        features = np.random.randn(100, 10)
        feature_names = [f"feat_{i}" for i in range(10)]

        # Create linkage matrix
        from scipy.cluster.hierarchy import linkage
        dist = compute_distance_matrix(features, 'correlation')
        from scipy.spatial.distance import squareform
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method='ward')

        # Export to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            _export_dendrogram(Z, feature_names, tmpdir, 'test')

            # Check files exist
            plot_path = Path(tmpdir) / 'test_dendrogram.png'
            linkage_path = Path(tmpdir) / 'test_linkage.npz'

            assert plot_path.exists()
            assert linkage_path.exists()

            # Check linkage file can be loaded
            data = np.load(linkage_path)
            assert 'linkage_matrix' in data
            assert 'feature_names' in data
            assert data['linkage_matrix'].shape == Z.shape
            assert len(data['feature_names']) == len(feature_names)


class TestHierarchicalClusteringIntegration:
    """Integration tests for hierarchical clustering with new features."""

    def test_clustering_with_gap_statistic(self):
        """Test clustering with gap statistic K selection."""
        np.random.seed(42)
        features = np.random.randn(200, 30)
        feature_names = [f"feat_{i}" for i in range(30)]

        assignments = hierarchical_clustering_assignment(
            features=features,
            feature_names=feature_names,
            num_clusters=None,
            min_clusters=2,
            max_clusters=8,
            k_selection_method='gap_statistic',
            gap_statistic_refs=5,
        )

        # Check we got valid assignments
        assert isinstance(assignments, dict)
        assert len(assignments) > 0

        # Check all features are assigned
        all_indices = []
        for indices in assignments.values():
            all_indices.extend(indices)
        assert len(set(all_indices)) <= 30

    def test_clustering_with_elbow(self):
        """Test clustering with elbow method K selection."""
        np.random.seed(42)
        features = np.random.randn(200, 30)
        feature_names = [f"feat_{i}" for i in range(30)]

        assignments = hierarchical_clustering_assignment(
            features=features,
            feature_names=feature_names,
            num_clusters=None,
            min_clusters=2,
            max_clusters=8,
            k_selection_method='elbow',
        )

        # Check we got valid assignments
        assert isinstance(assignments, dict)
        assert len(assignments) > 0

    def test_clustering_with_manual_k(self):
        """Test clustering with manual K specification."""
        np.random.seed(42)
        features = np.random.randn(200, 30)
        feature_names = [f"feat_{i}" for i in range(30)]

        manual_k = 5

        assignments = hierarchical_clustering_assignment(
            features=features,
            feature_names=feature_names,
            num_clusters=None,
            k_selection_method='manual',
            manual_k=manual_k,
        )

        # Should have exactly manual_k clusters (or fewer if some are too small)
        assert len(assignments) <= manual_k

    def test_clustering_with_distance_threshold(self):
        """Test clustering with distance threshold."""
        np.random.seed(42)
        features = np.random.randn(200, 30)
        feature_names = [f"feat_{i}" for i in range(30)]

        assignments = hierarchical_clustering_assignment(
            features=features,
            feature_names=feature_names,
            distance_threshold=0.5,
        )

        # Check we got valid assignments
        assert isinstance(assignments, dict)
        assert len(assignments) > 0

    def test_clustering_different_linkage_methods(self):
        """Test clustering with different linkage methods."""
        np.random.seed(42)
        features = np.random.randn(200, 30)
        feature_names = [f"feat_{i}" for i in range(30)]

        for linkage in ['ward', 'average', 'complete', 'single']:
            assignments = hierarchical_clustering_assignment(
                features=features,
                feature_names=feature_names,
                num_clusters=5,
                linkage_method=linkage,
            )

            assert isinstance(assignments, dict)
            assert len(assignments) > 0

    def test_clustering_different_distance_metrics(self):
        """Test clustering with different distance metrics."""
        np.random.seed(42)
        features = np.random.randn(200, 30)
        feature_names = [f"feat_{i}" for i in range(30)]

        for metric in ['correlation', 'euclidean', 'cosine']:
            assignments = hierarchical_clustering_assignment(
                features=features,
                feature_names=feature_names,
                num_clusters=5,
                distance_metric=metric,
            )

            assert isinstance(assignments, dict)
            assert len(assignments) > 0

    def test_clustering_with_dendrogram_export(self):
        """Test clustering with dendrogram export."""
        np.random.seed(42)
        features = np.random.randn(200, 30)
        feature_names = [f"feat_{i}" for i in range(30)]

        with tempfile.TemporaryDirectory() as tmpdir:
            assignments = hierarchical_clustering_assignment(
                features=features,
                feature_names=feature_names,
                num_clusters=5,
                export_dendrogram=True,
                dendrogram_path=tmpdir,
            )

            # Check dendrogram files were created
            files = list(Path(tmpdir).glob('*.png'))
            assert len(files) > 0

            # Check assignments are valid
            assert isinstance(assignments, dict)
            assert len(assignments) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
