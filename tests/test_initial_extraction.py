"""
Comprehensive tests for IC (Initial Condition) feature extraction.

Tests all components:
1. Manual feature extractor (14D features)
2. CNN encoder (28D learned features)
3. Hybrid extractor (42D combined features)
4. VAE mode (generative capability)
5. Feature registry
6. HDF5 storage integration
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import h5py

from spinlock.features.initial.config import InitialConfig, InitialManualConfig, InitialCNNConfig
from spinlock.features.initial.manual_extractors import InitialManualExtractor
from spinlock.features.initial.cnn_encoder import InitialCNNEncoder, InitialVAE, InitialCNNDecoder
from spinlock.features.initial.extractors import InitialExtractor
from spinlock.features.storage import HDF5FeatureWriter, HDF5FeatureReader


# Fixtures
@pytest.fixture
def device():
    """Device for testing (CPU to ensure tests work everywhere)."""
    return torch.device('cpu')


@pytest.fixture
def synthetic_ics():
    """Create synthetic ICs for testing [B=4, M=3, C=1, H=128, W=128]."""
    B, M, C, H, W = 4, 3, 1, 128, 128
    # Create diverse ICs with different patterns
    ics = torch.randn(B, M, C, H, W)
    # Normalize to reasonable range
    ics = torch.tanh(ics)
    return ics


@pytest.fixture
def ic_config():
    """Default IC configuration."""
    return InitialConfig()


@pytest.fixture
def ic_config_vae():
    """IC configuration with VAE enabled."""
    config = InitialConfig()
    config.cnn.use_vae = True
    return config


# Manual Extractor Tests
class TestManualExtractor:
    """Tests for InitialManualExtractor."""

    def test_manual_extractor_shape(self, device, synthetic_ics):
        """Test that manual extractor produces correct shape."""
        extractor = InitialManualExtractor(device=device)
        features = extractor.extract_all(synthetic_ics)

        B, M = synthetic_ics.shape[:2]
        assert features.shape == (B, M, 14), f"Expected [4, 3, 14], got {features.shape}"

    def test_manual_features_not_nan(self, device, synthetic_ics):
        """Test that manual features don't contain NaN values."""
        extractor = InitialManualExtractor(device=device)
        features = extractor.extract_all(synthetic_ics)

        assert not torch.isnan(features).any(), "Manual features contain NaN values"

    def test_manual_features_not_inf(self, device, synthetic_ics):
        """Test that manual features don't contain infinite values."""
        extractor = InitialManualExtractor(device=device)
        features = extractor.extract_all(synthetic_ics)

        assert not torch.isinf(features).any(), "Manual features contain infinite values"

    def test_manual_features_variance(self, device, synthetic_ics):
        """Test that manual features have reasonable variance (not all constant)."""
        extractor = InitialManualExtractor(device=device)
        features = extractor.extract_all(synthetic_ics)

        # Check variance across batch dimension
        variance = features.var(dim=0)
        assert (variance > 1e-6).any(), "Manual features have no variance"

    def test_spatial_features(self, device, synthetic_ics):
        """Test spatial feature extraction."""
        extractor = InitialManualExtractor(device=device)
        feature_dict = extractor.extract_spatial_features(synthetic_ics)

        # Manual extractors return dictionaries with feature names as keys
        assert isinstance(feature_dict, dict)
        assert len(feature_dict) == 4, f"Expected 4 spatial features, got {len(feature_dict)}"

    def test_spectral_features(self, device, synthetic_ics):
        """Test spectral feature extraction."""
        extractor = InitialManualExtractor(device=device)
        feature_dict = extractor.extract_spectral_features(synthetic_ics)

        assert isinstance(feature_dict, dict)
        assert len(feature_dict) == 3, f"Expected 3 spectral features, got {len(feature_dict)}"

    def test_information_features(self, device, synthetic_ics):
        """Test information feature extraction."""
        extractor = InitialManualExtractor(device=device)
        feature_dict = extractor.extract_information_features(synthetic_ics)

        assert isinstance(feature_dict, dict)
        assert len(feature_dict) == 4, f"Expected 4 information features, got {len(feature_dict)}"

    def test_morphological_features(self, device, synthetic_ics):
        """Test morphological feature extraction."""
        extractor = InitialManualExtractor(device=device)
        feature_dict = extractor.extract_morphological_features(synthetic_ics)

        assert isinstance(feature_dict, dict)
        assert len(feature_dict) == 3, f"Expected 3 morphological features, got {len(feature_dict)}"


# CNN Encoder Tests
class TestCNNEncoder:
    """Tests for InitialCNNEncoder."""

    def test_encoder_shape(self, device, synthetic_ics):
        """Test that CNN encoder produces correct shape."""
        encoder = InitialCNNEncoder(embedding_dim=28).to(device)

        B, M, C, H, W = synthetic_ics.shape
        # Reshape to [B*M, C, H, W] for CNN
        ics_flat = synthetic_ics.view(B * M, C, H, W)

        with torch.no_grad():
            embeddings = encoder(ics_flat)

        assert embeddings.shape == (B * M, 28), f"Expected [{B*M}, 28], got {embeddings.shape}"

    def test_encoder_output_not_nan(self, device, synthetic_ics):
        """Test that encoder output doesn't contain NaN values."""
        encoder = InitialCNNEncoder(embedding_dim=28).to(device)

        B, M, C, H, W = synthetic_ics.shape
        ics_flat = synthetic_ics.view(B * M, C, H, W)

        with torch.no_grad():
            embeddings = encoder(ics_flat)

        assert not torch.isnan(embeddings).any(), "Encoder output contains NaN values"

    def test_encoder_batch_consistency(self, device, synthetic_ics):
        """Test that encoder produces consistent results for same input."""
        encoder = InitialCNNEncoder(embedding_dim=28).to(device)
        encoder.eval()

        B, M, C, H, W = synthetic_ics.shape
        ics_flat = synthetic_ics.view(B * M, C, H, W)

        with torch.no_grad():
            embeddings1 = encoder(ics_flat)
            embeddings2 = encoder(ics_flat)

        assert torch.allclose(embeddings1, embeddings2, atol=1e-6), \
            "Encoder not deterministic in eval mode"


# VAE Tests
class TestICVAE:
    """Tests for InitialVAE (Variational Autoencoder)."""

    def test_vae_encode_shape(self, device, synthetic_ics):
        """Test VAE encoder produces correct mu and logvar shapes."""
        vae = InitialVAE(embedding_dim=28).to(device)

        B, M, C, H, W = synthetic_ics.shape
        ics_flat = synthetic_ics.view(B * M, C, H, W)

        with torch.no_grad():
            mu, logvar = vae.encode(ics_flat)

        assert mu.shape == (B * M, 28), f"Expected mu shape [{B*M}, 28], got {mu.shape}"
        assert logvar.shape == (B * M, 28), f"Expected logvar shape [{B*M}, 28], got {logvar.shape}"

    def test_vae_decode_shape(self, device):
        """Test VAE decoder produces correct IC shape."""
        vae = InitialVAE(embedding_dim=28).to(device)

        # Random latent code
        z = torch.randn(10, 28, device=device)

        with torch.no_grad():
            ics_reconstructed = vae.decode(z)

        assert ics_reconstructed.shape == (10, 1, 128, 128), \
            f"Expected [10, 1, 128, 128], got {ics_reconstructed.shape}"

    def test_vae_forward_shape(self, device, synthetic_ics):
        """Test VAE forward pass (encode → decode)."""
        vae = InitialVAE(embedding_dim=28).to(device)

        B, M, C, H, W = synthetic_ics.shape
        ics_flat = synthetic_ics.view(B * M, C, H, W)

        with torch.no_grad():
            ics_reconstructed, mu, logvar = vae(ics_flat)

        assert ics_reconstructed.shape == ics_flat.shape, \
            f"Expected reconstruction shape {ics_flat.shape}, got {ics_reconstructed.shape}"

    def test_vae_sample(self, device):
        """Test VAE sampling from prior."""
        vae = InitialVAE(embedding_dim=28).to(device)

        with torch.no_grad():
            sampled_ics = vae.sample(num_samples=5, device=device)

        assert sampled_ics.shape == (5, 1, 128, 128), \
            f"Expected [5, 1, 128, 128], got {sampled_ics.shape}"

    def test_vae_reconstruction_quality(self, device, synthetic_ics):
        """Test that VAE can reconstruct ICs (basic sanity check)."""
        vae = InitialVAE(embedding_dim=28).to(device)

        B, M, C, H, W = synthetic_ics.shape
        ics_flat = synthetic_ics.view(B * M, C, H, W)

        with torch.no_grad():
            ics_reconstructed, mu, logvar = vae(ics_flat)

        # Check that reconstruction is in reasonable range
        assert ics_reconstructed.min() >= -2.0, "Reconstruction values too small"
        assert ics_reconstructed.max() <= 2.0, "Reconstruction values too large"


# Hybrid Extractor Tests
class TestHybridExtractor:
    """Tests for InitialExtractor (hybrid manual + CNN)."""

    def test_hybrid_extractor_shape(self, device, synthetic_ics, ic_config):
        """Test that hybrid extractor produces correct combined shape."""
        extractor = InitialExtractor(config=ic_config, device=device)

        with torch.no_grad():
            feature_dict = extractor.extract_all(synthetic_ics)

        B, M = synthetic_ics.shape[:2]

        assert 'manual' in feature_dict
        assert 'cnn' in feature_dict
        assert 'combined' in feature_dict

        assert feature_dict['manual'].shape == (B, M, 14)
        assert feature_dict['cnn'].shape == (B, M, 28)
        assert feature_dict['combined'].shape == (B, M, 42)

    def test_hybrid_extractor_manual_only(self, device, synthetic_ics):
        """Test extraction with only manual features enabled."""
        config = InitialConfig()
        config.cnn.enabled = False

        extractor = InitialExtractor(config=config, device=device)

        with torch.no_grad():
            feature_dict = extractor.extract_all(synthetic_ics)

        B, M = synthetic_ics.shape[:2]

        assert feature_dict['manual'] is not None
        assert feature_dict['manual'].shape == (B, M, 14)
        # CNN should be None when disabled
        assert 'cnn' not in feature_dict or feature_dict['cnn'] is None
        assert feature_dict['combined'].shape == (B, M, 14)  # Only manual

    def test_hybrid_extractor_cnn_only(self, device, synthetic_ics):
        """Test extraction with only CNN features enabled."""
        config = InitialConfig()
        config.manual.enabled = False

        extractor = InitialExtractor(config=config, device=device)

        with torch.no_grad():
            feature_dict = extractor.extract_all(synthetic_ics)

        B, M = synthetic_ics.shape[:2]

        # Manual should be None when disabled
        assert 'manual' not in feature_dict or feature_dict['manual'] is None
        assert feature_dict['cnn'] is not None
        assert feature_dict['cnn'].shape == (B, M, 28)
        assert feature_dict['combined'].shape == (B, M, 28)  # Only CNN

    def test_registry(self, device, ic_config):
        """Test feature registry construction."""
        extractor = InitialExtractor(config=ic_config, device=device)

        registry = extractor.get_feature_registry()

        assert registry.num_features == 42
        assert 'spatial' in registry.categories
        assert 'spectral' in registry.categories
        assert 'information' in registry.categories
        assert 'morphological' in registry.categories
        assert 'cnn_learned' in registry.categories

    def test_vae_mode_generation(self, device, ic_config_vae):
        """Test IC generation from latent codes (VAE mode)."""
        extractor = InitialExtractor(config=ic_config_vae, device=device)

        assert extractor.is_generative, "Extractor should be in generative mode"

        # Generate ICs from random latent codes
        latent_codes = torch.randn(5, 28, device=device)

        with torch.no_grad():
            generated_ics = extractor.generate_ics(
                latent_codes=latent_codes,
                num_realizations=3
            )

        assert generated_ics.shape == (5, 3, 1, 128, 128)

    def test_vae_mode_sampling(self, device, ic_config_vae):
        """Test IC sampling from prior (VAE mode)."""
        extractor = InitialExtractor(config=ic_config_vae, device=device)

        with torch.no_grad():
            sampled_ics = extractor.sample_ics(
                num_operators=4,
                num_realizations=2
            )

        assert sampled_ics.shape == (4, 2, 1, 128, 128)

    def test_encoder_mode_generation_fails(self, device, ic_config):
        """Test that generation fails in encoder-only mode."""
        extractor = InitialExtractor(config=ic_config, device=device)

        assert not extractor.is_generative, "Extractor should not be in generative mode"

        latent_codes = torch.randn(5, 28, device=device)

        with pytest.raises(ValueError, match="requires VAE mode"):
            extractor.generate_ics(latent_codes=latent_codes)


# HDF5 Storage Tests
class TestHDF5Storage:
    """Tests for IC feature HDF5 storage."""

    def test_write_and_read_ic_features(self, device, synthetic_ics, ic_config):
        """Test writing and reading IC features from HDF5."""
        extractor = InitialExtractor(config=ic_config, device=device)

        # Extract features
        with torch.no_grad():
            feature_dict = extractor.extract_all(synthetic_ics)

        # Create temporary HDF5 file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Write features
            writer = HDF5FeatureWriter(dataset_path=temp_path, overwrite=True)

            B, M = synthetic_ics.shape[:2]
            registry = extractor.get_feature_registry()

            with writer.open_for_writing():
                writer.create_ic_group(
                    num_samples=B,
                    num_realizations=M,
                    registry=registry,
                    config=ic_config
                )

                writer.write_ic_batch(
                    batch_idx=0,
                    manual_features=feature_dict['manual'].cpu().numpy(),
                    cnn_features=feature_dict['cnn'].cpu().numpy(),
                    combined_features=feature_dict['combined'].cpu().numpy()
                )

            # Read features back
            with HDF5FeatureReader(dataset_path=temp_path) as reader:
                assert reader.has_ic()

                manual_read = reader.get_ic_manual_features()
                cnn_read = reader.get_ic_cnn_features()
                combined_read = reader.get_ic_combined_features()

                assert manual_read.shape == (B, M, 14)
                assert cnn_read.shape == (B, M, 28)
                assert combined_read.shape == (B, M, 42)

                # Check values match
                np.testing.assert_allclose(
                    manual_read,
                    feature_dict['manual'].cpu().numpy(),
                    rtol=1e-5
                )

        finally:
            # Cleanup
            temp_path.unlink()

    def test_ic_registry_storage(self, device, synthetic_ics, ic_config):
        """Test that IC registry is correctly stored and retrieved."""
        extractor = InitialExtractor(config=ic_config, device=device)
        registry = extractor.get_feature_registry()

        # Create temporary HDF5 file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Write registry
            writer = HDF5FeatureWriter(dataset_path=temp_path, overwrite=True)

            with writer.open_for_writing():
                writer.create_ic_group(
                    num_samples=4,
                    num_realizations=3,
                    registry=registry,
                    config=ic_config
                )

            # Read registry back
            with HDF5FeatureReader(dataset_path=temp_path) as reader:
                registry_read = reader.get_ic_registry()

                assert registry_read is not None
                assert registry_read.num_features == 42
                assert set(registry_read.categories) == set(registry.categories)

        finally:
            temp_path.unlink()


# Integration Tests
class TestICExtractionIntegration:
    """Integration tests for end-to-end IC extraction."""

    def test_end_to_end_extraction(self, device, ic_config):
        """Test complete IC extraction pipeline."""
        # Create synthetic data
        B, M, C, H, W = 10, 5, 1, 128, 128
        ics = torch.randn(B, M, C, H, W)

        # Extract features
        extractor = InitialExtractor(config=ic_config, device=device)

        with torch.no_grad():
            feature_dict = extractor.extract_all(ics)

        # Verify all outputs
        assert feature_dict['manual'].shape == (B, M, 14)
        assert feature_dict['cnn'].shape == (B, M, 28)
        assert feature_dict['combined'].shape == (B, M, 42)

        # Verify no NaN or Inf
        assert not torch.isnan(feature_dict['combined']).any()
        assert not torch.isinf(feature_dict['combined']).any()

        # Verify registry
        registry = extractor.get_feature_registry()
        assert registry.num_features == 42

    def test_extraction_determinism(self, device, ic_config):
        """Test that extraction is deterministic in eval mode."""
        ics = torch.randn(4, 3, 1, 128, 128)

        extractor = InitialExtractor(config=ic_config, device=device)
        extractor.eval()

        with torch.no_grad():
            features1 = extractor.extract_all(ics)
            features2 = extractor.extract_all(ics)

        torch.testing.assert_close(
            features1['combined'],
            features2['combined'],
            atol=1e-6,
            rtol=1e-6
        )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
