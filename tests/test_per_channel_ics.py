"""
Unit tests for per-channel independent IC generation.

Tests:
- ChannelICConfig validation and normalization
- ICTypeSampler sampling
- PerChannelICGenerator batch generation
- Verify channels have different patterns
"""

import pytest
import torch
from spinlock.config.schema import ChannelICConfig, InputGenerationConfig
from spinlock.dataset.generators import InputFieldGenerator, PerChannelICGenerator, ICTypeSampler


class TestChannelICConfig:
    """Test ChannelICConfig validation and normalization."""

    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1.0."""
        config = ChannelICConfig(
            ic_type_weights={
                "gaussian_random_field": 2.0,
                "localized": 3.0,
            }
        )

        # Weights should be normalized
        assert abs(sum(config.ic_type_weights.values()) - 1.0) < 1e-6
        assert abs(config.ic_type_weights["gaussian_random_field"] - 0.4) < 1e-6
        assert abs(config.ic_type_weights["localized"] - 0.6) < 1e-6

    def test_empty_weights(self):
        """Test that empty weights are handled."""
        config = ChannelICConfig(ic_type_weights={})
        assert config.ic_type_weights == {}

    def test_ic_params_storage(self):
        """Test that IC-specific parameters can be stored."""
        config = ChannelICConfig(
            ic_type_weights={"gaussian_random_field": 1.0},
        )
        # Add extra attributes (Pydantic allows this with extra='allow')
        config_dict = config.model_dump()
        assert "ic_type_weights" in config_dict


class TestICTypeSampler:
    """Test ICTypeSampler probabilistic sampling."""

    def test_sampling_distribution(self):
        """Test that sampling follows the specified distribution."""
        sampler = ICTypeSampler({
            "gaussian_random_field": 0.7,
            "localized": 0.3,
        })

        # Sample many times and check distribution
        samples = [sampler.sample() for _ in range(1000)]
        grf_count = sum(1 for s in samples if s == "gaussian_random_field")
        localized_count = sum(1 for s in samples if s == "localized")

        # Should be roughly 70/30 split (allow 10% tolerance)
        assert 0.6 < grf_count / len(samples) < 0.8
        assert 0.2 < localized_count / len(samples) < 0.4

    def test_single_type(self):
        """Test sampling with a single IC type."""
        sampler = ICTypeSampler({"gaussian_random_field": 1.0})

        # All samples should be the same type
        samples = [sampler.sample() for _ in range(100)]
        assert all(s == "gaussian_random_field" for s in samples)

    def test_zero_weights(self):
        """Test handling of zero weights (should use uniform)."""
        sampler = ICTypeSampler({
            "gaussian_random_field": 0.0,
            "localized": 0.0,
        })

        # Should still sample (uniform distribution)
        samples = [sampler.sample() for _ in range(100)]
        assert len(samples) == 100


class TestPerChannelICGenerator:
    """Test PerChannelICGenerator batch generation."""

    @pytest.fixture
    def input_generator(self):
        """Create a basic InputFieldGenerator."""
        return InputFieldGenerator(
            grid_size=64,
            num_channels=3,
            device=torch.device("cpu")
        )

    @pytest.fixture
    def channel_configs(self):
        """Create test channel configurations."""
        return {
            "channel_0": ChannelICConfig(
                ic_type_weights={"gaussian_random_field": 1.0},
            ).model_dump() | {"gaussian_random_field": {"length_scale": 0.05}},
            "channel_1": ChannelICConfig(
                ic_type_weights={"localized": 1.0},
            ).model_dump() | {"localized": {"num_blobs": 5}},
            "channel_2": ChannelICConfig(
                ic_type_weights={"gaussian_random_field": 1.0},
            ).model_dump() | {"gaussian_random_field": {"length_scale": 0.2}},
        }

    def test_generate_batch_shape(self, input_generator, channel_configs):
        """Test that generated batch has correct shape."""
        generator = PerChannelICGenerator(
            input_generator=input_generator,
            channel_configs=channel_configs,
            num_channels=3,
        )

        batch_size = 10
        fields, metadata = generator.generate_batch(batch_size)

        # Check shape
        assert fields.shape == (10, 3, 64, 64)
        assert len(metadata) == 10

    def test_channels_are_different(self, input_generator, channel_configs):
        """Test that channels have different statistical properties."""
        generator = PerChannelICGenerator(
            input_generator=input_generator,
            channel_configs=channel_configs,
            num_channels=3,
        )

        fields, _ = generator.generate_batch(20)

        # Channels should not be identical
        ch0 = fields[:, 0, :, :]
        ch1 = fields[:, 1, :, :]
        ch2 = fields[:, 2, :, :]

        # Check that channels are not all close
        assert not torch.allclose(ch0, ch1, rtol=0.1)
        assert not torch.allclose(ch1, ch2, rtol=0.1)

        # Check that different samples have different patterns
        assert not torch.allclose(fields[0], fields[1], rtol=0.1)

    def test_metadata_correctness(self, input_generator, channel_configs):
        """Test that metadata correctly describes IC types."""
        generator = PerChannelICGenerator(
            input_generator=input_generator,
            channel_configs=channel_configs,
            num_channels=3,
        )

        _, metadata = generator.generate_batch(5)

        # Check metadata structure
        for meta in metadata:
            assert "channel_0" in meta
            assert "channel_1" in meta
            assert "channel_2" in meta

            # Check IC types match config
            assert meta["channel_0"] == "gaussian_random_field"
            assert meta["channel_1"] == "localized"
            assert meta["channel_2"] == "gaussian_random_field"

    def test_mixed_ic_types(self, input_generator):
        """Test generation with mixed IC types per channel."""
        channel_configs = {
            "channel_0": ChannelICConfig(
                ic_type_weights={
                    "gaussian_random_field": 0.5,
                    "localized": 0.5,
                }
            ).model_dump() | {
                "gaussian_random_field": {"length_scale": 0.1},
                "localized": {"num_blobs": 3},
            },
            "channel_1": ChannelICConfig(
                ic_type_weights={"gaussian_random_field": 1.0},
            ).model_dump() | {"gaussian_random_field": {"length_scale": 0.15}},
            "channel_2": ChannelICConfig(
                ic_type_weights={"gaussian_random_field": 1.0},
            ).model_dump() | {"gaussian_random_field": {"length_scale": 0.2}},
        }

        generator = PerChannelICGenerator(
            input_generator=input_generator,
            channel_configs=channel_configs,
            num_channels=3,
        )

        fields, metadata = generator.generate_batch(50)

        # Check that channel 0 has mixed IC types in metadata
        ch0_types = [meta["channel_0"] for meta in metadata]
        assert "gaussian_random_field" in ch0_types
        assert "localized" in ch0_types

        # Check that channels 1 and 2 are always gaussian_random_field
        assert all(meta["channel_1"] == "gaussian_random_field" for meta in metadata)
        assert all(meta["channel_2"] == "gaussian_random_field" for meta in metadata)

    def test_zero_batch_size(self, input_generator, channel_configs):
        """Test handling of zero batch size."""
        generator = PerChannelICGenerator(
            input_generator=input_generator,
            channel_configs=channel_configs,
            num_channels=3,
        )

        fields, metadata = generator.generate_batch(0)

        # Should return empty tensors
        assert fields.shape == (0, 3, 64, 64)
        assert len(metadata) == 0

    def test_batching_efficiency(self, input_generator):
        """Test that batching groups samples by IC type efficiently."""
        # All channels use the same IC type
        channel_configs = {
            "channel_0": ChannelICConfig(
                ic_type_weights={"gaussian_random_field": 1.0},
            ).model_dump() | {"gaussian_random_field": {"length_scale": 0.1}},
            "channel_1": ChannelICConfig(
                ic_type_weights={"gaussian_random_field": 1.0},
            ).model_dump() | {"gaussian_random_field": {"length_scale": 0.1}},
            "channel_2": ChannelICConfig(
                ic_type_weights={"gaussian_random_field": 1.0},
            ).model_dump() | {"gaussian_random_field": {"length_scale": 0.1}},
        }

        generator = PerChannelICGenerator(
            input_generator=input_generator,
            channel_configs=channel_configs,
            num_channels=3,
        )

        # Should generate without errors
        fields, _ = generator.generate_batch(100)
        assert fields.shape == (100, 3, 64, 64)


class TestInputGenerationConfigIntegration:
    """Test integration with InputGenerationConfig."""

    def test_config_parsing(self):
        """Test that per-channel configs parse correctly."""
        config_dict = {
            "method": "per_channel",
            "channel_configs": {
                "channel_0": {
                    "ic_type_weights": {
                        "gaussian_random_field": 0.5,
                        "localized": 0.5,
                    },
                    "gaussian_random_field": {"length_scale": 0.05},
                    "localized": {"num_blobs": 5},
                },
                "channel_1": {
                    "ic_type_weights": {"gaussian_random_field": 1.0},
                    "gaussian_random_field": {"length_scale": 0.1},
                },
                "channel_2": {
                    "ic_type_weights": {"gaussian_random_field": 1.0},
                    "gaussian_random_field": {"length_scale": 0.2},
                },
            },
        }

        config = InputGenerationConfig(**config_dict)

        # Check that config was created
        assert config.method == "per_channel"
        assert len(config.channel_configs) == 3
        assert "channel_0" in config.channel_configs

        # Check weight normalization
        ch0_weights = config.channel_configs["channel_0"].ic_type_weights
        assert abs(sum(ch0_weights.values()) - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
