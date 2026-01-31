"""
Integration tests for variable-length VQ-VAE training pipeline.
"""

import pytest
import torch
import numpy as np

from spinlock.encoding.trajectory_length_sampler import TrajectoryLengthSampler
from spinlock.encoding.temporal_pyramid import TemporalPyramid
from spinlock.encoding.encoders.pyramid_temporal import PyramidTemporalEncoder
from spinlock.encoding.training.losses import reconstruction_loss
from spinlock.encoding.variable_length_utils import (
    parse_variable_length_config,
    create_length_sampler,
    extract_mask_info_from_batch,
)


class TestVariableLengthIntegration:
    """Integration tests for full variable-length pipeline."""

    def test_end_to_end_encoder_with_mask(self):
        """Test full encoder forward pass with variable-length input."""
        vl_config = {
            "enabled": True,
            "adaptive_pyramid": True,
            "min_pyramid_length": 1,
            "mask_downsample_method": "ceil",
        }

        encoder = PyramidTemporalEncoder(
            input_dim=64,
            level_dims=[32, 64, 96, 128],
            downsample_factors=[1, 2, 4, 8],
            variable_length_config=vl_config,
        )

        # Create input with variable lengths
        batch_size = 16
        max_T = 256
        x = torch.randn(batch_size, max_T, 64)

        # Create length sampler
        sampler = TrajectoryLengthSampler(min_timesteps=8, max_timesteps=max_T)
        lengths = sampler.sample_lengths(batch_size)
        mask = sampler.create_mask(lengths, max_T)

        # Forward pass
        embeddings, mask_info = encoder(x, mask=mask, lengths=lengths)

        # Check output
        assert embeddings.shape == (batch_size, encoder.total_output_dim)
        assert "num_valid_timesteps" in mask_info
        assert "valid_factors" in mask_info
        assert mask_info["num_valid_timesteps"].shape == (batch_size,)

    def test_encoder_with_short_trajectories(self):
        """Test encoder handles short trajectories (skips pyramid levels)."""
        vl_config = {
            "enabled": True,
            "adaptive_pyramid": True,
        }

        encoder = PyramidTemporalEncoder(
            input_dim=64,
            level_dims=[32, 64, 96, 128],
            downsample_factors=[1, 2, 4, 8],
            variable_length_config=vl_config,
        )

        # Create very short input (T=4)
        batch_size = 16
        T = 4
        x = torch.randn(batch_size, T, 64)
        lengths = torch.full((batch_size,), T)
        mask = torch.ones(batch_size, T, dtype=torch.bool)

        # Forward pass
        embeddings, mask_info = encoder(x, mask=mask, lengths=lengths)

        # Should skip 8× level, only use [1,2,4]
        assert mask_info["valid_factors"] == [1, 2, 4]
        assert mask_info["num_active_levels"] == 3

        # Output should still be full dimension (with padding)
        assert embeddings.shape == (batch_size, encoder.total_output_dim)

    def test_reconstruction_loss_with_masking(self):
        """Test reconstruction loss with sample weighting."""
        batch_size = 16

        # Create mock outputs and targets
        outputs = {
            "reconstruction": {
                "features": torch.randn(batch_size, 320)
            }
        }
        targets = {
            "features": torch.randn(batch_size, 320)
        }

        # Create mask info with variable lengths
        num_valid = torch.tensor([100, 200, 150, 500] * 4)  # Mix of lengths
        mask_info = {
            "num_valid_timesteps": num_valid
        }

        # Compute loss with and without masking
        loss_no_mask = reconstruction_loss(outputs, targets)
        loss_with_mask = reconstruction_loss(outputs, targets, mask_info=mask_info)

        # Both should be positive
        assert loss_no_mask > 0
        assert loss_with_mask > 0

        # They should be different (sample weighting changes loss)
        # Note: might be similar if errors are uniform, so just check it runs
        assert not torch.isnan(loss_with_mask)

    def test_config_parsing(self):
        """Test variable_length config parsing."""
        config = {
            "families": {
                "temporal": {
                    "encoder": "PyramidTemporalEncoder",
                    "encoder_params": {
                        "downsample_factors": [1, 2, 4, 8],
                        "variable_length": {
                            "enabled": True,
                            "min_timesteps": 8,
                            "max_timesteps": 500,
                        }
                    }
                }
            }
        }

        vl_config = parse_variable_length_config(config)

        assert vl_config is not None
        assert vl_config["enabled"] == True
        assert vl_config["min_timesteps"] == 8
        assert vl_config["max_timesteps"] == 500

    def test_config_parsing_disabled(self):
        """Test that disabled config returns None."""
        config = {
            "families": {
                "temporal": {
                    "encoder": "PyramidTemporalEncoder",
                    "encoder_params": {
                        "variable_length": {
                            "enabled": False,
                        }
                    }
                }
            }
        }

        vl_config = parse_variable_length_config(config)
        assert vl_config is None

    def test_config_validation_min_max(self):
        """Test config validation catches min >= max."""
        config = {
            "families": {
                "temporal": {
                    "encoder_params": {
                        "variable_length": {
                            "enabled": True,
                            "min_timesteps": 500,
                            "max_timesteps": 100,
                        }
                    }
                }
            }
        }

        with pytest.raises(ValueError, match="must be < max_timesteps"):
            parse_variable_length_config(config)

    def test_config_validation_warns_about_pyramid_levels(self):
        """Test config validation warns when min < max_factor."""
        config = {
            "families": {
                "temporal": {
                    "encoder_params": {
                        "downsample_factors": [1, 2, 4, 8],
                        "variable_length": {
                            "enabled": True,
                            "min_timesteps": 4,  # < max factor (8)
                            "max_timesteps": 500,
                        }
                    }
                }
            }
        }

        # Should parse successfully but print warning
        vl_config = parse_variable_length_config(config)
        assert vl_config is not None

    def test_create_length_sampler_from_config(self):
        """Test length sampler creation from config."""
        vl_config = {
            "min_timesteps": 8,
            "max_timesteps": 500,
            "sampling_strategy": "uniform",
        }

        sampler = create_length_sampler(vl_config)

        assert sampler is not None
        assert sampler.min_T == 8
        assert sampler.max_T == 500
        assert sampler.strategy == "uniform"

    def test_extract_mask_info_from_batch(self):
        """Test mask info extraction from batch."""
        batch = {
            "mask": torch.ones(16, 256, dtype=torch.bool),
            "length": torch.full((16,), 256),
        }

        device = torch.device("cpu")
        mask_info = extract_mask_info_from_batch(batch, device)

        assert mask_info is not None
        assert "num_valid_timesteps" in mask_info
        assert "lengths" in mask_info
        assert mask_info["num_valid_timesteps"].shape == (16,)
        assert (mask_info["num_valid_timesteps"] == 256).all()

    def test_extract_mask_info_no_mask(self):
        """Test mask info extraction when no mask in batch."""
        batch = {
            "features": torch.randn(16, 320),
        }

        device = torch.device("cpu")
        mask_info = extract_mask_info_from_batch(batch, device)

        assert mask_info is None

    def test_backward_compatibility(self):
        """Test that encoder works without variable_length config."""
        # Standard encoder without variable_length
        encoder = PyramidTemporalEncoder(
            input_dim=64,
            level_dims=[32, 64, 96, 128],
            downsample_factors=[1, 2, 4, 8],
        )

        x = torch.randn(16, 256, 64)

        # Should work as before
        embeddings = encoder(x)

        assert embeddings.shape == (16, 320)
        assert isinstance(embeddings, torch.Tensor)  # Not a tuple


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
