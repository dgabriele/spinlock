#!/usr/bin/env python3
"""Generate samples from trained diffusion model.

This script orchestrates the complete sampling pipeline:
1. Sample token sequences from diffusion model
2. Decode tokens to theta parameters and ICs
3. Generate PDE trajectories using CNOReplayer
4. Save outputs and create visualizations
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import torch
import numpy as np

from spinlock.tokens.tokenizer import VQTokenizer
from sampling import (
    DiffusionSampler,
    TrajectoryGenerator,
    SampleWriter,
    SampleVisualizer,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SampleGenerator:
    """Orchestrates sample generation pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = args.device

        # Load models (tokenizer first, diffusion needs it for vocab_sizes)
        self.tokenizer = self._load_tokenizer()
        self.diffusion, self.denoiser = self._load_diffusion_model()
        self.trajectory_gen = self._load_trajectory_generator()

        # Create utilities
        self.sampler = DiffusionSampler(
            self.diffusion, self.denoiser, self.device
        )
        self.writer = SampleWriter(Path(args.output_dir))
        self.visualizer = SampleVisualizer()

    def generate(self):
        """Run complete generation pipeline."""
        logger.info(f"Generating {self.args.num_samples} samples...")

        # 1. Sample tokens
        tokens = self._sample_tokens()
        logger.info(f"✓ Generated {len(tokens)} token sequences")

        # 2. Decode tokens
        theta, u0, temporal = self._decode_tokens(tokens)
        logger.info(f"✓ Decoded tokens (theta: {theta.shape if theta is not None else None})")

        # 3. Generate trajectories (optional)
        trajectories = None
        if self.args.generate_trajectories:
            trajectories = self._generate_trajectories(theta, u0)
            logger.info(f"✓ Generated trajectories: {trajectories.shape}")

        # 4. Save outputs
        self._save_outputs(tokens, theta, u0, trajectories)
        logger.info(f"✓ Saved outputs to {self.args.output_dir}")

        # 5. Create visualizations (optional)
        if self.args.visualize:
            self._create_visualizations(theta, trajectories)
            logger.info(f"✓ Created visualizations")

    def _sample_tokens(self):
        """Sample token sequences."""
        mode = self.args.mode

        if mode == "unconditional":
            return self.sampler.sample_unconditional(
                self.args.num_samples,
                self.args.batch_size
            )
        elif mode == "theta_cond":
            theta_tokens = self._load_condition_tokens()
            return self.sampler.sample_theta_conditioned(
                theta_tokens,
                self.args.num_samples,
                self.args.batch_size
            )
        elif mode == "inpaint":
            observed_tokens, observed_mask = self._load_inpaint_tokens()
            return self.sampler.sample_inpainting(
                observed_tokens,
                observed_mask,
                self.args.num_samples,
                self.args.batch_size
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _decode_tokens(self, tokens):
        """Decode tokens to continuous features."""
        # Stack tokens into batched dict
        batched_tokens = {}
        keys = tokens[0].keys()

        for key in keys:
            token_stack = torch.stack([t[key] for t in tokens])
            batched_tokens[key] = token_stack

        # Decode
        theta, u0_encoded, temporal = self.tokenizer.decode(batched_tokens)

        # u0_encoded is [B, initial_dim], need to convert to [B, C, H, W]
        # For now, use a simple projection/reshape
        # TODO: Add proper inverse decoder for InitialCNN
        if u0_encoded is not None:
            B = u0_encoded.shape[0]
            # Simple approach: create random ICs with correct statistics
            # This is a placeholder until proper inverse decoder is implemented
            # Use 1 channel to match CNO operator expectations
            u0 = torch.randn(B, 1, 64, 64, device=u0_encoded.device) * 0.1
            logger.warning(
                "Using placeholder random ICs (1-channel). "
                "Proper inverse decoder for initial conditions not yet implemented."
            )
        else:
            u0 = None

        # Convert to numpy
        theta_np = theta.cpu().numpy() if theta is not None else None
        u0_np = u0.cpu().numpy() if u0 is not None else None

        return theta_np, u0_np, temporal

    def _generate_trajectories(self, theta, u0):
        """Generate PDE trajectories."""
        theta_torch = torch.from_numpy(theta).float()
        u0_torch = torch.from_numpy(u0).float()

        trajectories = self.trajectory_gen.generate_trajectories(
            theta_torch,
            u0_torch,
            num_steps=self.args.num_timesteps,
            num_realizations=self.args.num_realizations,
            use_mno=self.args.use_mno,
            batch_size=self.args.traj_batch_size
        )

        return trajectories.cpu().numpy()

    def _save_outputs(self, tokens, theta, u0, trajectories):
        """Save all outputs."""
        metadata = self._create_metadata()

        self.writer.write_samples(
            tokens=tokens,
            theta=theta,
            u0=u0,
            trajectories=trajectories if self.args.save_trajectories else None,
            metadata=metadata
        )

    def _create_visualizations(self, theta, trajectories):
        """Create visualization plots."""
        output_dir = Path(self.args.output_dir)

        # Parameter distributions
        if theta is not None:
            self.visualizer.plot_parameter_distribution(
                theta,
                output_dir / "param_distributions.png"
            )

        # Trajectory plots
        if trajectories is not None:
            # Plot first trajectory
            self.visualizer.plot_trajectory(
                trajectories[0],
                theta[0],
                output_dir / "trajectory_example.png"
            )

            # Diversity comparison
            num_samples = min(10, len(trajectories))
            self.visualizer.plot_diversity_comparison(
                [trajectories[i] for i in range(num_samples)],
                output_dir / "diversity_comparison.png",
                num_samples=num_samples
            )

    def _create_metadata(self) -> Dict[str, Any]:
        """Create generation metadata."""
        return {
            "timestamp": datetime.now().isoformat(),
            "num_samples": self.args.num_samples,
            "mode": self.args.mode,
            "diffusion_checkpoint": str(self.args.diffusion_checkpoint),
            "tokenizer_checkpoint": str(self.args.tokenizer_checkpoint),
            "device": self.device,
            "seed": self.args.seed,
            "generate_trajectories": self.args.generate_trajectories,
            "num_timesteps": self.args.num_timesteps,
            "num_realizations": self.args.num_realizations,
        }

    def _load_diffusion_model(self):
        """Load diffusion model and denoiser."""
        import sys
        from pathlib import Path

        # Add experiments/diffusion to path
        diffusion_root = Path(__file__).parent.parent  # experiments/diffusion/
        sys.path.insert(0, str(diffusion_root))

        from models.discrete_d3pm import DiscreteD3PM, DiffusionSchedule
        from models.denoising_network import DenoisingNetwork

        logger.info(f"Loading diffusion model from {self.args.diffusion_checkpoint}")

        checkpoint = torch.load(
            self.args.diffusion_checkpoint,
            map_location='cpu',
            weights_only=False
        )

        # Extract config (pydantic model, convert to dict)
        config_obj = checkpoint['config']
        config = config_obj.model_dump() if hasattr(config_obj, 'model_dump') else config_obj

        # Extract vocab_sizes from the diffusion checkpoint
        # (These are the ACTUAL vocab sizes from the training dataset, not codebook sizes)
        vocab_sizes = {}
        category_level_info = {}

        # Get vocab sizes from diffusion model's transition matrices
        for key in checkpoint['diffusion_state_dict'].keys():
            if key.startswith('transition_matrices.'):
                category_key = key.replace('transition_matrices.', '')
                matrix_shape = checkpoint['diffusion_state_dict'][key].shape
                vocab_size = matrix_shape[1]  # [T, V, V] -> V
                vocab_sizes[category_key] = vocab_size

        # Parse category level info from vocab_sizes keys
        for key in vocab_sizes.keys():
            parts = key.split('_')
            family = parts[0]
            level = int(parts[-1][1:])
            category = '_'.join(parts[1:-1])
            category_level_info[key] = {
                'family': family,
                'category': category,
                'level': level,
            }

        logger.info(f"Detected {len(vocab_sizes)} token categories")

        # Create diffusion schedule from config dict
        diffusion_config = config['diffusion']
        schedule = DiffusionSchedule(
            num_timesteps=diffusion_config['num_timesteps'],
            beta_start=diffusion_config['beta_start'],
            beta_end=diffusion_config['beta_end'],
            schedule_type=diffusion_config['schedule_type']
        )

        # Create diffusion model
        diffusion = DiscreteD3PM(
            vocab_sizes=vocab_sizes,
            schedule=schedule,
            category_level_info=category_level_info,
        )

        # Create denoiser (unpack config['model'] dict)
        denoiser = DenoisingNetwork(
            vocab_sizes=vocab_sizes,
            category_level_info=category_level_info,
            **config['model']
        ).to(self.device)

        # Try to load weights (may fail if vocab sizes changed)
        try:
            diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
            denoiser.load_state_dict(checkpoint['denoiser_state_dict'])
            logger.info("✓ Loaded diffusion weights successfully")
        except RuntimeError as e:
            logger.warning(
                f"⚠ Could not load diffusion weights (vocab size mismatch). "
                f"This usually means the tokenizer was retrained with different codebook sizes. "
                f"Sampling from UNTRAINED model (will generate random tokens)."
            )
            logger.warning(f"Error details: {str(e)[:200]}...")

        return diffusion, denoiser

    def _load_tokenizer(self):
        """Load VQTokenizer."""
        logger.info(f"Loading tokenizer from {self.args.tokenizer_checkpoint}")
        return VQTokenizer.from_checkpoint(self.args.tokenizer_checkpoint)

    def _load_trajectory_generator(self):
        """Load trajectory generator."""
        if not self.args.generate_trajectories:
            return None

        logger.info(f"Loading CNO config from {self.args.cno_config}")
        return TrajectoryGenerator(
            Path(self.args.cno_config),
            device=self.device,
            mno_checkpoint=self.args.mno_checkpoint
        )

    def _load_condition_tokens(self):
        """Load conditioning tokens (for theta_cond mode)."""
        # TODO: Implement loading from tokenized dataset
        raise NotImplementedError("Theta conditioning not yet implemented")

    def _load_inpaint_tokens(self):
        """Load inpainting tokens and mask."""
        # TODO: Implement loading from tokenized dataset
        raise NotImplementedError("Inpainting not yet implemented")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate samples from diffusion model")

    # Model checkpoints
    parser.add_argument(
        "--diffusion-checkpoint", type=Path, required=True,
        help="Path to trained diffusion model checkpoint"
    )
    parser.add_argument(
        "--tokenizer-checkpoint", type=Path, required=True,
        help="Path to trained VQTokenizer checkpoint"
    )
    parser.add_argument(
        "--cno-config", type=Path,
        help="Path to CNO config YAML (for trajectory generation)"
    )
    parser.add_argument(
        "--mno-checkpoint", type=Path,
        help="Path to MNO checkpoint (optional, future)"
    )

    # Sampling parameters
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for diffusion sampling"
    )
    parser.add_argument(
        "--mode", choices=["unconditional", "theta_cond", "inpaint"],
        default="unconditional",
        help="Sampling mode"
    )

    # Trajectory generation
    parser.add_argument(
        "--generate-trajectories", action="store_true",
        help="Generate PDE trajectories"
    )
    parser.add_argument(
        "--num-timesteps", type=int, default=256,
        help="Rollout length"
    )
    parser.add_argument(
        "--num-realizations", type=int, default=1,
        help="Stochastic realizations per sample"
    )
    parser.add_argument(
        "--use-mno", action="store_true",
        help="Use MNO instead of CNOReplayer (requires --mno-checkpoint)"
    )
    parser.add_argument(
        "--traj-batch-size", type=int, default=8,
        help="Batch size for trajectory generation"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--save-trajectories", action="store_true",
        help="Save full trajectories (large files)"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Create visualizations"
    )

    # Device
    parser.add_argument(
        "--device", default="cuda",
        help="Computation device"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Validate arguments
    if args.generate_trajectories and args.cno_config is None:
        parser.error("--cno-config required when --generate-trajectories is set")

    if args.use_mno and args.mno_checkpoint is None:
        parser.error("--mno-checkpoint required when --use-mno is set")

    # Generate samples
    generator = SampleGenerator(args)
    generator.generate()

    logger.info("✓ Sample generation complete!")


if __name__ == "__main__":
    main()
