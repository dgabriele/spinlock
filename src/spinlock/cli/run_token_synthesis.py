"""
Run token synthesis self-play command for Spinlock CLI.

Launches the explore-refine loop that generates novel token sequences
via diffusion, grounds them in QBM physics, and refines the diffusion
model on high-surprisal cases.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .base import CLICommand


class RunTokenSynthesisCommand(CLICommand):
    """
    Command to run token synthesis self-play pipeline.

    Composes:
    - Discrete diffusion (D3PM) for token generation
    - VQTokenizer for encode/decode
    - QBMReplayer for physics simulation
    - Surprisal scoring and priority queue
    - Explore/refine mode scheduler
    """

    @property
    def name(self) -> str:
        return "run-token-synthesis"

    @property
    def help(self) -> str:
        return "Run token synthesis self-play (explore-refine loop)"

    @property
    def description(self) -> str:
        return """
Run the token synthesis self-play pipeline.

This pipeline autonomously discovers novel token sequences by:
  1. EXPLORE: Generate tokens from diffusion → decode → QBM simulate →
     retokenize → score surprisal → queue high-novelty items
  2. REFINE: Train diffusion model on physically grounded (retokenized)
     tokens from the priority queue

The loop teaches the diffusion model to generate tokens that survive
the physical roundtrip, discovering novel coherent dynamics.

Prerequisites:
  - Trained diffusion model checkpoint (D3PM + DenoisingNetwork)
  - Trained VQTokenizer checkpoint with inverse models (theta + initial)
  - QBM ground truth configuration

Examples:
  # Run with default QBM config
  spinlock run-token-synthesis \\
      --config experiments/token_synthesis/configs/qbm_synthesis.yaml

  # Override cycle count and batch size
  spinlock run-token-synthesis \\
      --config experiments/token_synthesis/configs/qbm_synthesis.yaml \\
      --max-cycles 5 --batch-size 4

  # Use CPU (for testing)
  spinlock run-token-synthesis \\
      --config experiments/token_synthesis/configs/qbm_synthesis.yaml \\
      --device cpu

Output:
  Results saved to output_dir specified in config or --output:
    - checkpoints/denoiser_cycle_N.pt: Per-cycle denoiser checkpoints
    - metrics.json: Full metrics history
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add run-token-synthesis command arguments."""
        parser.add_argument(
            "--config",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to token synthesis config YAML",
        )

        parser.add_argument(
            "--output",
            type=Path,
            metavar="PATH",
            help="Override output directory from config",
        )

        parser.add_argument(
            "--max-cycles",
            type=int,
            metavar="N",
            help="Override maximum number of explore-refine cycles",
        )

        parser.add_argument(
            "--batch-size",
            type=int,
            metavar="N",
            help="Override generation batch size",
        )

        parser.add_argument(
            "--device",
            type=str,
            choices=["cuda", "cpu"],
            help="Override device (cuda/cpu)",
        )

        parser.add_argument(
            "--seed",
            type=int,
            metavar="N",
            help="Override random seed",
        )

        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose logging",
        )

    def execute(self, args: Namespace) -> int:
        """Execute the run-token-synthesis command."""
        import logging
        import yaml

        from spinlock.experimental.token_synthesis.config import TokenSynthesisConfig
        from spinlock.experimental.token_synthesis.pipeline import SynthesisVerificationPipeline

        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        logger = logging.getLogger(__name__)

        # Validate config file
        if not self.validate_file_exists(args.config, "Configuration file"):
            return 1

        # Load config
        logger.info(f"Loading config from {args.config}")
        try:
            with open(args.config) as f:
                config_dict = yaml.safe_load(f)
        except Exception as e:
            return self.error(f"Failed to load config: {e}")

        # Apply CLI overrides
        if args.output:
            config_dict["output_dir"] = str(args.output)
        if args.max_cycles:
            config_dict.setdefault("scheduler", {})["max_cycles"] = args.max_cycles
        if args.batch_size:
            config_dict.setdefault("generation", {})["batch_size"] = args.batch_size
        if args.device:
            config_dict["device"] = args.device
        if args.seed is not None:
            config_dict["seed"] = args.seed

        # Parse config
        try:
            config = TokenSynthesisConfig(**config_dict)
        except Exception as e:
            return self.error(f"Invalid config: {e}")

        # Print summary
        logger.info("=" * 70)
        logger.info("Token Synthesis Self-Play")
        logger.info("=" * 70)
        logger.info(f"Diffusion:  {config.checkpoints.diffusion_checkpoint}")
        logger.info(f"VQ-VAE:     {config.checkpoints.vqvae_checkpoint}")
        logger.info(f"Ground truth: {config.checkpoints.qbm_substrate_config}")
        logger.info(f"Cycles:     {config.scheduler.max_cycles}")
        logger.info(f"Explore:    {config.scheduler.explore_steps} steps/cycle")
        logger.info(f"Refine:     {config.scheduler.refine_epochs} epochs/cycle")
        logger.info(f"Batch size: {config.generation.batch_size}")
        logger.info(f"Device:     {config.device}")
        logger.info(f"Output:     {config.output_dir}")
        logger.info("=" * 70)

        # Run pipeline
        pipeline = SynthesisVerificationPipeline(config)

        try:
            metrics = pipeline.run()
        except KeyboardInterrupt:
            logger.warning("Pipeline interrupted by user")
            return 130
        except Exception as e:
            import traceback
            logger.error(f"Pipeline failed:\n{traceback.format_exc()}")
            return self.error(f"Pipeline failed: {e}")

        # Print results
        print()
        print("=" * 70)
        print("Token Synthesis Complete!")
        print("=" * 70)

        if 'avg_surprisal' in metrics:
            surprisals = metrics['avg_surprisal']
            print(f"Total steps:        {len(surprisals)}")
            print(f"Final surprisal:    {surprisals[-1]:.4f}" if surprisals else "N/A")
            print(f"Mean surprisal:     {sum(surprisals) / len(surprisals):.4f}" if surprisals else "N/A")

        if 'avg_jaccard' in metrics:
            jaccards = metrics['avg_jaccard']
            print(f"Final Jaccard:      {jaccards[-1]:.4f}" if jaccards else "N/A")

        if 'refine_loss' in metrics:
            losses = metrics['refine_loss']
            print(f"Final refine loss:  {losses[-1]:.4f}" if losses else "N/A")

        print(f"Results saved to:   {config.output_dir}")
        print("=" * 70)

        return 0
