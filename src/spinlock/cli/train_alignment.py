"""
Train alignment CLI command for Spinlock.

Launches autonomous self-play to bootstrap the Rosetta contrastive
alignment (token space ↔ English space) without human interaction.
The LLM generates diverse exploration queries, the pipeline explores
them headlessly, RLAIF scores results, and the alignment model trains
on high-quality (token, description) pairs.

After this pretraining, the human chat interface benefits from a
meaningful alignment from day one.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .base import CLICommand


class TrainAlignmentCommand(CLICommand):
    """Command to run autonomous Rosetta alignment training."""

    @property
    def name(self) -> str:
        return "train-alignment"

    @property
    def help(self) -> str:
        return "Bootstrap Rosetta alignment via autonomous self-play"

    @property
    def description(self) -> str:
        return """
Bootstrap the Rosetta contrastive alignment (token ↔ English) via
autonomous self-play, without human interaction.

The LLM generates diverse exploration queries across 5 categories
(physics-targeted, abstract-physical, philosophical, consciousness,
contrastive). For each query, the pipeline generates candidates via
diffusion, simulates them via QBM, scores via RLAIF, and trains the
alignment model on high-quality (token, description) pairs.

Prerequisites:
  - Trained diffusion model checkpoint (D3PM + DenoisingNetwork)
  - Trained VQTokenizer checkpoint with inverse models
  - QBM substrate configuration
  - LLM dependencies: poetry install --with llm

Examples:
  # Default run (50 cycles, all categories, swap mode)
  spinlock train-alignment \\
      --synthesis-config experiments/token_synthesis/configs/qbm_synthesis.yaml

  # Quick test (2 cycles)
  spinlock train-alignment \\
      --synthesis-config experiments/token_synthesis/configs/qbm_synthesis.yaml \\
      --num-cycles 2 --queries-per-cycle 3 --candidates-per-query 4 --verbose

  # Consciousness + philosophy only
  spinlock train-alignment \\
      --synthesis-config experiments/token_synthesis/configs/qbm_synthesis.yaml \\
      --categories consciousness philosophical --num-cycles 20

Output:
  Results saved to --output directory:
    - checkpoints/alignment_cycle_N.pt: Alignment model checkpoints
    - checkpoints/query_state_cycle_N.json: Query generator state
    - metrics.json: Full per-cycle metrics history
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add train-alignment command arguments."""
        parser.add_argument(
            "--synthesis-config",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to token synthesis config YAML (pipeline config)",
        )

        parser.add_argument(
            "--output",
            type=Path,
            default=Path("experiments/alignment/results"),
            metavar="PATH",
            help="Output directory for checkpoints and metrics",
        )

        parser.add_argument(
            "--num-cycles",
            type=int,
            default=50,
            metavar="N",
            help="Number of DISCOVER/LEARN cycles (default: 50)",
        )

        parser.add_argument(
            "--queries-per-cycle",
            type=int,
            default=10,
            metavar="N",
            help="Queries generated per cycle (default: 10)",
        )

        parser.add_argument(
            "--candidates-per-query",
            type=int,
            default=16,
            metavar="N",
            help="Exploration candidates per query (default: 16)",
        )

        parser.add_argument(
            "--categories",
            type=str,
            nargs="+",
            metavar="CAT",
            help="Category filter (default: all). "
            "Options: physics_targeted, abstract_physical, philosophical, "
            "consciousness, contrastive",
        )

        parser.add_argument(
            "--memory-mode",
            type=str,
            choices=["resident", "swap"],
            default="swap",
            help="GPU memory strategy (default: swap)",
        )

        parser.add_argument(
            "--seed",
            type=int,
            metavar="N",
            help="Random seed",
        )

        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable DEBUG logging",
        )

    def execute(self, args: Namespace) -> int:
        """Execute the train-alignment command."""
        import logging

        import yaml

        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        log = logging.getLogger(__name__)

        # Validate config file
        if not self.validate_file_exists(args.synthesis_config, "Synthesis config"):
            return 1

        # Load synthesis config
        log.info(f"Loading synthesis config from {args.synthesis_config}")
        try:
            with open(args.synthesis_config) as f:
                config_dict = yaml.safe_load(f)
        except Exception as e:
            return self.error(f"Failed to load config: {e}")

        # Set seed if provided
        if args.seed is not None:
            import random

            import numpy as np
            import torch

            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            config_dict["seed"] = args.seed

        # Parse synthesis config
        try:
            from spinlock.experimental.token_synthesis.config import (
                TokenSynthesisConfig,
            )

            synthesis_config = TokenSynthesisConfig(**config_dict)
        except Exception as e:
            return self.error(f"Invalid synthesis config: {e}")

        # Build agent config
        from spinlock.experimental.llm_agent.config import (
            AgentConfig,
            AutonomousTrainingConfig,
        )

        agent_config = AgentConfig(
            memory_mode=args.memory_mode,
            alignment={"description_mode": "template"},
            evaluator={"enabled": True},
        )

        # Print banner
        log.info("=" * 70)
        log.info("Autonomous Rosetta Alignment Training")
        log.info("=" * 70)
        log.info(f"Synthesis config: {args.synthesis_config}")
        log.info(f"Memory mode:      {args.memory_mode}")
        log.info(f"Cycles:           {args.num_cycles}")
        log.info(f"Queries/cycle:    {args.queries_per_cycle}")
        log.info(f"Candidates/query: {args.candidates_per_query}")
        log.info(f"Categories:       {args.categories or 'all'}")
        log.info(f"Output:           {args.output}")
        log.info("=" * 70)

        # Initialize pipeline
        try:
            from spinlock.experimental.token_synthesis.pipeline import (
                SynthesisVerificationPipeline,
            )
            from spinlock.experimental.token_synthesis.scheduler import Mode

            pipeline = SynthesisVerificationPipeline(synthesis_config)
            pipeline.initialize()
            # Move models to GPU — initialize() loads on CPU,
            # _enter_mode(EXPLORE) moves denoiser + tokenizer to GPU
            pipeline._enter_mode(Mode.EXPLORE)
        except Exception as e:
            import traceback

            log.error(f"Pipeline init failed:\n{traceback.format_exc()}")
            return self.error(f"Pipeline initialization failed: {e}")

        # Initialize agent
        try:
            from spinlock.experimental.llm_agent.agent import ConversationalAgent

            agent = ConversationalAgent(agent_config, pipeline)
            agent.initialize()
        except ImportError as e:
            return self.error(
                f"LLM dependencies missing: {e}\n"
                "Install with: poetry install --with llm"
            )
        except Exception as e:
            import traceback

            log.error(f"Agent init failed:\n{traceback.format_exc()}")
            return self.error(f"Agent initialization failed: {e}")

        # Build training config
        training_config = AutonomousTrainingConfig(
            num_cycles=args.num_cycles,
            queries_per_cycle=args.queries_per_cycle,
            candidates_per_query=args.candidates_per_query,
            enabled_categories=args.categories,
            output_dir=args.output,
        )

        # Run autonomous training
        try:
            from spinlock.experimental.llm_agent.autonomous_trainer import (
                AutonomousAlignmentTrainer,
            )

            trainer = AutonomousAlignmentTrainer(agent, training_config)
            metrics = trainer.run()
        except KeyboardInterrupt:
            log.warning("Training interrupted by user")
            print("\nTraining interrupted. Partial results saved.")
            return 130
        except Exception as e:
            import traceback

            log.error(f"Training failed:\n{traceback.format_exc()}")
            return self.error(f"Training failed: {e}")

        # Print summary
        print()
        print("=" * 70)
        print("Autonomous Alignment Training Complete!")
        print("=" * 70)

        if "alignment_loss" in metrics:
            losses = [v for v in metrics["alignment_loss"] if v is not None]
            if losses:
                print(f"Final loss:        {losses[-1]:.4f}")
                print(f"Best loss:         {min(losses):.4f}")

        if "alignment_acc_t2e" in metrics:
            accs = [v for v in metrics["alignment_acc_t2e"] if v is not None]
            if accs:
                print(f"Final acc (t→e):   {accs[-1]:.3f}")

        if "alignment_acc_e2t" in metrics:
            accs = [v for v in metrics["alignment_acc_e2t"] if v is not None]
            if accs:
                print(f"Final acc (e→t):   {accs[-1]:.3f}")

        if "pairs_accumulated" in metrics:
            total_pairs = sum(
                v for v in metrics["pairs_accumulated"] if v is not None
            )
            print(f"Total pairs:       {total_pairs}")

        if "matches_found" in metrics:
            total_matches = sum(
                v for v in metrics["matches_found"] if v is not None
            )
            print(f"Total matches:     {total_matches}")

        print(f"Results saved to:  {args.output}")
        print("=" * 70)

        return 0
